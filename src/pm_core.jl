using RecipesBase
using Printf
using Random
using Flux
using Statistics
using UnderwaterAcoustics: RayArrival, location

# Core abstract types used across propagation models
abstract type DataDrivenUnderwaterEnvironment end
abstract type DataDrivenPropagationModel{T} end


export DataDrivenUnderwaterEnvironment, fit!, transfercoef, transmission_loss, check, plot, rays, eigenrays, arrivals, plane_wave_propagate, spherical_wave_propagate
export PlaneWaveCurvModel, calculate_field

# src/physics.jl

"""
    spherical_wave_propagate(rx_x, rx_y, sx, sy, k, A, phi)

Calculates complex pressure at a single receiver from a single source.
Differentiable by Zygote.
"""
function spherical_wave_propagate(rx_x, rx_y, sx, sy, k, A, phi)
    # 1. Distance
    dx = rx_x - sx
    dy = rx_y - sy
    r = sqrt(dx^2 + dy^2)

    # 2. Singularity guard (Zygote-friendly softplus or max)
    r_safe = max(r, 1e-6)

    # 3. Physics (1/r decay + Phase)
    amp = A / r_safe
    phase = (k * r_safe) + phi

    return amp * cis(phase) # cis(x) is exp(im*x)
end

function plane_wave_propagate(x, y, k, A, phi, theta, d)
    # 1. Project receiver location onto the ray direction vector (Longitudinal)
    # This matches Scenario 1 & 2 in your test
    r_long = x * cos(theta) + y * sin(theta)

    # 2. Project receiver location onto the perpendicular vector (Transverse)
    # This is for the curvature (d) term later
    r_trans = -x * sin(theta) + y * cos(theta)

    # 3. Calculate Phase
    # Plane wave term + Curvature term + Phase offset
    total_phase = (k * r_long) + ((k * r_trans^2) / (2 * d)) + phi

    # 4. Return Complex Pressure
    return A * exp(im * total_phase)
end


# src/pm_case1.jl


# --- Case 1: Far-Field / Plane Wave Model ---
# "I don't know where the source is, but I can hear it."
# We learn the Angle (theta) and the Curvature (d) directly.

mutable struct PlaneWaveCurvModel{T, E} <: DataDrivenPropagationModel{T}
    env::E            # The environment (can be Missing initially)
    nrays::Int        # Number of "Neurons" in our RBNN

    # --- Trainable Parameters ---
    # These are the "Weights" and "Biases" of the neural network
    A::Vector{T}      # Amplitude (Linear weight)
    phi::Vector{T}    # Phase offset (Bias)
    theta::Vector{T}  # Direction of Arrival (Non-linear parameter)
    d::Vector{T}      # Curvature Distance (Non-linear parameter)
end

Flux.@functor PlaneWaveCurvModel (A, phi, theta, d)

# --- The Constructor ---
# Initializes random rays to cover the whole horizon
# In PlaneWaveCurvModel constructor:
function PlaneWaveCurvModel(env, nrays::Int)
    # Default to Float64 parameters
    T = Float64

    # Simple random initialization
    return PlaneWaveCurvModel(
        env, nrays,
        randn(T, nrays) .* 0.01,       # A: Small random amplitudes
        zeros(T, nrays),               # phi: Start at zero
        rand(T, nrays) .* 2π,          # theta: Full circle coverage
        fill(T(1000.0), nrays)         # d: Initial curvature
    )
end

"""
$(TYPEDEF)
Create an underwater environment for data-driven physics-based propagation models by providing locations, acoustic measnreuments and other known environmental and channel geomtry knowledge.

- `locations`: location measurements (in the form of matrix with dimension [dimension of a single location data x number of data points])
- `measurements`: acoustic field measurements (in the form of matrix with dimension [1 x number of data points])
- `soundspeed`: medium sound speed (default: missing)
- `frequency`: source frequency (default: missing)
- `waterdepth`: water depth (default: missing)
- `salinity`: water salinity (default: 35)
- `surface`: surface property (default: PressureReleaseBoundary)
- `seabed`: seabed property (default: SandySilt)
- `tx`: source location (default: missing)
- set `dB` to `false` if `measurements` are not in dB scale (default: `true`)
"""

mutable struct BasicDataDrivenUnderwaterEnvironment{T_Loc, T_Meas, T2, T3, T4, T5, T6, T7} <: DataDrivenUnderwaterEnvironment
    # SPLIT HERE: distinct types for locations vs measurements
    locations::Union{T_Loc, Missing}
    measurements::Union{T_Meas, Missing}

    soundspeed::Union{T2, Missing}
    frequency::Union{T3, Missing}
    waterdepth::Union{T4, Missing}
    salinity::Union{Real, Missing}
    surface::Union{T5, Missing}
    seabed::Union{T6, Missing}
    tx::Union{T7, Missing}
    dB::Bool

    # Inner Constructor
    function BasicDataDrivenUnderwaterEnvironment(
        locations,
        measurements;
        soundspeed = missing,
        frequency = missing,
        waterdepth = missing,
        salinity = 35.0,
        surface = UnderwaterAcoustics.PressureReleaseBoundary,
        seabed = UnderwaterAcoustics.SandyMud,
        tx = missing,
        dB = true
    )
        # Type deduction happens automatically here based on inputs
        new{typeof(locations), typeof(measurements), typeof(soundspeed), typeof(frequency),
            typeof(waterdepth), typeof(surface), typeof(seabed), typeof(tx)}(
            locations, measurements, soundspeed, frequency, waterdepth, salinity, surface, seabed, tx, dB
        )
    end
end

DataDrivenUnderwaterEnvironment(locations, measurements; kwargs...) = BasicDataDrivenUnderwaterEnvironment(locations, measurements; kwargs...)

"""
$(SIGNATURES)
Create a lightweight data-driven environment without upfront measurements.
Intended for far-field 2D use where training data are provided directly to `fit!`.
"""
function BasicDataDrivenUnderwaterEnvironment(;
    soundspeed = missing,
    frequency = missing,
    waterdepth = missing,
    salinity = 35.0,
    surface = UnderwaterAcoustics.PressureReleaseBoundary,
    seabed = UnderwaterAcoustics.SandyMud,
    tx = missing,
    dB = true,
    dims::Int = 2)

    # 1. Create placeholders
    # Locations are Real (Float32)
    locations = zeros(Float32, dims, 0)

    # Measurements should be COMPLEX (ComplexF32) to prevent future type errors
    measurements = zeros(ComplexF32, 1, 0)

    # 2. Call the main constructor
    return BasicDataDrivenUnderwaterEnvironment(
        locations, measurements;
        soundspeed=soundspeed, frequency=frequency, waterdepth=waterdepth,
        salinity=salinity, surface=surface, seabed=seabed, tx=tx, dB=dB
    )
end


"""
$(SIGNATURES)
Train data-driven physics-based propagation model.

- `ini_lr`: initial learning rate
- `trainloss`: loss function used in training and model update
- `dataloss`: data loss function to calculate benchmarking validation error for early stopping
- `ratioₜ`: data split ratio = number of training data/(number of training data + number of validation data)
- set `seed` to `true` to seed random data selection order
- `maxepoch`: maximum number of training epoches allowed
- `ncount`: maximum number of tries before reducing learning rate
-  model training ends once learning rate is smaller than `minlearnrate`
- learning rate is reduced by `reducedlearnrate` once `ncount` is reached
- set `showloss` to true to display training and validation errors during the model training process, if the validation error is historically the best
"""
function ModelFit!(r::DataDrivenPropagationModel, inilearnrate, trainloss, dataloss, ratioₜ, seed, maxepoch, ncount, minlearnrate, reducedlearnrate, showloss)
    rₜ, pₜ, rᵥ, pᵥ = SplitData(r.env.locations, r.env.measurements, ratioₜ, seed)
    bestmodel = deepcopy(Flux.params(r))
    count = 0
    opt = Adam(inilearnrate)
    epoch = 0
    bestloss = dataloss(rᵥ, pᵥ, r)
    while true
        Flux.train!((x,y) -> trainloss(x, y, r), Flux.params(r), [(rₜ, pₜ)], opt)
        tmploss = dataloss(rᵥ, pᵥ, r)
        epoch += 1
        if tmploss < bestloss
            bestloss = tmploss
            # bestmodel = deepcopy(Flux.params(r))
            bestmodel = r
            count = 0
            showloss && (@show epoch, dataloss(rₜ, pₜ, r), dataloss(rᵥ, pᵥ, r))
        else
            count += 1
        end
        epoch > maxepoch && break
        if count > ncount
            count = 0
            # Flux.loadparams!(r, bestmodel)
            Flux.loadmodel!(r, bestmodel)
            opt.eta /= reducedlearnrate
            opt.eta < minlearnrate && break
            showloss && println("********* reduced learning rate: ",opt.eta, " *********" )
        end
    end
end

# -------------------------------------------------------------------------
# Bare Minimum Fix: Renaming functions to v0.4+ API
# -------------------------------------------------------------------------

# 1. RENAME: transfercoef -> acoustic_field
function UnderwaterAcoustics.acoustic_field(model::DataDrivenPropagationModel, tx::Union{Missing, Nothing, AcousticSource}, rx::AcousticReceiver; mode=:coherent) where {T1}
    mode === :coherent || throw(ArgumentError("Unsupported mode :" * string(mode)))
    if tx !== nothing &&  tx !== missing
        model.env.frequency == nominalfrequency(tx) || throw(ArgumentError("Mismatched frequencies in acoustic source and data driven environment"))
        if  model.env.tx !== missing
            location(model.env.tx) == location(tx) || throw(ArgumentError("Mismatched location in acoustic source and data driven environment"))
        else
            @warn "Source location is ignored in field calculation"
        end
    end
    if model isa GPR
        if model.twoDimension == true
            p = model.calculatefield(model, hcat([location(rx)[1], location(rx)[end]]))[1]
        else
            p = model.calculatefield(model, hcat([location(rx)[1], location(rx)[2], location(rx)[end]]))[1]
        end
        model.env.dB == true ? (return db2amp.(-p)) : (return p)
    else
        p = model.calculatefield(model, collect(location(rx)))[1]
    end
    return p
end

# 2. RENAME: transfercoef -> acoustic_field
function UnderwaterAcoustics.acoustic_field(model::DataDrivenPropagationModel, tx::Union{Missing, Nothing, AcousticSource}, rx::AcousticReceiverGrid2D; mode=:coherent) where {T1}
    mode === :coherent || throw(ArgumentError("Unsupported mode :" * string(mode)))
    if tx !== nothing &&  tx !== missing
        model.env.frequency == nominalfrequency(tx) || throw(ArgumentError("Mismatched frequencies in acoustic source and data driven environment"))
        if  model.env.tx !== missing
            location(model.env.tx) == location(tx) || throw(ArgumentError("Mismatched location in acoustic source and data driven environment"))
        else
            @warn "Source location is ignored in field calculation"
        end
    end
    (xlen, ylen) = size(rx)
    x = vec(location.(rx))
    p = reshape(model.calculatefield(model, hcat(first.(x), last.(x))'), xlen, ylen)
    if model isa GPR
        model.env.dB == true ? (return db2amp.(-p)) : (return p)
    else
        return p
    end
end

# 3. RENAME: transfercoef -> acoustic_field
function UnderwaterAcoustics.acoustic_field(model::DataDrivenPropagationModel, tx::Union{Missing, Nothing, AcousticSource}, rx::AcousticReceiverGrid3D; mode=:coherent) where {T1}
    mode === :coherent || throw(ArgumentError("Unsupported mode :" * string(mode)))
    if tx !== nothing &&  tx !== missing
        model.env.frequency == nominalfrequency(tx) || throw(ArgumentError("Mismatched frequencies in acoustic source and data driven environment"))
        if  model.env.tx !== missing
            location(model.env.tx) == location(tx) ||  throw(ArgumentError("Mismatched location in acoustic source and data driven environment"))
        else
            @warn "Source location is ignored in field calculation"
        end
    end
    (xlen, ylen, zlen) = size(rx)
    x = vec(location.(rx))
    if ylen == 1
        p = reshape(model.calculatefield(model, hcat(first.(x), getfield.(x, 2), last.(x))'), xlen, zlen)
    else
        p = reshape(model.calculatefield(model, hcat(first.(x), getfield.(x, 2), last.(x))'), xlen, ylen, zlen)
    end
    if model isa GPR
        model.env.dB == true ? (return db2amp.(-p)) : (return p)
    else
        return p
    end
end

# 4. UPDATE CALLS INSIDE ALIASES
UnderwaterAcoustics.acoustic_field(model::DataDrivenPropagationModel, tx::Union{Missing, Nothing, AcousticSource}, rx::AbstractArray{<:AcousticReceiver}) = UnderwaterAcoustics.tmap(rx1 -> acoustic_field(model, tx, rx1), rx)

UnderwaterAcoustics.acoustic_field(model::DataDrivenPropagationModel, rx::Union{AbstractVector, AbstractMatrix}) = model.calculatefield(model, rx)


# 5. RENAME: transmissionloss -> transmission_loss
# Also updated the internal call to use 'acoustic_field' instead of 'transfercoef'
UnderwaterAcoustics.transmission_loss(model::DataDrivenPropagationModel, rx::Union{AbstractVector, AbstractMatrix}) = -amp2db.(abs.(acoustic_field(model, rx)))

UnderwaterAcoustics.transmission_loss(model::DataDrivenPropagationModel, tx::Union{Missing, Nothing, AcousticSource}, rx::Union{AbstractVector, AbstractMatrix}) = -amp2db.(abs.(acoustic_field(model, tx, rx)))


# multiple dispatch; specifically for PLaneWaveCUrvModel
#
function UnderwaterAcoustics.arrivals(model::PlaneWaveCurvModel, tx, rx)
    # 1. Extract Geometry from the Receiver object
    # rx location is [x, y, z]. We need range r and depth z.
    loc = location(rx)
    r_val = sqrt(loc[1]^2 + loc[2]^2)
    z_val = loc[3]

    # 2. Get Physics Constants from the stored environment
    # Note: We assume model.env was initialized with these properties
    f = model.env.frequency
    c = model.env.soundspeed
    k = 2π * f / c

    # 3. Create a list of "Rays" (one per neuron)
    results = RayArrival[]

    for i in 1:model.nrays
        # --- REPLICATE THE PHYSICS KERNEL ---
        # We calculate exactly how much pressure this specific neuron/ray
        # contributes to the receiver location.

        r_local = r_val - 1000.0 # Center offset (matches calculate_field logic)
        θ = model.theta[i]
        d = model.d[i]
        A = model.A[i]
        ϕ = model.phi[i]

        # Calculate Phase (Plane Wave + Curvature)
        phase_plane = k * (r_local * cos(θ) + z_val * sin(θ))
        phase_curv  = (k * z_val^2) / (2 * d)

        # The Complex Pressure Contribution (The Phasor)
        p_contribution = A * cis(phase_plane + phase_curv + ϕ)

        # --- CONSTRUCT THE RAY OBJECT ---
        # Arguments: (time, phasor, launch_angle, arrival_angle, surface_bounces, bottom_bounces)
        # We set time=0 and bounces=0 because RBNNs abstract these away.
        push!(results, RayArrival(0.0, p_contribution, 0.0, θ, 0, 0))
    end

    return results
end

# 7. COMMENT OUT CONFLICTING TYPES
# UnderwaterAcoustics already defines Arrival, so we comment this out to avoid a crash.
# abstract type Arrival end

# function Base.show(io::IO, a::Arrival)
#     if a.time === missing
#         @printf(io, "                         |          | %5.1f dB ϕ%6.1f°", amp2db(abs(a.phasor)), rad2deg(angle(a.phasor)))
#     else
#         @printf(io, "                         | %6.2f ms | %5.1f dB ϕ%6.1f°", 1000*a.time, amp2db(abs(a.phasor)), rad2deg(angle(a.phasor)))
#     end
# end


#= struct DataDrivenArrival{T1,T2} <: UnderwaterAcoustics.Arrival
    time::T1
    phasor::T2
    surface::Missing
    bottom::Missing
    launchangle::Missing
    arrivalangle::Missing
    raypath::Missing
end =#


"""
$(SIGNATURES)
Show arrival rays at a location `rx` using a data-driven physics-based propagation model.

- `model`: data-driven physics-based propagation model
- `tx`: acoustic source. This is optional. Use `missing` or `nothing` for unknown source.
- `rx`: an acoustic receiver
"""
function UnderwaterAcoustics.arrivals(model::DataDrivenPropagationModel, tx::Union{Missing, Nothing, AcousticSource}, rx::Union{AbstractVector, AcousticReceiver}; threshold = 30)
    model isa GPR && throw(ArgumentError("GPR model does not support this function"))

    # Calculate the raw field
    arrival_field = model.calculatefield(model, collect(location(rx)); showarrivals = true)

    # Filter significant arrivals
    amp = amp2db.(abs.(arrival_field))
    idx = findall(amp .> (maximum(amp) - threshold))
    significant_arrivals = arrival_field[idx]

    # Sort by amplitude (strongest first)
    sorted_indices = sortperm(abs.(significant_arrivals), rev = true)
    final_indices = idx[sorted_indices]

    # Construct the result using YOUR custom struct
    # Note: We calculate time delays based on the model type if available
    results = map(1:length(final_indices)) do i
        k = final_indices[i] # Original index in the buffer
        phasor = arrival_field[k]

        # Calculate time based on model type (assuming model.d exists)
        if model isa RayBasis2D || model isa RayBasis2DCurv
            t = missing
        elseif model isa RayBasis3DRCNN
             t = model.d[k] ./ model.env.soundspeed
        else
             # Assuming standard RayBasisNN or similar
             t = (model.d[k] .+ model.ed[k]) ./ model.env.soundspeed
        end

        # Create your custom object
        DataDrivenArrival(t, phasor, missing, missing, missing, missing, missing)
    end

    return results
end

UnderwaterAcoustics.arrivals(model::DataDrivenPropagationModel, rx::Union{AbstractVector, AcousticReceiver}) =
    UnderwaterAcoustics.arrivals(model, nothing, rx)

@recipe function plot(env::DataDrivenUnderwaterEnvironment; receivers = [], transmission_loss = [], transmissionloss = missing, dynamicrange = 42.0)
    if transmissionloss !== missing && isempty(transmission_loss)
        transmission_loss = transmissionloss
    end
    size(transmission_loss) == size(receivers) || throw(ArgumentError("Mismatched receivers and transmission_loss"))
    receivers isa AcousticReceiverGrid2D || throw(ArgumentError("Receivers must be an instance of AcousticReceiverGrid2D"))
    minloss = minimum(transmission_loss)
    clims --> (-minloss-dynamicrange, -minloss)
    colorbar --> true
    cguide --> "dB"
    ticks --> :native
    legend --> false
    xguide --> "x (m)"
    yguide --> "z (m) "
    @series begin
        seriestype := :heatmap
        receivers.xrange, receivers.zrange, -transmission_loss'
    end
end

# This is the "Case 1" model: Metadata only, Far-field approximation.


mutable struct SphericalWaveModel{T, E} <: DataDrivenPropagationModel{T}
    env::E
    nrays::Int

    # We force these to be Vectors of type T.
    # This is safer than {AT, PT, TT} because it prevents type mixing.
    A::Vector{T}
    phi::Vector{T}
    theta::Vector{T}
end

function SphericalWaveModel(env, nrays::Int)
    # Detect T from the environment
    T = hasproperty(env, :measurements) && !ismissing(env.measurements) ?
        eltype(env.measurements) : Float64

    return SphericalWaveModel(
        env,
        nrays,
        zeros(T, nrays), # A
        zeros(T, nrays), # phi
        zeros(T, nrays)  # theta
    )
end


# src/pm_case2.jl

function calculate_field(model::SphericalWaveModel, rx_coords, k)
    # Extract source location once
    sx, sy = location(model.env.tx)[1], location(model.env.tx)[2]

    # We use a generator comprehension (sum) which Zygote can differentiate through
    # We broadcast over the receivers (columns of rx_coords)

    # Note: This looks complex but it's just efficient broadcasting
    # For each receiver i, sum over all rays m
    preds = [
        sum(
            spherical_wave_propagate(rx_coords[1,i], rx_coords[2,i], sx, sy, k, model.A[m], model.phi[m])
            for m in 1:model.nrays
        )
        for i in 1:size(rx_coords, 2)
    ]

    return preds
end

# src/pm_case2.jl

# src/pm_case2.jl

function fit!(model::SphericalWaveModel, measurements;
              max_epochs=10000,          # Increased to match paper standards
              learning_rate=0.05,        # Higher start (we will decay it)
              convergence_threshold=1e-9,
              verbose=false)

    # 1. Extract Geometry & Data
    rx_locs = model.env.locations
    meas_vec = vec(measurements)
    k = 2π * model.env.frequency / model.env.soundspeed

    # --- STRATEGY 1: SMART INITIALIZATION ---
    # Don't guess. Estimate A from the data using A ≈ P * r
    if all(model.A .== 0.0)
        sx, sy = location(model.env.tx)[1], location(model.env.tx)[2]

        # Calculate average range to all sensors
        avg_r = Statistics.mean(sqrt.((rx_locs[1,:] .- sx).^2 .+ (rx_locs[2,:] .- sy).^2))
        avg_p = Statistics.mean(abs.(meas_vec))

        # Set A to a physical estimate (e.g. 0.3 instead of 10.0)
        est_A = avg_p * avg_r

        T = eltype(model.A)
        model.A .= T(est_A)

        if verbose
            println("  Initialized Amplitude A ≈ $est_A (Data-Driven)")
        end
    end

    # --- STRATEGY 2: NORMALIZATION ---
    # Scale targets so the max amplitude is 1.0.
    # This prevents gradients from vanishing due to tiny acoustic numbers.
    scale_factor = 1.0 / maximum(abs.(meas_vec))
    target_normalized = meas_vec .* scale_factor

    # Setup Flux
    ps = Flux.params(model.A, model.phi)
    opt = Flux.Adam(learning_rate)

    # Loss function calculates error in the NORMALIZED space
    loss_fn() = Flux.mse(calculate_field(model, rx_locs, k) .* scale_factor, target_normalized)

    # Training Loop
    prev_loss = Inf

    for epoch in 1:max_epochs
        # Gradient Step
        gs = Flux.gradient(loss_fn, ps)
        Flux.Optimise.update!(opt, ps, gs)

        # Constraint: Keep A positive
        model.A .= abs.(model.A)

        # --- STRATEGY 3: SCHEDULER ---
        # Drop learning rate at 50% and 80% of epochs to fine-tune phase
        if epoch == div(max_epochs, 2)
            opt.eta *= 0.1
            verbose && println("  >> Scheduler: Learning Rate dropped to $(opt.eta)")
        elseif epoch == div(max_epochs * 8, 10)
            opt.eta *= 0.1
            verbose && println("  >> Scheduler: Learning Rate dropped to $(opt.eta)")
        end

        # Logging
        current_loss = loss_fn()
        loss_change = abs(prev_loss - current_loss)

        if verbose && (epoch % 500 == 0)
            # Print REAL (un-normalized) error for human readability
            real_mse = current_loss / (scale_factor^2)
            println("Epoch $epoch: NormLoss = $(round(current_loss, digits=6)) | RealMSE = $(real_mse)")
        end

        if loss_change < convergence_threshold
            verbose && println("Converged at epoch $epoch")
            break
        end
        prev_loss = current_loss
    end

    return model
end


# src/pm_case1.jl

# ... Include the Struct definition we wrote earlier ...

"""
    fit!(model::PlaneWaveCurvModel, measurements; kwargs...)

Train PlaneWaveCurvModel to fit acoustic field measurements using gradient descent.

# Arguments
- `model`: PlaneWaveCurvModel instance (modified in-place)
- `measurements`: Vector or matrix of complex pressure measurements

# Keyword Arguments
- `max_epochs`: Maximum number of training epochs (default: 1000)
- `learning_rate`: Initial learning rate for Adam optimizer (default: 0.01)
- `convergence_threshold`: Stop if loss change < threshold (default: 1e-6)
- `verbose`: Print training progress (default: false)

# Returns
- The trained model
"""
function fit!(model::PlaneWaveCurvModel, measurements;
              max_epochs=5000,
              learning_rate=0.1,
              convergence_threshold=1e-8,
              verbose=false)

    # Fix zero-amplitude initialization issue
    if all(model.A .== 0.0)
        T = eltype(model.A)
        model.A .= T(1.0)  # Initialize closer to expected amplitude
    end

    # Extract training data from environment
    rx_locs = model.env.locations
    meas_vec = vec(measurements)
    k = 2π * model.env.frequency / model.env.soundspeed

    # Define loss function
    function loss_fn()
        predictions = calculatefield(model, rx_locs, k)
        return Flux.mse(predictions, meas_vec)
    end

    # Setup optimizer
    opt = Flux.Adam(learning_rate)
    ps = Flux.params(model)

    # Training loop with gradient descent
    prev_loss = Inf
    for epoch in 1:max_epochs
        # Compute gradients and update parameters
        gs = Flux.gradient(loss_fn, ps)
        Flux.Optimise.update!(opt, ps, gs)

        # Apply parameter constraints (removed theta wrapping - let it evolve freely)
        model.d .= max.(model.d, 1.0)         # Keep d positive

        # Check convergence
        current_loss = loss_fn()
        loss_change = abs(prev_loss - current_loss)

        if verbose && (epoch % 100 == 0)
            println("Epoch $epoch: Loss = $(current_loss), Change = $(loss_change)")
        end

        if loss_change < convergence_threshold
            verbose && println("Converged at epoch $epoch")
            break
        end

        prev_loss = current_loss
    end

    return model
end

# Only train A, phi, and theta; keep d fixed to avoid confusing the optimizer
# Flux.trainable(m::PlaneWaveddddCurvModel) = (m.A, m.phi, m.theta)

"""
    calculatefield(model::PlaneWaveCurvModel, rx_coords::AbstractMatrix, k::Real)

Calculate acoustic field at receiver locations for PlaneWaveCurvModel.

# Arguments
- `model`: PlaneWaveCurvModel instance
- `rx_coords`: Matrix of receiver coordinates (3 x N) where rows are [x, y, z]
- `k`: Wavenumber (2π * frequency / soundspeed)

# Returns
- Vector of complex pressures (length N)
"""
function calculatefield(model::PlaneWaveCurvModel, rx_coords::AbstractMatrix, k::Real)
    n_receivers = size(rx_coords, 2)

    # Use a functional approach (Zygote-friendly - no mutation)
    pressure = [begin
        x, y = rx_coords[1, i], rx_coords[2, i]

        # Sum contributions from all rays
        sum(1:model.nrays) do j
            plane_wave_propagate(
                x, y, k,
                model.A[j], model.phi[j], model.theta[j], model.d[j]
            )
        end
    end for i in 1:n_receivers]

    return pressure
end

# factory function - takes env as input, decides which model to create based on data
# only returns sphericalwavemodel right now
function RayBasisNN(env::DataDrivenUnderwaterEnvironment; nrays=50, kwargs...)

    # CASE 3-5: We know the depth -> Use Ray Tracing (RayBasis2D)
    if !ismissing(env.waterdepth)
        # return RayBasis2D(env; nrays=nrays, kwargs...)
    end

    # CASE 1-2: We don't know the depth -> Use Spherical/Plane Wave Model
    #
    return SphericalWaveModel(env, nrays, Float64[], Float64[], Float64[])
end

# constructor - takes model type as argument, tries to construct that specific model type
function RayBasisNN(::Type{M}, env::E; nrays=50) where {M<:DataDrivenPropagationModel, E}
    # Initialize with concrete types (Float64) to maintain stability
    M(env, nrays, Float64[], Float64[], Float64[])
end


#= function UnderwaterAcoustics.check(::Type{RayBasis2D}, env::Union{<:DataDrivenUnderwaterEnvironment,Missing})
    if env !== missing
        size(env.locations)[1] == 2 || throw(ArgumentError("RayBasis2D only supports 2D environment"))
    end
    env
end

function UnderwaterAcoustics.check(::Type{RayBasis2DCurv}, env::Union{<:DataDrivenUnderwaterEnvironment,Missing})
    if env !== missing
        size(env.locations)[1] == 2 || throw(ArgumentError("RayBasis2DCurv only supports 2D environment"))
    end
    env
end

function UnderwaterAcoustics.check(::Type{RayBasis3D}, env::Union{<:DataDrivenUnderwaterEnvironment,Missing})
    if env !== missing
        size(env.locations)[1] == 3 || throw(ArgumentError("RayBasis3D only supports 3D environment"))
    end
    env
end

function UnderwaterAcoustics.check(::Type{RayBasis3DRCNN}, env::Union{<:DataDrivenUnderwaterEnvironment,Missing})
    if env !== missing
        env.tx === missing || throw(ArgumentError("RayBasis3DRCNN only supports environments with known source location"))
        length(location(env.tx)) == 3 || throw(ArgumentError("RayBasis3DRCNN only supports 3D source"))
        size(env.locations)[1] == 3|| throw(ArgumentError("RayBasis3DRCNN only supports 3D environment"))
        env.waterdepth !== missing || throw(ArgumentError("RayBasis3DRCNN only supports environments with known water depth"))
    end
    env
end


function UnderwaterAcoustics.check(::Type{GPR}, env::Union{<:DataDrivenUnderwaterEnvironment,Missing})
    env
end =#
