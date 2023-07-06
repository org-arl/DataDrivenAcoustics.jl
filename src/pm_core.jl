using RecipesBase
using Printf


export DataDrivenUnderwaterEnvironment, ModelFit!, transfercoef, transmissionloss, check, plot, rays, eigenrays, arrivals


"""
$(TYPEDEF)
Create an underwater environment for data-driven physics-based propagation models by providing locations, acoustic measnreuments and other known environmental and channel geomtry knowledge.

- `locations`: location measurements (in the form of matrix with dimension [dimension of a single location data x number of data points])
- `measurements`: acoustic field measurements (in the form of matrix with dimension [1 x number of data points])
- `soundspeed`: medium sound speed (default: missing)
- `frequency`: source frequency (default: missing)
- `waterdepth`: water depth (default: missing)
- `salinity`: water salinity (default: 35)
- `seasurface`: surface property (dafault: Vacuum)
- `seabed`: seabed property (default: SandySilt)
- `tx`: source location (default: missing)
- set `dB` to `false` if `measurements` are not in dB scale (default: `true`)
 """
Base.@kwdef struct BasicDataDrivenUnderwaterEnvironment{T1<:Matrix, T2, T3, T4, T5<:ReflectionModel, T6<:ReflectionModel, T7} <: DataDrivenUnderwaterEnvironment
    locations::T1 
    measurements::T1 
    soundspeed::T2
    frequency::T3 
    waterdepth::T4 
    salinity::Real
    seasurface::T5
    seabed::T6 
    tx::T7 
    dB::Bool
    function BasicDataDrivenUnderwaterEnvironment(locations, measurements; soundspeed = missing, frequency = missing, waterdepth = missing, salinity = 35.0, seasurface = Vacuum, seabed = SandySilt, tx = missing, dB = true)
        if  tx !== missing 
            length(location(tx)) == size(locations)[1] || throw(ArgumentError("Dimension of source location and measurement locations do not match"))
        end
        size(locations)[2] == size(measurements)[2] || throw(ArgumentError("Number of locations and fields measurements do not match"))
        size(locations)[1] < 4 || throw(ArgumentError("Dimension of location data should not be larger than 3"))
        size(measurements)[1] == 1 || throw(ArgumentError("size of acoustic measurements should be 1 × n"))
        new{typeof(locations), typeof(soundspeed), typeof(frequency), typeof(waterdepth), typeof(seasurface), typeof(seabed), typeof(tx)}(locations, measurements, soundspeed, frequency, waterdepth, salinity, seasurface, seabed, tx, dB)
    end
end


DataDrivenUnderwaterEnvironment(locations, measurements; kwargs...) = BasicDataDrivenUnderwaterEnvironment(locations, measurements; kwargs...)


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
    r
end 

"""
$(SIGNATURES)
Calculate transmission coefficient at location `rx` using a data-driven physics-based propagation model.

- `model`: data-driven physics-based propagation model
- `tx`: acoustic source. This is optional. Use `missing` or `nothing` for unknown source.
- `rx`: acoustic receiver location(s) 
"""
function UnderwaterAcoustics.transfercoef(model::DataDrivenPropagationModel, tx::Union{Missing, Nothing, AcousticSource}, rx::AcousticReceiver; mode=:coherent) where {T1}
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

function UnderwaterAcoustics.transfercoef(model::DataDrivenPropagationModel, tx::Union{Missing, Nothing, AcousticSource}, rx::AcousticReceiverGrid2D; mode=:coherent) where {T1}
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

function UnderwaterAcoustics.transfercoef(model::DataDrivenPropagationModel, tx::Union{Missing, Nothing, AcousticSource}, rx::AcousticReceiverGrid3D; mode=:coherent) where {T1}
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

UnderwaterAcoustics.transfercoef(model::DataDrivenPropagationModel, tx::Union{Missing, Nothing, AcousticSource}, rx::AbstractArray{<:AcousticReceiver}) = UnderwaterAcoustics.tmap(rx1 -> transfercoef(model, tx, rx1), rx)

UnderwaterAcoustics.transfercoef(model::DataDrivenPropagationModel, rx::Union{AbstractVector, AbstractMatrix}) = model.calculatefield(model, rx)


UnderwaterAcoustics.transmissionloss(model::DataDrivenPropagationModel, rx::Union{AbstractVector, AbstractMatrix}) = -amp2db.(abs.(transfercoef(model, rx)))

UnderwaterAcoustics.transmissionloss(model::DataDrivenPropagationModel, tx::Union{Missing, Nothing, AcousticSource}, rx::Union{AbstractVector, AbstractMatrix}) = -amp2db.(abs.(transfercoef(model, tx, rx)))



UnderwaterAcoustics.rays(model::DataDrivenPropagationModel, tx, rx) = throw(ArgumentError("This function is not yet supported"))
  
UnderwaterAcoustics.eigenrays(model::DataDrivenPropagationModel, tx, rx) = throw(ArgumentError("This function is not yet supported"))


abstract type Arrival end

function Base.show(io::IO, a::Arrival)
    if a.time === missing
        @printf(io, "                        |          | %5.1f dB ϕ%6.1f°", amp2db(abs(a.phasor)), rad2deg(angle(a.phasor)))
    else
        @printf(io, "                        | %6.2f ms | %5.1f dB ϕ%6.1f°", 1000*a.time, amp2db(abs(a.phasor)), rad2deg(angle(a.phasor)))
    end
end


struct RayArrival{T1,T2} <: Arrival
    time::T1
    phasor::T2
    surface::Missing
    bottom::Missing
    launchangle::Missing
    arrivalangle::Missing
    raypath::Missing
end
  
"""
$(SIGNATURES)
Show arrival rays at a location `rx` using a data-driven physics-based propagation model.

- `model`: data-driven physics-based propagation model
- `tx`: acoustic source. This is optional. Use `missing` or `nothing` for unknown source.
- `rx`: an acoustic receiver
"""
function UnderwaterAcoustics.arrivals(model::DataDrivenPropagationModel, tx::Union{Missing, Nothing, AcousticSource}, rx::Union{AbstractVector, AcousticReceiver}; threshold = 30) 
    model isa GPR && throw(ArgumentError("GPR model does not support this function"))
    arrival = model.calculatefield(model, collect(location(rx)); showarrivals = true)
    amp = amp2db.(abs.(arrival))
    idx = findall(amp .> (maximum(amp) - threshold))
    signficantarrival = arrival[idx]
    idx = sortperm(abs.(signficantarrival), rev = true)
    if model isa RayBasis2D || model isa RayBasis2DCurv
        rays = [RayArrival(missing, signficantarrival[idx[i]], missing, missing, missing, missing, missing) for i in 1 : length(idx)] 
    elseif model isa RayBasis3DRCNN
        rays =[RayArrival(model.d[idx[i]] ./ model.env.soundspeed, signficantarrival[idx[i]], missing, missing, missing, missing, missing) for i in 1 : length(idx)] 
    else
        rays =[RayArrival((model.d[idx[i]] .+ model.ed[idx[i]]) ./ model.env.soundspeed, signficantarrival[idx[i]], missing, missing, missing, missing, missing) for i in 1 : length(idx)] 
    end
    return rays
end

UnderwaterAcoustics.arrivals(model::DataDrivenPropagationModel, rx::Union{AbstractVector, AcousticReceiver}) = UnderwaterAcoustics.arrivals(model, nothing, rx)

@recipe function plot(env::DataDrivenUnderwaterEnvironment; receivers = [], transmissionloss = [],  dynamicrange = 42.0)
    size(transmissionloss) == size(receivers) || throw(ArgumentError("Mismatched receivers and transmissionloss"))
    receivers isa AcousticReceiverGrid2D || throw(ArgumentError("Receivers must be an instance of AcousticReceiverGrid2D"))
    minloss = minimum(transmissionloss)
    clims --> (-minloss-dynamicrange, -minloss)
    colorbar --> true
    cguide --> "dB"
    ticks --> :native
    legend --> false
    xguide --> "x (m)"
    yguide --> "z (m) "
    @series begin
        seriestype := :heatmap
        receivers.xrange, receivers.zrange, -transmissionloss'
    end
end

function UnderwaterAcoustics.check(::Type{RayBasis2D}, env::Union{<:DataDrivenUnderwaterEnvironment,Missing})
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
end


