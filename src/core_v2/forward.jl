# ============================================================================
# Forward Pass Implementations (Callable Structs)
# ============================================================================
#
# Each model is callable: model(rx) or model(rx, env; kwargs...)
# This is the standard Flux.jl pattern for neural network layers.
# ============================================================================

using Zygote
using UnderwaterAcoustics: absorption, location

# ============================================================================
# Helper Functions
# ============================================================================

"""
    get_tx_coords(env)

Extract transmitter coordinates from environment.
Handles both AcousticSource objects and raw coordinate vectors.
"""
function get_tx_coords(env)
    tx = env.tx
    if ismissing(tx)
        error("Environment must have tx (transmitter) defined")
    end
    if hasmethod(location, (typeof(tx),))
        return collect(location(tx))
    else
        return tx
    end
end

# ============================================================================
# RayBasis Forward (Case 1 - 2D)
# ============================================================================

"""
    (r::RayBasis)(xy)

Compute transmission loss for 2D receiver positions.

# Arguments
- `xy`: Receiver positions (2 x N matrix)

# Returns
- Transmission loss in dB (1 x N matrix)
"""
function (r::RayBasis)(xy::AbstractArray)
    xₒ = [0.0, 0.0]
    x = @view xy[1:1, :]
    y = @view xy[2:2, :]
    xx = x .- (xₒ[1] .- r.d .* cos.(r.θ))
    yy = y .- (xₒ[2] .- r.d .* sin.(r.θ))
    l = sqrt.(xx.^2 + yy.^2)
    kxcys = r.k .* l .+ r.ϕ
    real_im_amp = r.A .* cis.(kxcys)
    amp2db.(abs.(sum(real_im_amp; dims = 1)))
end

# ============================================================================
# RayBasis2DCurv Forward (Case 1 - 2D with Curvature)
# ============================================================================

"""
    (r::RayBasis2DCurv)(xy)

Compute transmission loss for 2D receiver positions with wavefront curvature.

# Arguments
- `xy`: Receiver positions (2 x N matrix)

# Returns
- Transmission loss in dB (1 x N matrix)
"""
function (r::RayBasis2DCurv)(xy::AbstractArray)
    xₒ = [0.0, 0.0]
    x = @view xy[1:1, :]
    y = @view xy[2:2, :]
    xx = x .- (xₒ[1] .- r.d .* cos.(r.θ))
    yy = y .- (xₒ[2] .- r.d .* sin.(r.θ))
    l = sqrt.(xx.^2 + yy.^2)
    kxcys = r.k .* l .+ r.ϕ
    real_im_amp = r.A .* cis.(kxcys)
    amp2db.(abs.(sum(real_im_amp; dims = 1)))
end

# ============================================================================
# RayBasis3d Forward (Case 2 - 3D Near-Field)
# ============================================================================

"""
    (r::RayBasis3d)(xyz; xₒ, nominal_ρ, nominal_θ, nominal_ψ)

Compute transmission loss for 3D receiver positions (without env).

# Arguments
- `xyz`: Receiver positions (3 x N matrix)

# Keyword Arguments
- `xₒ`: Reference origin (default: [0,0,0])
- `nominal_ρ`, `nominal_θ`, `nominal_ψ`: Nominal ray parameters from image sources

# Returns
- Transmission loss in dB (1 x N matrix)
"""
function (r::RayBasis3d)(xyz::AbstractArray; xₒ = Float64[0.0, 0.0, 0.0], nominal_ρ, nominal_θ, nominal_ψ)
    x = @view xyz[1:1,:]
    y = @view xyz[2:2,:]
    z = @view xyz[3:3,:]

    xx = x .- (xₒ[1] .- (r.ed .+ nominal_ρ) .* cos.(r.eθ .+ nominal_θ) .* sin.(r.eψ .+ nominal_ψ))
    yy = y .- (xₒ[2] .- (r.ed .+ nominal_ρ) .* sin.(r.eθ .+ nominal_θ) .* sin.(r.eψ .+ nominal_ψ))
    zz = z .- (xₒ[3] .- (r.ed .+ nominal_ρ) .* cos.(r.eψ .+ nominal_ψ))
    l = sqrt.(xx.^2 + yy.^2 + zz.^2)
    kx = r.k .* l .+ r.ϕ
    real_im_amp = vcat(sum(r.A ./ l .* cos.(kx); dims=1), sum(r.A ./ l .* sin.(kx); dims=1))
    amp2db.(sqrt.(sum(abs2, real_im_amp; dims=1)))
end

"""
    (m::RayBasis3d)(xyz, env; xₒ, nominal_ρ, nominal_θ, nominal_ψ)

Compute transmission loss for 3D receiver positions (with env for API consistency).
Delegates to the non-env version since Case 2 doesn't use env parameters.
"""
function (m::RayBasis3d)(xyz::AbstractArray, env; xₒ = Float64[0.0, 0.0, 0.0], nominal_ρ, nominal_θ, nominal_ψ)
    m(xyz; xₒ=xₒ, nominal_ρ=nominal_ρ, nominal_θ=nominal_θ, nominal_ψ=nominal_ψ)
end

# ============================================================================
# RayBasisRCNN Forward (Case 3 - RCNN)
# ============================================================================

"""
    (m::RayBasisRCNN)(xyz, env; xₒ, nominal_ρ, nominal_θ, nominal_ψ, n_rays)

Compute transmission loss using learned reflection coefficients.

# Arguments
- `xyz`: Receiver positions (3 x N matrix)
- `env`: Environment with soundspeed, frequency, waterdepth, tx

# Keyword Arguments
- `xₒ`: Reference origin (3-element vector)
- `nominal_ρ`, `nominal_θ`, `nominal_ψ`: Nominal ray parameters from image sources
- `n_rays`: Number of rays

# Returns
- Transmission loss in dB (1 x N matrix)
"""
function (m::RayBasisRCNN)(xyz::AbstractArray, env; xₒ, nominal_ρ, nominal_θ, nominal_ψ, n_rays)
    c = env.soundspeed
    f = env.frequency
    L = env.waterdepth
    tx = get_tx_coords(env)

    x = @view xyz[1:1, :]
    y = @view xyz[2:2, :]
    z = @view xyz[3:3, :]

    xx = x .- (xₒ[1] .- (m.ed .+ nominal_ρ) .* cos.(m.eθ .+ nominal_θ) .* sin.(m.eψ .+ nominal_ψ))
    yy = y .- (xₒ[2] .- (m.ed .+ nominal_ρ) .* sin.(m.eθ .+ nominal_θ) .* sin.(m.eψ .+ nominal_ψ))
    zz = z .- (xₒ[3] .- (m.ed .+ nominal_ρ) .* cos.(m.eψ .+ nominal_ψ))

    l = sqrt.(xx .^ 2.0f0 + yy .^ 2.0f0 + zz .^ 2.0f0)

    j = collect(1:n_rays)
    R² = abs2.(tx[1] .- x) .+ abs2.(tx[2] .- y)
    R = R² .^ 0.5f0
    upward = iseven.(j)
    s1 = 2 .* upward .- 1
    n_idx = div.(j, 2)
    s = div.(n_idx .+ upward, 2)
    b = div.(n_idx .+ (1 .- upward), 2)
    s2 = 2 .* iseven.(n_idx) .- 1
    dz = 2 .* b .* L .+ s1 .* tx[3] .- s1 .* s2 .* z
    θ = abs.(atan.(R ./ dz))

    s_loss = (-1.0f0) .^ s

    RC_mat = Matrix{Float32}(undef, n_rays, size(zz)[2])
    phase_mat = Matrix{Float32}(undef, n_rays, size(zz)[2])
    buf_phase = Zygote.Buffer(phase_mat, size(phase_mat))
    buf_RC = Zygote.Buffer(RC_mat, size(RC_mat))

    for i in 1:n_rays
        rcnn_out = m.rcnn(θ[i:i, :])
        buf_RC[i:i, :] = abs.(rcnn_out[1:1, :])
        buf_phase[i:i, :] = rcnn_out[2:2, :]
    end

    overall_phase = 2.0f0 * Float32(π) * l ./ c * f .+ copy(buf_phase) .* b
    amp = 1.0f0 ./ l .* s_loss .* copy(buf_RC) .^ b .* absorption.(f, l, 35.0f0)

    real_im_amp = vcat(
        sum(amp .* cos.(overall_phase); dims = 1),
        sum(amp .* sin.(overall_phase); dims = 1)
    )

    amp2db.(sqrt.(sum(abs2, real_im_amp; dims = 1)))
end

# ============================================================================
# RayBasisRayleigh Forward (Case 4 - Rayleigh Reflection)
# ============================================================================

"""
    (m::RayBasisRayleigh)(xyz, env; n_rays)

Compute amplitude using physics-based Rayleigh reflection coefficients.

# Arguments
- `xyz`: Receiver positions (3 x N matrix)
- `env`: Environment with soundspeed, frequency, waterdepth, tx

# Keyword Arguments
- `n_rays`: Number of rays

# Returns
- Linear amplitude (1 x N matrix) - NOT in dB
"""
function (m::RayBasisRayleigh)(xyz::AbstractArray, env; n_rays)
    c = env.soundspeed
    f = env.frequency
    L = env.waterdepth
    tx = get_tx_coords(env)

    x = @view xyz[1:1, :]
    y = @view xyz[2:2, :]
    z = @view xyz[3:3, :]

    j = collect(1:n_rays)
    R² = abs2.(tx[1] .- x) .+ abs2.(tx[2] .- y)
    R = R² .^ 0.5
    upward = iseven.(j)
    s1 = 2 .* upward .- 1
    n_idx = div.(j, 2)
    s = div.(n_idx .+ upward, 2)
    b = div.(n_idx .+ (1 .- upward), 2)
    s2 = 2 .* iseven.(n_idx) .- 1
    dz = 2 .* b .* L .+ s1 .* tx[3] .- s1 .* s2 .* z
    θ = abs.(atan.(R ./ dz))
    l = .√(R² .+ abs2.(dz))

    s_loss = (-1.0f0) .^ s

    ρr_val = abs(m.ρᵣ[1])
    cr_val = abs(m.cᵣ[1])
    δ_val = abs(m.δ[1])

    RC = reflectioncoef.(θ, ρr_val, cr_val, δ_val)

    overall_phase = 2π * l ./ c * f .+ angle.(RC) .* b
    amp = 1.0f0 ./ l .* s_loss .* abs.(RC) .^ b .* absorption.(f, l)

    real_im_amp = vcat(
        sum(amp .* cos.(overall_phase); dims = 1),
        sum(amp .* sin.(overall_phase); dims = 1)
    )

    sqrt.(sum(abs2, real_im_amp; dims = 1))
end
