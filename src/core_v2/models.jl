# ============================================================================
# Model Definitions
# ============================================================================
#
# All models inherit from PropagationModel and are callable structs.
# Each model learns different parameters for acoustic propagation.
# ============================================================================

# ============================================================================
# RayBasis (Case 1 - 2D Far-Field, Legacy)
# ============================================================================

"""
    RayBasis

2D far-field ray basis model (legacy, for BSON compatibility).
Learns ray arrival angles, amplitudes, phases, and distances.

# Fields
- `θ`: Azimuthal angles of arrival rays (radians)
- `A`: Amplitudes of arrival rays
- `ϕ`: Phases of arrival rays (radians)
- `d`: Distances for wavefront curvature modeling
- `k`: Angular wavenumber (rad/m)
"""
struct RayBasis{T1<:AbstractVector,T2<:Real} <: PropagationModel
    θ::T1
    A::T1
    ϕ::T1
    d::T1
    k::T2
end

RayBasis(rays::Integer, k::Real) = RayBasis(rand(Float32, rays) * π, rand(Float32, rays), rand(Float32, rays) * π, rand(Float32, rays), k)

Flux.@functor RayBasis
Flux.trainable(r::RayBasis) = (r.θ, r.A, r.ϕ, r.d)

# ============================================================================
# RayBasis2DCurv (Case 1 - 2D Far-Field with Curvature)
# ============================================================================

"""
    RayBasis2DCurv

2D far-field ray basis model with wavefront curvature.
Same structure as RayBasis but preferred for new code.

# Fields
- `θ`: Azimuthal angles of arrival rays (radians)
- `A`: Amplitudes of arrival rays
- `ϕ`: Phases of arrival rays (radians)
- `d`: Distances for wavefront curvature modeling
- `k`: Angular wavenumber (rad/m)
"""
struct RayBasis2DCurv{T1<:AbstractVector,T2<:Real} <: PropagationModel
    θ::T1
    A::T1
    ϕ::T1
    d::T1
    k::T2
end

RayBasis2DCurv(rays::Integer, k::Real) = RayBasis2DCurv(rand(Float32, rays) * π, rand(Float32, rays), rand(Float32, rays) * π, rand(Float32, rays), k)

Flux.@functor RayBasis2DCurv
Flux.trainable(r::RayBasis2DCurv) = (r.θ, r.A, r.ϕ, r.d)

# ============================================================================
# RayBasis3d (Case 2 - 3D Near-Field)
# ============================================================================

"""
    RayBasis3d

3D near-field ray basis model.
Uses nominal ray parameters from image source method, learns amplitude and phase corrections.

# Fields
- `eθ`: Error to nominal azimuthal angle (radians)
- `eψ`: Error to nominal elevation angle (radians)
- `ed`: Error to nominal propagation distance (meters)
- `A`: Amplitudes of arrival rays
- `ϕ`: Phases of arrival rays (radians)
- `k`: Angular wavenumber (rad/m)

# Notes
Uses Float64 precision due to large phase values (~2000-3000 radians).
Only `A` and `ϕ` are trainable by default.
"""
struct RayBasis3d{T1<:AbstractVector,T2<:Real} <: PropagationModel
    eθ::T1
    eψ::T1
    ed::T1
    A::T1
    ϕ::T1
    k::T2
end

RayBasis3d(rays::Integer, k::Real) = RayBasis3d(zeros(Float64, rays), zeros(Float64, rays), zeros(Float64, rays), rand(Float64, rays), rand(Float64, rays), Float64(k))

Flux.@functor RayBasis3d
Flux.trainable(r::RayBasis3d) = (r.A, r.ϕ)

# ============================================================================
# RayBasisRCNN (Case 3 - Geo-acoustic Inversion with RCNN)
# ============================================================================

"""
    RayBasisRCNN

3D ray basis model with Reflection Coefficient Neural Network (RCNN).
Learns seabed reflection coefficient as a function of incident angle.

# Fields
- `eθ`: Error to nominal azimuthal angle (radians)
- `eψ`: Error to nominal elevation angle (radians)
- `ed`: Error to nominal propagation distance (meters)
- `k`: Angular wavenumber (rad/m)
- `rcnn`: Neural network that maps incident angle → (reflection coef, phase shift)

# Notes
Only the RCNN is trainable. Geometry parameters are fixed from image sources.
"""
struct RayBasisRCNN{T1<:AbstractVector, T2<:Real, C} <: PropagationModel
    eθ::T1
    eψ::T1
    ed::T1
    k::T2
    rcnn::C
end

function RayBasisRCNN(rays::Integer, k::Real; rcnn=nothing)
    if rcnn === nothing
        # Default RCNN architecture
        rcnn = Chain(
            x -> (x ./ 0.5f0 .* Float32(π) .- 0.5f0) .* 2.0f0,
            Dense(1, 30, sigmoid),
            Dense(30, 50, sigmoid),
            Dense(50, 2),
        )
    end
    RayBasisRCNN(
        zeros(Float32, rays),
        zeros(Float32, rays),
        zeros(Float32, rays),
        Float32(k),
        rcnn
    )
end

Flux.@functor RayBasisRCNN
Flux.trainable(r::RayBasisRCNN) = (r.rcnn,)

# ============================================================================
# RayBasisRayleigh (Case 4 - Seabed Physical Parameters)
# ============================================================================

"""
    RayBasisRayleigh

3D ray basis model with physics-based Rayleigh reflection.
Learns physical seabed parameters directly.

# Fields
- `ρᵣ`: Density ratio (seabed/water)
- `cᵣ`: Sound speed ratio (seabed/water)
- `δ`: Attenuation coefficient
- `k`: Angular wavenumber (rad/m)

# Notes
All three seabed parameters (ρᵣ, cᵣ, δ) are trainable.
Uses Rayleigh reflection coefficient formula.
"""
struct RayBasisRayleigh{T1<:AbstractVector, T2<:Real} <: PropagationModel
    ρᵣ::T1
    cᵣ::T1
    δ::T1
    k::T2
end

RayBasisRayleigh(k::Real) = RayBasisRayleigh([1.0f0], [1.0f0], [1.0f0], Float32(k))

Flux.@functor RayBasisRayleigh
Flux.trainable(r::RayBasisRayleigh) = (r.ρᵣ, r.cᵣ, r.δ)
