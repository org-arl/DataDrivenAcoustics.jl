import Lux: LuxCore
import Random: AbstractRNG

export RayBasisNN_2D, RBNN_2D

"""
    RayBasisNN_2D(nrays)

Ray basis neural network (RBNN) Lux layer for 2D acoustic propagation.
`nrays` is the number of basis rays.
"""
struct RayBasisNN_2D <: LuxCore.AbstractLuxLayer
  nrays::Int
end

# alias for backward compatibility with pre-release versions
const RBNN_2D = RayBasisNN_2D

LuxCore.initialparameters(rng::AbstractRNG, l::RayBasisNN_2D) = (
  A = 1e-4 * randn(rng, Float32, l.nrays),  # amplitudes
  ϕ = rand(rng, Float32, l.nrays),          # phases (scaled by 1/2π)
  θ = rand(rng, Float32, l.nrays)           # ray angles (scaled by 1/2π)
)

LuxCore.initialstates(::AbstractRNG, ::RayBasisNN_2D) = NamedTuple()

function (l::RayBasisNN_2D)(inp::AbstractMatrix, ps, st::NamedTuple)
  xz = @view inp[1:2, :]
  k = @view inp[3:3, :]
  k̂ = hcat(cospi.(2ps.θ), sinpi.(2ps.θ))
  kr = k .* (k̂ * xz)
  y_re = sum(ps.A .* cos.(2f0 * π * ps.ϕ .+ kr); dims=1)
  y_im = sum(ps.A .* sin.(2f0 * π * ps.ϕ .+ kr); dims=1)
  vcat(y_re, y_im), st
end
