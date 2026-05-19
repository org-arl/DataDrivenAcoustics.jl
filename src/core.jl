using UnderwaterAcoustics
import UnderwaterAcoustics: AbstractPropagationModel, AbstractAcousticSource, AbstractAcousticReceiver
import ComponentArrays: ComponentArray
import Optimization: OptimizationFunction, OptimizationProblem, solve, AutoReverseDiff
import OptimizationOptimisers: Adam
import OptimizationOptimJL: BFGS, LBFGS
import ReverseDiff
import Random
import Lux

export DataDrivenPropagationModel, TransmissionLossMSE, Adam, BFGS, LBFGS
public fit!

struct DataDrivenPropagationModel{T1,T2} <: AbstractPropagationModel
  model::T1
  params::T2
  c::Float32
end

Base.show(io::IO, m::DataDrivenPropagationModel) = print(io, "DataDrivenPropagationModel(", m.model, ")")

"""
    DataDrivenPropagationModel(model; rng=Random.GLOBAL_RNG, soundspeed=soundspeed())

Data-driven acoustic propagation model using a Lux neural network layer `model`.
The model is initialized with random parameters using `rng` and assumes an isovelocity
medium with sound speed `soundspeed`.
"""
function DataDrivenPropagationModel(model; rng=Random.GLOBAL_RNG, soundspeed=soundspeed())
  pst = Lux.setup(rng, model)
  DataDrivenPropagationModel(model, ComponentArray(pst[1]), Float32(soundspeed))
end

# callable version to create a copy with new parameters
(pm::DataDrivenPropagationModel)(params) = DataDrivenPropagationModel(pm.model, params, pm.c)

function UnderwaterAcoustics.acoustic_field(pm::DataDrivenPropagationModel, tx::AbstractAcousticSource, rx::AbstractAcousticReceiver)
  f = frequency(tx)
  p = location(rx)
  k = 2f0 * π * f / pm.c
  inp = Float32[p.x; p.z; k;;]
  out = Lux.LuxCore.stateless_apply(pm.model, inp, pm.params)
  complex(out[1], out[2])
end

function UnderwaterAcoustics.acoustic_field(pm::DataDrivenPropagationModel, tx::AbstractAcousticSource, rxs::AbstractArray{<:AbstractAcousticReceiver})
  f = frequency(tx)
  p = vec(location.(rxs))
  k = 2f0 * π * f / pm.c
  inp = Float32[getfield.(p, :x) getfield.(p, :z) fill(k, length(p))]
  out = Lux.LuxCore.stateless_apply(pm.model, inp', pm.params)
  reshape(complex.(out[1,:], out[2,:]), size(rxs))
end

function fit!(pm::DataDrivenPropagationModel, loss, adtype=AutoReverseDiff(); optimizer=Adam(1e-4), maxiters=100, callback=nothing)
  ofun = OptimizationFunction(loss, adtype)
  oprob = OptimizationProblem(ofun, pm.params)
  sol = solve(oprob, optimizer; maxiters, callback)
  pm.params .= sol.u
  pm
end

function TransmissionLossMSE(pm::DataDrivenPropagationModel, tx::AbstractAcousticSource, rxs::AbstractArray{<:AbstractAcousticReceiver}, data)
  let tx = tx, rxs = rxs, data = data
    (ps, _) -> sum(abs2, transmission_loss(pm(ps), tx, rxs) - data)
  end
end
