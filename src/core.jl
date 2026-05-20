using UnderwaterAcoustics
import UnderwaterAcoustics: AbstractPropagationModel, AbstractAcousticSource, AbstractAcousticReceiver
import ComponentArrays: ComponentArray
import Optimization: OptimizationFunction, OptimizationProblem, solve, AutoReverseDiff
import OptimizationOptimisers: Adam
import OptimizationOptimJL: BFGS, LBFGS
import ReverseDiff
import Random
import Lux

export DataDrivenPropagationModel, TransmissionLossMSE
export Adam, BFGS, LBFGS
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

function fit!(pm::DataDrivenPropagationModel, loss, adtype=AutoReverseDiff(compile=true); optimizer=Adam(1e-4), maxiters=100, minloss=0, show_progress=0)
  ofun = OptimizationFunction(loss, adtype)
  oprob = OptimizationProblem(ofun, pm.params)
  cb = let show_progress = show_progress, minloss = minloss
    (st, l) -> begin
      show_progress > 0 && st.iter % show_progress == 0 && println("Iteration $(st.iter), Loss: $l")
      l < minloss
    end
  end
  sol = solve(oprob, optimizer; maxiters, callback=cb)
  pm.params .= sol.u
  pm
end

"""
    TransmissionLossMSE(pm, tx, rxs, data; sparsity=10f0)

Loss function for fitting a data-driven propagation model `pm` to observed
transmission loss data `data` measured at receivers `rxs`. The frequency of
operation is determined by the source `tx`. The `sparsity` parameter controls
the L1 regularization strength on the ray amplitudes to promote sparsity in
the solution.
"""
function TransmissionLossMSE(pm::DataDrivenPropagationModel, tx::AbstractAcousticSource, rxs::AbstractArray{<:AbstractAcousticReceiver}, data; sparsity=10f0)
  let tx = tx, rxs = rxs, data = data, sparsity = sparsity
    (ps, _) -> sum(abs2, transmission_loss(pm(ps), tx, rxs) - data) + sparsity * sum(abs, ps.A)
  end
end
