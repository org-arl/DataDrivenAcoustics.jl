using DataDrivenAcoustics
using UnderwaterAcoustics
using StableRNGs
using Test

# prepare dataset
rng = StableRNG(27)
env = UnderwaterEnvironment(seabed=Rock, bathymetry=200.0)
pm = PekerisRayTracer(env; max_bounces=3)
tx = AcousticSource(0, -11, 250)
rxpos = rand(rng, 2, 1000) .* [200.0, 40.0] .+ [5500.0, -110.0]
rxs = [AcousticReceiver(rxpos[1,i], rxpos[2,i]) for i ∈ 1:size(rxpos,2)]
xloss = Float32.(transmission_loss(pm, tx, rxs))

# train data-driven model
pm = DataDrivenPropagationModel(RayBasisNN_2D(60); rng=StableRNG(42))
rxs = [AcousticReceiver(x, z) for (x, z) ∈ zip(rxpos[1,:], rxpos[2,:])]
loss = TransmissionLossMSE(pm, AcousticSource(nothing, 250), rxs, xloss)
DataDrivenAcoustics.fit!(pm, loss; optimizer=Adam(5e-6), minloss=100, maxiters=5000)
DataDrivenAcoustics.fit!(pm, loss; optimizer=BFGS(), maxiters=200)

@test loss(pm.params, nothing) < 5
