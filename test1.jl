using UnderwaterAcoustics
using DataDrivenAcoustics

env = UnderwaterEnvironment(seabed=Rock)
pm = PekerisRayTracer(env; max_bounces=3)

using Random
Random.seed!(1)

txpos = [0.0, -11.0]
f = 1000.0
tx = AcousticSource(txpos[1], txpos[2], f)
rxpos = rand(2, 500) .* [100.0, -20.0] .+ [500.0, 0.0]
tloss = Array{Float32}(undef, size(rxpos)[2])

for i in 1 : 1 : size(rxpos)[2]
   tloss[i] = Float32(transmission_loss(pm, tx, AcousticReceiver(rxpos[1,i], rxpos[2,i])))
end

using Plots

rx = AcousticReceiverGrid2D(range(500.0, step=0.1, length=1000), range(-20.0, step=0.1, length=200))
let x = transmission_loss(pm, tx, rx)
   plot(env; xlims=(500,600), ylims=(-20,0))
   plot!(rx, x; clim = (0,80))
   scatter!(rxpos[1,:], rxpos[2,:]; markersize = 1.5, markercolor =:green, markerstrokewidth = 0)
end

pm = DataDrivenPropagationModel(RBNN_2D(7))
loss = TransmissionLossMSE(pm, tx, [AcousticReceiver(rxpos[1,i], rxpos[2,i]) for i in 1:size(rxpos,2)], tloss) # .- (mean(tloss) + 10*log10(pm.model.nrays)))
DataDrivenAcoustics.fit!(pm, loss, ADTypes.AutoReverseDiff(compile=true); maxiters=10000, optimizer=BFGS(), callback=(i, l) -> begin
   @info l
   false
end)
