using UnderwaterAcoustics
using DataDrivenAcoustics

env = UnderwaterEnvironment(seabed=Rock, bathymetry=200.0)
pm = PekerisRayTracer(env; max_bounces=3)

txpos = [0.0, -11.0]
f = 250.0
tx = AcousticSource(txpos[1], txpos[2], f)
rxpos = rand(2, 1000) .* [200.0, 30.0] .+ [5500.0, -100.0]
tloss = Array{Float32}(undef, size(rxpos)[2])

for i in 1 : 1 : size(rxpos)[2]
   tloss[i] = Float32(transmission_loss(pm, tx, AcousticReceiver(rxpos[1,i], rxpos[2,i])))
end

using Plots

rx = AcousticReceiverGrid2D(range(5000.0, length=1000), range(-200.0, length=200))
let x = transmission_loss(pm, tx, rx)
   plot(env; xlims=(5000,6000), ylims=(-200,0))
   plot!(rx, x; clim = (0,80))
   scatter!(rxpos[1,:], rxpos[2,:]; markersize = 1.5, markercolor =:green, markerstrokewidth = 0)
end

pm2 = DataDrivenPropagationModel(RBNN_2D(7))

rx1 = AcousticReceiver(5500, -100)
a = arrivals(pm, tx, rx1)

pm2 = let ps = pm2.params, c = pm.c, ω = 2π * tx.frequency, xz = location(rx1)
   ps.A = map(x -> abs(x.phasor), a)
   ps.θ = map(x -> -x.arrival_angle, a) / 2π
   ps.ϕ = map(x -> angle(x.phasor) + ω * (x.t - (xz[1] * cos(x.arrival_angle) - xz[3] * sin(x.arrival_angle)) / c), a) / 2π
   pm2(ps)
end

acoustic_field(pm, tx, rx1)
acoustic_field(pm2, tx, rx1)

let x = transmission_loss(pm2, tx, rx)
   plot(env; xlims=(5000,6000), ylims=(-200,0))
   plot!(rx, x; clim = (0,80))
   scatter!(rx1)
end

pm3 = DataDrivenPropagationModel(RBNN_2D(60))
pm3.params.A .*= 1f-4
# .- (mean(tloss) + 10*log10(pm3.model.nrays))

rxs = [AcousticReceiver(rxpos[1,i], rxpos[2,i]) for i in 1:size(rxpos,2)]
loss = TransmissionLossMSE(pm3, tx, rxs, tloss)
opt = Adam(1e-5)
DataDrivenAcoustics.fit!(pm3, loss; maxiters=5000, optimizer=opt, callback=(i, l) -> begin
   @info l
   l < 100
   #false
end)
DataDrivenAcoustics.fit!(pm3, loss; maxiters=100, optimizer=BFGS(), callback=(i, l) -> begin
   @info l
   false
end)

let x = transmission_loss(pm3, tx, rx)
   plot(env; xlims=(5000,6000), ylims=(-200,0))
   plot!(rx, x; clim = (0,80))
end
