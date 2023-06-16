using Plots
using UnderwaterAcoustics

models()

# generate synthetic training data:
env = UnderwaterEnvironment();
pm = PekerisRayModel(env, 7);
TX = [0.0, -5.0];
RX = rand(2, 500) .* [100.0, -20.0] .+ [1.0, 0.0];
TL = modelfield(RX, TX, 1000.0, pm);

#plot ground truth
tx = AcousticSource(0.0, -5.0, 1000.0);
rx = AcousticReceiverGrid2D(1.0, 0.1, 1000, -20.0, 0.1, 200)
tl = transmissionloss(pm, tx, rx)
plot(env; receivers=rx, transmissionloss=tl, title = "Ground truth")
scatter!(RX[1,:], RX[2,:], markersize = 1.5, markercolor =:green, markerstrokewidth = 0)


# data driven env and model:
dataenv = DataDrivenUnderwaterEnvironment(RX, TL; frequency = 1000.0, soundspeed = 1540.0);
models(dataenv)
datapm = RayBasis2DCurv(dataenv, ModelTrainingSetting(0.005, rmseloss, rmseloss))


x = transmissionloss(datapm, rx)
plot(env; receivers = rx, transmissionloss = x, title = "Estimation", clim = (-42, -0))

arrivals(datapm,AcousticReceiver(50, -10))



# plot extended fields:
rx = AcousticReceiverGrid2D(1.0, 0.1, 2000, -20.0, 0.1, 200)
tl = transmissionloss(pm, tx, rx)
plot(env; receivers=rx, transmissionloss=tl, title = "Ground truth")



x = transmissionloss(datapm, rx)
plot(env; receivers = rx, transmissionloss = x, title = "Estimation", clim = (-42, -0))
