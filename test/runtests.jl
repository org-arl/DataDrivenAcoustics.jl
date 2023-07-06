using DataDrivenAcoustics
using UnderwaterAcoustics
using Test
using Random
using DSP
using GaussianProcesses
using Flux

function test2d(datapm)
    x1 = transfercoef(datapm, nothing, AcousticReceiver(50.0, -5.0))
    x2 = transfercoef(datapm, nothing, AcousticReceiver(50.0, -10.0))
    x3 = transfercoef(datapm, nothing, AcousticReceiver(50.0, -15.0))
    x = transfercoef(datapm, nothing, [AcousticReceiver(50.0, -d) for d ∈ 5.0:5.0:15.0])
    @test x isa AbstractVector
    @test all(isapprox.([x1, x2, x3], x, atol= 0.000001))


    x = transfercoef(datapm, nothing, AcousticReceiverGrid2D(50.0, 0.0, 1, -5.0, -5.0, 3))
    @test x isa AbstractMatrix
    @test size(x) == (1, 3)
    @test all(isapprox.([x1 x2 x3], x, atol= 0.000001))


    x = transfercoef(datapm, nothing, AcousticReceiverGrid2D(50.0, 10.0, 3, -5.0, -5.0, 3))
    @test x isa AbstractMatrix
    @test size(x) == (3, 3)
    @test all(isapprox.([x1 x2 x3], x[1:1, :], atol= 0.000001))


    x1 = transmissionloss(datapm, nothing, AcousticReceiver(50.0, -5.0))
    x2 = transmissionloss(datapm, nothing, AcousticReceiver(50.0, -10.0))
    x3 = transmissionloss(datapm, nothing, AcousticReceiver(50.0, -15.0))
    x = transmissionloss(datapm, nothing, [AcousticReceiver(50.0, -d) for d ∈ 5.0:5.0:15.0])
    @test x isa AbstractVector
    @test all(isapprox.([x1, x2, x3], x, atol= 0.000001))

    x = transmissionloss(datapm, nothing, AcousticReceiverGrid2D(50.0, 0.0, 1, -5.0, -5.0, 3))
    @test x isa AbstractMatrix
    @test size(x) == (1, 3)
    @test all(isapprox.([x1 x2 x3], x, atol= 0.000001))

    x = transmissionloss(datapm, nothing, AcousticReceiverGrid2D(50.0, 10.0, 3, -5.0, -5.0, 3))
    @test x isa AbstractMatrix
    @test size(x) == (3, 3)
    @test all(isapprox.([x1 x2 x3], x[1:1,:], atol= 0.000001))
end


function test3d(datapm)
    x1 = transfercoef(datapm, nothing, AcousticReceiver(50.0, 0.0, -5.0))
    x2 = transfercoef(datapm, nothing, AcousticReceiver(50.0,0.0, -10.0))
    x3 = transfercoef(datapm, nothing, AcousticReceiver(50.0, 0.0, -15.0))
    x = transfercoef(datapm, nothing, [AcousticReceiver(50.0, 0.0, -d) for d ∈ 5.0:5.0:15.0])
    @test x isa AbstractVector
    @test all(isapprox.([x1, x2, x3], x, atol= 0.000001))


    x = transfercoef(datapm, nothing, AcousticReceiverGrid3D(50.0, 0.0, 1, 0.0, 1.0, 1, -5.0, -5.0, 3))
    @test x isa AbstractMatrix
    @test size(x) == (1, 3)
    @test all(isapprox.([x1 x2 x3], x, atol= 0.000001))

    x = transfercoef(datapm, nothing, AcousticReceiverGrid3D(50.0, 10.0, 3, 0.0, 1.0, 2, -5.0, -5.0, 3))
    @test x isa AbstractArray
    @test size(x) == (3, 2, 3)
    @test all(isapprox.([x1, x2, x3], x[1, 1,:], atol= 0.000001))

    x1 = transmissionloss(datapm, nothing, AcousticReceiver(50.0, 0.0,  -5.0))
    x2 = transmissionloss(datapm, nothing, AcousticReceiver(50.0, 0.0, -10.0))
    x3 = transmissionloss(datapm, nothing, AcousticReceiver(50.0, 0.0, -15.0))
    x = transmissionloss(datapm, nothing, [AcousticReceiver(50.0, 0.0, -d) for d ∈ 5.0:5.0:15.0])
    @test x isa AbstractVector
    @test all(isapprox.([x1, x2, x3], x, atol= 0.000001))


    x = transmissionloss(datapm, nothing, AcousticReceiverGrid3D(50.0, 0.0, 1, 0.0, 1.0, 1, -5.0, -5.0, 3))
    @test x isa AbstractMatrix
    @test size(x) == (1, 3)
    @test all(isapprox.([x1 x2 x3], x, atol= 0.000001))


    x = transmissionloss(datapm, nothing, AcousticReceiverGrid3D(50.0, 10.0, 3, 0.0, 1.0, 2, -5.0, -5.0, 3))
    @test x isa AbstractArray
    @test size(x) == (3, 2, 3)
    @test all(isapprox.([x1, x2, x3], x[1, 1,:], atol= 0.000001))
end


@test RayBasis2D in models()
@test RayBasis2DCurv in models()
@test RayBasis3D in models()
@test RayBasis3DRCNN in models()
@test GPR in models()


env = UnderwaterEnvironment()
pm = PekerisRayModel(env, 7)

Random.seed!(1)

txpos = [0.0, -5.0]
rxpos = rand(2, 500) .* [80.0, -20.0] .+ [1.0, 0.0]
tloss = Array{Float32}(undef, 1, size(rxpos)[2])
for i in 1 : 1 : size(rxpos)[2]
    tloss[1, i] = Float32(transmissionloss(pm, AcousticSource(txpos[1], txpos[2], 1000.0), AcousticReceiver(rxpos[1,i], rxpos[2,i]); mode=:coherent))
end
dataenv = DataDrivenUnderwaterEnvironment(rxpos, tloss; frequency = 1000.0, soundspeed = 1540.0);

datapm = RayBasis2D(dataenv; inilearnrate = 0.005, seed = true)
@test datapm isa RayBasis2D
test2d(datapm)
arr = arrivals(datapm, nothing, AcousticReceiver(50.0, -10.0))
@test arr isa AbstractVector{<:DataDrivenAcoustics.RayArrival}


datapm = RayBasis2DCurv(dataenv; inilearnrate = 0.005, seed = true)
@test datapm isa RayBasis2DCurv
test2d(datapm)
arr = arrivals(datapm, nothing, AcousticReceiver(50.0, -10.0))
@test arr isa AbstractVector{<:DataDrivenAcoustics.RayArrival}


kern = Matern(1/2, 0.0, 0.0)
datapm = GPR(dataenv, kern; logObsNoise = -5.0, seed = true, ratioₜ = 1.0)
@test datapm isa GPR
test2d(datapm)



Random.seed!(1)
txpos = [0.0, 0.0, -5.0]
rxpos = rand(3, 500) .* [100.0, 0.0, -20.0] .+ [1.0, 0.0, 0.0];
tloss = Array{Float32}(undef, 1, size(rxpos)[2])
for i in 1 : 1 : size(rxpos)[2]
    tloss[1, i] = Float32(transmissionloss(pm, AcousticSource(txpos[1], txpos[2], txpos[3], 1000.0), AcousticReceiver(rxpos[1,i], rxpos[2,i], rxpos[3,i]); mode=:coherent))
end

dataenv = DataDrivenUnderwaterEnvironment(rxpos, tloss; frequency = 1000.0, soundspeed = 1540.0, waterdepth = 20.0, tx = AcousticSource(0.0, 0.0, -5.0, 1000.0))
datapm = RayBasis3D(dataenv; inilearnrate = 0.005, seed  = true)
@test datapm isa RayBasis3D
test3d(datapm)
arr = arrivals(datapm, nothing, AcousticReceiver(50.0, 0.0, -10.0))
@test arr isa AbstractVector{<:DataDrivenAcoustics.RayArrival}



Random.seed!(1)

RCNN = Chain(  
    x -> (x ./ 0.5f0 .* π .- 0.5f0) .* 2.0f0, #normalization of incident angle
    Dense(1, 30, sigmoid),
    Dense(30, 50, sigmoid),  
    Dense(50, 2),
)
dataenv = DataDrivenUnderwaterEnvironment(rxpos, tloss; frequency = 1000.0, soundspeed = 1540.0, waterdepth = 20.0, tx = AcousticSource(0.0, 0.0, -5.0, 1000.0))
datapm = RayBasis3DRCNN(dataenv, RCNN; seed = true, inilearnrate = 0.05, ncount = 500)
@test datapm isa RayBasis3DRCNN
test3d(datapm)
arr = arrivals(datapm, nothing, AcousticReceiver(50.0, 0.0, -10.0))
@test arr isa AbstractVector{<:DataDrivenAcoustics.RayArrival}


kern = Matern(1/2, [0.0, 0.0, 0.0], 0.0)
datapm = GPR(dataenv, kern; logObsNoise = -5.0, seed = true, ratioₜ = 1.0)
@test datapm isa GPR
test3d(datapm)


