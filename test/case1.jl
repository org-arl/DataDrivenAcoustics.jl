using Test
using DataDrivenAcoustics
using UnderwaterAcoustics
using UnderwaterAcoustics: SandyClay, SampledField, PressureReleaseBoundary
using AcousticsToolbox
using Flux
using Statistics
using Random
using CSV
using DataFrames
using BSON

# Include core_v2 (unified API)
include("../src/core_v2.jl")

@testset "Case 1: Range-Dependent Bathymetry (Bellhop) - Env Fit" begin

    c = Float32(1541.0)
    f = Float32(10000.0)
    k = Float32(2.0) * π * f / c
    L = Float32(30.0)
    tx = Float32[0.0, 5.0]
    xmin = Float32(1000.0)
    xrange = Float32(50.0)
    zmin = Float32(0.0)
    zrange = L
    n_rays = 60

    env = UnderwaterEnvironment(
        surface = PressureReleaseBoundary,
        seabed = SandyClay,
        soundspeed = c,
        bathymetry = SampledField(Float32[40.0, 30.0, 33.0]; x = Float32[0.0, 550.0, 1100.0])
    )
    pm = AcousticsToolbox.Bellhop(env)

    rx = zig_zag_samples(xmin, xrange, Float32(0.05), Float32(1.0), Float32(29.0), Float32(0.5); T=Float32)
    data_dir = joinpath(@__DIR__, "..", "src", "data", "case1_far_field")
    TL_data = CSV.read(joinpath(data_dir, "A_train.csv"), DataFrame, header = false, types = Float32) |> Matrix

    rx_test, TL_test = generate_test_data(pm, tx, f, xmin, xrange, Float32(0.05), zmin, zrange, Float32(0.05))

    raw_rx = parse.(Float64, readlines(joinpath(data_dir, "capsule_rx_test.csv")))
    raw_TL = parse.(Float64, readlines(joinpath(data_dir, "capsule_TL_test.csv")))

    capsule_rx_test = reshape(raw_rx, size(rx_test))
    capsule_TL_test = reshape(raw_TL, size(TL_test))

    tx_dummy = nothing
    env_dd = BasicDataDrivenUnderwaterEnvironment(;
        soundspeed = c,
        frequency = f,
        tx = tx_dummy,
        dB = true,
        dims = 2
    )

    # Option 1: Random initialization (uncomment if no BSON available)
    # rbnn = RayBasis2DCurv(n_rays, k)

    # Option 2: Load initial weights from BSON
    bson_path = "src/bson_logs/case1/ini_RBNN.bson"
    @assert isfile(bson_path) "BSON file not found: $bson_path"
    data = BSON.load(bson_path)
    rbnn_loaded = data[:rbnn]
    rbnn = RayBasis2DCurv(rbnn_loaded.θ, rbnn_loaded.A, rbnn_loaded.ϕ, rbnn_loaded.d, rbnn_loaded.k)

    rbnn = fit!(env_dd, tx_dummy, rx, TL_data;
        model = rbnn,
        nrays = n_rays,
        initial_lr = Float32(0.5),
        threshold_count = 5000,
        threshold_lr = Float32(1e-6),
        show = false
    )

    data_loss_rbnn(x, y) = (Flux.Losses.mse(rbnn(x), y))^0.5f0

    test_loss = data_loss_rbnn(capsule_rx_test, capsule_TL_test)
    @test isfinite(test_loss)
end
