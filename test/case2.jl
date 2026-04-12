using Test
using DataDrivenAcoustics
using UnderwaterAcoustics
using Flux
using Statistics
using Random
using CSV
using DataFrames
using BSON
using DSP: amp2db
using LinearAlgebra: norm

# Include core_v2 (unified API)
include("../src/core_v2.jl")

# ============================================================================
# CASE 2: Near-Field 3D Environment Fitting
#
# This case fits a 3D ray basis model to near-field acoustic data.
# Uses Float64 precision due to large phase values (~2000-3000 radians).
# ============================================================================

@testset "Case 2: Near-Field 3D (Env Fit)" begin

    # Environmental parameters
    c = 1541.0
    f = 5000.0
    L = 30.0
    tx = Float64[0.0, 0.0, 15.0]
    n_rays = 60
    xₒ = Float64[0.0, 0.0, 0.0]

    # Create environment object
    env = BasicDataDrivenUnderwaterEnvironment(;
        soundspeed = c,
        frequency = f,
        waterdepth = L,
        tx = tx,
        dims = 3
    )

    # Sampling parameters
    xmin = 100.0
    xrange = 50.0
    zmin = 1.0
    zrange = L - 2.0

    # Generate measurement locations
    rx = zig_zag_samples(xmin, xrange, 0.3, zmin, zrange, 0.67; IsTwoD = false, T = Float64)

    # Load acoustic measurements
    data_dir = joinpath(@__DIR__, "..", "src", "data", "case2_near_field")
    TL_data = Float64.(CSV.read(joinpath(data_dir, "B_data.csv"), DataFrame, header = false, types = Float32) |> Matrix)

    # Load capsule test data
    raw_rx = parse.(Float64, readlines(joinpath(data_dir, "capsule_rx_test.csv")))
    raw_TL = parse.(Float64, readlines(joinpath(data_dir, "capsule_TL_test.csv")))
    capsule_rx_test = reshape(raw_rx, 3, :)
    capsule_TL_test = reshape(raw_TL, 1, :)

    # Calculate nominal arrival direction and distance of rays
    image_src = n_images_src(Float64[xmin, 0.0, abs(zmin) + zrange / 2.0], tx, L, n_rays; T = Float64)
    nominal_ρ, nominal_θ, nominal_ψ = cartesian2spherical(xₒ .- image_src)

    # Initialize model
    ω = 2.0 * π * f
    k = ω / c
    Random.seed!(size(rx, 2))

    # Option 1: Random initialization (uncomment if no BSON available)
    # rbnn = RayBasis3d(n_rays, k)

    # Option 2: Load initial weights from BSON
    bson_path = "src/bson_logs/case2/ini_RBNN.bson"
    @assert isfile(bson_path) "BSON file not found: $bson_path"
    data = BSON.load(bson_path)
    rbnn_f32 = data[:rbnn]
    rbnn = RayBasis3d(
        Float64.(rbnn_f32.eθ),
        Float64.(rbnn_f32.eψ),
        Float64.(rbnn_f32.ed),
        Float64.(rbnn_f32.A),
        Float64.(rbnn_f32.ϕ),
        Float64(rbnn_f32.k)
    )

    # Train using fit! - model is callable, no fwd_model needed
    rbnn = fit!(env, rx, TL_data, rbnn;
        nominal_ρ = nominal_ρ,
        nominal_θ = nominal_θ,
        nominal_ψ = nominal_ψ,
        xₒ = xₒ,
        l1_reg = 1.0,
        initial_lr = 0.5,
        threshold_count = 5000,
        threshold_lr = 1e-6,
        show = false
    )

    # Evaluate on capsule test data using callable model
    test_loss = sqrt(Flux.Losses.mse(
        rbnn(capsule_rx_test, env; xₒ=xₒ, nominal_ρ=nominal_ρ, nominal_θ=nominal_θ, nominal_ψ=nominal_ψ),
        capsule_TL_test
    ))
    @test test_loss < 2.1  # Should match capsule's ~2.08
end
