using Test
using DataDrivenAcoustics
using UnderwaterAcoustics
using UnderwaterAcoustics: absorption
using Flux
using Statistics
using Random
using CSV
using DataFrames
using BSON
using DSP: amp2db
using LinearAlgebra: norm
using Zygote

# Include core_v2 (unified API)
include("../src/core_v2.jl")

# ============================================================================
# CASE 3: Geo-acoustic Inversion for Seabed Reflection Model
#
# This case learns an unknown seabed reflection coefficient using a
# Reflection Coefficient Neural Network (RCNN).
# ============================================================================

@testset "Case 3: Geo-acoustic Inversion (RCNN)" begin

    # Environmental parameters
    c = 1541.0f0
    f = 5000.0f0
    L = 30.0f0
    tx = Float32[0.0, 0.0, 15.0]
    n_rays = 60
    xₒ = Float32[0.0, 0.0, 0.0]

    # Create environment object
    env = BasicDataDrivenUnderwaterEnvironment(;
        soundspeed = c,
        frequency = f,
        waterdepth = L,
        tx = tx,
        dims = 3
    )

    # Sampling parameters
    xmin = 0.0f0
    xrange = 300.0f0
    zmin = 1.0f0
    zrange = L - 2.0f0

    # Generate measurement locations
    rx = zig_zag_samples(xmin, xrange, 0.27f0, zmin, zrange, 0.6f0; IsTwoD = false, T = Float32)

    # Load acoustic measurements
    data_dir = joinpath(@__DIR__, "..", "src", "data", "case3")
    TL_data = CSV.read(joinpath(data_dir, "C_data.csv"), DataFrame, header = false, types = Float32) |> Matrix

    # Calculate nominal arrival direction and distance of rays
    image_src, ref = n_images_src_with_ref([xmin, 0.0f0, abs(zmin) + zrange / 2.0f0], tx, L, n_rays)
    nominal_ρ, nominal_θ, nominal_ψ = cartesian2spherical(xₒ .- image_src)

    # Initialize model - RCNN is embedded in RayBasisRCNN
    ω = 2.0f0 * Float32(π) * f
    k = ω / c

    # Option 1: Random initialization (uncomment if no BSON available)
    # rbnn = RayBasisRCNN(n_rays, k)

    # Option 2: Load initial RCNN weights from BSON
    bson_path = "src/bson_logs/case3/ini_rc_inversion_RCNN.bson"
    @assert isfile(bson_path) "BSON file not found: $bson_path"
    parsed = BSON.parse(bson_path)
    layers = parsed[:RCNN][:data][1][:data]

    W1 = extract_array_from_bson(layers[2][:data][1])
    b1 = extract_array_from_bson(layers[2][:data][2])
    W2 = extract_array_from_bson(layers[3][:data][1])
    b2 = extract_array_from_bson(layers[3][:data][2])
    W3 = extract_array_from_bson(layers[4][:data][1])
    b3 = extract_array_from_bson(layers[4][:data][2])

    rcnn = Chain(
        x -> (x ./ 0.5f0 .* Float32(π) .- 0.5f0) .* 2.0f0,
        Dense(W1, b1, sigmoid),
        Dense(W2, b2, sigmoid),
        Dense(W3, b3),
    )

    # Create RayBasisRCNN with embedded rcnn
    rbnn = RayBasisRCNN(n_rays, k; rcnn=rcnn)

    # Train using fit! - model is callable, no fwd_model needed
    rbnn = fit!(env, rx, TL_data, rbnn;
        nominal_ρ = nominal_ρ,
        nominal_θ = nominal_θ,
        nominal_ψ = nominal_ψ,
        n_rays = n_rays,
        xₒ = xₒ,
        initial_lr = 0.05f0,
        threshold_count = 5000,
        threshold_lr = 1e-6,
        show = false
    )

    # Compute final loss for testing using callable model
    rx_train, rx_val = data_split(rx)
    TL_val = TL_data[1+size(rx_train)[2]:end]'
    final_val_rmse = sqrt(Flux.Losses.mse(
        rbnn(rx_val, env; xₒ=xₒ, nominal_ρ=nominal_ρ, nominal_θ=nominal_θ, nominal_ψ=nominal_ψ, n_rays=n_rays),
        TL_val
    ))

    @test isfinite(final_val_rmse)
    @test final_val_rmse < 2.0
end
