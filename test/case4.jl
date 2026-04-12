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
using DSP: amp2db, db2amp
using LinearAlgebra: norm

# Include core_v2 (unified API)
include("../src/core_v2.jl")

# ============================================================================
# CASE 4: Geo-acoustic Inversion for Seabed Properties
#
# This case learns physical seabed parameters (ρr, cr, δ) using a strict
# physics-based Rayleigh reflection model.
#
# Success criterion: Our trained weights should match the capsule's trained
# weights from src/data/case4/trained_weights_D.bson
# ============================================================================

@testset "Case 4: Geo-acoustic Inversion for Seabed Properties" begin

    # Load capsule's trained weights (target)
    function load_capsule_trained_weights()
        parsed = BSON.parse("src/bson_logs/case4/trained_weights_D.bson")
        data = parsed[:rbnn][:data]
        ρr = extract_array_from_bson(data[1])[1]
        cr = extract_array_from_bson(data[2])[1]
        δ = extract_array_from_bson(data[3])[1]
        k = reinterpret(Float32, UInt8.(data[4][:data]))[1]
        return (ρr=ρr, cr=cr, δ=δ, k=k)
    end

    capsule_weights = load_capsule_trained_weights()

    # Environmental parameters
    c = 1541.0f0
    f = 5000.0f0
    L = 30.0f0
    tx = Float32[0.0, 0.0, 15.0]
    n_rays = 60

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
    xrange = 100.0f0
    zmin = 1.0f0
    zrange = L - 2.0f0

    # Generate measurement locations
    rx = zig_zag_samples(xmin, xrange, 0.6f0, zmin, zrange, 1.0f0; IsTwoD = false, T = Float32)

    # Load acoustic measurements
    TL_data = CSV.read("src/data/case4/D_data.csv", DataFrame, header = false, types = Float32) |> Matrix

    # Convert TL to linear amplitude for training
    amp_data = db2amp.(TL_data)

    # Initialize model
    ω = 2.0f0 * Float32(π) * f
    k = ω / c

    # Option 1: Random/default initialization (uncomment if no BSON available)
    # rbnn = RayBasisRayleigh(k)

    # Option 2: Load initial weights from BSON
    bson_path = "src/bson_logs/case4/ini_geoacoustic_inversion.bson"
    @assert isfile(bson_path) "BSON file not found: $bson_path"
    parsed = BSON.parse(bson_path)
    data = parsed[:rbnn][:data]
    ρr_init = extract_array_from_bson(data[1])
    cr_init = extract_array_from_bson(data[2])
    δ_init = extract_array_from_bson(data[3])
    k_loaded = reinterpret(Float32, UInt8.(data[4][:data]))[1]
    rbnn = RayBasisRayleigh(ρr_init, cr_init, δ_init, k_loaded)

    # Train using fit! - model is callable, no fwd_model needed
    rbnn = fit!(env, rx, amp_data, rbnn;
        n_rays = n_rays,
        initial_lr = 0.5f0,
        threshold_count = 5000,
        threshold_lr = 1e-5,
        show = false
    )

    # Extract learned parameters
    learned_ρr = abs(rbnn.ρᵣ[1])
    learned_cr = abs(rbnn.cᵣ[1])
    learned_δ = abs(rbnn.δ[1])

    # Tests
    @test isapprox(learned_ρr, capsule_weights.ρr, rtol=0.001)
    @test isapprox(learned_cr, capsule_weights.cr, rtol=0.001)
    @test isapprox(learned_δ, capsule_weights.δ, rtol=0.01)
end
