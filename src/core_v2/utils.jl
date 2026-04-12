# ============================================================================
# Utilities and Helper Functions
# ============================================================================

# ============================================================================
# Legacy ADAM Optimizer
# ============================================================================

"""
    LegacyADAM

Custom ADAM optimizer matching the original implementation behavior.
Used for Case 1 training to ensure reproducibility.
"""
mutable struct LegacyADAM <: AbstractOptimiser
    eta::Float64
    beta::Tuple{Float64, Float64}
    epsilon::Float64
    state::IdDict{Any, Any}
end

LegacyADAM(η = 0.001, β = (0.9, 0.999), ϵ = 1e-8) = LegacyADAM(η, β, ϵ, IdDict())

function apply!(o::LegacyADAM, x, Δ)
    η, β, ϵ = o.eta, o.beta, o.epsilon
    mt, vt, βp = get!(o.state, x) do
        (zero(x), zero(x), Float64[β[1], β[2]])
    end
    @. mt = β[1] * mt + (1 - β[1]) * Δ
    @. vt = β[2] * vt + (1 - β[2]) * Δ * conj(Δ)
    @. Δ =  mt / (1 - βp[1]) / (sqrt(vt / (1 - βp[2])) + ϵ) * η
    βp .= βp .* β
    return Δ
end

# ============================================================================
# Data Utilities
# ============================================================================

"""
    zig_zag_samples(xmin, xrange, xscale, zmin, zrange, zscale; IsTwoD=true, T=Float64)

Generate zig-zag sampling pattern for receiver positions.
Useful for creating training data that covers the domain efficiently.

# Arguments
- `xmin`: Minimum x coordinate
- `xrange`: Range in x direction
- `xscale`: Step size in x direction
- `zmin`: Minimum z coordinate
- `zrange`: Range in z direction
- `zscale`: Step size in z direction

# Keyword Arguments
- `IsTwoD`: Return 2D (true) or 3D (false) coordinates
- `T`: Numeric type (default: Float64)

# Returns
- Matrix of receiver positions (2 x N or 3 x N)
"""
function zig_zag_samples(xmin, xrange, xscale, zmin, zrange, zscale; IsTwoD = true, T = Float64)
    vt = zrange / zscale
    ht =  xrange / xscale
    z_in = collect(T, zmin : zscale : zmin + zrange)
    z_de = collect(T, zmin + zrange : -zscale : zmin)
    z = Array{T}(undef, 0)
    for i in 1 : 1 : ceil(ht / vt)
        iseven(i) ? (z = vcat(z, z_de)) : (z = vcat(z, z_in))
    end
    z = rle(z)[1]
    x = collect(T, xmin : xscale : xmin + xscale * (length(z) - 1))
    idx = findall(x .< (xmin + xrange))
    (IsTwoD == true) ? (return vcat(x', z')[:, idx]) : (return vcat(x', zeros(T, 1, length(x)), z')[:, idx])
end

"""
    data_split(rx; ratio=0.7)

Split receiver positions into training and validation sets.

# Arguments
- `rx`: Receiver positions (D x N matrix)

# Keyword Arguments
- `ratio`: Fraction for training (default: 0.7)

# Returns
- `(rx_train, rx_val)`: Tuple of training and validation positions
"""
function data_split(rx; ratio = 0.7)
    data_len = size(rx)[2]
    Random.seed!(data_len)
    data_idx = randperm(data_len)
    idx_train = data_idx[1 : Int(floor(data_len * ratio))]
    idx_val = data_idx[Int(floor(data_len * ratio)) + 1 : end]

    rx_train = rx[:,idx_train]
    rx_val =  rx[:, idx_val]

    return rx_train, rx_val
end

"""
    cartesian2spherical(pos)

Convert 3D Cartesian coordinates to spherical coordinates.

# Arguments
- `pos`: Position matrix (3 x N)

# Returns
- `(d, θ, ψ)`: Distance, azimuthal angle, elevation angle
"""
function cartesian2spherical(pos)
    x = @view pos[1,:]
    y = @view pos[2,:]
    z = @view pos[3,:]
    d = norm.(eachcol(pos))
    θ = atan.(y, x)
    ψ = atan.(norm.(eachcol(pos[1:2,:])), z)
    return d, θ, ψ
end

"""
    n_images_src(rx, tx, D, n_rays; T=Float64)

Compute image sources for ray tracing using the image source method.

# Arguments
- `rx`: Representative receiver position
- `tx`: Transmitter position
- `D`: Water depth
- `n_rays`: Number of rays to select

# Returns
- Matrix of image source positions (3 x n_rays)
"""
function n_images_src(rx, tx, D, n_rays; T = Float64)
    a = T(20.0)
    count = 0
    s = 2 * (Int(a) * 2 + 1)
    all_image_src = zeros(T, 3, s)
    image_src_amp = zeros(T, s)
    for w = T(0.0) : T(1.0)
        for n = -a : a
            count += 1
            image_src = T[tx[1], tx[2], (T(1.0) - T(2.0) * w) * tx[3] + T(2.0) * n * D]
            d = norm(image_src .- rx)
            image_src_amp[count] = T(0.2)^abs(n) * T(0.99)^abs(n - w) / d
            all_image_src[:, count] = image_src
        end
    end
    idx = sortperm(abs.(image_src_amp), rev = true)[1:n_rays]
    return all_image_src[:, idx]
end

"""
    n_images_src_with_ref(rx, tx, D, n_rays)

Compute image sources with reflection counts for geo-acoustic inversion.

# Arguments
- `rx`: Representative receiver position
- `tx`: Transmitter position
- `D`: Water depth
- `n_rays`: Number of rays to select

# Returns
- `(image_sources, ref)`: Image source positions and reflection counts [surface, bottom]
"""
function n_images_src_with_ref(rx, tx, D, n_rays)
    a = 20.0f0
    count = 0
    s = 2 * (Int(a) * 2 + 1)
    all_image_src = zeros(Float32, 3, s)
    image_src_amp = zeros(Float32, s)
    ref = zeros(Float32, 2, s)
    for w = 0.0f0 : 1.0f0
        for n = -a : a
            count += 1
            image_src = [tx[1], tx[2], (1.0f0 - 2.0f0 * w) * tx[3] + 2.0f0 * n * D]
            d = norm(image_src .- rx)
            image_src_amp[count] = 0.2f0^(abs(n)) * (0.99f0)^(abs(n - w)) / d
            all_image_src[:, count] = image_src
            ref[:, count] = [abs(n - w), abs(n)]
        end
    end
    idx = sortperm(abs.(image_src_amp), rev = true)[1:n_rays]
    same_idx = findall(ref[1, idx] .== ref[2, idx])
    selected_image_src = all_image_src[:, idx]
    for i in 2:2:length(same_idx)
        i == length(same_idx) && break
        if selected_image_src[3, same_idx[i]] < 0
            selected_image_src[3, same_idx[i]] = all_image_src[3, idx][same_idx[i+1]]
            selected_image_src[3, same_idx[i+1]] = all_image_src[3, idx][same_idx[i]]
        end
    end
    return selected_image_src, ref[:, idx]
end

# ============================================================================
# Physics Helpers
# ============================================================================

"""
    reflectioncoef(θ, ρr, cr, δ)

Compute Rayleigh reflection coefficient.
Wrapper for UnderwaterAcoustics.reflection_coef with sign convention fix.

# Arguments
- `θ`: Incident angle (radians)
- `ρr`: Density ratio (seabed/water)
- `cr`: Sound speed ratio (seabed/water)
- `δ`: Attenuation coefficient
"""
function reflectioncoef(θ, ρr, cr, δ)
    return UnderwaterAcoustics.reflection_coef(θ, ρr, cr, -δ)
end

# ============================================================================
# BSON Helpers
# ============================================================================

"""
    extract_array_from_bson(arr_dict)

Extract array from BSON parsed format.
Handles Flux version mismatches in saved models.

# Arguments
- `arr_dict`: Dictionary from BSON.parse

# Returns
- Extracted array or nothing if format doesn't match
"""
function extract_array_from_bson(arr_dict)
    isa(arr_dict, Dict) || return nothing
    haskey(arr_dict, :data) || return nothing
    haskey(arr_dict, :size) || return nothing
    raw_data = arr_dict[:data]
    size_arr = arr_dict[:size]
    arr = reinterpret(Float32, UInt8.(raw_data))
    length(size_arr) == 1 ? collect(arr) : collect(reshape(arr, Tuple(size_arr)...))
end

# ============================================================================
# Test Data Generation
# ============================================================================

"""
    generate_test_data(pm, tx, f, xmin, xrange, xs, zmin, zrange, zs; IsTwoD=true)

Generate test data using a propagation model (e.g., Bellhop).

# Arguments
- `pm`: Propagation model
- `tx`: Transmitter position
- `f`: Frequency
- `xmin`, `xrange`, `xs`: X grid parameters
- `zmin`, `zrange`, `zs`: Z grid parameters

# Returns
- `(rx_test, TL_test)`: Test receiver positions and transmission loss
"""
function generate_test_data(pm, tx, f, xmin, xrange, xs, zmin, zrange, zs; IsTwoD = true)
    x_range = xmin:xs:xmin + xrange
    z_range = -(zmin + zrange):zs:-zmin
    rx = AcousticReceiverGrid2D(x_range, z_range)
    TL = -transmission_loss(pm, AcousticSource(tx[1], -tx[end], f), rx)

    x = collect(Float32, x_range)'
    z = collect(Float32, zmin + zrange:-zs:zmin)'
    IsTwoD == true ?
        (rx_test = vcat(repeat(x, 1, length(z)), repeat(z, inner = (1, length(x))))) :
        (rx_test = vcat(repeat(x, 1, length(z)), zeros(Float32, 1, length(x) * length(z)), repeat(z, inner = (1, length(x)))))
    return rx_test, reshape(TL, 1, length(TL))
end

function remove_consecutive_duplicates(v)
    out = Vector{eltype(v)}()
    for item in v
        if isempty(out) || out[end] != item
            push!(out, item)
        end
    end
    return out
end
