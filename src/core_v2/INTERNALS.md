# DataDrivenAcoustics.jl — Internals

This document covers the internal architecture, shared functions, training pipeline, and design decisions. It is intended for developers who need to understand, debug, or extend the codebase.

## Shared Functions & Utilities

All utilities live in `src/core_v2/utils.jl` unless otherwise noted.

---

### `LegacyADAM(η=0.001, β=(0.9, 0.999), ϵ=1e-8) -> LegacyADAM`

Custom ADAM optimizer that replicates the behavior of the original CodeOcean capsule implementation. Used exclusively by Case 1 (`train_rbnn!`).

**Fields:**
- `eta::Float64` — learning rate (mutable)
- `beta::Tuple{Float64, Float64}` — exponential decay rates for first and second moment estimates
- `epsilon::Float64` — numerical stability constant
- `state::IdDict{Any, Any}` — per-parameter optimizer state `(mt, vt, βp)`

The update rule is standard ADAM with bias correction: `Δ = η * m̂t / (√v̂t + ε)`. The key implementation detail: bias correction powers `βp` are tracked as a mutable 2-element `Float64` array per parameter, updated as `βp .= βp .* β` each step. This differs from `Flux.Adam`, which tracks the step count and computes `β^t` on the fly. The floating-point ordering difference produces slightly different training trajectories.

**Why it exists:** The original capsule used a custom ADAM. Switching to `Flux.Adam` breaks numerical reproducibility of published results. Cases 2, 3, and 4 use `Flux.Adam` because they were developed after the capsule publication.

**Used by:** Case 1 (via `train_rbnn!`).

---

### `zig_zag_samples(xmin, xrange, xscale, zmin, zrange, zscale; IsTwoD=true, T=Float64) -> Matrix{T}`

Generate a zig-zag sampling pattern for receiver positions that efficiently covers a 2D spatial domain.

**Arguments:**
- `xmin::Real` — minimum range coordinate
- `xrange::Real` — total range extent
- `xscale::Real` — step size in range direction
- `zmin::Real` — minimum depth coordinate
- `zrange::Real` — total depth extent
- `zscale::Real` — step size in depth direction
- `IsTwoD::Bool` — if `true`, returns `2 x N`; if `false`, returns `3 x N` (with zero y-coordinate)
- `T::Type` — numeric type for the output

**Returns:** `Matrix{T}` of size `2 x N` or `3 x N`, where each column is a receiver position.

The pattern alternates ascending and descending depth sweeps as range increases, producing a zig-zag that covers the range-depth plane without redundant sampling. Consecutive duplicate depth values are removed via `rle` (run-length encoding from StatsBase).

**Used by:** Cases 1, 2, 3, 4.

---

### `data_split(rx; ratio=0.7) -> (rx_train, rx_val)`

Split receiver positions into training and validation sets using a deterministic random permutation.

**Arguments:**
- `rx::Matrix` — receiver positions (`D x N`)
- `ratio::Float64` — fraction allocated to training (default: 0.7)

**Returns:** Tuple `(rx_train, rx_val)` with the same row dimension as `rx`.

**Important implementation detail:** The random seed is set to `size(rx, 2)` (the number of receivers), making the split deterministic for a given dataset size. The corresponding TL data must be split using the same indices — `fit!` methods handle this by slicing `TL_data[1:size(rx_train,2)]` and `TL_data[size(rx_train,2)+1:end]`, which relies on `data_split` returning columns in the order defined by `randperm`.

**vs. old implementation:** The old `SplitData` function (`src/pm_utility.jl`) split both locations and measurements together and used a fixed seed (`Random.seed!(8)`). The new `data_split` only splits positions; measurement indices are computed in `fit!` via array slicing, decoupling data splitting from the measurement data structure.

**Used by:** Cases 1, 2, 3, 4.

---

### `cartesian2spherical(pos) -> (d, θ, ψ)`

Convert 3D Cartesian coordinates to spherical coordinates.

**Arguments:**
- `pos::Matrix` — position matrix (`3 x N`)

**Returns:**
- `d::Vector` — radial distances (Euclidean norms via `norm`)
- `θ::Vector` — azimuthal angles (radians), computed as `atan(y, x)`
- `ψ::Vector` — elevation angles (radians), computed as `atan(√(x²+y²), z)`

**vs. old implementation:** The old version returned `(θ, ψ, d)`. The new version returns `(d, θ, ψ)` matching the `nominal_ρ, nominal_θ, nominal_ψ` naming convention used in `fit!`.

**Used by:** Cases 2, 3, 4 (to convert image source positions to nominal ray parameters).

---

### `n_images_src(rx, tx, D, n_rays; T=Float64) -> Matrix{T}`

Compute image source positions for ray tracing using the image source method.

**Arguments:**
- `rx::Vector` — representative receiver position (used for distance-based ranking)
- `tx::Vector` — transmitter position
- `D::Real` — water depth
- `n_rays::Integer` — number of image sources to select
- `T::Type` — numeric type

**Returns:** `Matrix{T}` of size `3 x n_rays`, containing the positions of the `n_rays` strongest image sources.

Image sources are generated for reflection orders `n = -20:20` with both surface and bottom reflections (`w ∈ {0, 1}`), yielding 82 candidates. Each image source's strength is estimated as `0.2^|n| * 0.99^|n-w| / d`, where `n` is the reflection order, `w` indicates surface (`0`) or bottom (`1`) first reflection, and `d` is the distance to the receiver. The strongest `n_rays` are selected by sorting on estimated amplitude.

**Physics assumption:** The strength estimate uses hardcoded reflection coefficients (0.2 for bottom, 0.99 for surface) as a rough ranking heuristic — these don't affect the trained model, only which image sources are selected as ray basis functions.

**Used by:** Case 2.

---

### `n_images_src_with_ref(rx, tx, D, n_rays) -> (Matrix{Float32}, Matrix{Float32})`

Compute image source positions with reflection counts, for models that need to know how many times each ray reflects off the surface and bottom.

**Arguments:**
- `rx::Vector` — representative receiver position
- `tx::Vector` — transmitter position
- `D::Real` — water depth
- `n_rays::Integer` — number of image sources to select

**Returns:**
- `image_sources::Matrix{Float32}` — `3 x n_rays` image source positions
- `ref::Matrix{Float32}` — `2 x n_rays` reflection counts `[surface_bounces; bottom_bounces]`

Same image source generation as `n_images_src`, but always uses `Float32` and also returns reflection counts needed by Cases 3 and 4 to compute reflection-dependent amplitude losses. Includes a post-processing step that swaps z-coordinates for certain degenerate image pairs (where surface and bottom bounce counts are equal).

**Used by:** Cases 3, 4.

---

### `reflectioncoef(θ, ρr, cr, δ) -> Complex`

Compute the Rayleigh reflection coefficient for a plane wave incident on the seabed.

**Arguments:**
- `θ::Real` — incident angle (radians, measured from horizontal)
- `ρr::Real` — density ratio (seabed / water)
- `cr::Real` — sound speed ratio (seabed / water)
- `δ::Real` — attenuation coefficient

**Returns:** Complex reflection coefficient.

Thin wrapper around `UnderwaterAcoustics.reflection_coef` that negates `δ` to match the sign convention used in the reference paper.

**Used by:** Case 4.

---

### `get_tx_coords(env) -> Vector` *(in `src/core_v2/forward.jl`)*

Extract transmitter coordinates from an environment object.

**Arguments:**
- `env` — environment object with a `.tx` field

**Returns:** Vector of transmitter coordinates.

Handles both `AcousticSource` objects (extracts location via `UnderwaterAcoustics.location`) and raw coordinate vectors. Errors if `env.tx` is `missing`.

**Used by:** Cases 3, 4 (inside the forward pass).

---

### `extract_array_from_bson(arr_dict) -> Union{Array{Float32}, Nothing}`

Extract a numeric array from BSON's parsed dictionary format.

**Arguments:**
- `arr_dict::Dict` — dictionary with `:data` (raw bytes) and `:size` (dimensions)

**Returns:** `Array{Float32}` reshaped to the stored dimensions, or `nothing` if the format doesn't match.

Handles Flux version mismatches in saved model files by manually reinterpreting raw byte data (`UInt8 -> Float32`) rather than relying on BSON's struct deserialization.

**Used by:** Cases 3, 4 (for loading initial weights from BSON files).

---

### `generate_test_data(pm, tx, f, xmin, xrange, xs, zmin, zrange, zs; IsTwoD=true) -> (rx_test, TL_test)`

Generate a grid of test data using an external propagation model (e.g., Bellhop).

**Arguments:**
- `pm` — propagation model (e.g., `AcousticsToolbox.Bellhop`)
- `tx::Vector` — transmitter position `[range, depth]`
- `f::Real` — frequency (Hz)
- `xmin, xrange, xs` — range grid parameters (min, extent, step)
- `zmin, zrange, zs` — depth grid parameters (min, extent, step)
- `IsTwoD::Bool` — 2D or 3D output format

**Returns:**
- `rx_test::Matrix{Float32}` — receiver positions on a regular grid
- `TL_test::Matrix` — transmission loss values (`1 x N`)

**Used by:** Case 1 (test data generation).

---

### `remove_consecutive_duplicates(v) -> Vector`

Remove consecutive duplicate elements from a vector.

**Arguments:**
- `v::Vector` — input vector

**Returns:** Vector with consecutive duplicates removed.

<!-- TODO: clarify — this function appears unused in the core v2 code path -->

## Model Structs

All structs are defined in `src/core_v2/models.jl`. Every model inherits from `PropagationModel` (defined in `src/core_v2.jl`).

---

### `RayBasis{T1<:AbstractVector, T2<:Real} <: PropagationModel`

**Purpose:** Legacy 2D far-field ray basis model, retained for BSON compatibility with older saved models.

| Field | Type | Description | Trainable |
|-------|------|-------------|-----------|
| `θ` | `T1` | Azimuthal angles of arrival rays (radians) | Yes |
| `A` | `T1` | Amplitudes of arrival rays | Yes |
| `ϕ` | `T1` | Phases of arrival rays (radians) | Yes |
| `d` | `T1` | Distances for wavefront curvature modeling (meters) | Yes |
| `k` | `T2` | Angular wavenumber (rad/m) | No (excluded from `Flux.trainable`) |

**Outer constructor:** `RayBasis(rays::Integer, k::Real)` — initializes θ, ϕ in `[0, π)`, A and d in `[0, 1)`, all `Float32`.

**vs. old implementation:** The old `RayBasis2D` (`src/pm_RBNN.jl`) stored `env`, a `calculatefield` function, `nrays`, and a dynamic `trainable` tuple. The new `RayBasis` removes env storage (environment is passed to `fit!`), removes the `calculatefield` indirection (models are callable structs), and fixes trainable parameters at compile time via `Flux.trainable()`.

---

### `RayBasis2DCurv{T1<:AbstractVector, T2<:Real} <: PropagationModel`

**Purpose:** 2D far-field ray basis model with wavefront curvature. Preferred over `RayBasis` for new code. Identical field structure.

| Field | Type | Description | Trainable |
|-------|------|-------------|-----------|
| `θ` | `T1` | Azimuthal angles of arrival rays (radians) | Yes |
| `A` | `T1` | Amplitudes of arrival rays | Yes |
| `ϕ` | `T1` | Phases of arrival rays (radians) | Yes |
| `d` | `T1` | Distances for wavefront curvature modeling (meters) | Yes |
| `k` | `T2` | Angular wavenumber (rad/m) | No |

**Outer constructor:** `RayBasis2DCurv(rays::Integer, k::Real)` — same initialization as `RayBasis`.

**vs. old implementation:** The old `RayBasis2DCurv` (`src/pm_RBNN.jl`) used a `@kwdef` struct with 10+ type parameters and called `ModelFit!` inside the constructor, coupling model construction to training. The new struct is a simple parameter container; training is decoupled and invoked explicitly via `fit!()`.

---

### `RayBasis3d{T1<:AbstractVector, T2<:Real} <: PropagationModel`

**Purpose:** 3D near-field ray basis model. Uses nominal ray parameters from the image source method; learns amplitude and phase corrections.

| Field | Type | Description | Trainable |
|-------|------|-------------|-----------|
| `eθ` | `T1` | Error to nominal azimuthal angle (radians) | No |
| `eψ` | `T1` | Error to nominal elevation angle (radians) | No |
| `ed` | `T1` | Error to nominal propagation distance (meters) | No |
| `A` | `T1` | Amplitudes of arrival rays | Yes |
| `ϕ` | `T1` | Phases of arrival rays (radians) | Yes |
| `k` | `T2` | Angular wavenumber (rad/m) | No |

**Outer constructor:** `RayBasis3d(rays::Integer, k::Real)` — geometry errors initialized to zero (`Float64`), A and ϕ randomly initialized.

**Precision constraint:** Uses `Float64` because phase values (`k * l`) can reach ~2000–3000 radians at 5 kHz over 100+ m ranges. `Float32` (~7 decimal digits) loses gradient precision in the fractional part of these large phase values; `Float64` (~16 digits) preserves it.

**vs. old implementation:** The old `RayBasis3D` stored both nominal (`θ, ψ, d`) and error (`eθ, eψ, ed`) fields. The new `RayBasis3d` stores only error fields; nominal values (`nominal_ρ, nominal_θ, nominal_ψ`) are computed once from image sources and passed as keyword arguments to the forward pass and `fit!`, eliminating redundant storage.

---

### `RayBasisRCNN{T1<:AbstractVector, T2<:Real, C} <: PropagationModel`

**Purpose:** 3D ray basis model with a Reflection Coefficient Neural Network (RCNN) that learns the seabed reflection coefficient as a function of incident angle.

| Field | Type | Description | Trainable |
|-------|------|-------------|-----------|
| `eθ` | `T1` | Error to nominal azimuthal angle (radians) | No |
| `eψ` | `T1` | Error to nominal elevation angle (radians) | No |
| `ed` | `T1` | Error to nominal propagation distance (meters) | No |
| `k` | `T2` | Angular wavenumber (rad/m) | No |
| `rcnn` | `C` | Neural network: angle -> `[|RC|, phase_shift]` | Yes |

**Outer constructor:** `RayBasisRCNN(rays::Integer, k::Real; rcnn=nothing)` — if `rcnn` is not provided, creates a default architecture:
1. Input normalization: `x -> (x / 0.5 * π - 0.5) * 2`
2. `Dense(1, 30, sigmoid)`
3. `Dense(30, 50, sigmoid)`
4. `Dense(50, 2)`

**Constraint:** Only `rcnn` is in `Flux.trainable`. Geometry parameters are fixed from image source computation.

**vs. old implementation:** The old `RayBasis3DRCNN` required RCNN to be passed separately and computed image sources inside the constructor via `find_image_src()`. The new struct accepts an optional `rcnn` kwarg and stores it internally. Image source computation is moved to user code.

---

### `RayBasisRayleigh{T1<:AbstractVector, T2<:Real} <: PropagationModel`

**Purpose:** 3D ray basis model that learns physical seabed parameters directly, using the Rayleigh reflection coefficient formula.

| Field | Type | Description | Trainable |
|-------|------|-------------|-----------|
| `ρᵣ` | `T1` | Density ratio (seabed/water), 1-element vector | Yes |
| `cᵣ` | `T1` | Sound speed ratio (seabed/water), 1-element vector | Yes |
| `δ` | `T1` | Attenuation coefficient, 1-element vector | Yes |
| `k` | `T2` | Angular wavenumber (rad/m) | No |

**Outer constructor:** `RayBasisRayleigh(k::Real)` — initializes all three parameters to `[1.0f0]`.

**Design note:** Parameters are stored as 1-element vectors (not scalars) so that `Flux.params` can track and mutate them during training. The forward pass wraps each in `abs()` to ensure physical positivity.

**vs. old implementation:** This model type did not exist in the old implementation. It was added to enable direct inversion for physical seabed parameters using the Rayleigh formula, rather than learning an arbitrary neural network approximation (Case 3).

## Forward Pass

Defined in `src/core_v2/forward.jl`. Each model type is a callable struct following the Flux.jl pattern.

---

### `RayBasis` / `RayBasis2DCurv` — 2D Forward Pass

```julia
(r::RayBasis2DCurv)(xy::AbstractArray) -> Matrix
```

**Input:** `xy` — `2 x N` matrix (row 1: range, row 2: depth).

**Computation:**
1. For each ray `i`, compute the virtual source position: `(x0 - d[i]*cos(θ[i]), y0 - d[i]*sin(θ[i]))` where `x0 = [0, 0]`.
2. Compute the distance `l[i,j]` from each virtual source to each receiver.
3. Compute the complex field contribution: `A[i] * cis(k * l[i,j] + ϕ[i])`.
4. Sum across all rays and convert to dB: `amp2db(|Σ_i contribution_i|)`.

The wavefront curvature is modeled through the `d` parameter: each ray appears to originate from a point source at distance `d` behind the wavefront, producing curved (spherical) rather than planar wavefronts.

**No geometric spreading:** Unlike the 3D models, there is no `1/l` factor. In the far-field, the source is distant (~1000 m) and the measurement region is small (~50 m), so distance variations are negligible relative to total path length. Geometric spreading loss is effectively constant across receivers and absorbed into the trainable amplitude `A`.

**Output:** `1 x N` matrix of transmission loss in dB.

**Note:** `RayBasis` and `RayBasis2DCurv` have identical forward passes.

---

### `RayBasis3d` — 3D Forward Pass

```julia
(r::RayBasis3d)(xyz; xₒ, nominal_ρ, nominal_θ, nominal_ψ) -> Matrix
```

**Input:** `xyz` — `3 x N` matrix. Keyword arguments provide the nominal ray geometry from image sources.

**Computation:**
1. Compute virtual source positions in 3D spherical coordinates, using nominal parameters plus learned error terms: `(eθ + nominal_θ, eψ + nominal_ψ, ed + nominal_ρ)`.
2. Convert to Cartesian, compute distance `l` to each receiver.
3. Compute complex field with `1/l` geometric spreading: `A[i]/l * cos(k*l + ϕ[i])` and `A[i]/l * sin(k*l + ϕ[i])`.
4. Sum real and imaginary parts separately, then compute `amp2db(√(Re² + Im²))`.

The real/imaginary separation (rather than using Julia's `Complex` type) is required for correct gradient computation through Zygote.

There is also a 2-argument method `(m::RayBasis3d)(xyz, env; ...)` that ignores `env` and delegates to the keyword-only version, for API consistency with Cases 3 and 4.

**Output:** `1 x N` matrix of transmission loss in dB.

---

### `RayBasisRCNN` — 3D with Learned Reflection

```julia
(m::RayBasisRCNN)(xyz, env; xₒ, nominal_ρ, nominal_θ, nominal_ψ, n_rays) -> Matrix
```

**Input:** `xyz` — `3 x N` matrix. `env` provides `soundspeed`, `frequency`, `waterdepth`, and `tx`.

**Computation:**
1. Compute ray geometry (same as `RayBasis3d`).
2. Compute image source geometry from ray index arithmetic: for ray `j`, determine surface bounce count `s`, bottom bounce count `b`, and vertical offset `dz`.
3. Compute the grazing angle at the seabed: `θ = |atan(R / dz)|` where `R` is horizontal range.
4. Pass each ray's angle through the RCNN: `rcnn(θ[i:i, :]) -> [|RC|, phase_shift]`. This loop uses `Zygote.Buffer` for in-place mutation within the differentiable computation graph.
5. Compute per-ray amplitude: `(1/l) * (-1)^s * |RC|^b * absorption(f, l, 35.0)`.
6. Compute overall phase: `2πfl/c + phase_shift * b`.
7. Sum real and imaginary parts, convert to dB.

**Output:** `1 x N` matrix of transmission loss in dB.

---

### `RayBasisRayleigh` — 3D with Physics-Based Reflection

```julia
(m::RayBasisRayleigh)(xyz, env; n_rays) -> Matrix
```

**Input:** `xyz` — `3 x N` matrix. `env` provides `soundspeed`, `frequency`, `waterdepth`, `tx`.

**Computation:**
1. Compute image source geometry directly from ray indices (no stored geometry — computed on the fly from `j = 1:n_rays`).
2. Compute distances `l = √(R² + dz²)` and grazing angles `θ = |atan(R/dz)|`.
3. Compute Rayleigh reflection coefficient: `reflectioncoef(θ, |ρᵣ|, |cᵣ|, |δ|)`. The `abs()` ensures physical positivity.
4. Compute per-ray amplitude: `(1/l) * (-1)^s * |RC|^b * absorption(f, l)`.
5. Compute overall phase: `2πfl/c + angle(RC) * b`.
6. Sum real and imaginary parts, compute magnitude.

**Output:** `1 x N` matrix of **linear amplitude** (NOT in dB). This differs from all other model types.

**Note:** The `absorption()` call does not specify salinity (uses UnderwaterAcoustics default), unlike Case 3 which hardcodes 35 ppt.

## The `fit!` Interface

The library uses Julia's multiple dispatch to provide a unified training API. Users call `fit!` with different model types and dispatch routes to the appropriate training procedure.

---

### `train_rbnn!(model, rx_train, rx_val, TL_train, TL_val; ...)` *(internal)*

Internal training loop used only by Case 1's `fit!`. Not exported.

**Signature:**
```julia
train_rbnn!(model, rx_train, rx_val, TL_train, TL_val;
    initial_lr=0.5f0, threshold_count=5000, threshold_lr=1e-6f0, show=false)
```

**Training loop structure:**
1. Define loss as `√MSE(model(x), y)` (RMSE).
2. Initialize `LegacyADAM(initial_lr)`.
3. For each epoch (up to 10 billion):
   - Compute gradient and update via `Flux.train!` on the RMSE loss.
   - If validation RMSE improves: save best model (deepcopy of all params), reset patience counter.
   - If no improvement for `threshold_count` epochs: restore best model, divide LR by 10 (`opt.eta /= 10.0`).
   - If LR < `threshold_lr`: stop.
4. Return model (note: does NOT restore best model before returning — the model retains whatever state it had at the final LR decay step).

**Key detail:** The loss function is `√MSE` (not raw MSE). `Flux.train!` differentiates through the square root, so the gradient includes a `1/(2√MSE)` factor. This is intentional and matches the capsule behavior.

---

### `fit!` for `RayBasis2DCurv` (Case 1)

```julia
fit!(env, tx, rx, data;
     model=nothing, nrays=60, initial_lr=0.5f0,
     threshold_count=5000, threshold_lr=1e-6f0, show=false) -> model
```

**Dispatch:** Matched when called with 4 positional arguments `(env, tx, rx, data)` and no typed model argument.

**Trainable parameters:** θ, A, ϕ, d (all 4 ray parameters).

**Optimizer:** `LegacyADAM`.

**Training loss:** `√MSE` (RMSE) — via `train_rbnn!`.

**Data split:** 70/30 train/validation via `data_split(rx)`.

If no model is provided, creates `RayBasis2DCurv(nrays, 2πf/c)` from `env.frequency` and `env.soundspeed`.

---

### `fit!` for `RayBasis3d` (Case 2)

```julia
fit!(env, rx, TL_data, rbnn::RayBasis3d;
     nominal_ρ, nominal_θ, nominal_ψ,
     xₒ=Float64[0,0,0], l1_reg=1.0, initial_lr=0.5,
     threshold_count=5000, threshold_lr=1e-6, show=false) -> model
```

**Dispatch:** Matched by the `rbnn::RayBasis3d` type annotation on the 4th argument.

**Trainable parameters:** A, ϕ only (geometry errors eθ, eψ, ed are excluded from `Flux.trainable`).

**Optimizer:** `Flux.Adam` (not `LegacyADAM`).

**Training loss:** `MSE(model(x), y) + l1_reg * mean(|A|)` — MSE plus L1 regularization on amplitudes.

**Validation/evaluation loss:** `√MSE` (RMSE) — regularization is NOT included in the validation metric.

**LR schedule:** Same plateau-based decay. LR mutated via `opt.eta /= 10.0`.

**Post-training:** Restores best model parameters before returning (unlike `train_rbnn!`).

---

### `fit!` for `RayBasisRCNN` (Case 3)

```julia
fit!(env, rx, TL_data, rbnn::RayBasisRCNN;
     nominal_ρ, nominal_θ, nominal_ψ, n_rays,
     xₒ=Float32[0,0,0], initial_lr=0.05f0, threshold_count=500,
     threshold_lr=1e-6, max_epochs=10_000_000, target_rmse=nothing,
     show=false) -> model
```

**Dispatch:** Matched by `rbnn::RayBasisRCNN`.

**Trainable parameters:** The RCNN neural network weights only (via `Flux.params(rbnn.rcnn)`).

**Optimizer:** `Flux.Adam`.

**Training loss:** `√MSE` (RMSE) — no regularization term.

**Early stopping:** In addition to the LR decay schedule, training can stop early if validation RMSE drops below `target_rmse`.

**Best model tracking:** Uses `deepcopy(rbnn.rcnn)` and `Flux.loadmodel!` rather than parameter-level copying, since the RCNN is a `Chain`.

**LR reset:** On plateau, creates a *new* `Flux.Adam` with reduced LR (`Flux.Adam(opt.eta / 10.0f0)`) rather than mutating `opt.eta` — this resets the optimizer's moment estimates.

**Default hyperparameters differ from other cases:** lower initial LR (0.05 vs 0.5), lower patience (500 vs 5000).

---

### `fit!` for `RayBasisRayleigh` (Case 4)

```julia
fit!(env, rx, TL_data, rbnn::RayBasisRayleigh;
     n_rays, initial_lr=0.5f0, threshold_count=5000,
     threshold_lr=1e-5, show=false) -> model
```

**Dispatch:** Matched by `rbnn::RayBasisRayleigh`.

**Trainable parameters:** ρᵣ, cᵣ, δ (three 1-element vectors).

**Optimizer:** `Flux.Adam`.

**Training loss:** `√MSE` (RMSE) on **linear amplitude** (not dB).

**Best model selection:** Uses a blended metric: `mean([val_loss, 0.5 * train_loss])` instead of pure validation loss. This gives some weight to training performance, which helps stability when optimizing only 3 parameters against hundreds of measurements.

**LR reset:** Like Case 3, creates a new `Flux.Adam` on each LR decay (resets moments).

**Minimum LR:** `1e-5` (higher than other cases' `1e-6`).

**Important:** Input data must be in linear amplitude (use `DSP.db2amp` to convert from dB). The forward pass returns linear amplitude, not dB.

## Loss Functions

### Training Loss

The base training loss across all cases is mean squared error:

`MSE = (1/N) * Σᵢ (ŷᵢ - yᵢ)²`

However, Cases 1, 3, and 4 actually optimize `√MSE` (RMSE), meaning the gradient includes a `1/(2√MSE)` factor from the square root. Case 2 optimizes raw MSE (plus regularization) for the training step but evaluates with RMSE.

| Case | Training objective | Evaluation metric |
|------|--------------------|-------------------|
| 1 | `√MSE(ŷ, y)` | `√MSE(ŷ, y)` |
| 2 | `MSE(ŷ, y) + α * mean(|A|)` | `√MSE(ŷ, y)` |
| 3 | `√MSE(ŷ, y)` | `√MSE(ŷ, y)` |
| 4 | `√MSE(ŷ, y)` on linear amp | `√MSE(ŷ, y)` on linear amp |

### L1 Regularization (Case 2)

Case 2 adds an L1 penalty on ray amplitudes:

`L = MSE(ŷ, y) + α * mean(|A|)`

where `α = l1_reg` (default 1.0). This encourages sparsity in the learned amplitudes, pushing the model toward solutions with fewer dominant rays — matching the physical expectation that only certain ray paths contribute meaningfully.

**Why it's absent in other cases:** In Case 1, the `LegacyADAM` optimizer and LR schedule provide sufficient implicit regularization. In Cases 3 and 4, ray amplitudes are not freely trainable — they are derived from reflection coefficients and geometric spreading, so regularizing them would fight the physics.

### Evaluation Metric: RMSE

All cases report RMSE for validation and final evaluation:

`RMSE = √((1/N) * Σᵢ (ŷᵢ - yᵢ)²)`

This is the value printed as "Val RMSE" and used for early stopping / best model selection (except Case 4's blended metric).

**Important distinction:** The L1 regularization term is excluded from the evaluation metric. Training loss and evaluation metric must not be conflated — the regularization term can make training loss higher than RMSE even when the model is fitting well.

## Case-by-Case Implementation Details

### Case 1: 2D Far-Field with Range-Dependent Bathymetry

**Physics:** Models acoustic propagation at long range (1000+ m) in a waveguide with varying bottom depth (40m -> 30m -> 33m). At these ranges, the source-receiver geometry is approximately 2D (range vs. depth). The wavefront curvature parameter `d` captures the finite distance to the apparent source of each ray.

**Model:** `RayBasis2DCurv`

**Key hyperparameters (from `test/case1.jl`):**
- Frequency: 10 kHz, Sound speed: 1541 m/s
- Water depth: 30 m (range-dependent bathymetry via `SampledField`)
- Transmitter: `[0, 5]` m (range=0, depth=5)
- Measurement region: 1000–1050 m range, 1–30 m depth
- Rays: 60
- Initial LR: 0.5, Patience: 5000, Min LR: 1e-6

**Training optimizes:** θ, A, ϕ, d for all 60 rays (240 parameters total).

**Data files:**
- `src/data/case1_far_field/A_train.csv` — training TL
- `src/data/case1_far_field/capsule_rx_test.csv` — test receiver positions
- `src/data/case1_far_field/capsule_TL_test.csv` — test TL (ground truth)
- `src/bson_logs/case1/ini_RBNN.bson` — initial model weights

**Verification:** Test RMSE must be finite.

---

### Case 2: 3D Near-Field

**Physics:** Models acoustic propagation at short range (100–150 m) in 3D. At these ranges, the full 3D geometry matters. The image source method provides nominal ray geometry (60 strongest image sources); the model learns amplitude and phase corrections to account for environmental deviations from the ideal waveguide.

**Model:** `RayBasis3d`

**Key hyperparameters (from `test/case2.jl`):**
- Frequency: 5 kHz, Sound speed: 1541 m/s
- Water depth: 30 m
- Transmitter: `[0, 0, 15]` m
- Measurement region: 100–150 m range, 1–29 m depth
- Rays: 60, L1 regularization: 1.0
- Initial LR: 0.5, Patience: 5000, Min LR: 1e-6
- Float64 precision throughout

**Training optimizes:** A, ϕ only (120 parameters). Geometry errors `eθ, eψ, ed` are initialized to zero and remain fixed.

**Data files:**
- `src/data/case2_near_field/B_data.csv` — training TL
- `src/data/case2_near_field/capsule_rx_test.csv`, `capsule_TL_test.csv` — test data
- `src/bson_logs/case2/ini_RBNN.bson` — initial weights

**Verification:** Test RMSE < 2.1 (capsule reference: ~2.08).

---

### Case 3: Geo-acoustic Inversion with RCNN

**Physics:** Learns the seabed reflection coefficient as a function of grazing angle using a neural network. This is geo-acoustic inversion: instead of measuring sediment properties directly, the model infers reflection behavior from acoustic field measurements.

**Model:** `RayBasisRCNN`

**Key hyperparameters (from `test/case3.jl`):**
- Frequency: 5 kHz, Sound speed: 1541 m/s
- Water depth: 30 m
- Transmitter: `[0, 0, 15]` m
- Measurement region: 0–300 m range, 1–29 m depth
- Rays: 60
- Initial LR: 0.05, Patience: 5000, Min LR: 1e-6

**RCNN architecture:** Normalization -> Dense(1->30, sigmoid) -> Dense(30->50, sigmoid) -> Dense(50->2). Output row 1: `|RC|`, output row 2: phase shift.

**Training optimizes:** RCNN weights only.

**Data files:**
- `src/data/case3/C_data.csv` — training TL
- `src/bson_logs/case3/ini_rc_inversion_RCNN.bson` — initial RCNN weights

**Verification:** Validation RMSE < 2.0 and finite.

**Implementation notes:**
- The forward pass loops over rays (`for i in 1:n_rays`) rather than batching the RCNN evaluation, using `Zygote.Buffer` for in-place mutation. This is a performance bottleneck but required for correct gradient computation.
- The test file uses `threshold_count = 5000` (not the `fit!` default of 500).

---

### Case 4: Geo-acoustic Parameter Inversion

**Physics:** Like Case 3 but uses the physics-based Rayleigh reflection formula instead of a neural network. Learns three interpretable physical parameters: density ratio (ρᵣ), sound speed ratio (cᵣ), and attenuation (δ).

**Model:** `RayBasisRayleigh`

**Key hyperparameters (from `test/case4.jl`):**
- Frequency: 5 kHz, Sound speed: 1541 m/s
- Water depth: 30 m
- Transmitter: `[0, 0, 15]` m
- Measurement region: 0–100 m range, 1–29 m depth
- Rays: 60
- Initial LR: 0.5, Patience: 5000, Min LR: 1e-5

**Training optimizes:** ρᵣ, cᵣ, δ (3 parameters total — the most constrained model).

**Data files:**
- `src/data/case4/D_data.csv` — training TL (converted to linear amplitude via `db2amp`)
- `src/bson_logs/case4/ini_geoacoustic_inversion.bson` — initial weights
- `src/bson_logs/case4/trained_weights_D.bson` — reference trained weights

**Verification:** Learned parameters must match capsule's trained weights: `rtol=0.001` for ρᵣ and cᵣ, `rtol=0.01` for δ.

**Quirks:**
- Forward pass returns linear amplitude (not dB), so training data must be converted from dB using `db2amp` before calling `fit!`.
- Best-model selection uses blended metric of validation and training loss.
- The `absorption()` call does not specify salinity (unlike Case 3's hardcoded 35 ppt).

## Design Decisions

### Why MSE for training vs. RMSE for evaluation

MSE gradients are proportional to the error (`∂MSE/∂ŷ = 2(ŷ - y)/N`), producing stable updates. RMSE gradients include a `1/√MSE` factor that grows as loss approaches zero, which can cause numerical instability in late-stage training.

However, Cases 1, 3, and 4 actually train on `√MSE` — this was inherited from the capsule implementation. The `√MSE` gradient emphasizes smaller errors relatively more than raw MSE, which empirically helps convergence for these cases. Case 2 uses raw MSE because the L1 regularization term adds directly to MSE; if RMSE were used, the penalty would need to be inside the square root, complicating gradients.

RMSE is reported for evaluation because it has the same units as the data (dB or linear amplitude), making it directly interpretable.

### Why the LR decay schedule works the way it does

The plateau-based LR decay (divide by 10 after N epochs without improvement, stop when LR < threshold) handles the multi-scale nature of acoustic fitting. Aggressive initial learning rates (0.5) help escape poor basins in the oscillatory loss landscape caused by wave interference. The 10x decay then allows fine-grained convergence. The patience threshold varies: 5000 for Cases 1, 2, 4 (many parameters or sensitive optimization); 500 default for Case 3 (RCNN weights converge faster).

On each LR decay step, the best model weights are restored before continuing — this prevents the optimizer from building on a degraded state.

### Why L1 regularization on amplitudes is inactive in most cases

L1 regularization on ray amplitudes encourages sparsity (few dominant rays). This is only meaningful in Case 2, where amplitudes are freely trainable. In Case 1, the LegacyADAM optimizer provides sufficient implicit regularization. In Cases 3 and 4, amplitudes are computed from physics (reflection coefficients, geometric spreading, absorption) rather than being free parameters — regularizing them would fight the physics.

### Why LegacyADAM exists instead of using Flux.Adam

The custom `LegacyADAM` replicates the exact floating-point behavior of the original CodeOcean capsule's optimizer. It tracks bias correction as cumulative β powers in a mutable array, while `Flux.Adam` tracks the step count. This ordering difference produces slightly different training trajectories, breaking reproducibility of published Case 1 results. Cases 2–4 use `Flux.Adam` because they were developed post-publication.

### The unified `fit!` interface vs. separate training functions

Multiple dispatch on the model type is idiomatic Julia and provides a clean API: one function name, one calling convention, automatic routing. The alternative — `fit_2d!`, `fit_3d!`, `fit_rcnn!` — would require users to know which function to call and would not compose with generic code. Switching from a neural seabed model (Case 3) to a physics-based one (Case 4) requires only changing the model constructor.

### Callable struct pattern vs. old `calculatefield`

Models use `model(rx)` instead of `model.calculatefield(model, rx)` because:
1. It follows the Flux.jl convention (e.g., `Dense(10, 5)(x)`).
2. It eliminates storing a function reference in the struct.
3. It's concise: `rbnn(rx)` vs. `rbnn.calculatefield(rbnn, rx)`.

### Hardcoded origin x0 = [0, 0, 0]

The 2D models hardcode `x0 = [0.0, 0.0]` inside the forward pass. The 3D models accept it as a keyword argument defaulting to `[0, 0, 0]`. This assumes the coordinate system is centered on the transmitter or a fixed reference. If a different origin is needed, receiver positions should be shifted instead.

### No geometric spreading in 2D (Case 1)

`RayBasis2DCurv` omits the `1/l` geometric spreading factor that `RayBasis3d` uses. In the far-field (source at ~1000 m), over a 50 m measurement region, distance changes are negligible relative to total path length. Spreading loss is effectively constant and absorbed into the trainable amplitude `A`. The `d` parameter models wavefront *curvature* (phase delays), not amplitude attenuation. In contrast, near-field 3D models require explicit `1/l` because distance variations significantly affect intensity.

### Case 4 weighted loss for early stopping

Case 4 uses `mean([val_loss, 0.5 * train_loss])` instead of pure validation loss for best-model selection. With only 3 trainable parameters against hundreds of measurements, pure validation selection is noisy. The training loss term acts as a stabilizer.

## Known Limitations & TODOs

1. **Hardcoded reflection coefficients in `n_images_src`:** The image source ranking uses `0.2` and `0.99` as rough bottom/surface reflection coefficients. Not configurable.

2. **Hardcoded salinity inconsistency:** Case 3's `absorption()` hardcodes `35.0f0` ppt; Case 4 uses the library default. Could affect results in non-standard environments.

3. **Hardcoded image source range:** Both `n_images_src` functions use `a = 20` (reflection orders -20 to +20, yielding 82 candidates). Not configurable.

4. **RayBasis / RayBasis2DCurv duplication:** These structs and forward passes are identical. `RayBasis` exists only for BSON compatibility. Could be eliminated with a type alias.

5. **No `fit!` dispatch for `RayBasis`:** Only `RayBasis2DCurv` has a `fit!` method. A loaded `RayBasis` must be manually converted for retraining.

6. **Sequential RCNN evaluation:** Case 3's forward pass loops over rays (`for i in 1:n_rays`) rather than batching. Performance bottleneck.

7. **`data_split` / TL indexing coupling:** `data_split` seeds with dataset size, and `fit!` assumes `TL_data[1:N_train]` corresponds to training receivers. Reordering input data silently breaks this correspondence.

8. **No GPU support:** All computations are CPU-only (`Array`, not `CuArray`).

9. **`train_rbnn!` does not restore best model:** Unlike the other `fit!` methods, `train_rbnn!` (Case 1) does not restore the best model parameters before returning. The returned model retains the state from the last LR decay step.

10. **`remove_consecutive_duplicates` appears unused** in the core v2 code path. <!-- TODO: clarify -->

11. **Case 4 `abs()` wrapping:** Applying `abs()` to ρᵣ, cᵣ, δ ensures physical positivity but creates a non-smooth gradient at zero.

12. **RCNN architecture hardcoded:** The default `1->30->50->2` architecture is baked into the constructor. No way to configure layer sizes without passing a custom `rcnn`.

13. **No batch training:** The entire dataset is processed every epoch. No mini-batch support.
