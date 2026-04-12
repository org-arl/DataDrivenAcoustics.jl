# DataDrivenAcoustics.jl

A Julia library for data-driven acoustic propagation modeling using ray basis neural networks.

---

## Motivation

Conventional physics-based acoustic propagation models — ray solvers, normal modes, parabolic equations — require **accurate prior knowledge of the environment**: seabed density and sound speed, bathymetry, water-column sound-speed profiles, and source location. In practice, these parameters are often unknown, difficult to measure, or expensive to obtain.

Classical data-driven alternatives (Gaussian Process Regression, deep neural networks) sidestep the environmental knowledge requirement, but they have two serious limitations in underwater acoustics:

- **Data hunger:** ocean measurements are sparse and expensive — a profiling float collecting 100–200 samples is often the best available dataset.
- **Poor extrapolation:** standard ML models interpolate within the training region but fail to generalize beyond it, because they have no physics built in.

DataDrivenAcoustics.jl implements the **Ray Basis Neural Network (RBNN)** framework ([Li & Chitre 2023](https://ieeexplore.ieee.org/abstract/document/10224658)) to resolve both problems. The design goal is to make this framework as accessible as possible: a user passes their environment, transmitter, receivers, and measured data into a single `fit!` call, and the library handles the rest.


## 2. Overview

DataDrivenAcoustics.jl solves the problem of predicting underwater acoustic fields when environmental parameters (seabed properties, bathymetry, sound speed profiles) are unknown or difficult to measure. Instead of requiring detailed environmental knowledge, the library learns acoustic propagation directly from sparse field measurements.

The core computational approach is **ray basis propagation**: the acoustic field is modeled as a superposition of ray-like basis functions, each parameterized by direction, amplitude, and phase. These parameters are learned via gradient descent, treating the physics-inspired basis functions as a differentiable neural network layer. This is distinct from black-box neural networks because the model structure encodes wave propagation physics (phase accumulation, geometric spreading, reflection), constraining the solution space to physically plausible fields.

The library implements four use cases (referred to as "Cases"):

| Case | Scenario | Model Type | What is Learned |
|------|----------|------------|-----------------|
| 1 | Far-field 2D, unknown source | `RayBasis2DCurv` | Ray angles, amplitudes, phases, wavefront curvature |
| 2 | Near-field 3D, known source | `RayBasis3d` | Amplitude and phase corrections to image-source rays |
| 3 | Near-field 3D, unknown seabed | `RayBasisRCNN` | Neural network mapping incident angle to reflection coefficient |
| 4 | Near-field 3D, physical inversion | `RayBasisRayleigh` | Physical seabed parameters (density ratio, sound speed ratio, attenuation) |

**Training loss vs. evaluation metric:** The training loss is Mean Squared Error (MSE), optionally augmented with regularization terms (e.g., L1 penalty on amplitudes in Case 2). The evaluation metric reported to the user is Root Mean Squared Error (RMSE), computed as `sqrt(MSE)` on validation data with no regularization. These are distinct quantities: minimizing MSE during training is mathematically equivalent to minimizing RMSE, but the gradient magnitudes differ. Regularization terms (when present) are added only to the training loss to encourage sparse amplitude solutions; they are zeroed when computing the reported RMSE.

---

## Core Idea: Physics-Structured Learning

### The Helmholtz equation as a model prior

The Helmholtz equation governs steady-state acoustic fields:

```
k²p̄(r) + ∇²p̄(r) = 0
```

A solution can always be written as a superposition of ray arrivals:

```
p̄(r) = Σₘ Aₘ exp(iϕₘ) exp(ikₘ · r)
         ↑       ↑             ↑
      amplitude  phase     ray direction
```

This sum is the **RBNN layer**: each term is a "neuron" whose activation is a plane wave. Because every individual term satisfies the wave equation and the equation is linear, their sum is also a valid solution — unconditionally, with no physics penalty term or regularization weight to tune.

The key distinction from Physics-Informed Neural Networks (PINNs) is that the RBNN *is* the physics by construction, rather than being penalized toward it through the loss function. This means every prediction the model makes is physically plausible, even when extrapolating far outside the training region.

### What is learned vs. what is known

The framework is flexible about which quantities come from environmental knowledge and which are learned from data:

| Case | Environmental Knowledge | What is Learned |
|------|------------------------|-----------------|
| 1 | Sound speed, frequency only | Ray angles, amplitudes, phases, wavefront curvature |
| 2 | Source location, geometry | Amplitude and phase corrections to image-source rays |
| 3 | Source location, geometry | Neural network mapping incident angle → reflection coefficient |
| 4 | Source location, geometry | Physical seabed parameters (ρᵣ, cᵣ, δ) |

### Why a single `fit!` interface?

The library exposes one unified entry point:

```julia
model = fit!(env, tx, rx, data; model=RayBasis2DCurv(nrays, k))
```

The user supplies the environment, transmitter, receiver positions, and measured transmission loss. Julia's multiple dispatch then routes to the correct training procedure based on the type of `model`:

| Model type | Dispatches to |
|---|---|
| `RayBasis2DCurv` | Case 1 trainer — far-field, all parameters learned from scratch |
| `RayBasis3d` | Case 2 trainer — near-field, learns amplitude/phase corrections |
| `RayBasisRCNN` | Case 3 trainer — trains the embedded seabed reflection network |
| `RayBasisRayleigh` | Case 4 trainer — inverts for physical seabed properties |

This means the choice of how much environmental knowledge to incorporate is expressed entirely through the model type. Switching from a data-driven seabed model (Case 3) to a physics-based one (Case 4) only requires changing the model you construct — the `fit!` call itself stays the same.




## 3. Architecture & Design

### 3.1 Core Abstractions

All propagation models inherit from the abstract type `PropagationModel`:

```julia
abstract type PropagationModel end
```

#### `RayBasis`

```
RayBasis
  Purpose: 2D far-field ray basis model (legacy, for BSON compatibility).
  Fields:
    θ :: AbstractVector  — Azimuthal angles of arrival rays (radians)
    A :: AbstractVector  — Amplitudes of arrival rays
    ϕ :: AbstractVector  — Phases of arrival rays (radians)
    d :: AbstractVector  — Distances for wavefront curvature modeling (meters)
    k :: Real            — Angular wavenumber (rad/m), fixed after construction

  vs. old implementation:
    The old RayBasis2D (src/pm_RBNN.jl) stored env, calculatefield function,
    nrays, and a dynamic trainable tuple. The new RayBasis removes env storage
    (environment is passed to fit!), removes the calculatefield indirection
    (models are callable structs), and fixes trainable parameters at compile
    time via Flux.trainable(). This simplifies the type signature and makes
    the struct purely a parameter container.
```

#### `RayBasis2DCurv`

```
RayBasis2DCurv
  Purpose: 2D far-field ray basis model with wavefront curvature (preferred for Case 1).
  Fields:
    θ :: AbstractVector  — Azimuthal angles of arrival rays (radians)
    A :: AbstractVector  — Amplitudes of arrival rays
    ϕ :: AbstractVector  — Phases of arrival rays (radians)
    d :: AbstractVector  — Distances for wavefront curvature modeling (meters)
    k :: Real            — Angular wavenumber (rad/m)

  vs. old implementation:
    The old RayBasis2DCurv (src/pm_RBNN.jl) used a kwdef struct with 10+ type
    parameters and called ModelFit! inside the constructor, coupling model
    construction to training. The new struct is a simple parameter container.
    Training is decoupled and invoked explicitly via fit!().
```

#### `RayBasis3d`

```
RayBasis3d
  Purpose: 3D near-field ray basis model for Case 2.
  Fields:
    eθ :: AbstractVector  — Error to nominal azimuthal angle (radians)
    eψ :: AbstractVector  — Error to nominal elevation angle (radians)
    ed :: AbstractVector  — Error to nominal propagation distance (meters)
    A  :: AbstractVector  — Amplitudes of arrival rays
    ϕ  :: AbstractVector  — Phases of arrival rays (radians)
    k  :: Real            — Angular wavenumber (rad/m)

  Constraints:
    - Uses Float64 precision due to large phase values (~2000-3000 radians at
      5 kHz over 100+ m ranges). Float32 loses gradient precision.
    - Only A and ϕ are trainable by default.

  vs. old implementation:
    The old RayBasis3D stored both nominal (θ, ψ, d) and error (eθ, eψ, ed)
    fields. The new RayBasis3d stores only error fields; nominal values
    (nominal_ρ, nominal_θ, nominal_ψ) are computed once from image sources
    and passed as arguments to the forward pass and fit!. This eliminates
    redundant storage and makes the image-source computation explicit in the
    calling code.
```

#### `RayBasisRCNN`

```
RayBasisRCNN
  Purpose: 3D ray basis model with embedded Reflection Coefficient Neural Network (Case 3).
  Fields:
    eθ   :: AbstractVector  — Error to nominal azimuthal angle (radians)
    eψ   :: AbstractVector  — Error to nominal elevation angle (radians)
    ed   :: AbstractVector  — Error to nominal propagation distance (meters)
    k    :: Real            — Angular wavenumber (rad/m)
    rcnn :: Chain           — Neural network mapping incident angle → [|RC|, phase_shift]

  Constraints:
    - Only rcnn is trainable. Geometry parameters are fixed from image sources.
    - The default RCNN architecture is: input normalization → Dense(1,30,sigmoid) →
      Dense(30,50,sigmoid) → Dense(50,2).

  vs. old implementation:
    The old RayBasis3DRCNN required RCNN to be passed separately to the
    constructor and computed image sources inside the constructor using
    find_image_src(). The new RayBasisRCNN accepts an optional rcnn kwarg
    (defaulting to a standard architecture) and stores it internally. Image
    source computation is moved to user code, making the struct independent
    of environment details.
```

#### `RayBasisRayleigh`

```
RayBasisRayleigh
  Purpose: 3D ray basis model with physics-based Rayleigh reflection (Case 4).
  Fields:
    ρᵣ :: AbstractVector  — Density ratio (seabed/water), length 1
    cᵣ :: AbstractVector  — Sound speed ratio (seabed/water), length 1
    δ  :: AbstractVector  — Attenuation coefficient, length 1
    k  :: Real            — Angular wavenumber (rad/m)

  Constraints:
    - All three seabed parameters are trainable.
    - Parameters are stored as length-1 vectors to work with Flux's parameter system.
    - The forward pass uses abs() on parameters to ensure physical positivity.

  vs. old implementation:
    This model type did not exist in the old implementation. It was added to
    enable direct inversion for physical seabed parameters using the Rayleigh
    reflection coefficient formula, rather than learning an arbitrary neural
    network approximation (Case 3).
```

### 3.2 Shared Functions

#### `zig_zag_samples`

```julia
zig_zag_samples(xmin, xrange, xscale, zmin, zrange, zscale; IsTwoD=true, T=Float64) -> Matrix
```

Generates a zig-zag sampling pattern for receiver positions, useful for creating training data that efficiently covers a 2D or 3D domain.

**Arguments:**
- `xmin`: Minimum x coordinate
- `xrange`: Total range in x direction
- `xscale`: Step size in x direction
- `zmin`: Minimum z coordinate
- `zrange`: Total range in z direction
- `zscale`: Step size in z direction
- `IsTwoD`: Return 2D coordinates (true) or 3D with y=0 (false)
- `T`: Numeric type for output

**Returns:** Matrix of receiver positions (2xN or 3xN).

**vs. old implementation:** This function did not exist in the old implementation. Sampling patterns were generated ad-hoc in test files. Extracting it to a utility function promotes code reuse.

#### `data_split`

```julia
data_split(rx; ratio=0.7) -> (rx_train, rx_val)
```

Splits receiver positions into training and validation sets using a deterministic random permutation seeded by data length.

**Arguments:**
- `rx`: Receiver positions (DxN matrix)
- `ratio`: Fraction for training (default: 0.7)

**Returns:** Tuple of training and validation position matrices.

**vs. old implementation:** The old `SplitData` function (src/pm_utility.jl) split both locations and measurements together and used a different seed strategy. The new `data_split` only splits positions; the corresponding measurement indices are computed in fit! using array slicing. This decouples data splitting from the measurement data structure.

#### `n_images_src`

```julia
n_images_src(rx, tx, D, n_rays; T=Float64) -> Matrix
```

Computes image source positions for ray tracing using the image source method.

**Arguments:**
- `rx`: Representative receiver position (3-element vector)
- `tx`: Transmitter position (3-element vector)
- `D`: Water depth
- `n_rays`: Number of rays to select

**Returns:** Matrix of image source positions (3xn_rays), sorted by expected amplitude contribution.

**Implementation notes:** Uses a fixed search range of +/-20 surface/bottom reflection orders. Selects rays by sorting on estimated amplitude (accounting for reflection losses and geometric spreading).

**vs. old implementation:** The old `find_image_src` function (src/pm_utility.jl) was similar but returned all columns in a different format. The new function also supports an explicit type parameter `T` for precision control.

#### `n_images_src_with_ref`

```julia
n_images_src_with_ref(rx, tx, D, n_rays) -> (image_sources, ref)
```

Computes image sources with reflection counts, required for Case 3 where surface/bottom bounce counts affect the RCNN output weighting.

**Returns:** Tuple of (image source positions, reflection counts) where `ref` is a 2xn_rays matrix with rows [surface_bounces, bottom_bounces].

#### `cartesian2spherical`

```julia
cartesian2spherical(pos) -> (d, θ, ψ)
```

Converts 3D Cartesian coordinates to spherical coordinates.

**Arguments:**
- `pos`: Position matrix (3xN)

**Returns:** Tuple of (distance, azimuthal angle, elevation angle).

**vs. old implementation:** The old version returned (θ, ψ, d) in a different order. The new version returns (d, θ, ψ) matching the nominal_ρ, nominal_θ, nominal_ψ naming convention used in fit!.

#### `extract_array_from_bson`

```julia
extract_array_from_bson(arr_dict) -> Array or nothing
```

Extracts arrays from BSON parsed format, handling Flux version mismatches in saved models.

**Arguments:**
- `arr_dict`: Dictionary from `BSON.parse` containing `:data` and `:size` keys

**Returns:** Extracted array reshaped to original dimensions, or `nothing` if format doesn't match.

**vs. old implementation:** This function did not exist in the old implementation. It was added to handle loading weights from BSON files saved with different Flux versions, avoiding `MethodError` when directly loading structs.

#### `reflectioncoef`

```julia
reflectioncoef(θ, ρr, cr, δ) -> Complex
```

Computes Rayleigh reflection coefficient with sign convention fix.

**Arguments:**
- `θ`: Incident angle (radians)
- `ρr`: Density ratio (seabed/water)
- `cr`: Sound speed ratio (seabed/water)
- `δ`: Attenuation coefficient

**Implementation notes:** Wraps `UnderwaterAcoustics.reflection_coef` with negated δ to match the sign convention used in the original implementation.

### 3.3 The `fit!` Interface

The `fit!` function trains propagation models using multiple dispatch to select the appropriate method based on model type.

#### Case 1: `fit!(env, tx, rx, data; model, ...)`

```julia
fit!(env, tx, rx, data;
     model = nothing,
     nrays = 60,
     initial_lr = Float32(0.5),
     threshold_count = 5000,
     threshold_lr = Float32(1e-6),
     show = false) -> trained_model
```

**Arguments:**
- `env`: BasicDataDrivenUnderwaterEnvironment with soundspeed, frequency
- `tx`: Transmitter position (can be `nothing` for far-field)
- `rx`: Receiver positions (2xN matrix)
- `data`: Measured transmission loss data (row vector)
- `model`: Pre-initialized RayBasis2DCurv (or creates one if `nothing`)
- `nrays`: Number of rays (default: 60)
- `initial_lr`: Initial learning rate (default: 0.5)
- `threshold_count`: Epochs without improvement before LR decay (default: 5000)
- `threshold_lr`: Minimum LR before stopping (default: 1e-6)
- `show`: Print training progress (default: false)

**Internal behavior:**
1. Splits data 70/30 train/val using `data_split`
2. Initializes model if not provided
3. Calls internal `train_rbnn!` loop with LegacyADAM optimizer
4. Returns trained model

#### Case 2: `fit!(env, rx, TL_data, rbnn::RayBasis3d; ...)`

```julia
fit!(env, rx, TL_data, rbnn::RayBasis3d;
     nominal_ρ, nominal_θ, nominal_ψ,
     xₒ = Float64[0.0, 0.0, 0.0],
     l1_reg = 1.0,
     initial_lr = 0.5,
     threshold_count = 5000,
     threshold_lr = 1e-6,
     show = false) -> trained_model
```

**Required kwargs:**
- `nominal_ρ`, `nominal_θ`, `nominal_ψ`: Nominal ray parameters from image sources

**Additional kwargs:**
- `l1_reg`: L1 regularization coefficient on amplitudes (default: 1.0)

**Internal behavior:** Uses Float64 precision, standard Flux.Adam optimizer, MSE + L1 penalty as training loss.

#### Case 3: `fit!(env, rx, TL_data, rbnn::RayBasisRCNN; ...)`

```julia
fit!(env, rx, TL_data, rbnn::RayBasisRCNN;
     nominal_ρ, nominal_θ, nominal_ψ, n_rays,
     xₒ = Float32[0.0, 0.0, 0.0],
     initial_lr = 0.05f0,
     threshold_count = 500,
     threshold_lr = 1e-6,
     max_epochs = 10_000_000,
     target_rmse = nothing,
     show = false) -> trained_model
```

**Additional kwargs:**
- `n_rays`: Number of rays (required)
- `target_rmse`: Early stopping target (optional)

**Internal behavior:** Only trains the RCNN weights. Uses smaller default `threshold_count` (500) because neural network training typically requires more frequent LR adjustments.

#### Case 4: `fit!(env, rx, TL_data, rbnn::RayBasisRayleigh; ...)`

```julia
fit!(env, rx, TL_data, rbnn::RayBasisRayleigh;
     n_rays,
     initial_lr = 0.5f0,
     threshold_count = 5000,
     threshold_lr = 1e-5,
     show = false) -> trained_model
```

**Important:** `TL_data` must be linear amplitude (NOT dB). Convert using `db2amp()` before calling.

**Internal behavior:** Uses a weighted combination of validation and training loss for early stopping: `mean([val_loss, 0.5 * train_loss])`. This prevents overfitting to the validation set when learning only 3 parameters.

**vs. old implementation:**

The old implementation used `ModelFit!` called inside struct constructors:
```julia
# Old pattern (src/pm_RBNN.jl)
x = new{...}(env, calculatefield, nrays, θ, A, ϕ, k, trainable)
ModelFit!(x, inilearnrate, trainloss, dataloss, ...)
return x
```

This coupled model construction to training, making it impossible to:
- Create a model without training it
- Train a model multiple times with different hyperparameters
- Save/load models independently of training state

The new `fit!` pattern decouples these concerns:
```julia
# New pattern
model = RayBasis2DCurv(n_rays, k)  # Construction only
model = fit!(env, tx, rx, tl_fn, data; model=model, ...)  # Training explicit
```

### 3.4 Loss Functions

#### Training Loss

The training loss is MSE-based with optional regularization:

**Case 1, 3:** Pure MSE
```
L_train = MSE(model(rx_train), TL_train)
```

**Case 2:** MSE + L1 amplitude regularization
```
L_train = MSE(model(rx_train), TL_train) + λ * mean(|A|)
```
where λ = `l1_reg` (default 1.0).

**Case 4:** Pure MSE (on linear amplitude, not dB)
```
L_train = MSE(model(rx_train), amp_train)
```

#### Evaluation Metric

The evaluation metric is RMSE computed on validation data with no regularization:

```
RMSE_val = sqrt(MSE(model(rx_val), TL_val))
```

This is the value printed as "Val RMSE" and used for early stopping comparisons.

**Why they differ:** MSE is used for training because gradient descent minimizes loss directly, and MSE gradients are proportional to the error. RMSE would introduce a `1/(2*sqrt(MSE))` factor that vanishes as loss approaches zero, making convergence unstable.

The L1 penalty in Case 2 encourages sparse amplitude vectors (few dominant rays) by penalizing non-zero amplitudes. It is excluded from RMSE_val to measure pure prediction accuracy.

**vs. old implementation:**

The old implementation used `rmseloss` (src/pm_utility.jl) as the default for both `trainloss` and `dataloss`:
```julia
function rmseloss(rx, tl, model)
    Flux.mse(transmission_loss(model, rx), tl)^0.5f0
end
```

This conflated training loss and evaluation metric. The new implementation explicitly separates them, using MSE for training and RMSE for reporting.

### 3.5 Case Mapping

| Case | Test File | Model Type | Trainable Parameters | Data Format |
|------|-----------|------------|----------------------|-------------|
| 1 | test/case1.jl | `RayBasis2DCurv` | θ, A, ϕ, d | TL in dB |
| 2 | test/case2.jl | `RayBasis3d` | A, ϕ | TL in dB |
| 3 | test/case3.jl | `RayBasisRCNN` | rcnn (Chain) | TL in dB |
| 4 | test/case4.jl | `RayBasisRayleigh` | ρᵣ, cᵣ, δ | Linear amplitude |

---

## 4. How to Run Each Case

### Case 1: Far-Field 2D (Range-Dependent Bathymetry)

**What it models:** Far-field acoustic propagation in a 2D domain where the source location is unknown. The model learns ray arrival angles, amplitudes, phases, and wavefront curvature distances from transmission loss measurements.

**Prerequisites:**
- `src/data/case1_far_field/A_train.csv` — Training transmission loss data
- `src/data/case1_far_field/capsule_rx_test.csv` — Test receiver positions
- `src/data/case1_far_field/capsule_TL_test.csv` — Test transmission loss (ground truth)
- `src/bson_logs/case1/ini_RBNN.bson` — Initial model weights

**Run command:**
```bash
julia test/case1.jl
```

**Key hyperparameters:**
- `n_rays = 60`
- `initial_lr = 0.5`
- `threshold_count = 5000`
- `threshold_lr = 1e-6`

### Case 2: Near-Field 3D

**What it models:** Near-field acoustic propagation in a 3D domain with known source location. Uses image source method to compute nominal ray parameters, then learns amplitude and phase corrections.

**Prerequisites:**
- `src/data/case2_near_field/B_data.csv` — Training transmission loss data
- `src/data/case2_near_field/capsule_rx_test.csv` — Test receiver positions
- `src/data/case2_near_field/capsule_TL_test.csv` — Test transmission loss
- `src/bson_logs/case2/ini_RBNN.bson` — Initial model weights

**Run command:**
```bash
julia test/case2.jl
```

**Key hyperparameters:**
- `n_rays = 60`
- `initial_lr = 0.5`
- `l1_reg = 1.0`
- `threshold_count = 5000`
- Uses Float64 precision

**Pass criterion:** `test_loss < 2.1`

### Case 3: Geo-acoustic Inversion (RCNN)

**What it models:** Unknown seabed reflection characteristics via a neural network that learns the mapping from incident angle to reflection coefficient magnitude and phase shift.

**Prerequisites:**
- `src/data/case3/C_data.csv` — Training TL data
- `src/bson_logs/case3/ini_rc_inversion_RCNN.bson` — Initial RCNN weights

**Run command:**
```bash
julia test/case3.jl
```

**Key hyperparameters:**
- `n_rays = 60`
- `initial_lr = 0.05`
- `threshold_count = 5000`
- RCNN architecture: Dense(1 -> 30 -> 50 -> 2)

**Pass criteria:** `isfinite(final_val_rmse)` and `final_val_rmse < 2.0`

### Case 4: Geo-acoustic Inversion (Rayleigh Physics)

**What it models:** Physical seabed parameters (density ratio, sound speed ratio, attenuation) using the Rayleigh reflection coefficient formula. Unlike Case 3, this learns interpretable physical quantities.

**Prerequisites:**
- `src/data/case4/D_data.csv` — Training TL data (converted to linear amplitude internally)
- `src/bson_logs/case4/ini_geoacoustic_inversion.bson` — Initial weights
- `src/bson_logs/case4/trained_weights_D.bson` — Reference trained weights (for comparison)

**Run command:**
```bash
julia test/case4.jl
```

**Key hyperparameters:**
- `n_rays = 60`
- `initial_lr = 0.5`
- `threshold_count = 5000`
- `threshold_lr = 1e-5`

**Pass criteria:**
- `isapprox(learned_ρr, capsule_ρr, rtol=0.001)`
- `isapprox(learned_cr, capsule_cr, rtol=0.001)`
- `isapprox(learned_δ, capsule_δ, rtol=0.01)`

---

## 5. Library API Reference

### Model Construction

#### `RayBasis2DCurv(rays::Integer, k::Real) -> RayBasis2DCurv`

Constructs a 2D far-field ray basis model with random initialization.

- `rays`: Number of rays
- `k`: Angular wavenumber (2πf/c)

Parameters are initialized as: θ in [0, π], A in [0, 1], ϕ in [0, π], d in [0, 1].

#### `RayBasis3d(rays::Integer, k::Real) -> RayBasis3d`

Constructs a 3D near-field ray basis model.

- Error parameters (eθ, eψ, ed) initialized to zeros
- Amplitude A and phase ϕ randomly initialized
- Uses Float64 precision

#### `RayBasisRCNN(rays::Integer, k::Real; rcnn=nothing) -> RayBasisRCNN`

Constructs a 3D model with embedded RCNN.

- If `rcnn` is `nothing`, creates default architecture: input normalization -> Dense(1,30,sigmoid) -> Dense(30,50,sigmoid) -> Dense(50,2)
- Error parameters initialized to zeros

#### `RayBasisRayleigh(k::Real) -> RayBasisRayleigh`

Constructs a Rayleigh reflection model with default parameters ρᵣ=cᵣ=δ=1.0.

### Training

#### `fit!(env, tx, rx, data; model, nrays, initial_lr, threshold_count, threshold_lr, show) -> model`

Trains a Case 1 (RayBasis2DCurv) model. See Section 3.3 for details.

#### `fit!(env, rx, TL_data, rbnn::RayBasis3d; nominal_ρ, nominal_θ, nominal_ψ, xₒ, l1_reg, initial_lr, threshold_count, threshold_lr, show) -> model`

Trains a Case 2 (RayBasis3d) model. Mutates `rbnn` in place and returns it.

#### `fit!(env, rx, TL_data, rbnn::RayBasisRCNN; nominal_ρ, nominal_θ, nominal_ψ, n_rays, xₒ, initial_lr, threshold_count, threshold_lr, max_epochs, target_rmse, show) -> model`

Trains a Case 3 (RayBasisRCNN) model.

#### `fit!(env, rx, TL_data, rbnn::RayBasisRayleigh; n_rays, initial_lr, threshold_count, threshold_lr, show) -> model`

Trains a Case 4 (RayBasisRayleigh) model. **Important:** `TL_data` must be linear amplitude.

### Callable Models (Forward Pass)

#### `(model::RayBasis2DCurv)(xy::AbstractArray) -> Matrix`

Computes transmission loss at 2D receiver positions.

- `xy`: 2xN matrix of [x; z] coordinates
- Returns: 1xN matrix of TL in dB

#### `(model::RayBasis3d)(xyz::AbstractArray; xₒ, nominal_ρ, nominal_θ, nominal_ψ) -> Matrix`

Computes transmission loss at 3D receiver positions.

- `xyz`: 3xN matrix of [x; y; z] coordinates
- Returns: 1xN matrix of TL in dB

#### `(model::RayBasisRCNN)(xyz::AbstractArray, env; xₒ, nominal_ρ, nominal_θ, nominal_ψ, n_rays) -> Matrix`

Computes transmission loss using RCNN for reflection coefficients.

- Requires environment with soundspeed, frequency, waterdepth, tx

#### `(model::RayBasisRayleigh)(xyz::AbstractArray, env; n_rays) -> Matrix`

Computes **linear amplitude** (not dB) using Rayleigh physics.

### Utilities

#### `zig_zag_samples(xmin, xrange, xscale, zmin, zrange, zscale; IsTwoD=true, T=Float64) -> Matrix`

Generates zig-zag sampling pattern for receiver positions.

#### `data_split(rx; ratio=0.7) -> (rx_train, rx_val)`

Splits receiver positions into train/validation sets.

#### `n_images_src(rx, tx, D, n_rays; T=Float64) -> Matrix`

Computes image source positions for ray tracing.

#### `n_images_src_with_ref(rx, tx, D, n_rays) -> (image_sources, ref)`

Computes image sources with reflection counts.

#### `cartesian2spherical(pos) -> (d, θ, ψ)`

Converts Cartesian to spherical coordinates.

#### `extract_array_from_bson(arr_dict) -> Array`

Extracts arrays from BSON parsed format.

#### `reflectioncoef(θ, ρr, cr, δ) -> Complex`

Computes Rayleigh reflection coefficient.

---

## 6. Design Decisions & Notes

### Why MSE for training instead of RMSE

The training loop minimizes MSE rather than RMSE because:

1. **Gradient stability:** The gradient of RMSE includes a `1/sqrt(MSE)` factor that explodes as the loss approaches zero, causing numerical instability in late-stage training.

2. **Penalty term scaling:** When regularization is present (e.g., `l1_reg * mean(|A|)` in Case 2), it adds directly to MSE. If we used RMSE, the penalty would need to be inside the square root, complicating the gradient computation.

3. **Equivalence:** Minimizing MSE is equivalent to minimizing RMSE since sqrt is monotonic. The optimum is the same; only the gradient magnitudes differ.

### Learning rate decay schedule

All fit! methods use the same adaptive LR schedule:

1. Track best validation RMSE seen so far
2. If validation RMSE improves, reset counter to 0 and save best model
3. If validation RMSE does not improve for `threshold_count` epochs:
   - Restore best model weights
   - Divide LR by 10
   - Reset counter
4. Stop when LR < `threshold_lr`

This schedule was chosen because:
- It allows escaping local minima by restoring best weights before LR reduction
- The aggressive 10x LR reduction enables rapid fine-tuning
- The `threshold_count` parameter (typically 5000) allows sufficient exploration at each LR level

### L1 amplitude regularization in Case 2

The Case 2 training loss includes `l1_reg * mean(|A|)`. This encourages sparse amplitude vectors where only a few rays have significant amplitude, matching the physical expectation that only certain ray paths contribute meaningfully to the field.

The regularization coefficient `l1_reg=1.0` was chosen empirically to balance sparsity against prediction accuracy. Higher values produce sparser solutions but may underfit.

### Case 4 weighted loss for early stopping

Case 4 uses `mean([val_loss, 0.5 * train_loss])` for early stopping instead of pure validation loss. This prevents overfitting when learning only 3 parameters (ρᵣ, cᵣ, δ) from hundreds of measurements. The training loss term acts as a regularizer, ensuring the model doesn't memorize validation set noise.

### Float64 precision for Case 2

Case 2 requires Float64 because phase values at 5 kHz over 100m ranges reach ~2000-3000 radians. Float32 has ~7 decimal digits of precision, so phases like 2345.678 radians lose gradient information in the fractional part. Float64's ~16 digits preserve gradient precision.

### No geometric spreading in RayBasis2DCurv

`RayBasis2DCurv` (Case 1, far-field) does not include the `1/l` geometric spreading term that `RayBasis3d` (Case 2, near-field) uses. This is intentional:

- In the far-field, the source is distant (e.g., 1000m) from the measurement area
- Over a small measurement region (e.g., 50m), distance changes are negligible relative to the total path
- Geometric spreading loss is effectively constant across all receivers
- This constant loss is absorbed into the trainable amplitude parameter `A`

The `d` parameter in `RayBasis2DCurv` is used only for wavefront curvature (phase delays), not amplitude attenuation. In contrast, `RayBasis3d` requires explicit `1/l` spreading because near-field distance variations significantly affect intensity.

### Callable struct pattern

Models use the callable struct pattern `model(rx)` instead of `model.calculatefield(model, rx)` because:

1. **Conciseness:** `rbnn(rx)` is shorter than `rbnn.calculatefield(rbnn, rx)`
2. **Flux convention:** Flux.jl layers use this pattern (e.g., `Dense(10, 5)(x)`)
3. **No function storage:** The old pattern required storing a function reference in the struct, adding type complexity

### LegacyADAM optimizer

The `LegacyADAM` optimizer in utils.jl replicates the ADAM implementation from older Flux versions. This ensures reproducibility when verifying results against the original CodeOcean capsule, which used an earlier Flux version with different bias correction behavior.

---

## 7. Known Limitations & TODOs

### Magic numbers

- Image source search range is hardcoded to +/-20 in `n_images_src`
- Default reflection coefficients (0.2, 0.99) in image source amplitude estimation
- RCNN architecture (1->30->50->2) is hardcoded in default constructor

### Missing features

- No GPU support (all computations are CPU-only)
- No batch training (entire dataset processed each epoch)
- No learning rate warmup

---

## Publications

### Primary paper

- K. Li and M. Chitre, "Data-aided underwater acoustic ray propagation modeling," 2023. [Online]. Available: https://ieeexplore.ieee.org/abstract/document/10224658

### Other useful papers

- K. Li and M. Chitre, "Ocean acoustic propagation modeling using scientific machine learning," in OCEANS: San Diego-Porto. IEEE, 2021, pp. 1-5.

- K. Li and M. Chitre, "Physics-aided data-driven modal ocean acoustic propagation modeling," in International Congress of Acoustics, 2022.
