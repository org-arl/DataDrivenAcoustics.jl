# DataDrivenAcoustics.jl

Data-driven underwater acoustic propagation modeling using ray basis neural networks.

## Overview

Conventional physics-based acoustic models (ray solvers, normal modes, parabolic equations) require accurate prior knowledge of the environment: seabed properties, bathymetry, sound-speed profiles. In practice, these parameters are often unknown or expensive to measure. Classical data-driven alternatives (GPs, deep networks) sidestep the environmental knowledge requirement but suffer from data hunger and poor extrapolation — serious problems when ocean measurements are sparse and expensive.

DataDrivenAcoustics.jl implements the **Ray Basis Neural Network (RBNN)** framework ([Li & Chitre 2023](https://ieeexplore.ieee.org/abstract/document/10224658)). The core idea: represent the acoustic field as a superposition of ray-like basis functions, each parameterized by an arrival angle, amplitude, phase, and propagation distance. Because every individual term satisfies the Helmholtz equation and the equation is linear, their sum is also a valid solution — unconditionally, with no physics penalty term to tune. This is distinct from PINNs, where the network is penalized *toward* the physics; here, the model *is* the physics by construction.

The library implements four cases of increasing complexity. Case 1 learns the full ray field from scratch (2D far-field). Case 2 uses image-source geometry and learns amplitude/phase corrections (3D near-field). Case 3 learns seabed reflection via a neural network (geo-acoustic inversion). Case 4 inverts for physical seabed parameters using Rayleigh reflection. All four share the same `fit!` interface — Julia's multiple dispatch routes to the correct training procedure based on the model type.

## Installation

Clone the repository and activate the project environment:

```julia
using Pkg
Pkg.activate("path/to/DataDrivenAcoustics.jl")
Pkg.instantiate()
```

Key dependencies: [Flux.jl](https://github.com/FluxML/Flux.jl) (v0.13–0.14), [UnderwaterAcoustics.jl](https://github.com/org-arl/UnderwaterAcoustics.jl), [AcousticsToolbox.jl](https://github.com/org-arl/AcousticsToolbox.jl) (Bellhop interface), [Zygote.jl](https://github.com/FluxML/Zygote.jl) (automatic differentiation). Requires Julia 1.6+.

## Getting Started — Case 1: 2D Far-Field Propagation

This walkthrough fits a ray-basis model to transmission loss data from a range-dependent bathymetry environment. Case 1 demonstrates the core workflow: set up parameters, create a model, train with `fit!`, and evaluate.

### 1. Set up the environment

```julia
using DataDrivenAcoustics
include("src/core_v2.jl")

# Sound speed in water (m/s) and acoustic frequency (Hz)
c = Float32(1541.0)
f = Float32(10000.0)

# Angular wavenumber: k = 2πf/c, the spatial frequency of the acoustic field
k = Float32(2.0) * π * f / c

# Water depth (m) and transmitter position [range, depth]
L = Float32(30.0)
tx = Float32[0.0, 5.0]

# Number of ray basis functions — more rays capture more multipath structure
n_rays = 60
```

### 2. Generate receiver positions

Receivers are arranged in a zig-zag pattern to efficiently cover the spatial domain. The pattern sweeps in range (1000–1050 m) and depth (1–30 m):

```julia
xmin = Float32(1000.0)
xrange = Float32(50.0)

rx = zig_zag_samples(xmin, xrange, Float32(0.05), Float32(1.0), Float32(29.0), Float32(0.5); T=Float32)
```

This returns a `2 x N` matrix where row 1 is range and row 2 is depth.

### 3. Prepare training data

Load measured transmission loss corresponding to the receiver positions:

```julia
using CSV, DataFrames

data_dir = joinpath("src", "data", "case1_far_field")
TL_data = CSV.read(joinpath(data_dir, "A_train.csv"), DataFrame, header=false, types=Float32) |> Matrix
```

`TL_data` is a `1 x N` matrix of transmission loss values in dB, one per receiver location.

### 4. Create the model and environment

```julia
# Wrap environmental parameters for the fit! interface
env = BasicDataDrivenUnderwaterEnvironment(;
    soundspeed = c,
    frequency = f,
    tx = nothing,
    dB = true,
    dims = 2
)

# Initialize a ray-basis model with random parameters
rbnn = RayBasis2DCurv(n_rays, k)
```

`RayBasis2DCurv(60, k)` creates a model with 60 rays, each initialized with random angles (theta), amplitudes (A), phases (phi), and curvature distances (d).

### 5. Train

```julia
rbnn = fit!(env, nothing, rx, TL_data;
    model = rbnn,
    nrays = n_rays,
    initial_lr = Float32(0.5),
    threshold_count = 5000,
    threshold_lr = Float32(1e-6),
    show = false
)
```

Training uses ADAM with a learning rate decay schedule: the LR is divided by 10 each time validation loss plateaus for `threshold_count` epochs, and training halts when LR drops below `threshold_lr`.

### 6. Evaluate

The trained model is a callable struct — pass receiver positions to get predicted transmission loss:

```julia
TL_predicted = rbnn(rx_test)  # Returns 1 x N matrix of TL in dB

# Compute RMSE against reference data
using Flux
rmse = sqrt(Flux.Losses.mse(rbnn(rx_test), TL_test))
```

**What's happening under the hood:** Each call to `rbnn(rx)` evaluates the Helmholtz equation solution `p(r) = sum_m A_m * exp(i * (k * l_m(r) + phi_m))`, where `l_m(r)` is the distance from each ray's virtual source to the receiver. The 60 rays interfere at each receiver position, and the result is converted to dB. The `fit!` function splits data 70/30 into train/validation, runs gradient descent on RMSE loss, and applies LR decay on plateau. See [INTERNALS.md](INTERNALS.md) for full details.

## Architecture at a Glance

### Model type hierarchy

All models inherit from the abstract type `PropagationModel` and are callable structs:

- **`RayBasis2DCurv`** — 2D far-field with wavefront curvature (Case 1). Learns theta, A, phi, d.
- **`RayBasis3d`** — 3D near-field (Case 2). Uses image-source nominal geometry; learns A and phi corrections.
- **`RayBasisRCNN`** — 3D with a neural network that learns seabed reflection coefficients (Case 3). Only the RCNN is trainable.
- **`RayBasisRayleigh`** — 3D with physics-based Rayleigh reflection (Case 4). Learns seabed density ratio, sound speed ratio, and attenuation.
- **`RayBasis`** — Legacy 2D model for BSON compatibility.

### The `fit!` interface

Call `fit!(env, rx, data, model; ...)` and multiple dispatch routes to the correct training procedure. All methods share the same pattern: split data -> initialize optimizer -> train with LR decay on plateau -> return best model. The choice of how much environmental knowledge to incorporate is expressed entirely through the model type.

### File layout

```
src/
  core_v2.jl          # Entry point, abstract types, imports
  core_v2/
    models.jl          # Struct definitions, Flux.@functor, trainable fields
    forward.jl         # Callable model implementations (model(rx) -> TL)
    fit.jl             # fit! methods for each model type, train_rbnn!
    utils.jl           # LegacyADAM optimizer, data splitting, image sources, physics helpers
```

## Cases Overview

| Case | Model Type | What it models | What is learned | Test file |
|------|-----------|----------------|-----------------|-----------|
| 1 | `RayBasis2DCurv` | 2D far-field, unknown source | Ray angles, amplitudes, phases, curvature | `test/case1.jl` |
| 2 | `RayBasis3d` | 3D near-field, known source | Amplitude and phase corrections | `test/case2.jl` |
| 3 | `RayBasisRCNN` | 3D, unknown seabed | Neural network: incident angle -> reflection coefficient | `test/case3.jl` |
| 4 | `RayBasisRayleigh` | 3D, physical inversion | Seabed parameters (rho_r, c_r, delta) | `test/case4.jl` |

## References

- Li, K. & Chitre, M. (2023). "Data-aided underwater acoustic ray propagation modeling." *IEEE Journal of Oceanic Engineering*. [Link](https://ieeexplore.ieee.org/abstract/document/10224658)
- Li, K. & Chitre, M. (2021). "Ocean acoustic propagation modeling using scientific machine learning." *OCEANS: San Diego-Porto*, IEEE, pp. 1-5.
