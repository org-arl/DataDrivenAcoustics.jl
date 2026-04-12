# ============================================================================
# Core V2 - Unified Ray Basis Models
# ============================================================================
#
# This is the entry point for the v2 implementation of DataDrivenAcoustics.
#
# Key concepts:
# 1. Models inherit from PropagationModel
# 2. Models are callable structs: model(rx) instead of model.calculatefield(model, rx)
# 3. Training uses fit!(env, rx, data, model) with multiple dispatch
#
# Files:
# - models.jl   : Model struct definitions
# - forward.jl  : Forward pass implementations (callable structs)
# - fit.jl      : Training API (fit! functions)
# - utils.jl    : Data utilities and helpers
# ============================================================================

using StatsBase: rle
using LinearAlgebra: norm
using Random
using Flux
using DSP: amp2db
using UnderwaterAcoustics
using AcousticsToolbox
using Statistics

import Flux.Optimise: apply!, AbstractOptimiser

# ============================================================================
# Abstract Type Hierarchy
# ============================================================================

"""
    PropagationModel

Abstract base type for all ray basis propagation models.
All models are callable structs that compute acoustic field predictions.

# Subtypes
- `RayBasis` - 2D far-field (legacy)
- `RayBasis2DCurv` - 2D far-field with wavefront curvature (Case 1)
- `RayBasis3d` - 3D near-field (Case 2)
- `RayBasisRCNN` - 3D with learned reflection coefficients (Case 3)
- `RayBasisRayleigh` - 3D with Rayleigh reflection physics (Case 4)

# Usage
```julia
# Create model
model = RayBasis2DCurv(60, k)

# Forward pass (callable struct pattern)
TL = model(rx)

# Training
trained_model = fit!(env, rx, data, model; ...)
```
"""
abstract type PropagationModel end

# Include component files
include("core_v2/utils.jl")
include("core_v2/models.jl")
include("core_v2/forward.jl")
include("core_v2/fit.jl")
