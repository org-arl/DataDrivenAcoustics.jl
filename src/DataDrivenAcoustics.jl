module DataDrivenAcoustics

using UnderwaterAcoustics
using DocStringExtensions
using DSP: amp2db, db2amp, pow2db, db2pow
export BasicDataDrivenUnderwaterEnvironment, DataDrivenUnderwaterEnvironment
export RayBasisNN, SphericalWaveModel, plane_wave_propagate, PlaneWaveCurvModel
export fit!, calculate_field
export initialize_angles, smart_initialize_angles, stack_coordinates, prepare_measurements

# New unified API (v2)
export PropagationModel
export RayBasis, RayBasis2DCurv, RayBasis3d, RayBasisRCNN, RayBasisRayleigh
export zig_zag_samples, data_split, n_images_src, n_images_src_with_ref, cartesian2spherical
export extract_array_from_bson, reflectioncoef

include("pm_core.jl")
include("pm_utility.jl")
include("pm_RBNN.jl")
include("pm_GPR.jl")
include("core_v2.jl")


#= function __init__()
    UnderwaterAcoustics.addmodel!(RayBasis2D)
    UnderwaterAcoustics.addmodel!(RayBasis2DCurv)
    UnderwaterAcoustics.addmodel!(RayBasis3D)
    UnderwaterAcoustics.addmodel!(RayBasis3DRCNN)
    UnderwaterAcoustics.addmodel!(GPR)
end =#

end