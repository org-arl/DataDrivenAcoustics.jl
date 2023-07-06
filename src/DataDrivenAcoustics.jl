module DataDrivenAcoustics

using UnderwaterAcoustics
using DocStringExtensions
using DSP: amp2db, db2amp, pow2db, db2pow


include("pm_RBNN.jl")
include("pm_GPR.jl")
include("pm_core.jl")
include("pm_utility.jl")



function __init__()
    UnderwaterAcoustics.addmodel!(RayBasis2D)
    UnderwaterAcoustics.addmodel!(RayBasis2DCurv)
    UnderwaterAcoustics.addmodel!(RayBasis3D)
    UnderwaterAcoustics.addmodel!(RayBasis3DRCNN)
    UnderwaterAcoustics.addmodel!(GPR)
end

end 