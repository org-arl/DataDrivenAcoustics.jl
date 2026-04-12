
using GaussianProcesses

export GPR, GPRCal


"""
$(TYPEDEF)
A Gaussian process regression model to model acoustic propagation.
"""
Base.@kwdef struct GPR{T, T1, T2, T3, T4, T5} <: DataDrivenPropagationModel{T}
    env::T1
    kern::T2
    mZero::T3
    logObsNoise::Real
    GPmodel::T4
    calculatefield::T5
    twoDimension::Bool

    # Inner Constructor
    function GPR(env, kern; mZero=MeanZero(), logObsNoise = -2.0, ratioₜ = 1.0, seed = false, calculatefield = GPRCal)
        # 1. Detect Precision T (using measurements or default to Float64)
        T = hasproperty(env, :measurements) && !ismissing(env.measurements) ? eltype(env.measurements) : Float64

        # 2. Logic
        # Note: This relies on an external 'SplitData' function from your legacy code
        rₜ, pₜ, _, _ = SplitData(env.locations, env.measurements, ratioₜ, seed)

        twoDimension = size(env.locations, 1) == 2

        # 3. Create and Train the GP (using GaussianProcesses.jl likely)
        GPmodel = GP(rₜ, vec(pₜ), mZero, kern, logObsNoise)
        optimize!(GPmodel)

        # 4. Return new object with T
        new{T, typeof(env), typeof(kern), typeof(mZero), typeof(GPmodel), typeof(calculatefield)}(
            env, kern, mZero, logObsNoise, GPmodel, calculatefield, twoDimension
        )
    end
end

"""
$(SIGNATURES)
Generate mean or standard deviation of Gaussian process regression prediction at location `xyz`.
Set `std` to true to return standard deviation (uncertainty), otherwise returns mean.
"""
function GPRCal(r::GPR, xyz::AbstractArray; std = false)
    # predict_y returns a tuple: (mean, variance) usually, or (mean, std) depending on the package.
    # Assuming index 1 is Mean and index 2 is Std/Variance.
    if std == false
        return predict_y(r.GPmodel, xyz)[1]
    else
        return predict_y(r.GPmodel, xyz)[2]
    end
end





