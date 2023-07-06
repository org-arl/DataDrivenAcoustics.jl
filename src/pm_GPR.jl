
using GaussianProcesses

export GPR, GPRCal


"""
$(TYPEDEF)
A Gaussian process regression model to model acoustic propagation.
"""
Base.@kwdef struct GPR{T1, T2, T3, T4, T5} <: DataDrivenPropagationModel{T1}
    env::T1
    kern::T2
    mZero::T3 
    logObsNoise::Real
    GPmodel::T4 
    calculatefield::T5
    twoDimension::Bool
    function GPR(env, kern; mZero= MeanZero(), logObsNoise = -2.0, ratioₜ = 1.0, seed = false, calculatefield = GPRCal)
        rₜ, pₜ, _, _ = SplitData(env.locations, env.measurements, ratioₜ, seed)
        size(env.locations)[1] == 2 ? (twoDimension = true) : (twoDimension = false)
        GPmodel = GP(rₜ, vec(pₜ), mZero, kern, logObsNoise)
        optimize!(GPmodel)
        new{typeof(env), typeof(kern), typeof(mZero),typeof(GPmodel), typeof(calculatefield)}(env, kern, mZero,logObsNoise, GPmodel, calculatefield, twoDimension)
    end
end


"""
$(SIGNATURES)
Generate mean or standard deviation of Gaussian process regression prediction at location `xyz` using GPR model. Set `std` to true for mean predictions.
"""
function GPRCal(r::GPR, xyz::AbstractArray; std = false) 
    if  std == false 
        return predict_y(r.GPmodel, xyz)[1]
    else
        return predict_y(r.GPmodel, xyz)[2]
    end
end





