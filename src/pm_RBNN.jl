using Flux
using Random
using DSP
using BangBang
using StatsBase


export RayBasis2D, RayBasis2DCal,RayBasis2DCurv, RayBasis2DCurvCal, RayBasis3D, RayBasis3DCal, RayBasis3DRCNN, RayBasis3DRCNNCal

abstract type DataDrivenUnderwaterEnvironment end

abstract type DataDrivenPropagationModel{T<:DataDrivenUnderwaterEnvironment} end

#---------------------- RayBasis2D ------------------------

Base.@kwdef struct RayBasis2D{T1, T2, T3, T4<:AbstractVector, T5, T6} <: DataDrivenPropagationModel{T1}
    env::T1
    trainsettings::T2
    calculatefield::T3 = RayBasis2DCal
    nrays::Int = 60
    θ::T4 = Vector{Missing}(undef, nrays)
    A::T4 = Vector{Missing}(undef, nrays)
    ϕ::T4 = Vector{Missing}(undef, nrays)
    k::T5 = missing
    trainable::T6 = ()
    function RayBasis2D(env, trainsettings, calculatefield, nrays, θ, A, ϕ, k, trainable)
        if sum(ismissing.(θ)) > 0
            θ = rand(nrays) .* π
            trainable = push!!(trainable, θ)
        end
        if sum(ismissing.(A)) > 0
            A = rand(nrays) 
            trainable = push!!(trainable, A)
        end
        if sum(ismissing.(ϕ)) > 0
            ϕ = rand(nrays) .* π
            trainable = push!!(trainable, ϕ)
        end
        if k === missing
            if env.soundspeed !== missing && env.frequency !== missing 
                k = 2.0f0 * π * env.frequency / env.soundspeed
            else
                k = 2.0f0 * π * 2000.0f0 / 1500.0f0 
                trainable = push!!(trainable, k)
            end
        end       
        x = new{typeof(env), typeof(trainsettings), typeof(calculatefield), typeof(θ), typeof(k), typeof(trainable)}(env, trainsettings, calculatefield, nrays, θ, A, ϕ, k, trainable)
        ModelFit!(x)
        return x
    end
end


function RayBasis2DCal(r::RayBasis2D, xy::AbstractArray; showarrivals = false)
    x = @view xy[1:1,:]
    y = - @view xy[end:end,:]
    kx = r.k * (x .* cos.(r.θ) + y .* sin.(r.θ)) .+ r.ϕ
    showarrivals == false ? (return sum(r.A .* cis.(kx), dims = 1)) : (return r.A .* cis.(kx))
end


RayBasis2D(env, trainsettings; kwargs...) = RayBasis2D(; env = env, trainsettings = trainsettings, kwargs...)

Flux.@functor RayBasis2D
Flux.trainable(r::RayBasis2D) = r.trainable


#---------------------- RayBasis2DCurv ------------------------

Base.@kwdef struct RayBasis2DCurv{T1, T2, T3, T4<:AbstractVector, T5, T6} <: DataDrivenPropagationModel{T1}
    env::T1
    trainsettings::T2
    calculatefield::T3 = RayBasis2DCurvCal
    nrays::Int = 60
    θ::T4 = Vector{Missing}(undef, nrays)
    A::T4 = Vector{Missing}(undef, nrays)
    ϕ::T4 = Vector{Missing}(undef, nrays)
    d::T4 = Vector{Missing}(undef, nrays)
    k::T5 = missing
    trainable::T6 = ()
    function RayBasis2DCurv(env, trainsettings, calculatefield, nrays, θ, A, ϕ, d, k, trainable)
        if sum(ismissing.(θ)) > 0
            θ = rand(nrays) .* π
            trainable = push!!(trainable, θ)
        end
        if sum(ismissing.(A)) > 0
            A = rand(nrays) 
            trainable = push!!(trainable, A)
        end
        if sum(ismissing.(ϕ)) > 0
            ϕ = rand(nrays) .* π
            trainable = push!!(trainable, ϕ)
        end
        if sum(ismissing.(d)) > 0
            d = rand(nrays) 
            trainable = push!!(trainable, d)
        end
        if k === missing
            if env.soundspeed !== missing && env.frequency !== missing 
                k = 2.0f0 * π * env.frequency / env.soundspeed
            else
                k = 2.0f0 * π * 2000.0f0 / 1500.0f0 
                trainable = push!!(trainable, k)
            end
        end       
        x = new{typeof(env), typeof(trainsettings), typeof(calculatefield), typeof(θ), typeof(k), typeof(trainable)}(env, trainsettings, calculatefield, nrays, θ, A, ϕ, d, k, trainable)
        ModelFit!(x)
        return x
    end
end

# function to calculate field given RBNN parameters and location xy
function RayBasis2DCurvCal(r::RayBasis2DCurv, xy::AbstractArray; showarrivals = false)
    x = @view xy[1:1,:]
    y = - @view xy[end:end,:]
    xx = x .- (0.0 .- r.d .* cos.(r.θ))
    yy = y .- (0.0  .- r.d .* sin.(r.θ))
    l = sqrt.(xx.^2 + yy.^2)
    kx = r.k .* l .+ r.ϕ
    showarrivals == false ? (return sum(r.A ./ l .* cis.(kx), dims = 1)) : (return r.A ./ l.* cis.(kx))
end 

RayBasis2DCurv(env, trainsettings; kwargs...) = RayBasis2DCurv(; env = env, trainsettings = trainsettings, kwargs...)

Flux.@functor RayBasis2DCurv
Flux.trainable(r::RayBasis2DCurv) = r.trainable


#---------------------- RayBasis3D ------------------------


Base.@kwdef struct RayBasis3D{T1, T2, T3, T4<:AbstractVector, T5, T6} <: DataDrivenPropagationModel{T1}
    env::T1
    trainsettings::T2
    calculatefield::T3 = RayBasis3DCal
    nrays::Int = 60
    θ::T4 = Vector{Missing}(undef, nrays)
    ψ::T4 = Vector{Missing}(undef, nrays)
    d::T4 = Vector{Missing}(undef, nrays)
    eθ::T4 = Vector{Missing}(undef, nrays)
    eψ::T4 = Vector{Missing}(undef, nrays)
    ed::T4 = Vector{Missing}(undef, nrays)
    A::T4 = Vector{Missing}(undef, nrays)
    ϕ::T4 = Vector{Missing}(undef, nrays)
    k::T5 = missing
    trainable::T6 = ()
    function RayBasis3D(env, trainsettings, calculatefield, nrays, θ, ψ, d,  eθ, eψ, ed, A, ϕ, k, trainable)
        size(env.locations)[1] == 3 || throw(ArgumentError("Dimension of measurement locations must be 3."))
        if sum(ismissing.(θ)) > 0
            θ = rand(nrays) .* π
            trainable = push!!(trainable, θ)
        end
        if sum(ismissing.(ψ)) > 0
            θ = rand(nrays) .* π
            trainable = push!!(trainable, ψ)
        end
        if sum(ismissing.(d)) > 0
            d = rand(nrays) .* π
            trainable = push!!(trainable, d)
        end
        if sum(ismissing.(eθ)) > 0
            θ = zeros(nrays) .* π
            trainable = push!!(trainable, eθ)
        end
        if sum(ismissing.(eψ)) > 0
            θ = zeros(nrays) .* π
            trainable = push!!(trainable, eψ)
        end
        if sum(ismissing.(ed)) > 0
            d = zeros(nrays) .* π
            trainable = push!!(trainable, ed)
        end

        if sum(ismissing.(A)) > 0
            A = rand(nrays) 
            trainable = push!!(trainable, A)
        end
        if sum(ismissing.(ϕ)) > 0
            ϕ = rand(nrays) .* π
            trainable = push!!(trainable, ϕ)
        end
        if k === missing
            if env.soundspeed !== missing && env.frequency !== missing 
                k = 2.0f0 * π * env.frequency / env.soundspeed
            else
                k = 2.0f0 * π * 2000.0f0 / 1500.0f0 
                trainable = push!!(trainable, k)
            end
        end       
        x = new{typeof(env), typeof(trainsettings), typeof(calculatefield), typeof(θ), typeof(k), typeof(trainable)}(env, trainsettings, calculatefield, nrays, θ, ψ, d,  eθ, eψ, ed, A, ϕ, k, trainable)
        ModelFit!(x)
        return x
    end
end


function RayBasis3DCal(r::RayBasis3D, xyz::AbstractArray; showarrivals = false)
    x = @view xyz[1:1,:]                
    y = @view xyz[2:2,:] 
    z = - @view xyz[3:3,:] 
  
    xx = x .- (xₒ[1] .- (r.ed .+ r.d) .* cos.(r.eθ .+ r.θ) .* sin.(r.eψ .+ r.ψ))
    yy = y .- (xₒ[2] .- (r.ed .+ r.d) .* sin.(r.eθ .+ r.θ) .* sin.(r.eψ .+ r.ψ))
    zz = z .- (xₒ[3] .- (r.ed .+ r.d) .* cos.(r.eψ .+ r.ψ))
    l = sqrt.(xx.^2.0f0 + yy.^2.0f0 + zz.^2.0f0)
    kx = r.k .* l .+ r.ϕ
    showarrivals == false ? (return sum(r.A ./ l .* cis.(kx), dims = 1)) : (return r.A ./ l .* cis.(kx))
end

RayBasis3D(env, trainsettings, calculatefield; kwargs...) = RayBasis3D(; env = env, trainsettings = trainsettings, kwargs...)

Flux.@functor RayBasis3D
Flux.trainable(r::RayBasis3D) = r.trainable


#---------------------- RayBasis3DRCNN ------------------------

Base.@kwdef struct RayBasis3DRCNN{T1, T2, T3, T4, T5<:AbstractVector, T6, T7} <: DataDrivenPropagationModel{T1}
    env::T1
    trainsettings::T2
    RCNN::T3
    calculatefield::T4 = RayBasis3DRCNNCal
    nrays::Int = 60
    θ::T5 = Vector{Missing}(undef, nrays)
    ψ::T5 = Vector{Missing}(undef, nrays)
    d::T5 = Vector{Missing}(undef, nrays)
    eθ::T5 = Vector{Missing}(undef, nrays)
    eψ::T5 = Vector{Missing}(undef, nrays)
    ed::T5 = Vector{Missing}(undef, nrays)
    k::T6 = missing
    trainable::T7 = ()
    function RayBasis3DRCNN(env, trainsettings, RCNN, calculatefield, nrays, θ, ψ, d,  eθ, eψ, ed, k, trainable)
        trainable = push!!(trainable, RCNN)
        size(env.locations)[1] == 3 || throw(ArgumentError("Dimension of measurement locations must be 3."))
        sum(ismissing.(tx)) == 0 || throw(ArgumentError("Source location must be provided."))
        length(env.tx) == 3 || throw(ArgumentError("Source location must be 3 dimensional."))
        if sum(ismissing.(θ)) > 0
            θ = rand(nrays) .* π
            trainable = push!!(trainable, θ)
        end
        if sum(ismissing.(ψ)) > 0
            θ = rand(nrays) .* π
            trainable = push!!(trainable, ψ)
        end

        if sum(ismissing.(d)) > 0
            d = rand(nrays) .* π
            trainable = push!!(trainable, d)
        end
        if sum(ismissing.(eθ)) > 0
            θ = zeros(nrays) .* π
            trainable = push!!(trainable, eθ)
        end
        if sum(ismissing.(eψ)) > 0
            θ = zeros(nrays) .* π
            trainable = push!!(trainable, eψ)
        end

        if sum(ismissing.(ed)) > 0
            d = zeros(nrays) .* π
            trainable = push!!(trainable, ed)
        end

        if sum(ismissing.(A)) > 0
            A = rand(nrays) 
            trainable = push!!(trainable, A)
        end
        if sum(ismissing.(ϕ)) > 0
            ϕ = rand(nrays) .* π
            trainable = push!!(trainable, ϕ)
        end
        if k === missing
            if env.soundspeed !== missing && env.frequency !== missing 
                k = 2.0f0 * π * env.frequency / env.soundspeed
            else
                k = 2.0f0 * π * 2000.0f0 / 1500.0f0 
                trainable = push!!(trainable, k)
            end
        end       
        x = new{typeof(env), typeof(trainsettings), typeof(RCNN), typeof(calculatefield), typeof(θ), typeof(k), typeof(trainable)}(env, trainsettings, RCNN, calculatefield, nrays, θ, ψ, d,  eθ, eψ, ed, k, trainable)
        ModelFit!(x)
        return x
    end
end


function RayBasis3DRCNNCal(r::RayBasis3DRCNN, xyz::AbstractArray; showarrivals = false)
    x = @view xyz[1:1,:]                
    y = @view xyz[2:2,:] 
    z = - @view xyz[3:3,:] 
    xx = x .- (xₒ[1] .- (r.ed .+ r.d) .* cos.(r.eθ .+  r.θ) .* sin.(r.eψ .+  r.ψ))
    yy = y .- (xₒ[2] .- (r.ed .+ r.d) .* sin.(r.eθ .+  r.θ) .* sin.(r.eψ .+  r.ψ))
    zz = z .- (xₒ[3] .- (r.ed .+ r.d) .* cos.(r.eψ .+  r.ψ))
    
    l = sqrt.(xx.^2.0f0 + yy.^2.0f0 + zz.^2.0f0)
    j = collect(1: 1: r.nrays) 
    R = (abs2.(r.env.tx[1] .- x) .+ abs2.(r.env.tx[2] .- y)).^ 0.5f0
    upward = iseven.(j) 
    s1 = 2.0f0 .* upward .- 1.0f0 
    n = div.(j, 2.0f0) 
    s = div.(n .+ upward, 2.0f0) 
    b = div.(n .+ (1 .- upward), 2.0f0) 
    s2 = 2.0f0 .* iseven.(n) .- 1.0f0 
    dz = 2.0f0 .* b .* r.L .+ s1 .* r.env.tx[3] .- s1 .* s2 .* z
    incidentangle = abs.(atan.(R ./  dz)) 

    surfaceloss =  ipow(reflectioncoef(env.seasurface, env.freqe, incidentangle), s)

    RC = Matrix{Float32}(undef, r.nrays,size(zz)[2])
    phase = Matrix{Float32}(undef, r.nrays, size(zz)[2])
    bufRCNN = Zygote.Buffer(RC, 2, size(zz)[2])
    bufphase = Zygote.Buffer(phase, size(phase))
    bufRC = Zygote.Buffer(RC, size(RC))

    for i in 1 : r.nrays
        bufRCNN = r.RCNN(incidentangle[i:i,:])
        bufRC[i:i,:] = abs.(bufRCNN[1:1,:]) 
        bufphase[i:i,:] = bufRCNN[2:2,:]
    end

    totalphase =  2π * l ./ c * f .+ copy(bufphase).* b
    amp = 1.0f0 ./ l  .* surfaceloss .* copy(bufRC).^ b .* absorption.(r.env.frequency, l, r.env.salinity)
    sum(amp.* cis.(totalphase); dims=1)
    showarrivals == false ? (return sum(amp.* cis.(totalphase); dims=1)) : (return amp.* cis.(totalphase))

end

RayBasis3DRCNN(env, trainsettings, RCNN; kwargs...) = RayBasis3DRCNN(; env = env, trainsettings = trainsettings, RCNN = RCNN, kwargs...)

Flux.@functor RayBasis3DRCNN
Flux.trainable(r::RayBasis3DRCNN) = r.trainable


