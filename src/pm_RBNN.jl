using Flux
using Random
using BangBang
using Zygote


export RayBasis2D, RayBasis2DCal, RayBasis2DCurv, RayBasis2DCurvCal, RayBasis3D, RayBasis3DCal, RayBasis3DRCNN, RayBasis3DRCNNCal

abstract type DataDrivenUnderwaterEnvironment end

abstract type DataDrivenPropagationModel{T<:DataDrivenUnderwaterEnvironment} end

"""
$(TYPEDEF)
A 2D plane wave RBNN formualtion.

- `env`: data driven underwater environment
- `calculatefield`: function to estimate acoustic field (default: `RayBasis2DCal`)
- `nrays`: number of rays (default: 60)
- `θ`: azimuthal angle of arrival rays in radian (default: missing)
- `A`: amplitude of arrival rays (default: missing)
- `ϕ`: phase of a rays in radian (default: missing)
- `k`: angular wavenumber in rad/m (default: missing)
- `trainable`: trainable parameters (default: empty)

- `ini_lr`: initial learning rate (default: 0.001)
- `trainloss`: loss function used in training and model update (default: `rmseloss`)
- `dataloss`: data loss function to calculate benchmarking validation error for early stopping (default: `rmseloss`)
- `ratioₜ`: data split ratio = number of training data/(number of training data + number of validation data) (default: 0.7)
- set `seed` to `true` to seed random data selection order (default: `false`)
- `maxepoch`: maximum number of training epoches allowed (default: 10000000)
- `ncount`: maximum number of tries before reducing learning rate (default: 5000)
-  model training ends once learning rate is smaller than `minlearnrate` (default: 1e-6)
- learning rate is reduced by `reducedlearnrate` once `ncount` is reached (default: 10)
- set `showloss` to true to display training and validation errors during the model training process, if the validation error is historically the best. (default: `false`)
"""
Base.@kwdef struct RayBasis2D{T1, T2, T3<:AbstractVector, T4, T5} <: DataDrivenPropagationModel{T1}
    env::T1
    calculatefield::T2
    nrays::Int
    θ::T3
    A::T3
    ϕ::T3
    k::T4
    trainable::T5
    function RayBasis2D(env; calculatefield = RayBasis2DCal, nrays = 60, θ = Vector{Missing}(undef, nrays), A = Vector{Missing}(undef, nrays),
        ϕ = Vector{Missing}(undef, nrays), k = missing, inilearnrate::Real = 0.001, trainloss = rmseloss, dataloss = rmseloss, ratioₜ::Real = 0.7,
        seed = false, maxepoch::Int = 10000000, ncount::Int = 5000, minlearnrate::Real = 1e-6 , reducedlearnrate::Real = 10.0, showloss::Bool = false)

        trainable = ()
        size(env.locations)[1] == 2 || throw(ArgumentError("RayBasis2D only supports 2D environment"))
        ratioₜ <= 1.0 || throw(ArgumentError("Training data split ratio can not exceed 1"))
        ratioₜ > 0.0 || throw(ArgumentError("Training data split ratio should be larger than 0"))

        seed == true && Random.seed!(6)

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
        x = new{typeof(env), typeof(calculatefield), typeof(θ), typeof(k), typeof(trainable)}(env, calculatefield, nrays, θ, A, ϕ, k, trainable)

        # removed ModelFit() call to allow user to control training process outside of constructor

        return x
    end
end

"""
$(SIGNATURES)
Predict acoustic field at location `xyz` using `RayBasis2D` model. Set `showarrivals` to `true` to return an array of individual complex arrivals.
"""
function RayBasis2DCal(r::RayBasis2D, xy::AbstractArray; showarrivals = false)
    x = @view xy[1:1,:]
    y = - @view xy[end:end,:]
    kx = r.k * (x .* cos.(r.θ) + y .* sin.(r.θ)) .+ r.ϕ
    showarrivals == false ? (return sum(r.A .* cis.(kx), dims = 1)) : (return r.A .* cis.(kx))
end


Flux.@functor RayBasis2D
Flux.trainable(r::RayBasis2D) = r.trainable


"""
$(TYPEDEF)
A 2D plane wave RBNN formualtion by modeling curvature of wavefornt.

- `env`: data driven underwater environment
- `calculatefield`: function to estimate acoustic field (default: `RayBasis2DCurvCal`)
- `nrays`: number of rays (default: 60)
- `θ`: azimuthal angle of arrival ray in radian (default: missing)
- `A`: amplitude of arrival rays (default: missing)
- `ϕ`: phase of a rays in radian (default: missing)
- `d`: distance in meters to help in modeling curvature of wavefornt (default: missing)
- `k`: angular wavenumber in rad/m (default: missing)
- `trainable`: trainable parameters (default: empty)

- `ini_lr`: initial learning rate (default: 0.001)
- `trainloss`: loss function used in training and model update (default: `rmseloss`)
- `dataloss`: data loss function to calculate benchmarking validation error for early stopping (default: `rmseloss`)
- `ratioₜ`: data split ratio = number of training data/(number of training data + number of validation data) (default: 0.7)
- set `seed` to `true` to seed random data selection order (default: `false`)
- `maxepoch`: maximum number of training epoches allowed (default: 10000000)
- `ncount`: maximum number of tries before reducing learning rate (default: 5000)
-  model training ends once learning rate is smaller than `minlearnrate` (default: 1e-6)
- learning rate is reduced by `reducedlearnrate` once `ncount` is reached (default: 10)
- set `showloss` to true to display training and validation errors during the model training process, if the validation error is historically the best. (default: `false`)

"""
Base.@kwdef struct RayBasis2DCurv{T1, T2, T3<:AbstractVector, T4, T5} <: DataDrivenPropagationModel{T1}
    env::T1
    calculatefield::T2
    nrays::Int
    θ::T3
    A::T3
    ϕ::T3
    d::T3
    k::T4
    trainable::T5
    function RayBasis2DCurv(env;  calculatefield = RayBasis2DCurvCal, nrays = 60, θ = Vector{Missing}(undef, nrays),
        A = Vector{Missing}(undef, nrays), ϕ= Vector{Missing}(undef, nrays), d = Vector{Missing}(undef, nrays), k = missing,
        inilearnrate::Real = 0.001, trainloss = rmseloss, dataloss = rmseloss, ratioₜ::Real = 0.7, seed = false, maxepoch::Int = 10000000,
        ncount::Int = 5000, minlearnrate::Real = 1e-6 , reducedlearnrate::Real = 10.0, showloss::Bool = false)

        size(env.locations)[1] == 2 || throw(ArgumentError("RayBasis2DCurv only supports 2D environment"))
        trainable = ()
        seed == true && Random.seed!(6)

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
        x = new{typeof(env), typeof(calculatefield), typeof(θ), typeof(k), typeof(trainable)}(env, calculatefield, nrays, θ, A, ϕ, d, k, trainable)
        ModelFit!(x, inilearnrate,trainloss, dataloss, ratioₜ, seed, maxepoch, ncount, minlearnrate, reducedlearnrate, showloss)
        return x
    end
end

"""
$(SIGNATURES)
Predict acoustic field at location `xyz` using `RayBasis2DCurv` model. Set `showarrivals` to `true` to return an array of individual complex arrivals.
`xₒ` is the reference location and can be an arbitrary location.
"""
function RayBasis2DCurvCal(r::RayBasis2DCurv, xy::AbstractArray; showarrivals = false, xₒ = [0.0, 0.0])
    x = @view xy[1:1,:]
    y = - @view xy[end:end,:]
    xx = x .- (xₒ[1] .- r.d .* cos.(r.θ))
    yy = y .- (xₒ[2] .- r.d .* sin.(r.θ))
    l = sqrt.(xx.^2 + yy.^2)
    kx = r.k .* l .+ r.ϕ
    showarrivals == false ? (return sum(r.A ./ l .* cis.(kx), dims = 1)) : (return r.A ./ l.* cis.(kx))
end

Flux.@functor RayBasis2DCurv
Flux.trainable(r::RayBasis2DCurv) = r.trainable


"""
$(TYPEDEF)
A 3D spherical wave RBNN formualtion.

- `env`: data driven underwater environment
- `calculatefield`: function to estimate acoustic field (default: `RayBasis3DCal`)
- `nrays`: number of rays (default: 60)
- `θ`: nominal azimuthal angle of arrival rays in radian (default: missing)
- `ψ`: nominal elevation angle of arrival rays in radian (default: missing)
- `d`: nominal propagation distance of arrival rays  in meters (default: missing)
- `eθ`: error to nominal azimuthal angle of arrival rays in radian (default: missing)
- `eψ`: error to nominal elevation angle of arrival rays in radian (default: missing)
- `ed`: error to nominal propagation distance of arrival rays in meters (default: missing)
- `A`: amplitude of arrival rays (default: missing)
- `ϕ`: phase of a rays in radian (default: missing)
- `k`: angular wavenumber in rad/m (default: missing)
- `trainable`: trainable parameters (default: empty)

- `ini_lr`: initial learning rate (default: 0.001)
- `trainloss`: loss function used in training and model update (default: `rmseloss`)
- `dataloss`: data loss function to calculate benchmarking validation error for early stopping (default: `rmseloss`)
- `ratioₜ`: data split ratio = number of training data/(number of training data + number of validation data) (default: 0.7)
- set `seed` to `true` to seed random data selection order (default: `false`)
- `maxepoch`: maximum number of training epoches allowed (default: 10000000)
- `ncount`: maximum number of tries before reducing learning rate (default: 5000)
-  model training ends once learning rate is smaller than `minlearnrate` (default: 1e-6)
- learning rate is reduced by `reducedlearnrate` once `ncount` is reached (default: 10)
- set `showloss` to true to display training and validation errors during the model training process, if the validation error is historically the best. (default: `false`)

"""
Base.@kwdef struct RayBasis3D{T1, T2, T3<:AbstractVector, T4, T5} <: DataDrivenPropagationModel{T1}
    env::T1
    calculatefield::T2
    nrays::Int
    θ::T3
    ψ::T3
    d::T3
    eθ::T3
    eψ::T3
    ed::T3
    A::T3
    ϕ::T3
    k::T4
    trainable::T5
    function RayBasis3D(env; calculatefield = RayBasis3DCal, nrays = 60, θ = Vector{Missing}(undef, nrays), ψ = Vector{Missing}(undef, nrays),
        d = Vector{Missing}(undef, nrays), eθ = Vector{Missing}(undef, nrays), eψ = Vector{Missing}(undef, nrays), ed = Vector{Missing}(undef, nrays),
        A = Vector{Missing}(undef, nrays), ϕ = Vector{Missing}(undef, nrays), k = missing,  inilearnrate::Real = 0.001, trainloss = rmseloss,
        dataloss = rmseloss, ratioₜ::Real = 0.7, seed = false, maxepoch::Int = 10000000, ncount::Int = 5000, minlearnrate::Real = 1e-6 ,
        reducedlearnrate::Real = 10.0, showloss::Bool = false)

        trainable = ()
        seed == true && Random.seed!(6)
        size(env.locations)[1] == 3 || throw(ArgumentError("RayBasis3D only supports 3D environment."))


        if sum(ismissing.(θ)) > 0
            θ = rand(nrays) .* π
            trainable = push!!(trainable, θ)
            eθ = zeros(nrays) .* π
        end
        if sum(ismissing.(ψ)) > 0
            ψ = rand(nrays) .* π
            trainable = push!!(trainable, ψ)
            eψ = zeros(nrays) .* π
        end
        if sum(ismissing.(d)) > 0
            d = rand(nrays) .* π
            trainable = push!!(trainable, d)
            ed = zeros(nrays) .* π
        end
        if sum(ismissing.(eθ)) > 0
            eθ = zeros(nrays) .* π
            trainable = push!!(trainable, eθ)
        end
        if sum(ismissing.(eψ)) > 0
            eψ = zeros(nrays) .* π
            trainable = push!!(trainable, eψ)
        end
        if sum(ismissing.(ed)) > 0
            ed = zeros(nrays) .* π
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
        x = new{typeof(env), typeof(calculatefield), typeof(θ), typeof(k), typeof(trainable)}(env, calculatefield, nrays, θ, ψ, d,  eθ, eψ, ed, A, ϕ, k, trainable)
        ModelFit!(x, inilearnrate,trainloss, dataloss, ratioₜ, seed, maxepoch, ncount, minlearnrate, reducedlearnrate, showloss)
        return x
    end
end

"""
$(SIGNATURES)
Predict acoustic field at location `xyz` using `RayBasis3D` model. Set `showarrivals` to `true` to return an array of individual complex arrivals.
`xₒ` is the reference location and can be an arbitrary location.
"""
function RayBasis3DCal(r::RayBasis3D, xyz::AbstractArray; showarrivals = false, xₒ = [0.0, 0.0, 0.0])
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


Flux.@functor RayBasis3D
Flux.trainable(r::RayBasis3D) = r.trainable


"""
$(TYPEDEF)
A 3D spherical wave RBNN formualtion with reflection coefficient neural network (RCNN) as part of the model.

- `env`: data driven underwater environment
- `RCNN`: neural network to model seabed reflection coefficient
- `calculatefield`: function to estimate acoustic field (default: `RayBasis3DRCNNCal`)
- `nrays`: number of rays (default: 60)
- `eθ`: error to nominal azimuthal angle of arrival rays in radian (default: missing)
- `eψ`: error to nominal elevation angle of arrival rays in radian (default: missing)
- `ed`: error to nominal propagation distance of arrival rays in meters (default: missing)
- `k`: angular wavenumber in rad/m (default: missing)
- `trainable`: trainable parameters (default: empty)

- `ini_lr`: initial learning rate (default: 0.001)
- `trainloss`: loss function used in training and model update (default: `rmseloss`)
- `dataloss`: data loss function to calculate benchmarking validation error for early stopping (default: `rmseloss`)
- `ratioₜ`: data split ratio = number of training data/(number of training data + number of validation data) (default: 0.7)
- set `seed` to `true` to seed random data selection order (default: `false`)
- `maxepoch`: maximum number of training epoches allowed (default: 10000000)
- `ncount`: maximum number of tries before reducing learning rate (default: 5000)
-  model training ends once learning rate is smaller than `minlearnrate` (default: 1e-6)
- learning rate is reduced by `reducedlearnrate` once `ncount` is reached (default: 10)
- set `showloss` to true to display training and validation errors during the model training process, if the validation error is historically the best. (default: `false`)

"""
Base.@kwdef struct RayBasis3DRCNN{T1, T2, T3, T4<:AbstractVector, T5, T6} <: DataDrivenPropagationModel{T1}
    env::T1
    RCNN::T2
    calculatefield::T3
    nrays::Int
    θ::T4
    ψ::T4
    d::T4
    k::T5
    trainable::T6
    function RayBasis3DRCNN(env, RCNN; calculatefield = RayBasis3DRCNNCal, nrays = 60, θ = Vector{Missing}(undef, nrays), ψ = Vector{Missing}(undef, nrays),
        d = Vector{Missing}(undef, nrays), k = missing, inilearnrate::Real = 0.001, trainloss = rmseloss, dataloss = rmseloss, ratioₜ::Real = 0.7,
        seed = false, maxepoch::Int = 10000000, ncount::Int = 5000, minlearnrate::Real = 1e-6 , reducedlearnrate::Real = 10.0, showloss::Bool = false)

        trainable = ()
        seed == true && Random.seed!(6)
        size(env.locations)[1] == 3 || throw(ArgumentError("RayBasis3DRCNN only supports 3D environment."))
        trainable = push!!(trainable, RCNN)
        env.tx !== missing  || throw(ArgumentError("Source location must be provided."))
        length(location(env.tx)) == 3 || throw(ArgumentError("Source location must be 3 dimensional."))
        env.waterdepth !== missing || throw(ArgumentError("Water depth needs to be provided"))

        θ, ψ, d = cartesian2spherical([0.0, 0.0, 0.0].- find_image_src(env.locations[:,1], location(env.tx), nrays, env.waterdepth))
        if k === missing
            if env.soundspeed !== missing && env.frequency !== missing
                k = 2.0f0 * π * env.frequency / env.soundspeed
            else
                k = 2.0f0 * π * 2000.0f0 / 1500.0f0
                trainable = push!!(trainable, k)
            end
        end
        x = new{typeof(env), typeof(RCNN), typeof(calculatefield), typeof(θ), typeof(k), typeof(trainable)}(env, RCNN, calculatefield, nrays, θ, ψ, d, k, trainable)
        ModelFit!(x, inilearnrate,trainloss, dataloss, ratioₜ, seed, maxepoch, ncount, minlearnrate, reducedlearnrate, showloss)
        return x
    end
end

"""
$(SIGNATURES)
Predict acoustic field at location `xyz` using `RayBasis3DRCNN` model. Set `showarrivals` to `true` to return an array of individual complex arrivals.
`xₒ` is the reference location and can be an arbitrary location.
"""
function RayBasis3DRCNNCal(r::RayBasis3DRCNN, xyz::AbstractArray; showarrivals = false, xₒ = [0.0, 0.0, 0.0])
    x = @view xyz[1:1,:]
    y = @view xyz[2:2,:]
    z = - @view xyz[3:3,:]
    xx = x .- (xₒ[1] .- r.d .* cos.(r.θ) .* sin.(r.ψ))
    yy = y .- (xₒ[2] .- r.d .* sin.(r.θ) .* sin.(r.ψ))
    zz = z .- (xₒ[3] .- r.d .* cos.(r.ψ))


    l = sqrt.(xx.^2.0f0 + yy.^2.0f0 + zz.^2.0f0)
    j = collect(1: 1: r.nrays)
    R = (abs2.(location(r.env.tx)[1] .- x) .+ abs2.(location(r.env.tx)[2] .- y)).^ 0.5f0
    upward = iseven.(j)
    s1 = 2.0f0 .* upward .- 1.0f0
    n = div.(j, 2)
    s = div.(n .+ upward, 2.0f0)
    b = div.(n .+ (1 .- upward), 2.0f0)
    s2 = 2.0f0 .* iseven.(n) .- 1.0f0
    dz = 2.0f0 .* b .* r.env.waterdepth .+ s1 .*location(r.env.tx)[3] .- s1 .* s2 .* z
    incidentangle = Float32.(abs.(atan.(R ./ dz)))

    surfaceloss = reflectioncoef(r.env.seasurface, r.env.frequency, incidentangle).^s

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

    totalphase = r.k * l .+ copy(bufphase) .* b
    amp = 1.0f0 ./ l .* surfaceloss .* copy(bufRC).^ b .* absorption.(r.env.frequency, l, r.env.salinity)
    showarrivals == false ? (return sum(amp.* cis.(totalphase); dims=1)) : (return amp.* cis.(totalphase))
end

Flux.@functor RayBasis3DRCNN
Flux.trainable(r::RayBasis3DRCNN) = r.trainable


