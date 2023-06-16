using UnderwaterAcoustics
using DataFrames
using RecipesBase

export DataDrivenUnderwaterEnvironment, ModelTrainingSetting,  ModelFit!, transfercoef, transmissionloss, check, plot, rays, eigenrays, arrivals

Base.@kwdef struct BasicDataDrivenUnderwaterEnvironment{T1<:Matrix, T2, T3, T4, T5<:ReflectionModel, T6<:ReflectionModel, T7} <: DataDrivenUnderwaterEnvironment
    locations::T1 
    measurements::T1 
    soundspeed::T2 = missing
    frequency::T3  = missing
    waterdepth::T4 = missing
    salinity::Number = 35.0
    seasurface::T5 = Vacuum
    seabed::T6 = SandySilt
    tx::T7 = Vector{Missing}(undef, 3)
    function BasicDataDrivenUnderwaterEnvironment(locations, measurements, soundspeed, frequency, waterdepth, salinity, seasurface, seabed, tx)
        if  sum(ismissing.(tx)) == 0 
            length(tx) == size(locations)[1] || throw(ArgumentError("Dimension of source location and measurement locations do not match"))
        end
        (size(locations)[2] == size(measurements)[2]) || throw(ArgumentError("Number of locations and fields measurements do not match"))
        (size(locations)[1] < 4) || throw(ArgumentError("Dimension of location data should not be larger than 3"))
        (size(measurements)[1] == 1) || throw(ArgumentError("size of acoustic measurements should be 1 × n"))
        new{typeof(locations), typeof(soundspeed), typeof(frequency), typeof(waterdepth), typeof(seasurface), typeof(seabed), typeof(tx)}(locations, measurements, soundspeed, frequency, waterdepth, salinity, seasurface, seabed, tx)
    end
end

DataDrivenUnderwaterEnvironment(locations, measurements; kwargs...) = BasicDataDrivenUnderwaterEnvironment(; locations = locations, measurements = measurements, kwargs...)



Base.@kwdef struct ModelTrainingSetting{T1<:Real, T2, T3}
    inilearnrate::T1                       # initial learning rate
    trainloss:: T2                      
    dataloss::T3
    ratioₜ::T1 = 0.7
    seed::Bool = false
    maxepoch::Int = 10000000               # Maximum number of training epoches allowed
    ncount::Int = 5000              # maximum number of tries before reducing learning rate
    minlearnrate::T1 = 1e-6      # minimal learning rate threshold (once reached, training ends)
    reducedlearnrate::T1 = 10.0              # the amount that learning rate is reduced
    showloss::Bool = false         # whether show current losses while training the model
    function ModelTrainingSetting(inilearnrate, trainloss, dataloss, ratioₜ, seed, maxepoch, ncount, minlearnrate, reducedlearnrate, showloss)
        ratioₜ < 1.0 || throw(ArgumentError("Training data split ratio can not exceed 1"))
        ratioₜ > 0.0 || throw(ArgumentError("Training data split ratio should be larger than 0"))
        new{typeof(inilearnrate), typeof(trainloss), typeof(dataloss)}(inilearnrate, trainloss, dataloss, ratioₜ, seed, maxepoch, ncount, minlearnrate, reducedlearnrate, showloss)
    end
end

ModelTrainingSetting(inilearnrate, trainloss, dataloss; kwargs...) = ModelTrainingSetting(; inilearnrate = inilearnrate, trainloss = trainloss, dataloss = dataloss, kwargs...)

#Train RBNN model
function ModelFit!(r)
    rₜ, pₜ, rᵥ, pᵥ = SplitData(r.env.locations, r.env.measurements, r.trainsettings.ratioₜ, r.trainsettings.seed)
    bestmodel = deepcopy(Flux.params(r))
    count = 0
    opt = ADAM(r.trainsettings.inilearnrate)
    epoch = 0
    bestloss = r.trainsettings.dataloss(rᵥ, pᵥ, r)
    while true
        Flux.train!((x,y) -> r.trainsettings.trainloss(x, y, r), Flux.params(r), [(rₜ, pₜ)], opt)
        tmploss = r.trainsettings.dataloss(rᵥ, pᵥ, r) 
        epoch += 1
        if tmploss < bestloss 
            bestloss = tmploss
            bestmodel = deepcopy(Flux.params(r))
            count = 0
            r.trainsettings.showloss && (@show epoch, r.trainsettings.dataloss(rₜ, pₜ, r), r.trainsettings.dataloss(rᵥ, pᵥ, r))
        else
            count += 1
        end
        epoch > r.trainsettings.maxepoch && break     
        if count > r.trainsettings.ncount
            count = 0
            Flux.loadparams!(r, bestmodel)
            opt.eta /= r.trainsettings.reducedlearnrate
            opt.eta < r.trainsettings.minlearnrate && break 
            r.trainsettings.showloss && println("********* reduced learning rate: ",opt.eta, " *********" )     
        end
    end
    r
end 


function UnderwaterAcoustics.transfercoef(model::DataDrivenPropagationModel, tx, rx::AcousticReceiver; mode=:coherent) where {T1}
    mode === :coherent || throw(ArgumentError("Unsupported mode :" * string(mode)))
    model.calculatefield(model, collect(location(rx)))
end

function UnderwaterAcoustics.transfercoef(model::DataDrivenPropagationModel, tx, rx::AcousticReceiverGrid2D; mode=:coherent) where {T1}
    mode === :coherent || throw(ArgumentError("Unsupported mode :" * string(mode)))
    (xlen, ylen) = size(rx)
    x = vec(location.(rx))
    reshape(model.calculatefield(model, hcat(first.(x), last.(x))'), xlen, ylen)
end

function UnderwaterAcoustics.transfercoef(model::DataDrivenPropagationModel, tx, rx::AcousticReceiverGrid3D; mode=:coherent) where {T1}
    mode === :coherent || throw(ArgumentError("Unsupported mode :" * string(mode)))
    (xlen, ylen, zlen) = size(rx)
    x = vec(location.(rx))
    reshape(model.calculatefield(model, hcat(first.(x), getfield.(x, 2), last.(x))'), xlen, ylen, zlen)
end

UnderwaterAcoustics.transfercoef(model::DataDrivenPropagationModel, rx::AcousticReceiver) =  UnderwaterAcoustics.transfercoef(model, [ ], rx; mode=:coherent)
UnderwaterAcoustics.transfercoef(model::DataDrivenPropagationModel, rx::AcousticReceiverGrid2D) =  UnderwaterAcoustics.transfercoef(model, [ ], rx; mode=:coherent)
UnderwaterAcoustics.transfercoef(model::DataDrivenPropagationModel, rx::AcousticReceiverGrid3D) =  UnderwaterAcoustics.transfercoef(model, [ ], rx; mode=:coherent)
UnderwaterAcoustics.transfercoef(model::DataDrivenPropagationModel, rx::Union{AbstractVector, AbstractMatrix}) = model.calculatefield(model, rx)

UnderwaterAcoustics.transmissionloss(model::DataDrivenPropagationModel, rx::Union{AbstractVector, AbstractMatrix,AcousticReceiver, AcousticReceiverGrid2D, AcousticReceiverGrid3D}) = -amp2db.(abs.(transfercoef(model, rx)))




UnderwaterAcoustics.rays(model::DataDrivenPropagationModel, tx, rx) = throw(ArgumentError("This function is not yet supported."))
  
UnderwaterAcoustics.eigenrays(model::DataDrivenPropagationModel, tx, rx) = throw(ArgumentError("This function is not yet supported."))

function UnderwaterAcoustics.arrivals(model::DataDrivenPropagationModel, tx, rx::Union{AbstractVector, AcousticReceiver}; threshold = 30) 
    arrival = model.calculatefield(model, collect(location(rx)); showarrivals = true)
    amp = amp2db.(abs.(arrival))
    idx = findall(amp .> (maximum(amp) - threshold))
    signficantarrival = arrival[idx]
    idx = sortperm(abs.(signficantarrival), rev = true)
    DataFrame(Amplitude = amp2db.(abs.(signficantarrival[idx])), Phase = angle.(signficantarrival[idx]))
end

UnderwaterAcoustics.arrivals(model::DataDrivenPropagationModel, rx::Union{AbstractVector, AcousticReceiver}) = UnderwaterAcoustics.arrivals(model, [], rx)

@recipe function plot(env::DataDrivenUnderwaterEnvironment; receivers = [], transmissionloss = [],  dynamicrange = 42.0)
    size(transmissionloss) == size(receivers) || throw(ArgumentError("Mismatched receivers and transmissionloss"))
    receivers isa AcousticReceiverGrid2D || throw(ArgumentError("Receivers must be an instance of AcousticReceiverGrid2D"))
    minloss = minimum(transmissionloss)
    clims --> (-minloss-dynamicrange, -minloss)
    colorbar --> true
    cguide --> "dB"
    ticks --> :native
    legend --> false
    xguide --> "x (m)"
    yguide --> "z (m) "
    @series begin
        seriestype := :heatmap
        receivers.xrange, receivers.zrange, -transmissionloss'
    end

end


function UnderwaterAcoustics.check(::Type{RayBasis2D}, env::Union{<:DataDrivenUnderwaterEnvironment,Missing})
    if env !== missing
        size(env.locations)[1] == 2|| throw(ArgumentError("RayBasis2D only supports 2D environment"))
    end
    env
end

function UnderwaterAcoustics.check(::Type{RayBasis2DCurv}, env::Union{<:DataDrivenUnderwaterEnvironment,Missing})
    if env !== missing
        size(env.locations)[1] == 2|| throw(ArgumentError("RayBasis2D only supports 2D environment"))
    end
    env
end

function UnderwaterAcoustics.check(::Type{RayBasis3D}, env::Union{<:DataDrivenUnderwaterEnvironment,Missing})
    if env !== missing
        (length(env.tx) == 3 &&  size(env.locations)[1] == 3)|| throw(ArgumentError("RayBasis3D only supports 3D environment"))
    end
    env
end

function UnderwaterAcoustics.check(::Type{RayBasis3DRCNN}, env::Union{<:DataDrivenUnderwaterEnvironment,Missing})
    if env !== missing
      sum(ismissing.(env.tx)) == 0 || throw(ArgumentError("RayBasis3DRCNN only supports environments with known source location"))
      (length(env.tx) == 3 &&  size(env.locations)[1] == 3)|| throw(ArgumentError("RayBasis3DRCNN only supports 3D environment"))
      env.waterdepth !== missing || throw(ArgumentError("RayBasis3DRCNN only supports environments with known water depth"))
    end
    env
end




  