export modelfield, rmseloss

#Split input data into training and validation dataset
function SplitData(location, measurement, ratioₜ, seed)
    dsize = size(location)[2]
    seed && Random.seed!(1)
    idxsequence = randperm(dsize)
    rₜ = location[:,idxsequence[1 : Int(round(dsize * ratioₜ))]]
    pₜ = measurement[:,idxsequence[1 : Int(round(dsize * ratioₜ))]]
    rᵥ = location[:,idxsequence[Int(round(dsize * ratioₜ)) + 1 : end]]
    pᵥ = measurement[:,idxsequence[Int(round(dsize * ratioₜ)) + 1: end]]
    return rₜ, pₜ, rᵥ, pᵥ
end



#Generate sythetic field from propagation models (PekerisRayModel, Bellhop, Kraken, RayTracer)
function modelfield(rx, tx, f, pm; error = 0.0f0)
    TL = Array{Float32}(undef, 1, size(rx)[2])
    Random.seed!(1)
    rand_err = rand(Float32, size(rx))
    rx = rx .+ error .* rand_err
    if length(tx) == 2
        for i in 1 : 1 : size(rx)[2]
            TL[1, i] = Float32(transmissionloss(pm, AcousticSource(tx[1], tx[2], f), AcousticReceiver(rx[1,i], rx[2,i]); mode=:coherent))
        end
    else
        for i in 1 : 1 : size(rx)[2]
            TL[1, i] = Float32(transmissionloss(pm, AcousticSource(tx[1], tx[2], tx[3], f), AcousticReceiver(rx[1,i], rx[2,i], rx[3,i]); mode=:coherent))
        end
    end
    TL
end


#Calculate image source locations given channel geometry
function find_image_src(rx, tx, L, n_rays)
    IsTwoD = false
    (length(tx) == 3) && (IsTwoD == true)
    a = 20.0f0               
    count = 0
    s = 2 * (Int(a) * 2 + 1)                                   # no of image sources for a given "a"
    IsTwoD ? (all_image_src = zeros(Float32, 2, s)) : (all_image_src = zeros(Float32, 3, s)) # store all image sources
    image_src_amp = zeros(Float32, s)                                   # amplitude of arrival rays
    ref = zeros(Float32, 2, s)                                          # number of reflection on surface and bottom.
    for w = 0.0f0 : 1.0f0         # for all 8 possible combinations of +- (eqn(11)), they can only take values 0 or 1
        for n = -a : a                           # "a" can be any positive number
            count += 1
            IsTwoD ? (image_src = [tx[1], (1.0f0 - 2.0f0 * w) * tx[2] + 2.0f0 * n * L]) : (image_src = [tx[1], tx[2], (1.0f0 - 2.0f0 * w) * tx[3] + 2.0f0 * n * L])
            d = norm(image_src .- rx)
            image_src_amp[count]= 0.2f0^(abs(n)) * (0.99f0)^ (abs(n - w)) / d 
            all_image_src[:, count] = image_src
            ref[:, count] = [abs(n-w), abs(n)]
        end
    end
    idx = sortperm(abs.(image_src_amp), rev = true)[1 : n_rays] # sort based on received amplitude to select n_rays image sources
    return all_image_src[:, idx]
end


function cartesian2spherical(pos)
    x = @view pos[1,:]
    y = @view pos[2,:]
    z = @view pos[3,:]
    ρ = norm.(eachcol(pos))
    θ = atan.(y, x)
    ψ = atan.(norm.(eachcol(pos[1:2,:])), z)
    return ρ, θ, ψ
end


rmseloss(rx, tl, model) = Flux.mse(transmissionloss(model, rx), tl)^0.5f0
