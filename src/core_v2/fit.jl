# ============================================================================
# Training API (fit! functions)
# ============================================================================
#
# Unified training interface using Julia's multiple dispatch.
# Call fit!(env, rx, data, model; ...) and the correct method is selected
# based on the model type.
# ============================================================================

# ============================================================================
# Internal Training Functions
# ============================================================================

"""
    train_rbnn!(model, rx_train, rx_val, TL_train, TL_val; kwargs...)

Internal training loop for Case 1 models.
Uses LegacyADAM optimizer and RMSE loss.
"""
function train_rbnn!(model, rx_train, rx_val, TL_train, TL_val;
              initial_lr = Float32(0.5),
              threshold_count = 5000,
              threshold_lr = Float32(1e-6),
              show = false)
    data_loss(x, y) = (Flux.Losses.mse(model(x), y))^0.5f0
    loss_func = data_loss

    best_model = [deepcopy(p) for p in Flux.params(model)]
    best_loss = data_loss(rx_val, TL_val)
    count = 0
    opt = LegacyADAM(initial_lr)

    println("=== Training: INITIAL ===")
    println("Train RMSE: ", data_loss(rx_train, TL_train))
    println("Val RMSE: ", data_loss(rx_val, TL_val))

    A_prev = copy(model.A)

    for epoch in 1:10_000_000_000
        Flux.train!(loss_func, Flux.params(model), [(rx_train, TL_train)], opt)

        if epoch % 1000 == 0 && show
            println("--- Epoch $epoch: Sum A = $(sum(model.A))")
        end

        tmploss = data_loss(rx_val, TL_val)
        if best_loss > tmploss
            best_loss = tmploss
            best_model = [deepcopy(p) for p in Flux.params(model)]
            count = 0
            if show
                @show epoch, data_loss(rx_train, TL_train), data_loss(rx_val, TL_val)
            end
        else
            count += 1
        end

        if count > threshold_count
            count = 0
            for (p, b) in zip(Flux.params(model), best_model)
                p .= b
            end
            opt.eta /= 10.0
            A_prev = copy(model.A)
            opt.eta < threshold_lr && break
        end
    end

    println("=== Training: FINAL ===")
    println("Val RMSE: ", best_loss)

    return model
end

# ============================================================================
# fit! for Case 1 (RayBasis2DCurv)
# ============================================================================

"""
    fit!(env, tx, rx, data; model, kwargs...)

Fit a 2D far-field ray basis model (Case 1).

# Arguments
- `env`: Environment with soundspeed, frequency
- `tx`: Transmitter position (can be nothing)
- `rx`: Receiver positions (2 x N matrix)
- `data`: Measured transmission loss data

# Keyword Arguments
- `model`: Pre-initialized model (required)
- `nrays`: Number of rays (default: 60)
- `initial_lr`: Initial learning rate (default: 0.5)
- `threshold_count`: Epochs before LR decay (default: 5000)
- `threshold_lr`: Minimum LR before stopping (default: 1e-6)
- `show`: Print progress (default: false)

# Returns
- Trained model
"""
function fit!(env, tx, rx, data;
              model = nothing,
              nrays = 60,
              initial_lr = Float32(0.5),
              threshold_count = 5000,
              threshold_lr = Float32(1e-6),
              show = false)
    rx_train, rx_val = data_split(rx)

    TL_train = data[1:size(rx_train, 2)]'
    TL_val = data[1+size(rx_train, 2):end]'

    if model === nothing
        k = Float32(2.0) * π * env.frequency / env.soundspeed
        model = RayBasis2DCurv(nrays, k)
    end

    return train_rbnn!(model, rx_train, rx_val, TL_train, TL_val;
        initial_lr = initial_lr,
        threshold_count = threshold_count,
        threshold_lr = threshold_lr,
        show = show
    )
end

# ============================================================================
# fit! for RayBasis3d (Case 2)
# ============================================================================

"""
    fit!(env, rx, TL_data, rbnn::RayBasis3d; kwargs...)

Fit a 3D near-field ray basis model (Case 2).
Uses Float64 precision and L1 regularization on amplitudes.

# Arguments
- `env`: Environment (for API consistency)
- `rx`: Receiver positions (3 x N matrix)
- `TL_data`: Transmission loss data
- `rbnn`: RayBasis3d model

# Keyword Arguments
- `nominal_ρ`, `nominal_θ`, `nominal_ψ`: Nominal ray parameters (required)
- `xₒ`: Reference origin (default: [0,0,0])
- `l1_reg`: L1 regularization on amplitudes (default: 1.0)
- `initial_lr`: Initial learning rate (default: 0.5)
- `threshold_count`: Epochs before LR decay (default: 5000)
- `threshold_lr`: Minimum LR before stopping (default: 1e-6)
- `show`: Print progress (default: false)

# Returns
- Trained RayBasis3d model
"""
function fit!(env, rx, TL_data, rbnn::RayBasis3d;
              nominal_ρ, nominal_θ, nominal_ψ,
              xₒ = Float64[0.0, 0.0, 0.0],
              l1_reg = 1.0,
              initial_lr = 0.5,
              threshold_count = 5000,
              threshold_lr = 1e-6,
              show = false)

    rx_train, rx_val = data_split(rx)
    TL_train = TL_data[1:size(rx_train, 2)]'
    TL_val = TL_data[1+size(rx_train, 2):end]'

    data_loss(x, y) = sqrt(Flux.Losses.mse(rbnn(x, env; xₒ=xₒ, nominal_ρ=nominal_ρ, nominal_θ=nominal_θ, nominal_ψ=nominal_ψ), y))
    full_loss(x, y) = Flux.Losses.mse(rbnn(x, env; xₒ=xₒ, nominal_ρ=nominal_ρ, nominal_θ=nominal_θ, nominal_ψ=nominal_ψ), y) + l1_reg * mean(abs, rbnn.A)

    best_model = [deepcopy(p) for p in Flux.params(rbnn)]
    best_loss = data_loss(rx_val, TL_val)
    count = 0
    opt = Flux.Adam(initial_lr)

    println("=== fit! RayBasis3d: INITIAL ===")
    println("Train RMSE: ", data_loss(rx_train, TL_train))
    println("Val RMSE: ", data_loss(rx_val, TL_val))

    A_prev = copy(rbnn.A)

    for epoch in 1:10_000_000_000
        Flux.train!(full_loss, Flux.params(rbnn), [(rx_train, TL_train)], opt)

        if epoch % 1000 == 0 && show
            println("--- Epoch $epoch: Sum A = $(sum(rbnn.A))")
        end

        tmploss = data_loss(rx_val, TL_val)
        if best_loss > tmploss
            best_loss = tmploss
            best_model = [deepcopy(p) for p in Flux.params(rbnn)]
            count = 0
            if show
                @show epoch, data_loss(rx_train, TL_train), data_loss(rx_val, TL_val)
            end
        else
            count += 1
        end

        if count > threshold_count
            count = 0
            for (p, b) in zip(Flux.params(rbnn), best_model)
                p .= b
            end
            opt.eta /= 10.0
            A_prev = copy(rbnn.A)
            opt.eta < threshold_lr && break
        end
    end

    for (p, b) in zip(Flux.params(rbnn), best_model)
        p .= b
    end

    println("=== fit! RayBasis3d: FINAL ===")
    println("Train RMSE: ", data_loss(rx_train, TL_train))
    println("Val RMSE: ", data_loss(rx_val, TL_val))

    return rbnn
end

# ============================================================================
# fit! for RayBasisRCNN (Case 3)
# ============================================================================

"""
    fit!(env, rx, TL_data, rbnn::RayBasisRCNN; kwargs...)

Fit an RCNN model for geo-acoustic inversion (Case 3).

# Arguments
- `env`: Environment with soundspeed, frequency, waterdepth, tx
- `rx`: Receiver positions (3 x N matrix)
- `TL_data`: Transmission loss data
- `rbnn`: RayBasisRCNN model (contains the trainable RCNN)

# Keyword Arguments
- `nominal_ρ`, `nominal_θ`, `nominal_ψ`: Nominal ray parameters (required)
- `n_rays`: Number of rays (required)
- `xₒ`: Reference origin (default: [0,0,0])
- `initial_lr`: Initial learning rate (default: 0.05)
- `threshold_count`: Epochs before LR decay (default: 500)
- `threshold_lr`: Minimum LR before stopping (default: 1e-6)
- `max_epochs`: Maximum epochs (default: 10_000_000)
- `target_rmse`: Early stopping target (default: nothing)
- `show`: Print progress (default: false)

# Returns
- Trained RayBasisRCNN model
"""
function fit!(env, rx, TL_data, rbnn::RayBasisRCNN;
              nominal_ρ, nominal_θ, nominal_ψ, n_rays,
              xₒ = Float32[0.0, 0.0, 0.0],
              initial_lr = 0.05f0,
              threshold_count = 500,
              threshold_lr = 1e-6,
              max_epochs = 10_000_000,
              target_rmse = nothing,
              show = false)

    rx_train, rx_val = data_split(rx)
    TL_train = TL_data[1:size(rx_train)[2]]'
    TL_val = TL_data[1+size(rx_train)[2]:end]'

    data_loss(x, y) = sqrt(Flux.Losses.mse(rbnn(x, env; xₒ=xₒ, nominal_ρ=nominal_ρ, nominal_θ=nominal_θ, nominal_ψ=nominal_ψ, n_rays=n_rays), y))

    best_model = deepcopy(rbnn.rcnn)
    best_loss = data_loss(rx_val, TL_val)
    count = 0
    opt = Flux.Adam(initial_lr)

    println("=== fit! RayBasisRCNN: INITIAL ===")
    println("Train RMSE: ", data_loss(rx_train, TL_train))
    println("Val RMSE: ", data_loss(rx_val, TL_val))

    for epoch in 1:max_epochs
        Flux.train!(data_loss, Flux.params(rbnn.rcnn), [(rx_train, TL_train)], opt)

        tmploss = data_loss(rx_val, TL_val)
        if best_loss > tmploss
            best_loss = tmploss
            best_model = deepcopy(rbnn.rcnn)
            count = 0
            if show
                @show epoch, data_loss(rx_train, TL_train), data_loss(rx_val, TL_val)
            end
            if target_rmse !== nothing && best_loss < target_rmse
                println("✅ Reached target RMSE $(best_loss) < $(target_rmse) at epoch $epoch")
                break
            end
        else
            count += 1
        end

        if count > threshold_count
            count = 0
            Flux.loadmodel!(rbnn.rcnn, best_model)
            opt = Flux.Adam(opt.eta / 10.0f0)
            opt.eta < threshold_lr && break
        end
    end

    Flux.loadmodel!(rbnn.rcnn, best_model)

    println("=== fit! RayBasisRCNN: FINAL ===")
    println("Train RMSE: ", data_loss(rx_train, TL_train))
    println("Val RMSE: ", best_loss)

    return rbnn
end

# ============================================================================
# fit! for RayBasisRayleigh (Case 4)
# ============================================================================

"""
    fit!(env, rx, TL_data, rbnn::RayBasisRayleigh; kwargs...)

Fit seabed physical parameters using Rayleigh reflection (Case 4).

# Arguments
- `env`: Environment with soundspeed, frequency, waterdepth, tx
- `rx`: Receiver positions (3 x N matrix)
- `TL_data`: Linear amplitude data (NOT in dB)
- `rbnn`: RayBasisRayleigh model

# Keyword Arguments
- `n_rays`: Number of rays (required)
- `initial_lr`: Initial learning rate (default: 0.5)
- `threshold_count`: Epochs before LR decay (default: 5000)
- `threshold_lr`: Minimum LR before stopping (default: 1e-5)
- `show`: Print progress (default: false)

# Returns
- Trained RayBasisRayleigh model
"""
function fit!(env, rx, TL_data, rbnn::RayBasisRayleigh;
              n_rays,
              initial_lr = 0.5f0,
              threshold_count = 5000,
              threshold_lr = 1e-5,
              show = false)

    rx_train, rx_val = data_split(rx)
    TL_train = TL_data[1:size(rx_train)[2]]'
    TL_val = TL_data[1+size(rx_train)[2]:end]'

    data_loss(x, y) = sqrt(Flux.Losses.mse(rbnn(x, env; n_rays=n_rays), y))

    best_model = deepcopy(Flux.params(rbnn))
    best_loss = data_loss(rx_val, TL_val)
    count = 0
    opt = Flux.Adam(initial_lr)

    println("=== fit! RayBasisRayleigh: INITIAL ===")
    println("Train RMSE: ", data_loss(rx_train, TL_train))
    println("Val RMSE: ", data_loss(rx_val, TL_val))
    println("ρr=$(abs(rbnn.ρᵣ[1])), cr=$(abs(rbnn.cᵣ[1])), δ=$(abs(rbnn.δ[1]))")

    for epoch in 1:10_000_000_000
        Flux.train!(data_loss, Flux.params(rbnn), [(rx_train, TL_train)], opt)

        val_loss = data_loss(rx_val, TL_val)
        train_loss = data_loss(rx_train, TL_train)
        tmploss = mean([val_loss, 0.5 * train_loss])

        if tmploss < best_loss
            best_loss = tmploss
            best_model = deepcopy(Flux.params(rbnn))
            count = 0
            if show
                @show epoch, train_loss, val_loss
            end
        else
            count += 1
        end

        if count > threshold_count
            count = 0
            for (p, b) in zip(Flux.params(rbnn), best_model)
                p .= b
            end
            opt = Flux.Adam(opt.eta / 10.0f0)
            if show
                println(">>> LR drop, new LR: $(opt.eta)")
            end
            opt.eta < threshold_lr && break
        end
    end

    for (p, b) in zip(Flux.params(rbnn), best_model)
        p .= b
    end

    println("=== fit! RayBasisRayleigh: FINAL ===")
    println("Train RMSE: ", data_loss(rx_train, TL_train))
    println("Val RMSE: ", data_loss(rx_val, TL_val))
    println("ρr=$(abs(rbnn.ρᵣ[1])), cr=$(abs(rbnn.cᵣ[1])), δ=$(abs(rbnn.δ[1]))")

    return rbnn
end
