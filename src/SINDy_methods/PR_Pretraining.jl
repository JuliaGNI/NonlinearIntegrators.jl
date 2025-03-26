# Define polynomial layer (P-layer)
struct PolynomialLayer
    degree::Int
end

function (layer::PolynomialLayer)(x::Vector{Float64})
    #TODO: Generalize to multiple dimensions
    return hcat([x .^ d for d in 0:layer.degree]...)'
end

# Define radial layer (R-layer)
struct RadialLayer
    functions::Vector{Function}
end

function (layer::RadialLayer)(x::Matrix{Float64})
    @assert size(x, 1) == length(layer.functions)
    transformed_features = [layer.functions[i].(x[i, :]) for i in eachindex(layer.functions)]
    return hcat(transformed_features...)'
end


# Define PR model architecture                                                                                                     
struct PRModel
    P::PolynomialLayer
    P_linear::Dense
    R::RadialLayer
    R_linear::Dense
end

function PRModel(degree::Int, functions::Vector{Function}, output_dim::Int=1)
    P_layer = PolynomialLayer(degree)
    Plinear_layer = f64(Dense(degree + 1, length(functions), identity, bias=false))
    R_layer = RadialLayer(functions)
    Rlinear_layer = f64(Dense(length(functions), output_dim, identity, bias=false))
    return PRModel(P_layer, Plinear_layer, R_layer, Rlinear_layer)
end

# Forward pass for the PR model
function (model::PRModel)(x::Vector{Float64})
    mono_features = model.P(x)
    poly_features = model.P_linear(mono_features)
    radial_features = model.R(poly_features)
    return model.R_linear(radial_features)
end

# Dictionary of radial basis functions
radial_functions = [
    # x -> x,
    # x -> x^2,
    # atan,
    sin,
    cos,
    identity,
    # exp,
    # x -> log(abs(x) + 1e-5),
    # x -> 1 / (1 + x^2)
]

Flux.trainable(a::PRModel) = (;a.P_linear,a.R_linear)




# model([0.0,0.3,0.2])
# Loss function
function loss_fn(model, x, y;lambda=1e-4)
    y_pred = model(x)
    params_sum = 0.0
    for i in eachindex(Flux.trainable(model))
        params_sum+=sum(abs.(Flux.trainable(model)[i].weight))
    end
    return 0.5 * sqrt(sum((y_pred[1,:] - y) .^2)) + lambda * params_sum
end

# Training settings
λ_lasso = 1e-4
batch_size = 100
max_epochs = 5000
prune_threshold = 0.05
patience = 50
loss_threshold = 1e-2

λ_fct(epoch::Int) = λ_lasso * (1 + 0.5 * sin(epoch/10))

# Seed for reproducibility
Random.seed!(1234)


#Define hyperparameters to generate data
t_step = 0.01
SINDy_h = 1.0
T = 10.0
N_SindyStep = T / SINDy_h
DP_lode = GeometricProblems.DoublePendulum.lodeproblem(tstep=t_step, tspan=(0, T))
DP_pref = integrate(DP_lode, Gauss(8))

trajectory_vector = collect(DP_pref.q[:, 1])[2:end]

# Parameters
segment_length = Int(SINDy_h / t_step)
overlap = 50
step = segment_length - overlap

# Generate trajectory segments
trajectory_segments = []
for start in 1:overlap:(length(trajectory_vector)-segment_length+1)
    push!(trajectory_segments, trajectory_vector[start:start+segment_length-1])
end

for (i, sub_vector) in enumerate(trajectory_segments)
    println("Segment $i: $(sub_vector[1]-1) to $(sub_vector[end]-1), size = $(length(sub_vector))")
end

# X = reshape(collect(t_step:t_step:Int(SINDy_h)), 1, :)
# Y = reshape(trajectory_segments[1], 1, :)
# problem = DirectDataDrivenProblem(X, Y)

x_data = collect(t_step:t_step:SINDy_h)
y_data = trajectory_segments[1]

# # # Generate dataset: f(x) = cos(x^2) over [0, 3]
# N = 10^4
# x_data = rand(N) * 3  # Random data points in [0, 3]
# f_label(x) = 0.5 .* cos.(0.5 .* x .+ 0.6)
# y_data = f_label(x_data)   # Target values

# Data batching
function get_batches(x_data, y_data, batch_size)
    idx = shuffle(1:length(x_data))
    x_batches = [x_data[idx[i:i+batch_size-1]] for i in 1:batch_size:length(x_data) if i + batch_size - 1 <= length(x_data)]
    y_batches = [y_data[idx[i:i+batch_size-1]] for i in 1:batch_size:length(y_data) if i + batch_size - 1 <= length(y_data)]
    return x_batches, y_batches
end


# Instantiate the PR model
model = PRModel(3, radial_functions)
opt_state = Flux.setup(Flux.Adam(1e-3), model)

# Training loop
best_loss = Inf
no_improvement_epochs = 0

for epoch in 1:max_epochs
    x_batches, y_batches = get_batches(x_data, y_data, batch_size)
    epoch_loss = 0.0
    current_lambda = λ_fct(epoch)
    for (x_batch, y_batch) in zip(x_batches, y_batches)
        grad = Flux.gradient(m -> loss_fn(m, x_batch, y_batch,lambda = current_lambda), model)
        Flux.update!(opt_state, model, grad[1])
        epoch_loss += loss_fn(model, x_batch, y_batch,lambda = current_lambda)
    end

    # Lasso regularization and pruning
    for param in Flux.trainable(model)
        params = param.weight[:]
        params .-= λ_lasso * sign.(params)
        params[abs.(params) .< prune_threshold] .= 0.0
        param.weight[:] = params
    end

    epoch % 50 == 0 ? println("Epoch:", epoch, "Loss = ",epoch_loss) : nothing

    # Early stopping condition
    if abs(epoch_loss - best_loss) / best_loss < loss_threshold
        no_improvement_epochs += 1
    else
        no_improvement_epochs = 0
        best_loss = epoch_loss
    end

    if no_improvement_epochs >= patience
        println("Early stopping after $epoch epochs.")
        break
    end
end

for i in eachindex(Flux.trainable(model))
    println("layer $i: ", Flux.trainable(model)[i].weight)
end

println("Training complete.")
# plot(model(collect(0:0.01:3))[1,:])
# plot!(f_label(collect(0:0.01:3)))
plot(model(collect(0:0.01:1))[1,:])
plot(y_data)
