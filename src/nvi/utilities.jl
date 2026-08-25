"""
    create_internal_stage_vector(DT, D, S) -> Vector{Vector{DT}}

`S` zero vectors of length `D`, one per internal stage of a `D`-dimensional problem. Used by the
integrator caches to hold the stage values `Q`, `P`, `V` and `F`.
"""
create_internal_stage_vector(DT, D, S) = [zeros(DT, D) for _ in 1:S]

function simpson_quadrature(N::Int, ::Type{T}=Float64) where {T}
    if N % 2 != 0
        error("N must be even for Simpson's rule.")
    end

    # Step size
    h = one(T) / N

    # Generate weights
    w = zeros(T, N + 1)
    for i in 1:(N + 1)
        if i == 1 || i == N + 1
            w[i] = h / 3 # First and last weights
        elseif i % 2 == 0
            w[i] = 4 * h / 3 # Even-indexed weights
        else
            w[i] = 2 * h / 3 # Odd-indexed weights
        end
    end
    
    return w
end

"""
    _param_arrays(params) -> Tuple

Every layer parameter array of `params`, `vec`'d, in layer-then-field order. The backing tuple
for [`flatten_params!`](@ref) and [`flatten_params`](@ref); it aliases `params` rather than
copying.

`@generated` because the layer/field structure is in the *type*, so the sequence can be emitted
once at compile time. Walking it at run time is what made the old `flatten_params`
type-unstable — `values(params)` iterates a heterogeneous `NamedTuple`, so
`fieldnames(typeof(layer))` inside the loop cannot be folded.

This is not `NeuralNetworkParameters.flatten!`, which does the same walk. That one is
allocation-free only when it is handed a `ParameterLayout`, and building the layout per call is
what has to be avoided here: the four call sites in `DenseNet`'s `components!` flatten a
*freshly built* gradient set, and there is nowhere on the cache to keep a layout for it today.
A `@generated` walk needs no layout at all. The training loops, which flatten one long-lived
parameter set, do use the upstream pair — see `initial_params!` in `shallownet.jl`.
"""
@generated function _param_arrays(params::NamedTuple{LN,LT}) where {LN,LT}
    entries = Expr[]
    for (i, lname) in enumerate(LN)
        for f in fieldnames(LT.parameters[i])
            push!(entries, :(vec(params.$lname.$f)))
        end
    end
    Expr(:tuple, entries...)
end

_param_arrays(params::NetworkParameters) =
    _param_arrays(NeuralNetworkParameters.params(params))

"""
    flatten_params!(dest, params) -> dest

Copy every layer parameter array of `params`, in layer-then-field order, into the flat vector
`dest`.

Replaces an allocating `flatten_params` that built a `flat_list = []` — a `Vector{Any}` — and
returned `vcat(flat_list...)`, which infers as `Any` when splatted. `DenseNet`'s `components!`
calls this `2 + 2R` times per dimension per residual evaluation, i.e. per Newton iteration and
per Jacobian column, so it is worth having a form that writes into a caller-supplied buffer.
"""
function flatten_params!(dest::AbstractVector, params)
    off = 0
    # A homogeneous tuple (`vec` of a `Matrix` and of a `Vector` are both `Vector{T}`), so this
    # loop is unrolled and concretely typed.
    for v in _param_arrays(params)
        copyto!(dest, off + 1, v, 1, length(v))
        off += length(v)
    end
    return dest
end

"""
    flatten_params(params) -> Vector

Allocating form of [`flatten_params!`](@ref). Not on any hot path; kept because it reads better
in one-off code and in the benchmarks. `reduce(vcat, ...)` over the concretely typed tuple keeps
the element type generic, where the old splatted `vcat` over a `Vector{Any}` inferred as `Any`.
"""
flatten_params(params) = reduce(vcat, _param_arrays(params))


"""
    box_init_plain(input_dim, output_dim, ::Type{T}; rng = Random.default_rng())

Draw a "box" initialisation of a `output_dim × input_dim` weight matrix and its bias at
element type `T`.

`T` is mandatory. It used to default to `Float32`, and every call site omitted it, so a
`Float64` network was initialised at single precision and then converted on assignment —
invisible to a test that checks the `eltype` of the result. The package's central invariant
is that a run started at `T` stays at `T`, so the initialiser has to be told which `T`.

`rng` is drawn from, not re-seeded. It used to be a keyword defaulting to the *expression*
`Random.seed!(1)`, which is evaluated afresh on every call that omits it: each call silently
reseeded Julia's global RNG and then drew from it, so consecutive calls returned correlated
draws and any seeding the caller had done was discarded. Seed at the call site instead.
"""
function box_init_plain(input_dim::Int, output_dim::Int, ::Type{T};
                        rng::Random.AbstractRNG = Random.default_rng()) where {T}
    W = zeros(T, output_dim, input_dim)
    b = zeros(T, output_dim)

    for i in 1:output_dim
        p = rand(rng, T, input_dim)
        n = randn(rng, T, input_dim)
        n ./= norm(n)
        p_max = map((n_i) -> n_i ≥ 0 ? one(T) : zero(T), n)
        k = 1 / dot((p_max .- p), n)
        W[i, :] = k * n
        b[i] = k * dot(p, n)
    end
    return W, b
end

function lsgd_loss(network_inputs,labels,NN,ps)
    NN_output = NN(network_inputs, ps)
    return sqrt(mean((labels .- NN_output).^2))
end

"""
    mae_loss(x, y, NN, ps; λ = 0)

Mean **absolute** error of `NN(x, ps)` against target `y`, with optional boundary penalty
`λ * |NN(x[1], ps) - y[1]|²`. Used as the training objective for `TrainingMethod`.

Named `mse_loss` until the audit: the name said squared error, the docstring said absolute
error, and the body computed absolute error. The body is authoritative — renaming leaves the
numerics of every `TrainingMethod` seed exactly as they were, where switching to a squared
error would have changed them silently. A dead `μ = 0.00001` keyword was also dropped.

`λ` defaults to an untyped `0`, not `0.0`: a `Float64` literal here promoted the whole loss
to `Float64` for a `Float32` or `Float16` network.
"""
function mae_loss(x, y::AbstractArray{T}, NN, ps; λ = 0) where {T}
    y_pred = NN(x, ps)
    loss = mean(abs, y_pred - y) + λ * abs2(y_pred[1] - y[1])
    return loss
end

# Least-specific fallback: `NetworkIntegratorCore.jl` covers the three supported
# extrapolations for every NetworkIntegratorMethod, and several integrators override those.
# Anything else lands here and gets a readable message rather than a MethodError.
function initial_trajectory!(sol, history, params, ::GeometricIntegrator, initial_trajectory::Extrapolation)
    error("For extrapolation $(initial_trajectory) method is not implemented!")
end

