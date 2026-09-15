# ---- Integrator adapters ----------------------------------------------------
#
# The greedy algorithm itself lives in `greedy.jl` and knows nothing about integrators.
# What remains here is the per-integrator glue: assemble the fit target from the cache,
# say how atoms map to neurons, and scatter the result into the nonlinear solution vector
# `x` in that integrator's layout. Everything else — dictionary, selection, fit, guard
# rails, precision discipline — is shared.
#
# The four integrators differ in exactly two ways:
#
#   * the *ansatz*. `ShallowNetAutodiff` and `ShallowNetAutodiffReversible` represent the step as
#     `q(t) = (1-t) q̄ + t q̃ + t(1-t) u(t)`, so the network only has to fit what is left
#     after the linear part is subtracted, and every dictionary atom carries the `t(1-t)`
#     factor. The two symbolic-derivative integrators (`ShallowNet` and `ShallowNetReversible`) fit the labels directly.
#   * the *symmetry*. The two time-reversible integrators add neurons in mirrored pairs,
#     sharing one output weight (`ShallowNetAutodiffReversible`) or keeping two
#     (`ShallowNetReversible`), and store only the independent half of the hidden
#     parameters in `x`.
#
# This file is included *after* the integrator definitions, since the methods below
# dispatch on them.

"""
    oga_seed(int, oga, symmetry, targets, modulation) -> Vector{OGAResult}

Run the greedy fit once per solution component, sharing the dictionary configuration and
quadrature taken from the integrator's method.

`targets[d]` is the fit target for component `d` at the network's input nodes, and
`modulation` is the optional per-node ansatz factor (see [`oga_fit`](@ref)).
"""
function oga_seed(int::GeometricIntegrator, oga::OGA, symmetry::OGASymmetry,
        targets::AbstractVector, modulation)
    local S = nbasis(method(int))
    local D = length(cache(int).q̃)
    local T = eltype(nlsolution(int))
    local nodes = vec(T.(method(int).network_inputs))
    local quad_weights = simpson_quadrature(extrapolation_substep(method(int)), T)
    local σ = method(int).basis.activation
    local bias_interval = method(int).bias_interval
    local dict_amount = method(int).dict_amount

    return [oga_fit(oga, σ, nodes, quad_weights, targets[d], S;
                bias_interval = bias_interval, dict_amount = dict_amount,
                modulation = modulation, symmetry = symmetry) for d in 1:D]
end

# Keep the parameter cache consistent with the seed. `components!` and `record_finer_solution!`
# both repopulate `ps` from `x`, so this is not load-bearing — but a stale `ps` would be a
# trap for anything that inspects the cache between the seed and the first residual
# evaluation.
function _store_params!(ps, results)
    for d in eachindex(results)
        # Linear indexing, not plain broadcast: the hidden layer's `W` is `S×1` and the
        # output layer's is `1×S`, so a length-`S` vector broadcasts into the first but
        # not the second.
        ps[d][1].W[:] .= results[d].W
        ps[d][1].b[:] .= results[d].b
        ps[d][2].W[:] .= results[d].c
    end
    return nothing
end

# Layout used by `ShallowNet` and `ShallowNetAutodiff`: all `S` hidden neurons are
# independent, so `x` carries `W` and `b` for every one of them.
function _store_full!(x, results, D::Int, S::Int)
    for d in 1:D
        r = results[d]
        for i in 1:S
            x[D * (i - 1) + d] = r.c[i]
            x[D * (S + 1) + D * (i - 1) + d] = r.W[i]
            x[D * (S + 1 + S) + D * (i - 1) + d] = r.b[i]
        end
    end
    return nothing
end

# Layout used by the two time-reversible integrators: neurons come in mirrored pairs, so
# only the odd (independent) half of `W`/`b` is stored and `components!` reconstructs the
# rest. Note the stride of the bias block is `S/2`, not `S`.
function _store_symmetric!(x, results, D::Int, S::Int)
    half = S ÷ 2
    for d in 1:D
        r = results[d]
        for i in 1:S
            x[D * (i - 1) + d] = r.c[i]
        end
        for i in 1:half
            x[D * (S + 1) + D * (i - 1) + d] = r.W[2i - 1]
            x[D * (S + 1 + half) + D * (i - 1) + d] = r.b[2i - 1]
        end
    end
    return nothing
end

# The `t(1-t)` factor of the boundary ansatz, and the target it leaves for the network:
# the labels minus the straight line between the step's endpoints.
_ansatz_modulation(nodes::AbstractVector{T}) where {T} = nodes .* (one(T) .- nodes)

function _ansatz_target(labels, nodes::AbstractVector{T}, q_begin, q_end) where {T}
    return labels .- ((one(T) .- nodes) .* q_begin .+ nodes .* q_end)
end

function _oga_status(results, show_status)
    show_status && for (d, r) in enumerate(results)
        println("OGA dimension $d: $(r.neurons) neurons, weighted residual $(r.residual)" *
                (r.rejected > 0 ?
                 ", $(r.rejected) atom(s) rejected for adding no new direction" : ""))
    end
end

# ---- ShallowNet -------------------------------------------------

function initial_params!(int::GeometricIntegrator{<:ShallowNet}, oga::OGA, sol)
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local x = nlsolution(int)
    local T = eltype(x)
    local labels = cache(int).network_labels'

    targets = [T.(labels[d, :]) for d in 1:D]
    results = oga_seed(int, oga, NoSymmetry(), targets, nothing)

    _store_params!(cache(int).ps, results)
    _store_full!(x, results, D, S)
    # `x[D*S+d]` is the momentum, set by `initial_trajectory!` — deliberately untouched.
    @debug "Initial guess for DOF from OGA " x
    _oga_status(results, method(int).show_status)
    return nothing
end

# ---- ShallowNetAutodiff -----------------------------------------------------------

function initial_params!(int::GeometricIntegrator{<:ShallowNetAutodiff}, oga::OGA, sol)
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local x = nlsolution(int)
    local T = eltype(x)
    local nodes = vec(T.(method(int).network_inputs))
    local labels = cache(int).network_labels'

    # The endpoint estimate comes from the last label, i.e. from the initial-trajectory
    # integrator, rather than from `cache(int).q̃`.
    targets = [_ansatz_target(T.(labels[d, :]), nodes, T(sol.q[d]), T(labels[d, end]))
               for d in 1:D]
    results = oga_seed(int, oga, NoSymmetry(), targets, _ansatz_modulation(nodes))

    _store_params!(cache(int).ps, results)
    _store_full!(x, results, D, S)
    for d in 1:D
        x[D * S + d] = cache(int).q̃[d]      # the ansatz's endpoint unknown
    end
    _oga_status(results, method(int).show_status)
    return nothing
end

# ---- ShallowNetReversible ----------------------------------------------

function initial_params!(int::GeometricIntegrator{<:ShallowNetReversible}, oga::OGA, sol)
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local x = nlsolution(int)
    local T = eltype(x)
    local labels = cache(int).network_labels'

    targets = [T.(labels[d, :]) for d in 1:D]
    results = oga_seed(int, oga, MirrorPairs(), targets, nothing)

    _store_params!(cache(int).ps, results)
    _store_symmetric!(x, results, D, S)
    # `x[D*S+d]` is the momentum, set by `initial_trajectory!`.
    _oga_status(results, method(int).show_status)
    return nothing
end

# ---- ShallowNetAutodiffReversible ----------------------------------------------

function initial_params!(int::GeometricIntegrator{<:ShallowNetAutodiffReversible}, oga::OGA, sol)
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local x = nlsolution(int)
    local T = eltype(x)
    local nodes = vec(T.(method(int).network_inputs))
    local labels = cache(int).network_labels'
    local q̃ = cache(int).q̃

    # Unlike `ShallowNetAutodiff`, the endpoint of the linear part is the cache's endpoint
    # estimate rather than the last label.
    targets = [_ansatz_target(T.(labels[d, :]), nodes, T(sol.q[d]), T(q̃[d])) for d in 1:D]
    results = oga_seed(int, oga, SharedMirrorPairs(), targets, _ansatz_modulation(nodes))

    _store_params!(cache(int).ps, results)
    _store_symmetric!(x, results, D, S)
    for d in 1:D
        x[D * S + d] = q̃[d]
    end
    _oga_status(results, method(int).show_status)
    return nothing
end
