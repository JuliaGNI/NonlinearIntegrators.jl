"""
    ShallowNetAutodiff <: ShallowNetMethod

Shallow-net variational integrator that computes network derivatives with `ForwardDiff`
instead of the pre-compiled symbolic derivatives used by `ShallowNet`.
The ansatz and optimisation are otherwise identical to `ShallowNet`.

# Constructor

    ShallowNetAutodiff(basis, quadrature; kwargs...)

Keyword arguments are the same as `ShallowNet`:
`initial_trajectory_method`, `initial_guess_method`, `extrapolation_substep`,
`training_epochs`, `show_status`, `bias_interval`, `dict_amount`, `record_grid_points`.
"""
struct ShallowNetAutodiff{T, NNODES, basisType <: Basis{T},
                    ET <: Extrapolation,
                    IPMT <: InitialParametersMethod} <: ShallowNetMethod
    common        :: NetworkIntegratorCore{T, NNODES, basisType, ET, IPMT}
    bias_interval :: SVector{2, T}
    dict_amount   :: Int

    function ShallowNetAutodiff(basis::Basis{T}, quadrature::QuadratureRule{T};
        extrapolation_substep      :: Int  = 10,
        training_epochs           :: Int  = 50000,
        show_status               :: Bool = false,
        initial_trajectory_method :: ET   = IntegratorExtrapolation(),
        # Alone among the four integrators, this one's greedy step selects on the *normalized*
        # inner product. The rule decides which neurons are picked and hence which Newton basin
        # the step lands in, so it is a tuned baseline rather than a free choice.
        initial_guess_method      :: IPMT = OGA1dNormalized(),
        record_grid_points = 41,
        bias_interval = [-pi, pi],
        dict_amount   :: Int = 50000,) where {T, ET, IPMT}
        common = NetworkIntegratorCore(basis, quadrature;
            extrapolation_substep=extrapolation_substep,
            training_epochs=training_epochs,
            show_status = show_status,
            initial_trajectory_method=initial_trajectory_method,
            initial_guess_method=initial_guess_method,
            record_grid_points =  record_grid_points)
        new{T, nnodes(quadrature), typeof(basis), ET, IPMT}(
            common, SVector{2,T}(bias_interval), dict_amount)
    end
end

# The cache lives in `network_integrator_core.jl` and is shared with this integrator's
# sibling; the only thing that differs is the number of unknowns per dimension, passed
# below. `CacheType` returns `AutodiffShallowNetCache{ST}` — a *concrete* type, which it was
# not while the basis size and sub-step count were (runtime-computed) type parameters.

function GeometricIntegratorsBase.Cache{ST}(problem::AbstractProblemIODE, method::ShallowNetAutodiff; kwargs...) where {ST}
    local S = nbasis(method)
    AutodiffShallowNetCache{ST}(initial_conditions(problem), 3 * S + 1, S, nnodes(method),
        extrapolation_substep(method);
        record_grid_points = method.record_grid_points, kwargs...)
end

@inline GeometricIntegratorsBase.CacheType(ST, problem::AbstractProblemIODE, method::ShallowNetAutodiff) =
    AutodiffShallowNetCache{ST}


# The extrapolated trajectory has to land in `network_labels`, because that is what the OGA
# seed reads (`initial_params!` in `src/oga/adapters.jl`). This used to write the extrapolated
# positions into the *output-weight* slots of `x` and leave `network_labels` at zero, so the
# seed fitted the boundary ansatz to an all-zero target and then overwrote the very slots the
# extrapolation had filled. It also put `p̃` into `x[D*S+k]`, which for this ansatz is the
# endpoint *position* unknown rather than the momentum — cf. the `IntegratorExtrapolation`
# method below, which sets `q̃` there.
#
# Note that `solutionstep!` only extrapolates when `iguess(int)` is a framework extrapolation;
# with the default `NoInitialGuess()` it is a no-op. Pass `initialguess = HermiteExtrapolation()`
# to `GeometricIntegrator` for a real Hermite warm start.
function initial_trajectory!(sol, history, params, int::GeometricIntegrator{<:ShallowNetAutodiff}, initial_trajectory_method::HermiteExtrapolation)
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local x = nlsolution(int)
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local h = timestep(int)

    for i in eachindex(network_inputs)
        soltmp = (
            t=sol.t + (network_inputs[i] - 1) * h,
            q=cache(int).q̃,
            p=cache(int).p̃,
            q̇=cache(int).ṽ,
            ṗ=cache(int).f̃,
        )
        solutionstep!(soltmp, history, problem(int), iguess(int))

        for k in 1:D
            network_labels[i, k] = cache(int).q̃[k]
        end
    end

    soltmp = (
        t=sol.t,
        q=cache(int).q̃,
        p=cache(int).p̃,
        q̇=cache(int).ṽ,
        ṗ=cache(int).f̃,
    )
    solutionstep!(soltmp, history, problem(int), iguess(int))

    for k in 1:D
        x[D*S+k] = cache(int).q̃[k]
    end
end

function initial_trajectory!(sol, history, params, int::GeometricIntegrator{<:ShallowNetAutodiff}, initial_trajectory_method::IntegratorExtrapolation)
    local network_labels = cache(int).network_labels
    local integrator = default_iguess_integrator(method(int))
    local h = int.problem.timestep
    local extrapolation_substep = method(int).extrapolation_substep
    local D = length(cache(int).q̃)
    local problem = int.problem
    local S = nbasis(method(int))
    local x = nlsolution(int)

    tem_ode = similar(problem, [zero(h), h], h / extrapolation_substep, (q=StateVariable(sol.q[:]), p=StateVariable(sol.p[:])))
    tem_sol = integrate(tem_ode, integrator)

    for k in 1:D
        network_labels[:, k] = tem_sol.q[:, k]#[1].s
        cache(int).q̃[k] = tem_sol.q[:, k][end]
        cache(int).p̃[k] = tem_sol.p[:, k][end]
        x[D*S+k] = cache(int).q̃[k]
    end
end

# Allocation-free. The previous form materialised three slices of `ps` (`ps[1:S]`,
# `ps[S+1:2S]`, `ps[2S+1:3S]`) plus three broadcast temporaries on every call. This runs
# inside `ForwardDiff.gradient`, and via `∂VNN_ansatz_∂params` inside a *nested*
# gradient-of-a-derivative, so each of those six allocations was paid once per Dual chunk,
# twice over — per quadrature node, per dimension, per Newton residual, per Jacobian column.
#
# `ps` is laid out as [W2 (S) | W1 (S) | b1 (S)], so neuron `i` is (ps[i], ps[S+i], ps[2S+i]).
function apply_NN(t, ps, S, activation)
    return sum(ps[i] * activation(ps[S+i] * t + ps[2S+i]) for i in 1:S)
end

function NN_ansatz(ps, S::Int, activation, t, q̄, q)
    # q_h(t) = (1-t)q_n + t*q_{n+1} + t(1-t)NN(t)
    return (one(t) - t) * q̄ + t * q + t * (one(t) - t) * apply_NN(t, ps, S, activation)
end

"""
    VNN_ansatz(ps, S, activation, t, q̄, q)

`d/dt` of [`NN_ansatz`](@ref), in closed form.

`components!` used to get this from `Zygote.gradient` — reverse mode, for a *scalar* ℝ→ℝ
derivative, once per quadrature node per dimension per Newton residual and per Jacobian
column. The obvious replacement, `ForwardDiff.derivative(tt -> NN_ansatz(…, tt, …), t)`, does
not work at that call site: SimpleSolvers builds its Jacobian with **untagged** (`Tag =
Nothing`) `Dual`s, so `ps` and `q` arrive untagged, and ForwardDiff cannot order `Nothing`
against the tag its own inner derivative introduces. That is what kept the Zygote call alive.

Differentiating by hand sidesteps the nesting. With

    q_h(t) = (1-t)·q̄ + t·q + t(1-t)·N(t),    N(t) = Σᵢ W2ᵢ·σ(W1ᵢ·t + b1ᵢ)

we have

    q_h'(t) = q - q̄ + (1-2t)·N(t) + t(1-t)·N'(t),    N'(t) = Σᵢ W2ᵢ·W1ᵢ·σ'(W1ᵢ·t + b1ᵢ)

so only the *scalar* activation still needs differentiating, and that stays within a single
tag level whatever `ps` is. No allocation, no reverse-mode tape, and one fewer dependency.
"""
function VNN_ansatz(ps, S, activation, t, q̄, q)
    N  = apply_NN(t, ps, S, activation)
    N′ = sum(ps[i] * ps[S+i] * ForwardDiff.derivative(activation, ps[S+i] * t + ps[2S+i])
             for i in 1:S)
    return q - q̄ + (one(t) - 2t) * N + t * (one(t) - t) * N′
end
# In-place gradients, written into a caller-supplied buffer. These are what `components!` uses.
#
# `ForwardDiff.gradient(f, x)` builds a `GradientConfig` whose chunk size is chosen from
# `length(x)` — a *runtime* value here, since the parameter vector is `3S` long and `S` is a
# field of the basis. The resulting `Dual` width is therefore not inferable and the call returns
# `Any`, which propagated into every `view(g, 1:S)` downstream: JET reported twelve runtime
# dispatches in this function from that one cause alone. `gradient!` returns the buffer it was
# given, so the type is whatever the caller already knew, and the per-call allocation of the
# gradient vector goes away with it.
∂NN_ansatz_∂params!(g, ps, S, activation, t, q̄, q) =
    ForwardDiff.gradient!(g, p -> NN_ansatz(p, S, activation, t, q̄, q), ps)
∂VNN_ansatz_∂params!(g, ps, S, activation, t, q̄, q) =
    ForwardDiff.gradient!(g, p -> VNN_ansatz(p, S, activation, t, q̄, q), ps)

# Allocating wrappers, kept for `benchmark/compare_derivative_backends.jl`, which measures the
# derivative backends against each other per call and wants a self-contained expression.
∂NN_ansatz_∂params(ps, S, activation, t, q̄, q) =
    ∂NN_ansatz_∂params!(similar(ps), ps, S, activation, t, q̄, q)
∂VNN_ansatz_∂params(ps, S, activation, t, q̄, q) =
    ∂VNN_ansatz_∂params!(similar(ps), ps, S, activation, t, q̄, q)


function GeometricIntegratorsBase.components!(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:ShallowNetAutodiff}) where {ST}
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local C = cache(int, ST)

    local quad_nodes = QuadratureRules.nodes(int.method.quadrature)
    local q̄ = sol.q

    local q = cache(int, ST).q̃
    local p = cache(int, ST).p̃
    local Q = cache(int, ST).Q
    local V = cache(int, ST).V
    local P = cache(int, ST).P
    local F = cache(int, ST).F
    local X = cache(int, ST).X

    local NN = method(int).basis.NN
    local ps = cache(int, ST).ps
    local ps_vec = cache(int, ST).ps_vec
    local g_buf  = cache(int, ST).g_buf
    local gv_buf = cache(int, ST).gv_buf

    local dqdW2c = cache(int, ST).dqdW2c
    local dvdW2c = cache(int, ST).dvdW2c
    local dqdW1c = cache(int, ST).dqdW1c
    local dvdW1c = cache(int, ST).dvdW1c
    local dqdbc = cache(int, ST).dqdbc
    local dvdbc = cache(int, ST).dvdbc


    local activation = method(int).basis.activation

    # copy x to q
    for k in eachindex(q)
        q[k] = x[D*S+k]
    end

    for k in 1:D
        for i in 1:S
            ps[k][2].W[i] = x[D*(i-1)+k]
            ps[k][1].W[i] = x[D*(S+1)+D*(i-1)+k]
            ps[k][1].b[i] = x[D*(S+1+S)+D*(i-1)+k]
        end
    end

    # One pass over `d`, one `ps_vec` fill per dimension.
    #
    # This used to be three separate `d`-loops, each re-deriving the *same* flat parameter
    # vector: `ps_vec = zeros(ST, 3S)` was allocated once before the first loop and then again
    # *inside* each of the other two, and all three repeated the same three-slice gather, whose
    # right-hand sides (`ps[d][2].W[:]` and friends) are themselves copies. `ps_vec` is now a
    # cache field, filled once per dimension with `copyto!`.
    #
    # There are no boundary-point gradients here either. `∂NN_ansatz_∂params` used to be
    # evaluated at t=0 and t=1 as well, into six `dqd*r₀`/`dqd*r₁` cache arrays that nothing
    # ever read: `residual!` works entirely from the quadrature-node arrays and `quad_nodes`,
    # because this ansatz interpolates the endpoints exactly — q(0) = q̄ and q(1) = q by
    # construction — so the boundary derivatives carry no information the residual needs.
    #
    # The `g[1:S]` / `g[S+1:2S]` / `g[2S+1:3S]` reads are views now; each used to allocate.
    for d in 1:D
        copyto!(view(ps_vec, 1:S),      ps[d][2].W)
        copyto!(view(ps_vec, S+1:2S),   ps[d][1].W)
        copyto!(view(ps_vec, 2S+1:3S),  ps[d][1].b)

        # `q̃plain`, not `q[d]`, in the two *gradient* calls below. `q[d]` is the `ST` cache, so
        # during the Newton solve it is a `ForwardDiff.Dual`; `∂NN_ansatz_∂params` nests a
        # second `ForwardDiff.gradient` inside that, and mixing an outer untagged `Dual` with
        # the inner tagged one raises "Cannot determine ordering of Dual tags". Reading the
        # *problem-datatype* cache keeps this argument a plain number.
        #
        # That is sound rather than merely expedient: `q` enters the ansatz only through the
        # linear term `t·q`, so ∂/∂(W2, W1, b1) is identically independent of it, and the same
        # holds after the `d/dt` in `∂VNN_ansatz_∂params`. The *value* computations below do
        # need the live endpoint estimate, and use `q[d]`.
        q̃plain = cache(int).q̃[d]

        for j in eachindex(quad_nodes)
            g = ∂NN_ansatz_∂params!(g_buf, ps_vec, S, activation, quad_nodes[j], q̄[d], q̃plain)
            copyto!(view(dqdW2c, j, :, d), view(g, 1:S))
            copyto!(view(dqdW1c, j, :, d), view(g, S+1:2S))
            copyto!(view(dqdbc,  j, :, d), view(g, 2S+1:3S))

            gv = ∂VNN_ansatz_∂params!(gv_buf, ps_vec, S, activation, quad_nodes[j], q̄[d], q̃plain)
            copyto!(view(dvdW2c, j, :, d), view(gv, 1:S))
            copyto!(view(dvdW1c, j, :, d), view(gv, S+1:2S))
            copyto!(view(dvdbc,  j, :, d), view(gv, 2S+1:3S))

            # Position and velocity at the same node, from the same `ps_vec`.
            Q[j][d] = NN_ansatz(ps_vec, S, activation, quad_nodes[j], q̄[d], q[d])
            V[j][d] = VNN_ansatz(ps_vec, S, activation, quad_nodes[j], q̄[d], q[d]) / timestep(int)
        end
    end

    # compute P=ϑ(Q,V) and F=f(Q,V)
    for i in eachindex(C.Q, C.V, C.P, C.F)
        tᵢ = sol.t + timestep(int) * (method(int).c[i] - 1)
        equations(int).ϑ(C.P[i], tᵢ, C.Q[i], C.V[i], params)
        equations(int).f(C.F[i], tᵢ, C.Q[i], C.V[i], params)
    end
end


function GeometricIntegratorsBase.residual!(b::Vector{ST}, sol, params, int::GeometricIntegrator{<:ShallowNetAutodiff}) where {ST}
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local q̄ = sol.q
    local p̄ = sol.p
    local p̃ = cache(int, ST).p̃
    local P = cache(int, ST).P
    local F = cache(int, ST).F
    local X = cache(int, ST).X


    local dqdW2c = cache(int, ST).dqdW2c
    local dvdW2c = cache(int, ST).dvdW2c
    local dqdW1c = cache(int, ST).dqdW1c
    local dvdW1c = cache(int, ST).dvdW1c
    local dqdbc = cache(int, ST).dqdbc
    local dvdbc = cache(int, ST).dvdbc
    local quad_nodes = QuadratureRules.nodes(int.method.quadrature)


    # compute b = - [(P-AF)], the residual in actual action, vatiation with respect to Q_{n,i}
    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdW2c[j, i, k]
                z += method(int).b[j] * P[j][k] * dvdW2c[j, i, k]
            end
            b[D*(i-1)+k] =  z
        end
    end

    for k in eachindex(p̄)
        z = zero(ST)
        for j in eachindex(P, F)
            z += timestep(int) * method(int).b[j] * F[j][k] * (1 - quad_nodes[j])
            z += method(int).b[j] * P[j][k] * (-1)
        end
        b[D*S+k] = p̄[k] + z
    end

    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdW1c[j, i, k]
                z += method(int).b[j] * P[j][k] * dvdW1c[j, i, k]
            end
            b[D*(S+1)+D*(i-1)+k] =  z
        end
    end

    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdbc[j, i, k]
                z += method(int).b[j] * P[j][k] * dvdbc[j, i, k]
            end
            b[D*(S+1+S)+D*(i-1)+k] = z
        end
    end

end



function update_solution!(sol, params, int::GeometricIntegrator{<:ShallowNetAutodiff}, ::Type{DT}) where {DT}
    local D = length(cache(int).q̃)
    local quad_nodes = QuadratureRules.nodes(int.method.quadrature)
    local P = cache(int).P
    local F = cache(int).F

    sol.q .= cache(int, DT).q̃

    for k in 1:D
        z = zero(eltype(sol.p))
        for j in eachindex(P, F)
            # dQ/dq_{n+1} = τ, dV/dq_{n+1} = 1/h
            z += timestep(int) * method(int).b[j] * F[j][k] * (quad_nodes[j])
            z += method(int).b[j] * P[j][k]
        end
        sol.p[k] = z
    end
    # sol.p .= cache(int, DT).p̃
end



function record_finer_solution!(sol, int::GeometricIntegrator{<:ShallowNetAutodiff})
    local x = nlsolution(int)
    local stage_values = cache(int).stage_values
    # local network_inputs = method(int).network_inputs
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local q̄ = sol.q  # start point q_n
    local q = cache(int).q̃ # endpoint estimate q_{n+1}
    local activation = method(int).basis.activation

    local N_plot = method(int).record_grid_points
    local T = eltype(x)
    network_inputs = reshape(collect(range(zero(T), one(T), N_plot)), 1, N_plot)

    @debug "solution x after solving by Newton" x
    for k in 1:D
        for i in 1:S
            ps[k][2].W[i] = x[D*(i-1)+k]
            ps[k][1].W[i] = x[D*(S+1)+D*(i-1)+k]
            ps[k][1].b[i] = x[D*(S+1+S)+D*(i-1)+k]
        end

        ps_vec = zeros(eltype(x), 3S)
        ps_vec[1:S] = ps[k][2].W[:]
        ps_vec[S+1:2S] = ps[k][1].W[:]
        ps_vec[2S+1:3S] = ps[k][1].b[:]

        for i in eachindex(network_inputs)
            stage_values[i, k] = NN_ansatz(ps_vec, S, activation, network_inputs[i], q̄[k], q[k])
        end

        @debug "parameters after solving" dim = k W2 = ps[k][2].W W1 = ps[k][1].W b1 = ps[k][1].b
    end

    @debug "stages prediction after solving" stage_values q = sol.q p = sol.p

end


