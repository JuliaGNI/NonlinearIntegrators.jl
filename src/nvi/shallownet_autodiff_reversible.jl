"""
    ShallowNetAutodiffReversible <: ShallowNetMethod

Time-symmetric variant of `ShallowNetAutodiff`: computes network derivatives with
`ForwardDiff` (no symbolic pre-compilation) and enforces the palindromic
time-reversal symmetry (`issymmetric(method) == true`). Combines the
forward-differentiation approach of `ShallowNetAutodiff` with the symmetry structure
of `ShallowNetReversible`.

# Constructor

    ShallowNetAutodiffReversible(basis, quadrature; kwargs...)

Keyword arguments are the same as `ShallowNetAutodiff`. Only the [`OGA`](@ref) seeds are
supported as `initial_guess_method`. The basis must have an even number of neurons —
they come in mirrored pairs sharing one output weight.
"""
struct ShallowNetAutodiffReversible{T, NNODES, basisType <: Basis{T},
                                ET <: Extrapolation,
                                IPMT <: InitialParametersMethod} <: ShallowNetMethod
    common        :: NetworkIntegratorCore{T, NNODES, basisType, ET, IPMT}
    bias_interval :: SVector{2, T}
    dict_amount   :: Int

    function ShallowNetAutodiffReversible(basis::Basis{T}, quadrature::QuadratureRule{T};
        extrapolation_substep      :: Int  = 10,
        training_epochs           :: Int  = 50000,
        show_status               :: Bool = false,
        initial_trajectory_method :: ET   = IntegratorExtrapolation(),
        initial_guess_method      :: IPMT = OGA1d(),
        record_grid_points        :: Int  = 41,
        bias_interval = [-pi, pi],
        dict_amount   :: Int = 50000) where {T, ET, IPMT}
        # See `ShallowNetReversible`: mirrored pairs, only the independent half stored.
        iseven(basis.S) || throw(ArgumentError(
            "ShallowNetAutodiffReversible requires a basis with an even number of neurons, " *
            "got S = $(basis.S). Neurons come in mirrored pairs sharing one output weight, " *
            "and only the S/2 independent hidden parameters are stored in the nonlinear " *
            "solution vector."))

        common = NetworkIntegratorCore(basis, quadrature;
            extrapolation_substep=extrapolation_substep,
            training_epochs=training_epochs,
            show_status=show_status,
            initial_trajectory_method=initial_trajectory_method,
            initial_guess_method=initial_guess_method,
            record_grid_points = record_grid_points)
        new{T, nnodes(quadrature), typeof(basis), ET, IPMT}(
            common, SVector{2,T}(bias_interval), dict_amount)
    end
end

GeometricIntegratorsBase.issymmetric(::Union{ShallowNetAutodiffReversible, Type{<:ShallowNetAutodiffReversible}}) = true

# The cache lives in `network_integrator_core.jl` and is shared with this integrator's
# sibling; the only thing that differs is the number of unknowns per dimension, passed
# below. `CacheType` returns `AutodiffShallowNetCache{ST}` — a *concrete* type, which it was
# not while the basis size and sub-step count were (runtime-computed) type parameters.

function GeometricIntegratorsBase.Cache{ST}(problem::AbstractProblemIODE, method::ShallowNetAutodiffReversible; kwargs...) where {ST}
    local S = nbasis(method)
    AutodiffShallowNetCache{ST}(initial_conditions(problem), 2 * S + 1, S, nnodes(method),
        extrapolation_substep(method);
        record_grid_points = method.record_grid_points, kwargs...)
end

@inline GeometricIntegratorsBase.CacheType(ST, problem::AbstractProblemIODE, method::ShallowNetAutodiffReversible) =
    AutodiffShallowNetCache{ST}

# The extrapolated trajectory has to land in `network_labels`, because that is what the OGA
# seed reads (`initial_params!` in `src/oga/adapters.jl`). This used to write the extrapolated
# positions into the *output-weight* slots of `x` and leave `network_labels` at zero, so the
# seed fitted the boundary ansatz to an all-zero target and then overwrote the very slots the
# extrapolation had filled. It also put `p̃` into `x[D*S+k]`, which for this ansatz is the
# endpoint *position* unknown rather than the momentum — cf. the `IntegratorExtrapolation`
# method below, and `ShallowNetAutodiff`, which both set `q̃` there.
#
# Note that `solutionstep!` only extrapolates when `iguess(int)` is a framework extrapolation;
# with the default `NoInitialGuess()` it is a no-op. Pass `initialguess = HermiteExtrapolation()`
# to `GeometricIntegrator` for a real Hermite warm start.
function initial_trajectory!(sol, history, params, int::GeometricIntegrator{<:ShallowNetAutodiffReversible}, initial_trajectory_method::HermiteExtrapolation)
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

function initial_trajectory!(sol, history, params, int::GeometricIntegrator{<:ShallowNetAutodiffReversible}, initial_trajectory_method::IntegratorExtrapolation)
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
# The ansatz (`apply_NN`, `NN_ansatz`, `VNN_ansatz`, `∂NN_ansatz_∂params`,
# `∂VNN_ansatz_∂params`) is shared with `ShallowNetAutodiff` and defined once, in
# shallownet_autodiff.jl. A commented-out second copy used to sit here, which made it look as
# though this file were self-contained when in fact it depends on that file being `include`d
# first (see the include order in src/NonlinearIntegrators.jl).

function GeometricIntegratorsBase.components!(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:ShallowNetAutodiffReversible}) where {ST}
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
        end
        for i in 1:(S ÷ 2)
            ps[k][1].W[2i-1] = x[D*(S+1)+D*(i-1)+k]
            ps[k][1].b[2i-1] = x[D*(S+1+S÷2)+D*(i-1)+k]
            ps[k][1].W[2i] = -ps[k][1].W[2i-1]
            ps[k][1].b[2i] = ps[k][1].W[2i-1] + ps[k][1].b[2i-1]
        end
    end

    # One pass over `d`, one `ps_vec` fill per dimension. See the equivalent block in
    # shallownet_autodiff.jl: this used to be three separate `d`-loops each re-deriving the
    # same flat parameter vector, two of them re-allocating it inside the loop, and the
    # velocity was taken with `Zygote.gradient` where the ForwardDiff `VNN_ansatz` computes
    # the same number. The `g[1:S]`-style reads are views now; each used to allocate.
    #
    # There are no boundary-point gradients here either: they fed six cache arrays that
    # nothing ever read, because this ansatz interpolates the endpoints exactly.
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


function GeometricIntegratorsBase.residual!(b::Vector{ST}, sol, params, int::GeometricIntegrator{<:ShallowNetAutodiffReversible}) where {ST}
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
            z += timestep(int) * method(int).b[j] * F[j][k] * (1-quad_nodes[j])
            z += method(int).b[j] * P[j][k] * (-1)
        end
        b[D*S+k] = p̄[k] + z
    end

    for i in 1:(S ÷ 2)
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdW1c[j, 2i-1, k]
                z += method(int).b[j] * P[j][k] * dvdW1c[j, 2i-1, k]
            end
            b[D*(S+1)+D*(i-1)+k] =  z
        end
    end

    for i in 1:(S ÷ 2)
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdbc[j, 2i-1, k]
                z += method(int).b[j] * P[j][k] * dvdbc[j, 2i-1, k]
            end
            b[D*(S+1+S÷2)+D*(i-1)+k] = z
        end
    end

    # See the note in shallownet_reversible.jl: `residual!` is called per Newton iteration and
    # per Jacobian column, so this has to be `@debug`, not a `show_status`-gated `println`.
    @debug "residual" b norm_b = norm(b)
end



function update_solution!(sol, params, int::GeometricIntegrator{<:ShallowNetAutodiffReversible}, ::Type{DT}) where {DT}
    local D = length(cache(int).q̃)
    local quad_nodes = QuadratureRules.nodes(int.method.quadrature)
    local P = cache(int).P
    local F = cache(int).F

    sol.q .= cache(int, DT).q̃

    for k in 1:D
        z = zero(DT)
        for j in eachindex(P, F)
            # dQ/dq_{n+1} = τ, dV/dq_{n+1} = 1/h
            z += timestep(int) * method(int).b[j] * F[j][k] * (quad_nodes[j])
            z += method(int).b[j] * P[j][k]
        end
        sol.p[k] = z
    end
    # sol.p .= cache(int, DT).p̃
end



function record_finer_solution!(sol, int::GeometricIntegrator{<:ShallowNetAutodiffReversible})
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
        end
        for i in 1:(S ÷ 2)
            ps[k][1].W[2i-1] = x[D*(S+1)+D*(i-1)+k]
            ps[k][1].b[2i-1] = x[D*(S+1+S÷2)+D*(i-1)+k]
            ps[k][1].W[2i] = -ps[k][1].W[2i-1]
            ps[k][1].b[2i] = ps[k][1].W[2i-1] + ps[k][1].b[2i-1]
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


