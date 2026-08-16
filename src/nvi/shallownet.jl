"""
    ShallowNet <: ShallowNetMethod

Continuous Galerkin Variational Integrator using a `ShallowNetBasis` basis.
The network ansatz `q(t) = NN(t; θ)` is a single-hidden-layer network; the
optimal parameters `θ` are found by Newton's method applied to the discrete
Euler-Lagrange equations at each time step.

# Constructor

    ShallowNet(basis, quadrature; kwargs...)

**Required:**
- `basis::AbstractShallowNetBasis{T}` — e.g. `ShallowNetBasis{T}(activation, S)`
- `quadrature::QuadratureRule{T}` — e.g. `GaussLegendreQuadrature(T, R)`

**Keyword arguments:**
- `initial_trajectory_method` — `IntegratorExtrapolation()` (default), `HermiteExtrapolation()`, or `NoExtrapolation()`
- `initial_guess_method` — an [`OGA`](@ref) seed: `OGA1d()` (default), `OGA1dNormalized()`,
  `OGA1dStable()`, `OGA2d()`, `OGASphere()`, or a hand-built `OGA(dictionary, selection, fit)`.
  Also `OGA1dNormalEquations()` (the original-paper reference path) and `TrainingMethod()`.
- `extrapolation_substep::Int = 10` — sub-steps for the `IntegratorExtrapolation` warm start
- `training_epochs::Int = 50000` — gradient-descent epochs when `initial_guess_method = TrainingMethod()`
- `bias_interval` — bias search range for OGA dictionary, default `[-π, π]`
- `dict_amount::Int = 50000` — number of atoms in the OGA dictionary
- `record_grid_points::Int = 41` — number of grid points per step stored in `stage_values`

# Example

```julia
using NonlinearIntegrators, QuadratureRules
basis = ShallowNetBasis{Float64}(tanh, 8)
quad  = GaussLegendreQuadrature(Float64, 8)
method = ShallowNet(basis, quad; bias_interval = [-π, π], dict_amount = 400)
```
"""
struct ShallowNet{T, NNODES, basisType <: Basis{T},
                               ET <: Extrapolation,
                               IPMT <: InitialParametersMethod} <: ShallowNetMethod
    common        :: NetworkIntegratorCore{T, NNODES, basisType, ET, IPMT}
    bias_interval :: SVector{2, T}
    dict_amount   :: Int

    function ShallowNet(basis::Basis{T}, quadrature::QuadratureRule{T};
        extrapolation_substep      :: Int  = 10,
        training_epochs           :: Int  = 50000,
        show_status               :: Bool = false,
        initial_trajectory_method :: ET   = IntegratorExtrapolation(),
        initial_guess_method      :: IPMT = OGA1d(),
        record_grid_points = 41,
        bias_interval = [-pi, pi],
        dict_amount   :: Int = 50000,) where {T, ET, IPMT}
        require_symbolic_derivatives(basis, "ShallowNet")
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
# below. `CacheType` returns `SymbolicShallowNetCache{ST}` — a *concrete* type, which it was
# not while the basis size and sub-step count were (runtime-computed) type parameters.

function GeometricIntegratorsBase.Cache{ST}(problem::AbstractProblemIODE, method::ShallowNet; kwargs...) where {ST}
    local S = nbasis(method)
    SymbolicShallowNetCache{ST}(initial_conditions(problem), 3 * S + 1, S, nnodes(method),
        extrapolation_substep(method);
        record_grid_points = method.record_grid_points, kwargs...)
end

@inline GeometricIntegratorsBase.CacheType(ST, problem::AbstractProblemIODE, method::ShallowNet) =
    SymbolicShallowNetCache{ST}

function initial_params!(int::GeometricIntegrator{<:ShallowNet}, initialParams::TrainingMethod, sol)
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))

    local x = nlsolution(int)
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local extrapolation_substep = method(int).extrapolation_substep
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local nepochs = method(int).training_epochs

    Random.seed!(42)

    for k in 1:D
        @debug "For dimension" k network_labels[:, k]
        labels = reshape(network_labels[:, k], 1, extrapolation_substep + 1)

        PNN = AbstractNeuralNetworks.NeuralNetwork(NN)
        # `Adam` and the line search are built at the parameter element type: `Optimizer` does not
        # convert `Adam`, so an `Adam{Float64}` handed `Float32` parameters would not dispatch.
        # `ps_flat` aliases the network's arrays (see `optimizer_params`), so the in-place updates
        # in the loop below are visible through `PNN.params`. `Adam` supplies only a direction, so
        # the learning rate is the line search, decaying from 1e-3 to 5e-5 over the epoch budget.
        local PT = eltype(PNN.params[1].W)
        ps_flat = optimizer_params(PNN.params)
        loss(p) = mae_loss(network_inputs, labels, NN, network_params(p, PNN.params))
        algorithm = GeometricOptimizers.Adam(PT)
        # `max_iterations` is the epoch budget: `solve!` runs its own loop and stops on
        # `meets_stopping_criteria`, so the budget has to be an option rather than a `for` range.
        # `warn_iterations = 0` because reaching that budget is the normal outcome here, not a
        # diagnosis — at the default of 1000 against 50 000 epochs, `solve!` would print its
        # warning once per dimension per time step.
        opt = GeometricOptimizers.Optimizer(ps_flat, loss; algorithm = algorithm,
            linesearch = GeometricOptimizers.DecayingStatic(PT; η₁ = PT(1e-3), η₂ = PT(5e-5), n = nepochs),
            max_iterations = nepochs, warn_iterations = 0)
        state = GeometricOptimizers.OptimizerState(algorithm, ps_flat)
        # `solve!` rather than the hand-rolled epoch loop this used to run: it makes the same
        # `solver_step!` calls in the same order, but it also assesses the solve and hands back an
        # `OptimizerResult` carrying the outcome. The two DenseNet loops cannot be written this way
        # — one breaks early on the loss, the other re-solves a layer by least squares inside every
        # epoch — so they stay hand-rolled; see the CHANGELOG.
        result = GeometricOptimizers.solve!(ps_flat, state, opt)
        optstatus = GeometricOptimizers.status(result)
        # The training is a *seed* for the Newton solve that follows, so a budget spent without
        # converging is the expected case and not an error: it is reported at debug level, like the
        # loss beside it, rather than warned about. The epoch count comes from the state because
        # `solve!` may stop before the budget, which the old message assumed it never did.
        @debug "dimension" k "final loss:" mae_loss(network_inputs, labels, NN, PNN.params) "in" GeometricOptimizers.iteration_number(state) "of" nepochs "epochs; converged:" GeometricOptimizers.isconverged(optstatus)

        for i in 1:S
            x[D*(i-1)+k] = PNN.params[2].W[i]
            x[D*(S+1)+D*(i-1)+k] = PNN.params[1].W[i]
            x[D*(S+1+S)+D*(i-1)+k] = PNN.params[1].b[i]
        end
    end
    @debug "Initial guess from network training" x
end

function GeometricIntegratorsBase.components!(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:ShallowNet}) where {ST}
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local C = cache(int, ST)

    local quad_nodes = QuadratureRules.nodes(int.method.quadrature)

    local q = cache(int, ST).q̃
    local p = cache(int, ST).p̃
    local Q = cache(int, ST).Q
    local V = cache(int, ST).V
    local P = cache(int, ST).P
    local F = cache(int, ST).F
    local X = cache(int, ST).X

    local NN = method(int).basis.NN
    local ps = cache(int, ST).ps

    local tbuf = cache(int, ST).tbuf
    local r₀ = cache(int, ST).r₀
    local r₁ = cache(int, ST).r₁
    local m = cache(int, ST).m
    local a = cache(int, ST).a
    local dqdWc = cache(int, ST).dqdWc
    local dqdbc = cache(int, ST).dqdbc
    local dvdWc = cache(int, ST).dvdWc
    local dvdbc = cache(int, ST).dvdbc
    local dqdWr₁ = cache(int, ST).dqdWr₁
    local dqdWr₀ = cache(int, ST).dqdWr₀
    local dqdbr₁ = cache(int, ST).dqdbr₁
    local dqdbr₀ = cache(int, ST).dqdbr₀

    local DVDθ = method(int).basis.dvdθ
    local DQDθ = method(int).basis.dqdθ

    # copy x to X
    for i in eachindex(X)
        for k in eachindex(X[i])
            X[i][k] = x[D*(i-1)+k]
        end
    end

    # copy x to p # momenta
    for k in eachindex(p)
        p[k] = x[D*S+k]
    end

    for k in 1:D
        for i in 1:S
            ps[k][2].W[i] = x[D*(i-1)+k]
            ps[k][1].W[i] = x[D*(S+1)+D*(i-1)+k]
            ps[k][1].b[i] = x[D*(S+1+S)+D*(i-1)+k]
        end
    end

    # Coefficients and their parameter derivatives, at the quadrature nodes and the two
    # interval boundaries. This used to be two passes over `d`; merging them is what lets the
    # per-`(j, d)` work be shared:
    #
    #   * `DVDθ` is evaluated once per node instead of twice. The first pass called it for `a`
    #     (taking `.L2.W`) and the second called it again at the same node for `dvdWc`/`dvdbc`
    #     (taking `.L1.W`/`.L1.b`) — one call returns all three. This is the most expensive
    #     kernel in the integrator, so it is a straight 2× on the dominant cost.
    #   * `NeuralNetworkParameters(ps[d])` is loop-invariant in `j` and is now built once per
    #     dimension rather than four times per node.
    #   * `copyto!` into a `view` replaces `dst[j, :, d] = src.L1.W[:]`, whose `[:]` allocated
    #     a copy of the source only to read it once.
    for d in 1:D
        psd = NeuralNetworkParameters(ps[d])

        tbuf[1] = zero(ST)
        copyto!(view(r₀, :, d), (NN.layers[1])(tbuf, ps[d][1]))
        g0 = DQDθ(tbuf, psd)
        copyto!(view(dqdWr₀, :, d), g0.L1.W)
        copyto!(view(dqdbr₀, :, d), g0.L1.b)

        tbuf[1] = one(ST)
        copyto!(view(r₁, :, d), (NN.layers[1])(tbuf, ps[d][1]))
        g1 = DQDθ(tbuf, psd)
        copyto!(view(dqdWr₁, :, d), g1.L1.W)
        copyto!(view(dqdbr₁, :, d), g1.L1.b)

        for j in eachindex(quad_nodes)
            tbuf[1] = quad_nodes[j]
            copyto!(view(m, j, :, d), (NN.layers[1])(tbuf, ps[d][1]))

            g = DQDθ(tbuf, psd)
            copyto!(view(dqdWc, j, :, d), g.L1.W)
            copyto!(view(dqdbc, j, :, d), g.L1.b)

            gv = DVDθ(tbuf, psd)
            copyto!(view(a, j, :, d), gv.L2.W)
            copyto!(view(dvdWc, j, :, d), gv.L1.W)
            copyto!(view(dvdbc, j, :, d), gv.L1.b)
        end
    end

    # compute Q : q at quaadurature points
    for i in eachindex(Q)
        for d in eachindex(Q[i])
            y = zero(ST)
            for j in eachindex(X)
                y += m[i, j, d] * X[j][d]
            end
            Q[i][d] = y
        end
    end

    # compute q[t_{n+1}]
    for d in eachindex(q)
        y = zero(ST)
        for i in eachindex(X)
            y += r₁[i, d] * X[i][d]
        end
        q[d] = y
    end

    # compute V volicity at quadrature points
    for i in eachindex(V)
        for k in eachindex(V[i])
            y = zero(ST)
            for j in eachindex(X)
                y += a[i, j, k] * X[j][k]
            end
            V[i][k] = y / timestep(int)
        end
    end

    # compute P=ϑ(Q,V) and F=f(Q,V)
    for i in eachindex(C.Q, C.V, C.P, C.F)
        tᵢ = sol.t + timestep(int) * (method(int).c[i] - 1)
        equations(int).ϑ(C.P[i], tᵢ, C.Q[i], C.V[i], params)
        equations(int).f(C.F[i], tᵢ, C.Q[i], C.V[i], params)
    end
end


function GeometricIntegratorsBase.residual!(b::Vector{ST}, sol, params, int::GeometricIntegrator{<:ShallowNet}) where {ST}
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local q̄ = sol.q
    local p̄ = sol.p
    local p̃ = cache(int, ST).p̃
    local P = cache(int, ST).P
    local F = cache(int, ST).F
    local X = cache(int, ST).X

    local r₀ = cache(int, ST).r₀
    local r₁ = cache(int, ST).r₁
    local m = cache(int, ST).m
    local a = cache(int, ST).a

    local dqdWc = cache(int, ST).dqdWc
    local dqdbc = cache(int, ST).dqdbc
    local dvdWc = cache(int, ST).dvdWc
    local dvdbc = cache(int, ST).dvdbc
    local dqdWr₁ = cache(int, ST).dqdWr₁
    local dqdWr₀ = cache(int, ST).dqdWr₀
    local dqdbr₁ = cache(int, ST).dqdbr₁
    local dqdbr₀ = cache(int, ST).dqdbr₀

    # compute b = - [(P-AF)], the residual in actual action, vatiation with respect to Q_{n,i}
    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += method(int).b[j] * m[j, i, k] * F[j][k] * timestep(int)
                z += method(int).b[j] * a[j, i, k] * P[j][k]
            end
            b[D*(i-1)+k] = (r₁[i, k] * p̃[k] - r₀[i, k] * p̄[k]) - z
        end
    end

    # the continue constraint from hamilton pontryagin principle
    for k in eachindex(q̄)
        y = zero(ST)
        for j in eachindex(X)
            y += r₀[j, k] * X[j][k]
        end
        b[D*S+k] = q̄[k] - y
    end

    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdWc[j, i, k]
                z += method(int).b[j] * P[j][k] * dvdWc[j, i, k]
            end
            b[D*(S+1)+D*(i-1)+k] = dqdWr₁[i, k] * p̃[k] - z
        end
    end

    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdbc[j, i, k]
                z += method(int).b[j] * P[j][k] * dvdbc[j, i, k]
            end
            b[D*(S+1+S)+D*(i-1)+k] = (dqdbr₁[i, k] * p̃[k] - dqdbr₀[i, k] * p̄[k]) - z
        end
    end
    # @debug " Residual vector b: " b
    # @debug " Norm of Residual vector b: " norm(b)

end





function record_finer_solution!(sol, int::GeometricIntegrator{<:ShallowNet})
    local x = nlsolution(int)
    local stage_values = cache(int).stage_values
    # local network_inputs = method(int).network_inputs
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local NN = method(int).basis.NN
    local ps = cache(int).ps

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
        stage_values[:, k] = NN(network_inputs, ps[k])[:]
    end

    @debug "stages prediction after solving" stage_values
    @debug "sol from this step q:", sol.q, "p:", sol.p
end


