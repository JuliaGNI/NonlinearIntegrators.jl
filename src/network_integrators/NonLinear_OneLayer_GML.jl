"""
    NonLinear_OneLayer_GML <: OneLayerMethod

Continuous Galerkin Variational Integrator using a `OneLayerNetwork_GML` basis.
The network ansatz `q(t) = NN(t; θ)` is a single-hidden-layer network; the
optimal parameters `θ` are found by Newton's method applied to the discrete
Euler-Lagrange equations at each time step.

# Constructor

    NonLinear_OneLayer_GML(basis, quadrature; kwargs...)

**Required:**
- `basis::OneLayerNetBasis{T}` — e.g. `OneLayerNetwork_GML{T}(activation, S)`
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
basis = OneLayerNetwork_GML{Float64}(tanh, 8)
quad  = GaussLegendreQuadrature(Float64, 8)
method = NonLinear_OneLayer_GML(basis, quad; bias_interval = [-π, π], dict_amount = 400)
```
"""
struct NonLinear_OneLayer_GML{T, NNODES, basisType <: Basis{T},
                               ET <: Extrapolation,
                               IPMT <: InitialParametersMethod} <: OneLayerMethod
    common        :: NetworkIntegratorCore{T, NNODES, basisType, ET, IPMT}
    bias_interval :: SVector{2, T}
    dict_amount   :: Int

    function NonLinear_OneLayer_GML(basis::Basis{T}, quadrature::QuadratureRule{T};
        extrapolation_substep      :: Int  = 10,
        training_epochs           :: Int  = 50000,
        show_status               :: Bool = true,
        initial_trajectory_method :: ET   = IntegratorExtrapolation(),
        initial_guess_method      :: IPMT = OGA1d(),
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
        new{T, QuadratureRules.nnodes(quadrature), typeof(basis), ET, IPMT}(
            common, SVector{2,T}(bias_interval), dict_amount)
    end
end

struct NonLinear_OneLayer_GMLCache{ST,S,R,N} <: NetworkIntegratorCache{ST}
    x::Vector{ST}

    q̄::Vector{ST}
    p̄::Vector{ST}

    q̃::Vector{ST}
    p̃::Vector{ST}
    ṽ::Vector{ST}
    f̃::Vector{ST}
    s̃::Vector{ST}

    X::Vector{Vector{ST}}
    Q::Vector{Vector{ST}}
    P::Vector{Vector{ST}}
    V::Vector{Vector{ST}}
    F::Vector{Vector{ST}}

    ps::Vector{@NamedTuple{L1::@NamedTuple{W::Matrix{ST}, b::Vector{ST}},L2::@NamedTuple{W::Matrix{ST}}}}

    r₀::Matrix{ST}
    r₁::Matrix{ST}
    m::Array{ST,3}
    a::Array{ST,3}

    dqdWc::Array{ST,3}
    dqdbc::Array{ST,3}
    dvdWc::Array{ST,3}
    dvdbc::Array{ST,3}

    dqdWr₁::Matrix{ST}
    dqdWr₀::Matrix{ST}

    dqdbr₁::Matrix{ST}
    dqdbr₀::Matrix{ST}

    stage_values::Matrix{ST}
    network_labels::Matrix{ST}

    function NonLinear_OneLayer_GMLCache{ST,S,R,N}(ics; record_grid_points::Int = 41) where {ST,S,R,N}
        D = length(vec(ics.q))
        x = zeros(ST, D * (S + 1 + 2 * S)) # Last layer Weight S (no bias for now) + P + hidden layer W (S*S₁) + hidden layer bias S

        q̄ = zeros(ST, D)
        p̄ = zeros(ST, D)

        # create temporary vectors
        q̃ = zeros(ST, D)
        p̃ = zeros(ST, D)
        ṽ = zeros(ST, D)
        f̃ = zeros(ST, D)
        s̃ = zeros(ST, D)

        # create internal stage vectors
        X = create_internal_stage_vector(ST, D, S)
        Q = create_internal_stage_vector(ST, D, R)
        P = create_internal_stage_vector(ST, D, R)
        V = create_internal_stage_vector(ST, D, R)
        F = create_internal_stage_vector(ST, D, R)

        # create parameter vectors
        ps = [(L1=(W=zeros(ST, S, 1), b=zeros(ST, S)), L2=(W=zeros(ST, 1, S),)) for k in 1:D]

        r₀ = zeros(ST, S, D)
        r₁ = zeros(ST, S, D)
        m = zeros(ST, R, S, D)
        a = zeros(ST, R, S, D)

        dqdWc = zeros(ST, R, S, D)
        dqdbc = zeros(ST, R, S, D)
        dvdWc = zeros(ST, R, S, D)
        dvdbc = zeros(ST, R, S, D)

        dqdWr₁ = zeros(ST, S, D)
        dqdWr₀ = zeros(ST, S, D)
        dqdbr₁ = zeros(ST, S, D)
        dqdbr₀ = zeros(ST, S, D)

        stage_values = zeros(ST, record_grid_points, D)
        network_labels = zeros(ST, N + 1, D)

        new(x, q̄, p̄, q̃, p̃, ṽ, f̃, s̃, X, Q, P, V, F, ps, r₀, r₁, m, a,
            dqdWc, dqdbc, dvdWc, dvdbc, dqdWr₁, dqdWr₀, dqdbr₁, dqdbr₀,
            stage_values, network_labels)
    end
end

function GeometricIntegrators.Integrators.Cache{ST}(problem::AbstractProblemIODE, method::NonLinear_OneLayer_GML; kwargs...) where {ST}
    NonLinear_OneLayer_GMLCache{ST, nbasis(method), nnodes(method),
        extrapolation_substep(method),}(initial_conditions(problem);
        record_grid_points = method.record_grid_points, kwargs...)
end

@inline GeometricIntegrators.Integrators.CacheType(ST, problem::AbstractProblemIODE, method::NonLinear_OneLayer_GML) =
    NonLinear_OneLayer_GMLCache{ST, nbasis(method), nnodes(method),
        extrapolation_substep(method),}

function initial_params!(int::GeometricIntegrator{<:NonLinear_OneLayer_GML}, initialParams::TrainingMethod, sol)
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
        loss(p) = mse_loss(network_inputs, labels, NN, network_params(p, PNN.params))
        opt = GeometricOptimizers.Optimizer(ps_flat, loss;
            algorithm  = GeometricOptimizers.Adam(PT),
            linesearch = GeometricOptimizers.DecayingStatic(PT; η₁ = PT(1e-3), η₂ = PT(5e-5), n = nepochs))
        state = GeometricOptimizers.OptimizerState(GeometricOptimizers.Adam(PT), ps_flat)
        for ep in 1:nepochs
            GeometricOptimizers.increase_iteration_number!(state)
            GeometricOptimizers.solver_step!(ps_flat, state, opt)
            GeometricOptimizers.update!(state, opt, ps_flat)
        end
        @debug "dimension" k "final loss:" mse_loss(network_inputs, labels, NN, PNN.params) "in" nepochs "epochs"

        for i in 1:S
            x[D*(i-1)+k] = PNN.params[2].W[i]
            x[D*(S+1)+D*(i-1)+k] = PNN.params[1].W[i]
            x[D*(S+1+S)+D*(i-1)+k] = PNN.params[1].b[i]
        end
    end
    @debug "Initial guess from network training" x
end

function GeometricIntegrators.Integrators.components!(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:NonLinear_OneLayer_GML}) where {ST}
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

    # compute coefficients
    for d in 1:D
        r₀[:, d] = (NN.layers[1])([zero(ST)], ps[d][1])
        r₁[:, d] = (NN.layers[1])([one(ST)], ps[d][1])
        for j in eachindex(quad_nodes)
            m[j, :, d] = (NN.layers[1])([quad_nodes[j]], ps[d][1])
            a[j, :, d] = DVDθ([quad_nodes[j]], NeuralNetworkParameters(ps[d])).L2.W[:]
        end
    end

    # compute the derivatives of the coefficients on the quadrature nodes and at the boundaries
    for d in 1:D
        for j in eachindex(quad_nodes)
            g = DQDθ([quad_nodes[j]], NeuralNetworkParameters(ps[d]))
            dqdWc[j, :, d] = g.L1.W[:]
            dqdbc[j, :, d] = g.L1.b[:]

            gv = DVDθ([quad_nodes[j]], NeuralNetworkParameters(ps[d]))
            dvdWc[j, :, d] = gv.L1.W[:]
            dvdbc[j, :, d] = gv.L1.b[:]
        end

        g0 = DQDθ([zero(ST)], NeuralNetworkParameters(ps[d]))
        dqdWr₀[:, d] = g0.L1.W[:]
        dqdbr₀[:, d] = g0.L1.b[:]

        g1 = DQDθ([one(ST)], NeuralNetworkParameters(ps[d]))
        dqdWr₁[:, d] = g1.L1.W[:]
        dqdbr₁[:, d] = g1.L1.b[:]
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


function GeometricIntegrators.Integrators.residual!(b::Vector{ST}, sol, params, int::GeometricIntegrator{<:NonLinear_OneLayer_GML}) where {ST}
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





function record_finer_solution!(sol, int::GeometricIntegrator{<:NonLinear_OneLayer_GML})
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


