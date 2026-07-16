struct NonLinear_DenseNet_GML{T, NNODES, basisType <: Basis{T},
                               ET <: Extrapolation,
                               IPMT <: InitialParametersMethod} <: DenseNetMethod
    common :: NetworkIntegratorCore{T, NNODES, basisType, ET, IPMT}

    function NonLinear_DenseNet_GML(basis::Basis{T}, quadrature::QuadratureRule{T};
        extrapolation_substep :: Int  = 10,
        training_epochs       :: Int  = 50000,
        initial_trajectory_method :: ET   = IntegratorExtrapolation(),
        initial_guess_method  :: IPMT = LSGD()) where {T, ET, IPMT}
        common = NetworkIntegratorCore(basis, quadrature;
            extrapolation_substep=extrapolation_substep,
            training_epochs=training_epochs,
            initial_trajectory_method=initial_trajectory_method,
            initial_guess_method=initial_guess_method)
        new{T, QuadratureRules.nnodes(quadrature), typeof(basis), ET, IPMT}(common)
    end
end

struct NonLinear_DenseNet_GMLCache{ST,S₁,S,NP,R,N} <: NetworkIntegratorCache{ST}
    x::Vector{ST}

    q̄::Vector{ST}
    p̄::Vector{ST}

    q̃::Vector{ST}
    p̃::Vector{ST}
    ṽ::Vector{ST}
    f̃::Vector{ST}
    s̃::Vector{ST}
    q0::Vector{ST}

    X::Vector{Vector{ST}}
    Q::Vector{Vector{ST}}
    P::Vector{Vector{ST}}
    V::Vector{Vector{ST}}
    F::Vector{Vector{ST}}

    ps::Vector{@NamedTuple{L1::@NamedTuple{W::Matrix{ST}, b::Vector{ST}},L2::@NamedTuple{W::Matrix{ST}, b::Vector{ST}},
        L3::@NamedTuple{W::Matrix{ST}}}}

    g0_params::Matrix{ST}
    g1_params::Matrix{ST}

    dqdθc::Array{ST,3}
    dvdθc::Array{ST,3}

    stage_values::Matrix{ST}
    network_labels::Matrix{ST}

    function NonLinear_DenseNet_GMLCache{ST,S₁,S,NP,R,N}(ics) where {ST,S₁,S,NP,R,N}
        D = length(vec(ics.q))
        x = zeros(ST,D*(NP+1))

        q̄ = zeros(ST,D)
        p̄ = zeros(ST,D)

        # create temporary vectors
        q̃ = zeros(ST,D)
        p̃ = zeros(ST,D)
        ṽ = zeros(ST,D)
        f̃ = zeros(ST,D)
        s̃ = zeros(ST,D)

        q0 = zeros(ST,D)

        # create internal stage vectors
        X = create_internal_stage_vector(ST,D,S)
        Q = create_internal_stage_vector(ST,D,R)
        P = create_internal_stage_vector(ST,D,R)
        V = create_internal_stage_vector(ST,D,R)
        F = create_internal_stage_vector(ST,D,R)

        # create hidden layer parameter vectors
        ps = [(L1 = (W = zeros(ST,S₁,1),b = zeros(ST,S₁)),
                L2 = (W = zeros(ST,S,S₁),b = zeros(ST,S)),
                L3 = (W = zeros(ST,1,S),))  for k in 1:D]

        g0_params = zeros(ST,NP, D)
        g1_params = zeros(ST,NP, D)

        dqdθc = zeros(ST, R, NP, D)
        dvdθc = zeros(ST, R, NP, D)

        stage_values = zeros(ST, 41, D)
        network_labels = zeros(ST, N+1, D)

        return new(x, q̄, p̄, q̃, p̃, ṽ, f̃, s̃, q0, X, Q, P, V, F, ps,
        g0_params, g1_params, dqdθc, dvdθc,
        stage_values,network_labels)
    end
end

function GeometricIntegrators.Integrators.Cache{ST}(problem::AbstractProblemIODE, method::NonLinear_DenseNet_GML; kwargs...) where {ST}
    NonLinear_DenseNet_GMLCache{ST, method.basis.S₁, method.basis.S, method.basis.NP,
        nnodes(method), extrapolation_substep(method)}(initial_conditions(problem); kwargs...)
end

@inline GeometricIntegrators.Integrators.CacheType(ST, problem::AbstractProblemIODE, method::NonLinear_DenseNet_GML) =
    NonLinear_DenseNet_GMLCache{ST, method.basis.S₁, method.basis.S, method.basis.NP,
        nnodes(method), extrapolation_substep(method)}

function initial_trajectory!(sol, history, params, int::GeometricIntegrator{<:NonLinear_DenseNet_GML}, initial_trajectory::HermiteExtrapolation)
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local x = nlsolution(int)
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local NP = method(int).basis.NP

    # TODO: here we should not initialise with the solution q but with the degree of freedom x,
    # obtained e.g. from an L2 projection of q onto the basis

    for i in eachindex(network_inputs)
        soltmp = (
            t=sol.t + (network_inputs[i]-1) * timestep(int),
            q=cache(int).q̃,
            p=cache(int).p̃,
            q̇=cache(int).ṽ,
            ṗ=cache(int).f̃,
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
        q̇=cache(int).ṽ,
        ṗ=cache(int).f̃,
    )
    solutionstep!(soltmp, history, problem(int), iguess(int))

    for k in 1:D
        x[D*NP+k] = cache(int).p̃[k]
    end
end

function initial_trajectory!(sol, history, params, int::GeometricIntegrator{<:NonLinear_DenseNet_GML}, initial_trajectory::IntegratorExtrapolation)
    local network_labels = cache(int).network_labels
    local integrator = default_iguess_integrator(method(int))
    local h = timestep(int)
    local extrapolation_substep = method(int).extrapolation_substep
    local D = length(cache(int).q̃)
    local problem = int.problem
    local S = method(int).basis.S
    local x = nlsolution(int)
    local NP = method(int).basis.NP

    tem_ode = similar(problem, [zero(h), h], h / extrapolation_substep, (q=StateVariable(sol.q[:]), p=StateVariable(sol.p[:])))
    tem_sol = integrate(tem_ode, integrator)

    for k in 1:D
        network_labels[:, k] = tem_sol.q[:, k]#[1].s
        cache(int).q̃[k] = tem_sol.q[:, k][end]
        cache(int).p̃[k] = tem_sol.p[:, k][end]
        x[D*NP+k] = cache(int).p̃[k]
    end
end

function initial_params!(int::GeometricIntegrator{<:NonLinear_DenseNet_GML}, InitialParams::TrainingMethod, sol)
    local D = length(cache(int).q̃)
    local S = int.method.basis.S
    local S₁ = int.method.basis.S₁

    local x = nlsolution(int)
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local extrapolation_substep = method(int).extrapolation_substep
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local nepochs = method(int).training_epochs
    local backend = method(int).basis.backend
    local NP = method(int).basis.NP

    Random.seed!(42)

    for k in 1:D
        @debug "For dimension" k network_labels[:, k]

        labels = reshape(network_labels[:,k],1,extrapolation_substep+1)

        PNN = GeometricMachineLearning.NeuralNetwork(NN)
        # opt = GeometricMachineLearning.Optimizer(AdamOptimizer(0.001, 0.9, 0.99, 1e-8), ps[k])
        opt = GeometricMachineLearning.Optimizer(GeometricMachineLearning.AdamOptimizerWithDecay(nepochs,1e-3, 5e-5), PNN)
        err = 0
        λ = GeometricMachineLearning.GlobalSection(PNN.params)
        for ep in 1:nepochs
            gs = Zygote.gradient(p -> mse_loss(network_inputs,labels,PNN,p),PNN.params)[1]
            GeometricMachineLearning.optimization_step!(opt,λ, PNN.params, gs)
            err = mse_loss(network_inputs,labels,PNN,PNN.params)

            if err < 5e-8
                @debug "dimension $k,final loss: $err by $ep epochs"
                break
            elseif ep == nepochs
                @debug "dimension $k,final loss: $err by $ep epochs"
            end
        end

        ps[k] = PNN.params[:]

        for i in 1:S₁
            x[(k-1)*NP+1:(k-1)*NP+S₁] = ps[k].L1.W[:,1]
            x[(k-1)*NP+S₁+1:(k-1)*NP+S₁+S₁] = ps[k].L1.b[:]
            x[(k-1)*NP+S₁+S₁ + (i-1)*S+1:(k-1)*NP+S₁+S₁+i*S] = ps[k].L2.W[:,i]
            x[(k-1)*NP+2*S₁+S*S₁+1:(k-1)*NP+2*S₁+S*S₁+S] = ps[k].L2.b[:]
            x[(k-1)*NP+2*S₁+S*S₁+S+1:(k-1)*NP+2*S₁+S*S₁+S+S] = ps[k].L3.W[1, :]
        end
    end

    @debug "network parameters" ps
    @debug "Initial guess from network training" x

end

function initial_params!(int::GeometricIntegrator{<:NonLinear_DenseNet_GML}, InitialParams::LSGD, sol)
    local D = length(cache(int).q̃)
    local S = int.method.basis.S
    local S₁ = int.method.basis.S₁

    local x = nlsolution(int)
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local extrapolation_substep = method(int).extrapolation_substep
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local nepochs = method(int).training_epochs
    local backend = method(int).basis.backend
    local NP = method(int).basis.NP

    Random.seed!(42)
    for k in 1:D
        @debug "network labels for dimension $k" network_labels[:,k]

        labels = reshape(network_labels[:,k],1,extrapolation_substep+1)

        PNN = NeuralNetwork(NN)
        PNN.params.L1.W[:], PNN.params.L1.b[:] = box_init_plain(1, S₁)
        PNN.params.L2.W[:], PNN.params.L2.b[:] = box_init_plain(S₁, S)
        PNN.params.L3.W[:], _ = box_init_plain(S, 1)
        tem_ps = (L1 = PNN.params.L1, L2 = PNN.params.L2)
        opt = GeometricMachineLearning.Optimizer(GeometricMachineLearning.GradientOptimizer(.001), tem_ps)
        err = 0
        λ = GeometricMachineLearning.GlobalSection(tem_ps)

        for ep in 1:nepochs
            Φ = AbstractNeuralNetworks.Chain(NN.layers[1:end-1]...)(network_inputs,tem_ps)
            # Φ = NN(network_inputs, PNN.params)
            # PNN.params.L3.W[:] = labels/Φ
            PNN.params.L3.W[:] = (Φ' \ labels')'
            gs = Zygote.gradient(p -> lsgd_loss(network_inputs,labels,NN,p),PNN.params)[1]
            tem_ps = (L1 = PNN.params.L1, L2 = PNN.params.L2)
            tem_gs = (L1 = gs.L1, L2 = gs.L2)
            GeometricMachineLearning.optimization_step!(opt,λ, tem_ps, tem_gs)
            err = lsgd_loss(network_inputs,labels,NN,PNN.params)
              if err < 5e-5
                @debug "dimension $k,final loss: $err by $ep epochs"
                break
            elseif ep == nepochs
                @debug "dimension $k,final loss: $err by $ep epochs"
            end
        end

        ps[k] = PNN.params[:]

        for i in 1:S₁
            x[(k-1)*NP+1:(k-1)*NP+S₁] = ps[k].L1.W[:,1]
            x[(k-1)*NP+S₁+1:(k-1)*NP+S₁+S₁] = ps[k].L1.b[:]
            x[(k-1)*NP+S₁+S₁ + (i-1)*S+1:(k-1)*NP+S₁+S₁+i*S] = ps[k].L2.W[:,i]
            x[(k-1)*NP+2*S₁+S*S₁+1:(k-1)*NP+2*S₁+S*S₁+S] = ps[k].L2.b[:]
            x[(k-1)*NP+2*S₁+S*S₁+S+1:(k-1)*NP+2*S₁+S*S₁+S+S] = ps[k].L3.W[1, :]
        end
    end

    @debug "network parameters" ps
    @debug "Initial guess from network training" x

end

function GeometricIntegrators.Integrators.components!(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:NonLinear_DenseNet_GML}) where {ST}
    # set some local variables for convenience and clarity
    local D = length(cache(int).q̃)
    local S₁ = int.method.basis.S₁
    local S = int.method.basis.S
    local σ = int.method.basis.activation
    local R = length(method(int).c)
    local NP = method(int).basis.NP

    local quad_nodes = QuadratureRules.nodes(int.method.quadrature)
    local C = cache(int, ST)

    local q = cache(int, ST).q̃
    local q0 = cache(int, ST).q0
    local p = cache(int, ST).p̃
    local Q = cache(int, ST).Q
    local V = cache(int, ST).V

    local NN = method(int).basis.NN
    local ps = cache(int, ST).ps

    local g0_params = cache(int, ST).g0_params
    local g1_params = cache(int, ST).g1_params
    local dqdθc = cache(int, ST).dqdθc
    local dvdθc = cache(int, ST).dvdθc

    local DVDθ = method(int).basis.dvdθ
    local DQDθ = method(int).basis.dqdθ
    local V_func = method(int).basis.V_func

    # copy x to X and bias
    for i in 1:S
        for d in 1:D
            C.X[i][d] = x[D*(i-1)+d]
        end
    end

    # copy x to p # momenta
    for k in eachindex(p)
        p[k] = x[D*NP+k]
    end

    # Fill ps from x (network parameters)
    for d in 1:D
        for i in 1:S₁
            ps[d].L1.W[:,1] = x[(d-1)*NP+1:(d-1)*NP+S₁]
            ps[d].L1.b[:] = x[(d-1)*NP+S₁+1:(d-1)*NP+S₁+S₁]

            ps[d].L2.W[:,i] = x[(d-1)*NP+S₁+S₁+(i-1)*S+1:(d-1)*NP+S₁+S₁+i*S]

            ps[d].L2.b[:] = x[(d-1)*NP+2*S₁+S*S₁+1:(d-1)*NP+2*S₁+S*S₁+S]
            ps[d].L3.W[1, :] = x[(d-1)*NP+2*S₁+S*S₁+S+1:(d-1)*NP+2*S₁+S*S₁+S+S]
        end
    end

    # compute coefficients
    for d in 1:D
        # intermidiate_ps = (L1 = ps[d].L1, L2 = ps[d].L2)
        # r₀[:,d] = AbstractNeuralNetworks.Chain(NN.layers[1:end-1]...)([0.0],intermidiate_ps)
        # r₁[:,d] = AbstractNeuralNetworks.Chain(NN.layers[1:end-1]...)([1.0],intermidiate_ps)
        # for i in 1:S
        g0 = DQDθ([zero(ST)], NeuralNetworkParameters(ps[d]))
        g0_params[:,d] = flatten_params(g0)

        g1 = DQDθ([one(ST)], NeuralNetworkParameters(ps[d]))
        g1_params[:,d] = flatten_params(g1)

        for j in eachindex(quad_nodes)
            g = DQDθ([quad_nodes[j]], NeuralNetworkParameters(ps[d]))
            dqdθc[j, :, d] = flatten_params(g)

            gv = DVDθ([quad_nodes[j]], NeuralNetworkParameters(ps[d]))[1,1]
            dvdθc[j, :, d] = flatten_params(gv)
        end
    end

    # compute Q q at quaadurature points
    for i in eachindex(Q)
        for d in eachindex(Q[i])
            Q[i][d] = NN([quad_nodes[i]], NeuralNetworkParameters(ps[d]))[1]
        end
    end

    # compute q[t_{n+1}]
    for d in eachindex(q)
        q[d] = NN([one(ST)], NeuralNetworkParameters(ps[d]))[1]
    end

    for d in eachindex(q)
        q0[d] = NN([zero(ST)], NeuralNetworkParameters(ps[d]))[1]
    end

    # compute V volicity at quadrature points
    for i in eachindex(V)
        for d in eachindex(V[i])
            V[i][d] = V_func([quad_nodes[i]],NeuralNetworkParameters(ps[d]))[1] / timestep(int)
        end
    end

    # compute P=ϑ(Q,V) pl/pv and F=f(Q,V) pl/px
    for i in eachindex(C.Q, C.V, C.P, C.F)
        tᵢ = sol.t + timestep(int) * (method(int).c[i] - 1)
        equations(int).ϑ(C.P[i], tᵢ, C.Q[i], C.V[i], params)
        equations(int).f(C.F[i], tᵢ, C.Q[i], C.V[i], params)
    end
end

function GeometricIntegrators.Integrators.residual!(b::Vector{ST},sol, params, int::GeometricIntegrator{<:NonLinear_DenseNet_GML}) where {ST}
    local D = length(cache(int).q̃)
    local S = int.method.basis.S
    local S₁ = int.method.basis.S₁
    local R = length(method(int).c)

    local q̄ = sol.q
    local p̄ = sol.p
    local p̃ = cache(int, ST).p̃ #initial guess for p[t_{n+1}]
    local P = cache(int, ST).P # p at internal stages/quad_nodes
    local F = cache(int, ST).F
    local X = cache(int, ST).X

    local g0_params = cache(int, ST).g0_params
    local g1_params = cache(int, ST).g1_params
    local dqdθc = cache(int, ST).dqdθc
    local dvdθc = cache(int, ST).dvdθc
    local NP = method(int).basis.NP
    local q0 = cache(int, ST).q0

    # compute b = - [(P-AF)]
    for k in 1:D
        for i in 1:NP
            z = zero(ST)
            for j in 1:R
                z += method(int).b[j] * F[j][k] * dqdθc[j,i,k] * timestep(int)
                z += method(int).b[j] * P[j][k] * dvdθc[j,i,k]
            end
            b[(k-1)*NP+i] = (g1_params[i,k] * p̃[k] - g0_params[i,k] * p̄[k]) - z
        end
    end # the residual in actual action, vatiation with respect to Q_{n,i}

    for k in 1:D
        b[D*NP+k] = q̄[k] - q0[k] # the continue constraint from hamilton pontryagin principle
    end

end





function record_finer_solution!(sol,int::GeometricIntegrator{<:NonLinear_DenseNet_GML})
    local x = nlsolution(int)
    local stage_values = cache(int).stage_values
    local D = length(cache(int).q̃)
    local S = int.method.basis.S
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local S₁ = method(int).basis.S₁
    local NP = method(int).basis.NP

    network_inputs = reshape(collect(0:1/40:1),1,41)

    @debug "solution x after solving by Newton" x

    for d in 1:D
        for i in 1:S₁
            ps[d].L1.W[:,1] = x[(d-1)*NP+1:(d-1)*NP+S₁]
            ps[d].L1.b[:] = x[(d-1)*NP+S₁+1:(d-1)*NP+S₁+S₁]

            ps[d].L2.W[:,i] = x[(d-1)*NP+S₁+S₁ + (i-1)*S+1:(d-1)*NP+S₁+S₁+i*S]

            ps[d].L2.b[:] = x[(d-1)*NP+2*S₁+S*S₁+1:(d-1)*NP+2*S₁+S*S₁+S]
            ps[d].L3.W[1, :] = x[(d-1)*NP+2*S₁+S*S₁+S+1:(d-1)*NP+2*S₁+S*S₁+S+S]
        end
        stage_values[:,d] = NN(network_inputs,ps[d])[:]
    end

    @debug "stages prediction after solving" stage_values
    @debug "sol from this step q:", sol.q, "p:", sol.p

end

