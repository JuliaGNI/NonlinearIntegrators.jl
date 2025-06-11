struct NonLinear_DenseNet_GML{T, NNODES, basisType <: Basis{T},ET<:IntegratorExtrapolation,IPMT<:InitialParametersMethod} <: DenseNetMethod
    basis::basisType
    quadrature::QuadratureRule{T,NNODES}

    b::SVector{NNODES,T}
    c::SVector{NNODES,T}

    nstages::Int
    show_status::Bool
    network_inputs::Matrix{T}
    training_epochs::Int

    initial_trajectory::ET
    initial_guess_method::IPMT

    function NonLinear_DenseNet_GML(basis::Basis{T}, quadrature::QuadratureRule{T};
        nstages::Int = 10,show_status::Bool=true,training_epochs::Int=50000,
        initial_trajectory::ET=IntegratorExtrapolation(),
        initial_guess_method::IPMT=LSGD()) where {T, ET, IPMT}
        # get number of quadrature nodes and number of basis functions
        NNODES = QuadratureRules.nnodes(quadrature)

        # get quadrature nodes and weights
        quad_weights = QuadratureRules.weights(quadrature)
        quad_nodes = QuadratureRules.nodes(quadrature)
        network_inputs = reshape(collect(0:1/nstages:1),1,nstages+1)

        new{T,NNODES, typeof(basis),ET,IPMT}(basis, quadrature, quad_weights, quad_nodes,nstages, show_status, network_inputs, training_epochs,
            initial_trajectory, initial_guess_method)
    end
end

CompactBasisFunctions.basis(method::NonLinear_DenseNet_GML) = method.basis
quadrature(method::NonLinear_DenseNet_GML) = method.quadrature
CompactBasisFunctions.nbasis(method::NonLinear_DenseNet_GML) = method.basis.S
nnodes(method::NonLinear_DenseNet_GML) = QuadratureRules.nnodes(method.quadrature)
activation(method::NonLinear_DenseNet_GML) = method.basis.activation
nstages(method::NonLinear_DenseNet_GML) = method.nstages

isexplicit(::Union{NonLinear_DenseNet_GML, Type{<:NonLinear_DenseNet_GML}}) = false
isimplicit(::Union{NonLinear_DenseNet_GML, Type{<:NonLinear_DenseNet_GML}}) = true
issymmetric(::Union{NonLinear_DenseNet_GML, Type{<:NonLinear_DenseNet_GML}}) = missing
issymplectic(::Union{NonLinear_DenseNet_GML, Type{<:NonLinear_DenseNet_GML}}) = true


default_solver(::NonLinear_DenseNet_GML) = Newton()
# default_iguess(::NonLinear_DenseNet_GML) = HermiteExtrapolation()# HarmonicOscillator
default_iguess(::NonLinear_DenseNet_GML) = MidpointExtrapolation()#CoupledHarmonicOscillator
default_iguess_integrator(::NonLinear_DenseNet_GML) = ImplicitMidpoint()

struct NonLinear_DenseNet_GMLCache{ST,D,S₁,S,NP,R,N} <: IODEIntegratorCache{ST,D}
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

    function NonLinear_DenseNet_GMLCache{ST,D,S₁,S,NP,R,N}() where {ST,D,S₁,S,NP,R,N}
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

        stage_values = zeros(ST, N, D)
        network_labels = zeros(ST, N+1, D)

        return new(x, q̄, p̄, q̃, p̃, ṽ, f̃, s̃, q0, X, Q, P, V, F, ps, 
        g0_params, g1_params, dqdθc, dvdθc,
        stage_values,network_labels)
    end
end

function GeometricIntegrators.Integrators.reset!(cache::NonLinear_DenseNet_GMLCache, t, q, p)
    copyto!(cache.q̄, q)
    copyto!(cache.p̄, p)
end

GeometricIntegrators.Integrators.nlsolution(cache::NonLinear_DenseNet_GMLCache) = cache.x

function GeometricIntegrators.Integrators.Cache{ST}(problem::AbstractProblemIODE, method::NonLinear_DenseNet_GML; kwargs...) where {ST}
    NonLinear_DenseNet_GMLCache{ST, ndims(problem), method.basis.S₁,method.basis.S,method.basis.NP, nnodes(method),nstages(method)}(; kwargs...)
end

@inline GeometricIntegrators.Integrators.CacheType(ST, problem::AbstractProblemIODE, method::NonLinear_DenseNet_GML) = NonLinear_DenseNet_GMLCache{ST, ndims(problem), method.basis.S₁,method.basis.S,method.basis.NP, nnodes(method),nstages(method)}

@inline function Base.getindex(c::NonLinear_DenseNet_GMLCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end

function GeometricIntegrators.Integrators.initial_guess!(sol, history, params,int::GeometricIntegrator{<:NonLinear_DenseNet_GML}) 
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local show_status = method(int).show_status 
    local initial_trajectory = method(int).initial_trajectory
    local initial_guess_method = method(int).initial_guess_method

    # choose initial guess method based on the value of h
    initial_trajectory!(sol, history, params, int, initial_trajectory)
    
    if show_status
        print("\n network inputs")
        print(network_inputs)

        print("\n network labels from initial guess methods")
        print(network_labels)
    end

    initial_params!(int, initial_guess_method)
end

function initial_trajectory!(sol, history, params, int::GeometricIntegrator{<:NonLinear_DenseNet_GML}, initial_trajectory::HermiteExtrapolation)
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local D = ndims(int)
    local h = int.problem.tstep

    for i in eachindex(network_inputs)
        initialguess!(solstep(int).t̄+network_inputs[i]*h, cache(int).q̃, cache(int).p̃, solstep(int), int.problem, int.iguess)
        for k in 1:D
            network_labels[i,k] = cache(int).q̃[k]
        end
    end
    network_labels[1,:] = solstep(int).q #safe check for MidpointExtrapolation
end

function initial_trajectory!(sol, history, params, int::GeometricIntegrator{<:NonLinear_DenseNet_GML}, initial_trajectory::IntegratorExtrapolation)
    local network_labels = cache(int).network_labels
    local integrator = default_iguess_integrator(method(int))
    local h = int.problem.tstep
    local nstages = method(int).nstages
    local D = ndims(int)
    local problem = int.problem
    local S = method(int).basis.S
    local x = nlsolution(int)
    local NP = method(int).basis.NP

    tem_ode = similar(problem, [0.0, h], h / nstages, (q=StateVariable(sol.q[:]), p=StateVariable(sol.p[:])))
    tem_sol = integrate(tem_ode, integrator)

    for k in 1:D
        network_labels[:, k] = tem_sol.q[:, k]#[1].s
        cache(int).q̃[k] = tem_sol.q[:, k][end]
        cache(int).p̃[k] = tem_sol.p[:, k][end]
        x[D*NP+k] = cache(int).p̃[k]
    end
end 

function initial_params!(int::GeometricIntegrator{<:NonLinear_DenseNet_GML}, InitialParams::TrainingMethod)
    local D = ndims(int)
    local S = int.method.basis.S
    local S₁ = int.method.basis.S₁

    local show_status = method(int).show_status 
    local x = nlsolution(int)
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local nstages = method(int).nstages
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local nepochs = method(int).training_epochs
    local backend = method(int).basis.backend
    local NP = method(int).basis.NP
    for k in 1:D
        if show_status
            print("\n network lables for dimension $k \n")
            print(network_labels[:,k])
        end

        labels = reshape(network_labels[:,k],1,nstages+1)

        PNN = GeometricMachineLearning.NeuralNetwork(NN)
        # opt = GeometricMachineLearning.Optimizer(AdamOptimizer(0.001, 0.9, 0.99, 1e-8), ps[k])
        opt = GeometricMachineLearning.Optimizer(GeometricMachineLearning.AdamOptimizerWithDecay(nepochs,1e-3, 5e-5), PNN)
        err = 0
        λ = GeometricMachineLearning.GlobalSection(PNN.params)
        for ep in 1:nepochs
            gs = Zygote.gradient(p -> mse_loss(network_inputs,labels,PNN,p)[1],PNN.params)[1]
            GeometricMachineLearning.optimization_step!(opt,λ, PNN.params, gs)
            err = mse_loss(network_inputs,labels,PNN,PNN.params)[1]

            if err < 5e-8
                show_status ? print("\n dimension $k,final loss: $err by $ep epochs") : nothing
                break
            elseif ep == nepochs
                show_status ? print("\n dimension $k,final loss: $err by $ep epochs") : nothing
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


    if show_status
        print("\n network parameters \n")
        print(ps)

        print("\n initial guess x from network training \n")
        print(x)
    end

end

function initial_params!(int::GeometricIntegrator{<:NonLinear_DenseNet_GML}, InitialParams::LSGD)
    local D = ndims(int)
    local S = int.method.basis.S
    local S₁ = int.method.basis.S₁

    local show_status = method(int).show_status 
    local x = nlsolution(int)
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local nstages = method(int).nstages
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local nepochs = method(int).training_epochs
    local backend = method(int).basis.backend
    local NP = method(int).basis.NP
    for k in 1:D
        if show_status
            print("\n network lables for dimension $k \n")
            print(network_labels[:,k])
        end

        labels = reshape(network_labels[:,k],1,nstages+1)

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
                show_status ? print("\n dimension $k,final loss: $err by $ep epochs") : nothing
                break
            elseif ep == nepochs
                show_status ? print("\n dimension $k,final loss: $err by $ep epochs") : nothing
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


    if show_status
        print("\n network parameters \n")
        print(ps)

        print("\n initial guess x from network training \n")
        print(x)
    end

end

function GeometricIntegrators.Integrators.components!(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:NonLinear_DenseNet_GML}) where {ST}
    # set some local variables for convenience and clarity
    local D = ndims(int)
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
        g0 = DQDθ([0.0], NeuralNetworkParameters(ps[d]))
        g0_params[:,d] = flatten_params(g0)

        g1 = DQDθ([1.0], NeuralNetworkParameters(ps[d]))
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
        q[d] = NN([1.0], NeuralNetworkParameters(ps[d]))[1]
    end

    for d in eachindex(q)
        q0[d] = NN([0.0], NeuralNetworkParameters(ps[d]))[1]
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
    local D = ndims(int)
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

function GeometricIntegrators.Integrators.residual!(b::AbstractVector{ST}, x::AbstractVector{ST},sol, params, int::GeometricIntegrator{<:NonLinear_DenseNet_GML}) where {ST}
    @assert axes(x) == axes(b)

    # compute stages from nonlinear solver solution x
    GeometricIntegrators.Integrators.components!(x, sol, params, int)

    # compute residual vector
    GeometricIntegrators.Integrators.residual!(b, sol, params, int)
end

function GeometricIntegrators.Integrators.update!(sol, params, int::GeometricIntegrator{<:NonLinear_DenseNet_GML}, DT)
    sol.q .= cache(int, DT).q̃
    sol.p .= cache(int, DT).p̃
end

function GeometricIntegrators.Integrators.update!(sol, params, x::AbstractVector{DT}, int::GeometricIntegrator{<:NonLinear_DenseNet_GML}) where {DT}
    # compute vector field at internal stages
    GeometricIntegrators.Integrators.components!(x, sol, params, int)

    # compute final update
    GeometricIntegrators.Integrators.update!(sol, params, int, DT)
end

function GeometricIntegrators.Integrators.integrate_step!(sol, history, params,int::GeometricIntegrator{<:NonLinear_DenseNet_GML, <:AbstractProblemIODE})
    # call nonlinear solver
    solve!(nlsolution(int), (b, x) -> GeometricIntegrators.Integrators.residual!(b, x, sol, params, int), solver(int))

    # print solver status
    # print_solver_status(int.solver.status, int.solver.params)

    # check if solution contains NaNs or error bounds are violated
    # check_solver_status(int.solver.status, int.solver.params)

    # compute final update
    GeometricIntegrators.Integrators.update!(sol, params, nlsolution(int), int)

    #compute the trajectory after solving by newton method
    stages_compute!(sol, int)

    #check for NaNs
    # if sum(isnan.(cache(int).q̃[:])) > 0 
    #     error("NaN value encountered, terminating program.")
    # end
end

function stages_compute!(sol,int::GeometricIntegrator{<:NonLinear_DenseNet_GML})
    local x = nlsolution(int)
    local stage_values = cache(int).stage_values
    local network_inputs = method(int).network_inputs
    local D = ndims(int)
    local S = int.method.basis.S
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local show_status = method(int).show_status
    local S₁ = method(int).basis.S₁
    local NP = method(int).basis.NP

    if show_status
        print("\n solution x after solving by Newton \n")
        print(x)
    end

    for d in 1:D
        for i in 1:S₁
            ps[d].L1.W[:,1] = x[(d-1)*NP+1:(d-1)*NP+S₁]
            ps[d].L1.b[:] = x[(d-1)*NP+S₁+1:(d-1)*NP+S₁+S₁]

            ps[d].L2.W[:,i] = x[(d-1)*NP+S₁+S₁ + (i-1)*S+1:(d-1)*NP+S₁+S₁+i*S]

            ps[d].L2.b[:] = x[(d-1)*NP+2*S₁+S*S₁+1:(d-1)*NP+2*S₁+S*S₁+S]
            ps[d].L3.W[1, :] = x[(d-1)*NP+2*S₁+S*S₁+S+1:(d-1)*NP+2*S₁+S*S₁+S+S]
        end
        stage_values[:,d] = NN(network_inputs,ps[d])[2:end]
    end

    if show_status
        print("\n stages prediction after solving \n")
        print(stage_values)
    end

end