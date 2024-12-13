struct NonLinear_OneLayer_GML{T,NBASIS,NNODES,basisType<:Basis{T}, ET <: IntegratorExtrapolation, IPMT <: InitialParametersMethod} <: OneLayerMethod
    basis::basisType
    quadrature::QuadratureRule{T,NNODES}

    b::SVector{NNODES,T}
    c::SVector{NNODES,T}

    nstages::Int
    show_status::Bool
    network_inputs::Matrix{T}

    initial_trajectory::ET
    initial_guess_method::IPMT

    training_epochs::Int
    problem_initial_hamitltonian::Float64
    use_hamiltonian_loss::Bool

    bias_interval::Vector{T}
    dict_amount::Int
    function NonLinear_OneLayer_GML(basis::Basis{T}, quadrature::QuadratureRule{T};
        nstages::Int=10, show_status::Bool=true, training_epochs::Int=50000, problem_initial_hamitltonian::Float64=0.0, use_hamiltonian_loss::Bool=true, 
        initial_trajectory::ET=IntegratorExtrapolation(), 
        initial_guess_method::IPMT=OGA1d(),
        bias_interval =[-pi, pi],dict_amount=50000) where {T, ET <: Extrapolation, IPMT <: OGA1d}
        # get number of quadrature nodes and number of basis functions
        # initial_trajectory_list = subtypes(Extrapolation)
        # @assert initial_trajectory in initial_trajectory_list "initial_trajectory should be one of $(initial_trajectory_list)"

        # initial_guess_methods_list = subtypes(InitialParametersMethod)
        # @assert initial_guess_method in initial_guess_methods_list "initial_guess_methods should be one of $(initial_guess_methods_list)"

        NNODES = QuadratureRules.nnodes(quadrature)
        NBASIS = basis.S

        # get quadrature nodes and weights
        quad_weights = QuadratureRules.weights(quadrature)
        quad_nodes = QuadratureRules.nodes(quadrature)

        network_inputs = reshape(collect(0:1/nstages:1), 1, nstages + 1)
        new{T,NBASIS,NNODES,typeof(basis), ET, IPMT}(basis, quadrature, quad_weights, quad_nodes, nstages, show_status, network_inputs, initial_trajectory, initial_guess_method,
            training_epochs, problem_initial_hamitltonian, use_hamiltonian_loss,bias_interval,dict_amount)
    end
end

CompactBasisFunctions.basis(method::NonLinear_OneLayer_GML) = method.basis
quadrature(method::NonLinear_OneLayer_GML) = method.quadrature
CompactBasisFunctions.nbasis(method::NonLinear_OneLayer_GML) = method.basis.S
nnodes(method::NonLinear_OneLayer_GML) = QuadratureRules.nnodes(method.quadrature)
activation(method::NonLinear_OneLayer_GML) = method.basis.activation
nstages(method::NonLinear_OneLayer_GML) = method.nstages
show_status(method::NonLinear_OneLayer_GML) = method.show_status
training_epochs(method::NonLinear_OneLayer_GML) = method.training_epochs

isexplicit(::Union{NonLinear_OneLayer_GML,Type{<:NonLinear_OneLayer_GML}}) = false
isimplicit(::Union{NonLinear_OneLayer_GML,Type{<:NonLinear_OneLayer_GML}}) = true
issymmetric(::Union{NonLinear_OneLayer_GML,Type{<:NonLinear_OneLayer_GML}}) = missing
issymplectic(::Union{NonLinear_OneLayer_GML,Type{<:NonLinear_OneLayer_GML}}) = missing

default_solver(::NonLinear_OneLayer_GML) = Newton()
default_iguess(::NonLinear_OneLayer_GML) = IntegratorExtrapolation()#CoupledHarmonicOscillator
default_iparams(::NonLinear_OneLayer_GML) = OGA1d()
# default_iguess_integrator(::NonLinear_OneLayer_GML) =  CGVI(Lagrange(QuadratureRules.nodes(QuadratureRules.GaussLegendreQuadrature(4))),QuadratureRules.GaussLegendreQuadrature(4))

default_iguess_integrator(::NonLinear_OneLayer_GML) = ImplicitMidpoint()

struct NonLinear_OneLayer_GMLCache{ST,D,S,R,N} <: IODEIntegratorCache{ST,D}
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

    ps::Vector{@NamedTuple{L1::@NamedTuple{W::Matrix{ST}, b::Vector{ST}}, 
                L2::@NamedTuple{W::Matrix{ST}}}}

    r₀::VecOrMat{ST}
    r₁::VecOrMat{ST}
    m::Array{ST} 
    a::Array{ST}

    dqdWc::Array{ST}
    dqdbc::Array{ST}
    dvdWc::Array{ST}
    dvdbc::Array{ST}

    dqdWr₁::VecOrMat{ST}
    dqdWr₀::VecOrMat{ST}
    dqdbr₁::VecOrMat{ST}
    dqdbr₀::VecOrMat{ST}

    current_step::Vector{ST}
    stage_values::VecOrMat{ST}
    network_labels::VecOrMat{ST}
    function NonLinear_OneLayer_GMLCache{ST,D,S,R,N}() where {ST,D,S,R,N}
        x = zeros(ST,D*(S+1+2*S)) # Last layer Weight S (no bias for now) + P + hidden layer W (S*S₁) + hidden layer bias S

        q̄ = zeros(ST,D)
        p̄ = zeros(ST,D)

        # create temporary vectors
        q̃ = zeros(ST,D)
        p̃ = zeros(ST,D)
        ṽ = zeros(ST,D)
        f̃ = zeros(ST,D)
        s̃ = zeros(ST,D)

        # create internal stage vectors
        X = create_internal_stage_vector(ST,D,S)
        Q = create_internal_stage_vector(ST,D,R)
        P = create_internal_stage_vector(ST,D,R)
        V = create_internal_stage_vector(ST,D,R)
        F = create_internal_stage_vector(ST,D,R)

        # create parameter vectors
        ps = [(L1 = (W = zeros(ST,S,1),b = zeros(ST,S)),L2 = (W = zeros(ST,1,S),))  for k in 1:D]

        r₀ = zeros(ST, S, D)
        r₁ = zeros(ST, S, D)
        m  = zeros(ST, R, S, D)
        a  = zeros(ST, R, S, D)

        dqdWc=zeros(ST, R, S, D)
        dqdbc=zeros(ST, R, S, D)
        dvdWc=zeros(ST, R, S, D)
        dvdbc=zeros(ST, R, S, D)
        
        dqdWr₁= zeros(ST, S, D)
        dqdWr₀= zeros(ST, S, D)
        dqdbr₁= zeros(ST, S, D)
        dqdbr₀= zeros(ST, S, D)

        current_step = zeros(ST, 1)
        stage_values = zeros(ST, N, D)
        network_labels = zeros(ST, N+1, D)

        new(x, q̄, p̄, q̃, p̃, ṽ, f̃, s̃, X, Q, P, V, F, ps, r₀, r₁, m, a, 
            dqdWc, dqdbc, dvdWc, dvdbc, dqdWr₁, dqdWr₀, dqdbr₁, dqdbr₀,
            current_step,stage_values,network_labels)
    end
end

GeometricIntegrators.Integrators.nlsolution(cache::NonLinear_OneLayer_GMLCache) = cache.x

function GeometricIntegrators.Integrators.Cache{ST}(problem::AbstractProblemIODE, method::NonLinear_OneLayer_GML; kwargs...) where {ST}
    NonLinear_OneLayer_GMLCache{ST, ndims(problem), nbasis(method), nnodes(method),nstages(method)}(; kwargs...)
end

@inline GeometricIntegrators.Integrators.CacheType(ST, problem::AbstractProblemIODE, method::NonLinear_OneLayer_GML) = NonLinear_OneLayer_GMLCache{ST, ndims(problem), nbasis(method), nnodes(method),nstages(method)}

@inline function Base.getindex(c::NonLinear_OneLayer_GMLCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end

function GeometricIntegrators.Integrators.reset!(cache::NonLinear_OneLayer_GMLCache, t, q, p)
    copyto!(cache.q̄, q)
    copyto!(cache.p̄, p)
end

function GeometricIntegrators.Integrators.initial_guess!(sol, history, params, int::GeometricIntegrator{<:NonLinear_OneLayer_GML})
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local show_status = method(int).show_status
    local current_step = cache(int).current_step
    local initial_trajectory = method(int).initial_trajectory
    local initial_guess_method = method(int).initial_guess_method

    show_status ? print("\n current time step: $current_step") : nothing
    current_step[1] += 1

    initial_trajectory!(sol, history, params, int, initial_trajectory)

    if show_status
        print("\n network inputs \n")
        print(network_inputs)

        print("\n network labels from initial guess methods \n")
        print(network_labels')
    end

    initial_params!(int, initial_guess_method)
end

function initial_trajectory!(sol, history, params, int::GeometricIntegrator{<:NonLinear_OneLayer_GML}, initial_trajectory::HermiteExtrapolation)
    local D = ndims(int)
    local S = nbasis(method(int))
    local x = nlsolution(int)

    # TODO: here we should not initialise with the solution q but with the degree of freedom x,
    # obtained e.g. from an L2 projection of q onto the basis

    for i in eachindex(network_inputs)
        soltmp = (
            t=sol.t + network_inputs[i] * timestep(int),
            q=cache(int).q̃,
            p=cache(int).p̃,
            v=cache(int).ṽ,
            f=cache(int).f̃,
        )
        solutionstep!(soltmp, history, problem(int), iguess(int))

        for k in 1:D
            x[D*(i-1)+k] = cache(int).q̃[k]
        end
    end

    soltmp = (
        t=sol.t,
        q=cache(int).q̃,
        p=cache(int).p̃,
        v=cache(int).ṽ,
        f=cache(int).f̃,
    )
    solutionstep!(soltmp, history, problem(int), iguess(int))

    for k in 1:D
        x[D*S+k] = cache(int).p̃[k]
    end
end

function initial_trajectory!(sol, history, params, int::GeometricIntegrator{<:NonLinear_OneLayer_GML}, initial_trajectory::IntegratorExtrapolation)
    local network_labels = cache(int).network_labels
    local integrator = default_iguess_integrator(method(int))
    local h = int.problem.tstep
    local nstages = method(int).nstages
    local D = ndims(int)
    local problem = int.problem
    local S = nbasis(method(int))
    local x = nlsolution(int)

    tem_ode = similar(problem, [0.0, h], h / nstages, (q=StateVariable(sol.q[:]), p=StateVariable(sol.p[:])))
    tem_sol = integrate(tem_ode, integrator)

    for k in 1:D
        network_labels[:, k] = tem_sol.q[:, k]
        cache(int).q̃[k] = tem_sol.q[:, k][end]
        cache(int).p̃[k] = tem_sol.p[:, k][end]
        x[D*S+k] = cache(int).p̃[k]
    end
end

function initial_params!(int::GeometricIntegrator{<:NonLinear_OneLayer_GML}, InitialParams::TrainingMethod)
    local D = ndims(int)
    local S = nbasis(method(int))

    local show_status = method(int).show_status 
    local x = nlsolution(int)
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local nstages = method(int).nstages
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local nepochs = method(int).training_epochs
    local backend = method(int).basis.backend

    for k in 1:D
        if show_status
            print("\n network lables for dimension $k \n")
            print(network_labels[:,k])
        end
        
        labels = reshape(network_labels[:,k],1,nstages+1)

        ps[k] = AbstractNeuralNetworks.initialparameters(NN,backend,Float64)

        # opt = GeometricMachineLearning.Optimizer(AdamOptimizer(0.001, 0.9, 0.99, 1e-8), ps[k])
        opt = GeometricMachineLearning.Optimizer(AdamOptimizerWithDecay(nepochs,1e-3, 5e-5), ps[k])

        err = 0
        for ep in 1:nepochs
            gs = Zygote.gradient(p -> mse_loss(network_inputs,labels,NN,p)[1],ps[k])[1]
            optimization_step!(opt, NN, ps[k], gs)
            err = mse_loss(network_inputs,labels,NN,ps[k])[1]
        end

        show_status ? print("\n dimension $k,final loss: $err by $nepochs epochs") : nothing

        for i in 1:S
            x[D*(i-1)+k] = ps[k][2].W[i]
            x[D*(S+1)+D*(i-1)+k] = ps[k][1].W[i]
            x[D*(S+1 + S)+D*(i-1)+k] = ps[k][1].b[i]
        end
    end

    if show_status
        print("\n network parameters \n")
        print(ps)
        print("\n initial guess x from network training \n")
        print(x)
    end

end


function initial_params!(int::GeometricIntegrator{<:NonLinear_OneLayer_GML}, InitialParams::OGA1d)
    local S = nbasis(method(int))
    local D = ndims(int)
    local quad_nodes = method(int).network_inputs
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local network_labels = cache(int).network_labels'
    local activation = method(int).basis.activation
    local x = nlsolution(int)
    local show_status = method(int).show_status
    local nstages = method(int).nstages
    local bias_interval = method(int).bias_interval
    local dict_amount = method(int).dict_amount

    quad_weights = simpson_quadrature(nstages)# Simpson's rule for 11 quad points 0:0.1:1
    
    for d in 1:D
        W = zeros(S, 1)        # all parameters w
        Bias = zeros(S, 1)      # all parameters b
        C = zeros(S, nstages + 1)
        for k = 1:S
            #     The subproblem is key to the greedy algorithm, where the 
            #     inner products |(u,g) - (f,g)| should be maximized.
            #     Part of the inner products can be computed in advance.
            uk_quad = NN(quad_nodes, ps[d])'

            #select the Optimal basis
            B = bias_interval[1]:(bias_interval[2]-bias_interval[1])/dict_amount:bias_interval[2]
            w_list = vcat(-1 * ones(length(B), 1), ones(length(B), 1))
            b_list = vcat(collect(B), collect(B))
            A = hcat(w_list, b_list)
            quad_nodes_mat = hcat(quad_nodes', ones(length(quad_nodes)))'
            gx_quad = activation.(A * quad_nodes_mat)

            f_weight = network_labels[d, :] .* quad_weights
            uk_weight = uk_quad .* quad_weights

            loss = -(1 / 2) * (gx_quad * (uk_weight - f_weight)) .^ 2
            argmin_index = argmin(loss)
            W[k] = A[argmin_index[1], :][1]
            Bias[k] = A[argmin_index[1], :][2]

            ak = hcat(W[k], Bias[k])
            C[k, :] = ak * quad_nodes_mat
            selected_g = activation.(C[1:k, :])

            Gk = selected_g * (selected_g .* quad_weights')'
            rhs = selected_g * (network_labels[d, :] .* quad_weights)
            xk = Gk \ rhs

            ps[d][1].W[:] .= W
            ps[d][1].b[:] .= Bias
            ps[d][2].W[1:k] = xk

            # opt = Optimisers.Descent(0.0001)
            # st_opt = Optimisers.setup(opt, ps[d])

            # errs = sum(network_labels[d, :] - NN(quad_nodes, ps[d])') .^ 2
            # show_status ? print("\n OGA error $errs before training \n ") : nothing

            # gs = Zygote.gradient(p -> sum(network_labels[d, :] - NN(quad_nodes, p)') .^ 2, ps[d])[1]

            # gs[1].W[:] = Float64[x === nothing ? 0.0 : x for x in gs[1].W[:]]
            # gs[1].b[:] = Float64[x === nothing ? 0.0 : x for x in gs[1].b[:]]

            # st_opt, ps[d] = Optimisers.update(st_opt, ps[d], gs)

            errs = sum(network_labels[d, :] - NN(quad_nodes, ps[d])') .^ 2
            show_status ? print("\n OGA error $errs ") : nothing
        end
        show_status ? print("\n Finish OGA for dimension $d ") : nothing

    end

    for k in 1:D
        for i in 1:S
            x[D*(i-1)+k] = ps[k][2].W[i]
            x[D*(S+1)+D*(i-1)+k] = ps[k][1].W[i]
            x[D*(S+1+S)+D*(i-1)+k] = ps[k][1].b[i]
        end
    end
    # st = st_tem[1]
    show_status ? print("\n initial guess for DOF from OGA  ") : nothing
    show_status ? print("\n ", x) : nothing

end

function GeometricIntegrators.Integrators.components!(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:NonLinear_OneLayer_GML}) where {ST}
    local D = ndims(int)
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
    local m  = cache(int, ST).m
    local a  = cache(int, ST).a
    local dqdWc=cache(int, ST).dqdWc
    local dqdbc=cache(int, ST).dqdbc
    local dvdWc=cache(int, ST).dvdWc
    local dvdbc=cache(int, ST).dvdbc
    local dqdWr₁=cache(int, ST).dqdWr₁
    local dqdWr₀=cache(int, ST).dqdWr₀
    local dqdbr₁=cache(int, ST).dqdbr₁
    local dqdbr₀=cache(int, ST).dqdbr₀

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

    for i in 1:S 
        for k in 1:D
            ps[k][2].W[i] = x[D*(i-1)+k]  
            ps[k][1].W[i] = x[D*(S+1)+D*(i-1)+k] 
            ps[k][1].b[i] = x[D*(S+1+S)+D*(i-1)+k]
        end
    end

    # compute coefficients
    for d in 1:D 
        r₀[:,d] = (NN.layers[1])([0.0],ps[d][1])
        r₁[:,d] = (NN.layers[1])([1.0],ps[d][1])
        for j in eachindex(quad_nodes)
            m[j,:,d] =(NN.layers[1])([quad_nodes[j]],ps[d][1])
            a[j,:,d] = DVDθ([quad_nodes[j]],NeuralNetworkParameters(ps[d]))[1,1].L2.W[:]
        end
    end

    # compute the derivatives of the coefficients on the quadrature nodes and at the boundaries
    ϵ=0.00001
    for d in 1:D 
        for j in eachindex(quad_nodes)
            g = DQDθ([quad_nodes[j]],NeuralNetworkParameters(ps[d]))
            dqdWc[j,:,d] = g[1,1].L1.W[:]
            dqdbc[j,:,d] = g[1,1].L1.b[:]

            gv = DVDθ([quad_nodes[j]],NeuralNetworkParameters(ps[d]))
            dvdWc[j,:,d] = gv[1,1].L1.W[:]
            dvdbc[j,:,d] = gv[1,1].L1.b[:]
        end

        g0= DQDθ([0.0],NeuralNetworkParameters(ps[d]))
        dqdWr₀[:,d] = g0[1,1].L1.W[:]
        dqdbr₀[:,d] = g0[1,1].L1.b[:]

        g1 = DQDθ([1.0],NeuralNetworkParameters(ps[d]))
        dqdWr₀[:,d] = g1[1,1].L1.W[:]
        dqdbr₀[:,d] = g1[1,1].L1.b[:]
    end

    # compute Q : q at quaadurature points
    for i in eachindex(Q)
        for d in eachindex(Q[i])
            y = zero(ST)
            for j in eachindex(X)
                y += m[i,j,d] * X[j][d]
            end
            Q[i][d] = y
        end
    end

    # compute q[t_{n+1}]
    for d in eachindex(q)
        y = zero(ST)
        for i in eachindex(X)
            y += r₁[i,d] * X[i][d]
        end
        q[d] = y
    end

    # compute V volicity at quadrature points
    for i in eachindex(V)
        for k in eachindex(V[i])
            y = zero(ST)
            for j in eachindex(X)
                y += a[i,j,k] * X[j][k]
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
    local D = ndims(int)
    local S = nbasis(method(int))
    local q̄ = sol.q
    local p̄ = sol.p
    local p̃ = cache(int, ST).p̃ 
    local P = cache(int, ST).P 
    local F = cache(int, ST).F
    local X = cache(int, ST).X

    local r₀ = cache(int, ST).r₀
    local r₁ = cache(int, ST).r₁
    local m  = cache(int, ST).m
    local a  = cache(int, ST).a

    local dqdWc=cache(int, ST).dqdWc
    local dqdbc=cache(int, ST).dqdbc
    local dvdWc=cache(int, ST).dvdWc
    local dvdbc=cache(int, ST).dvdbc
    local dqdWr₁=cache(int, ST).dqdWr₁
    local dqdWr₀=cache(int, ST).dqdWr₀
    local dqdbr₁=cache(int, ST).dqdbr₁
    local dqdbr₀=cache(int, ST).dqdbr₀

    # compute b = - [(P-AF)], the residual in actual action, vatiation with respect to Q_{n,i}
    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P,F)
                z += method(int).b[j] * m[j,i,k] * F[j][k] * timestep(int)
                z += method(int).b[j] * a[j,i,k] * P[j][k]
            end
            b[D*(i-1)+k] = (r₁[i,k] * p̃[k] - r₀[i,k] * p̄[k]) - z
        end
    end

    # the continue constraint from hamilton pontryagin principle
    for k in eachindex(q̄)
        y = zero(ST)
        for j in eachindex(X)
            y += r₀[j,k] * X[j][k]
        end
        b[D*S+k] = q̄[k] - y 
    end

    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P,F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdWc[j,i,k] 
                z += method(int).b[j] * P[j][k] *dvdWc[j,i,k] 
            end
            b[D*(S+1)+D*(i-1)+k] = dqdWr₁[i,k] * p̃[k]  - z
        end
    end 

    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P,F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdbc[j,i,k] 
                z += method(int).b[j] * P[j][k] * dvdbc[j,i,k]
            end
            b[D*(S+1 + S)+D*(i-1)+k] = (dqdbr₁[i,k] * p̃[k] - dqdbr₀[i,k] * p̄[k]) - z
        end
    end 

end

# Compute stages of Variational Partitioned Runge-Kutta methods.
function GeometricIntegrators.Integrators.residual!(b::AbstractVector{ST}, x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:NonLinear_OneLayer_GML}) where {ST}
    # check that x and b are compatible
    @assert axes(x) == axes(b)

    # compute stages from nonlinear solver solution x
    GeometricIntegrators.Integrators.components!(x, sol, params, int)

    # compute residual vector
    GeometricIntegrators.Integrators.residual!(b, sol, params, int)
end


function GeometricIntegrators.Integrators.update!(sol, params, int::GeometricIntegrator{<:NonLinear_OneLayer_GML}, DT)
    sol.q .= cache(int, DT).q̃
    sol.p .= cache(int, DT).p̃
end

function GeometricIntegrators.Integrators.update!(sol, params, x::AbstractVector{DT}, int::GeometricIntegrator{<:NonLinear_OneLayer_GML}) where {DT}
    # compute vector field at internal stages
    GeometricIntegrators.Integrators.components!(x, sol, params, int)

    # compute final update
    GeometricIntegrators.Integrators.update!(sol, params, int, DT)
end


function GeometricIntegrators.Integrators.integrate_step!(sol, history, params, int::GeometricIntegrator{<:NonLinear_OneLayer_GML, <:AbstractProblemIODE})
    # call nonlinear solver
    solve!(nlsolution(int), (b,x) -> GeometricIntegrators.Integrators.residual!(b, x, sol, params, int), solver(int))

    # print solver status
    # print_solver_status(int.solver.status, int.solver.params)

    # check if solution contains NaNs or error bounds are violated
    # check_solver_status(int.solver.status, int.solver.params)

    # compute final update
    GeometricIntegrators.Integrators.update!(sol, params, nlsolution(int), int)

    #compute the trajectory after solving by newton method
    stages_compute!(sol,int)
end

function stages_compute!(sol,int::GeometricIntegrator{<:NonLinear_OneLayer_GML})
    local x = nlsolution(int)
    local stage_values = cache(int).stage_values
    local network_inputs = method(int).network_inputs
    local D = ndims(int)
    local S = nbasis(method(int))
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local show_status = method(int).show_status

    if show_status
        print("\n solution x after solving by Newton \n")
        print(x)
    end

    for k in 1:D
        for i in 1:S 
            ps[k][2].W[i] = x[D*(i-1)+k]  
            ps[k][1].W[i] = x[D*(S+1)+D*(i-1)+k] 
            ps[k][1].b[i] = x[D*(S+1+S)+D*(i-1)+k]
        end
        stage_values[:,k] = NN(network_inputs,ps[k])[2:end]
    end

    if show_status
        print("\n stages prediction after solving \n")
        print(stage_values)
        print("\n sol from this step \n")
        print("q:",sol.q,"\n")
        print("p:",sol.p,"\n")

    end

end