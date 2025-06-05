struct NonLinear_DenseNet_GML{T, NBASIS,NHIDDEN, NNODES, basisType <: Basis{T},ET<:IntegratorExtrapolation,IPMT<:InitialParametersMethod} <: DenseNetMethod
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

    function NonLinear_DenseNet_GML(basis::Basis{T}, quadrature::QuadratureRule{T};nstages::Int = 10,show_status::Bool=true,training_epochs::Int=50000,
        initial_trajectory::ET=IntegratorExtrapolation(),
        initial_guess_method::IPMT=TrainingMethod()) where {T, ET, IPMT}
        # get number of quadrature nodes and number of basis functions
        NNODES = QuadratureRules.nnodes(quadrature)
        NBASIS = basis.S
        NHIDDEN = basis.S₁    

        # get quadrature nodes and weights
        quad_weights = QuadratureRules.weights(quadrature)
        quad_nodes = QuadratureRules.nodes(quadrature)
        network_inputs = reshape(collect(0:1/nstages:1),1,nstages+1)

        new{T, NBASIS, NHIDDEN, NNODES, typeof(basis),ET,IPMT}(basis, quadrature, quad_weights, quad_nodes,nstages, show_status, network_inputs, training_epochs,
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

struct NonLinear_DenseNet_GMLCache{ST,D,S₁,S,R,N} <: IODEIntegratorCache{ST,D}
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

    ps::Vector{@NamedTuple{L1::@NamedTuple{W::Matrix{ST}, b::Vector{ST}},L2::@NamedTuple{W::Matrix{ST}, b::Vector{ST}},
        L3::@NamedTuple{W::Matrix{ST}}}}

    r₀::VecOrMat{ST}
    r₁::VecOrMat{ST}
    m::Array{ST} 
    a::Array{ST}

    dqdWc::Array{ST}
    dqdbc::Array{ST}
    dvdWc::Array{ST}
    dvdbc::Array{ST}

    dqdWr₁::Array{ST}
    dqdWr₀::Array{ST}

    dqdbr₁::VecOrMat{ST}
    dqdbr₀::VecOrMat{ST}

    stage_values::VecOrMat{ST}
    network_labels::VecOrMat{ST}

    function NonLinear_DenseNet_GMLCache{ST,D,S₁,S,R,N}() where {ST,D,S₁,S,R,N}
        x = zeros(ST,D*(S+1+(S₁+1)*S)) # Last layer Weight S (no bias for now) + P + hidden layer W (S*S₁) + hidden layer bias S

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

        # create hidden layer parameter vectors
        ps = [(L1 = (W = zeros(ST,S₁,1),b = zeros(ST,S₁)),
                L2 = (W = zeros(ST,S,S₁),b = zeros(ST,S)),
                L3 = (W = zeros(ST,1,S),))  for k in 1:D]

        r₀ = zeros(ST, S, D)
        r₁ = zeros(ST, S, D)
        m  = zeros(ST, R, S, D)
        a  = zeros(ST, R, S, D)

        dqdWc=zeros(ST, R, S, S₁, D)
        dqdbc=zeros(ST, R, S, D)
        dvdWc=zeros(ST, R, S, S₁, D)
        dvdbc=zeros(ST, R, S, D)
        
        dqdWr₁= zeros(ST, S, S₁, D)
        dqdWr₀= zeros(ST, S, S₁, D)
    
        dqdbr₁= zeros(ST, S, D)
        dqdbr₀= zeros(ST, S, D)

        stage_values = zeros(ST, N, D)
        network_labels = zeros(ST, N+1, D)

        return new(x, q̄, p̄, q̃, p̃, ṽ, f̃, s̃, X, Q, P, V, F, ps, r₀, r₁, m, a, 
        dqdWc, dqdbc, dvdWc, dvdbc, dqdWr₁, dqdWr₀, dqdbr₁, dqdbr₀,stage_values,network_labels)
    end
end

function GeometricIntegrators.Integrators.reset!(cache::NonLinear_DenseNet_GMLCache, t, q, p)
    copyto!(cache.q̄, q)
    copyto!(cache.p̄, p)
end

GeometricIntegrators.Integrators.nlsolution(cache::NonLinear_DenseNet_GMLCache) = cache.x

function GeometricIntegrators.Integrators.Cache{ST}(problem::AbstractProblemIODE, method::NonLinear_DenseNet_GML; kwargs...) where {ST}
    NonLinear_DenseNet_GMLCache{ST, ndims(problem), method.basis.S₁,method.basis.S, nnodes(method),nstages(method)}(; kwargs...)
end

@inline GeometricIntegrators.Integrators.CacheType(ST, problem::AbstractProblemIODE, method::NonLinear_DenseNet_GML) = NonLinear_DenseNet_GMLCache{ST, ndims(problem), method.basis.S₁,method.basis.S, nnodes(method),nstages(method)}

@inline function Base.getindex(c::NonLinear_DenseNet_GMLCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end

function GeometricIntegrators.Integrators.initial_guess!(sol, history, params,int::GeometricIntegrator{<:NonLinear_DenseNet_GML}) 
    local h = int.problem.tstep
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

    tem_ode = similar(problem, [0.0, h], h / nstages, (q=StateVariable(sol.q[:]), p=StateVariable(sol.p[:])))
    tem_sol = integrate(tem_ode, integrator)

    for k in 1:D
        network_labels[:, k] = tem_sol.q[:, k]#[1].s
        cache(int).q̃[k] = tem_sol.q[:, k][end]
        cache(int).p̃[k] = tem_sol.p[:, k][end]
        x[D*S+k] = cache(int).p̃[k]
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

    for k in 1:D
        if show_status
            print("\n network lables for dimension $k \n")
            print(network_labels[:,k])
        end

        labels = reshape(network_labels[:,k],1,nstages+1)

        NN_with_params = GeometricMachineLearning.NeuralNetwork(NN)
        # opt = GeometricMachineLearning.Optimizer(AdamOptimizer(0.001, 0.9, 0.99, 1e-8), ps[k])
        opt = GeometricMachineLearning.Optimizer(GeometricMachineLearning.AdamOptimizerWithDecay(nepochs,1e-3, 5e-5), NN_with_params)
        err = 0
        λ = GeometricMachineLearning.GlobalSection(NN_with_params.params)
        for ep in 1:nepochs
            gs = Zygote.gradient(p -> mse_loss(network_inputs,labels,NN_with_params,p)[1],NN_with_params.params)[1]
            GeometricMachineLearning.optimization_step!(opt,λ, NN_with_params.params, gs)
            err = mse_loss(network_inputs,labels,NN_with_params,NN_with_params.params)[1]

            if err < 5e-5
                show_status ? print("\n dimension $k,final loss: $err by $ep epochs") : nothing
                break
            elseif ep == nepochs
                show_status ? print("\n dimension $k,final loss: $err by $ep epochs") : nothing
            end
        end

        ps[k] = NN_with_params.params[:]
        for i in 1:S
            x[D*(i-1)+k] = ps[k][end].W[i]
            for kk in 1:S₁
                x[D*(S+1)+D*(i-1)*S₁+D*(kk-1)+k] = ps[k][end-1].W[i,kk]
            end
            x[D*(S+1+S₁*S)+D*(i-1)+k] = ps[k][end-1].b[i]
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

    local quad_nodes = QuadratureRules.nodes(int.method.quadrature)
    local C = cache(int, ST)

    local q = cache(int, ST).q̃
    local p = cache(int, ST).p̃
    local Q = cache(int, ST).Q

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

    local float_ps = cache(int,Float64).ps


    # copy x to X and bias 
    for i in 1:S
        for d in 1:D
            C.X[i][d] = x[D*(i-1)+d]
        end
    end

    # copy x to p # momenta
    for k in eachindex(p)
        p[k] = x[D*S+k]
    end

    if ST != Float64 
        for k in 1:D
            for layers in 1:length(float_ps[k])-2
                ps[k][layers].W[:] = float_ps[k][layers].W[:]
                ps[k][layers].b[:] = float_ps[k][layers].b[:]
            end
        end
    end

    # copy x to hidden layer weights W : [D,S*S₁]
    
    for d in 1:D
        for i in 1:S
            ps[d][end].W[i]= x[D*(i-1)+d]
            for kk in 1:S₁
                ps[d][end-1].W[i,kk] = x[D*(S+1)+D*(i-1)*S₁+D*(kk-1)+d]
            end
            ps[d][end-1].b[i]= x[D*(S+1+S₁*S)+D*(i-1)+d]
        end
    end 
    
    # compute coefficients
    ϵ=0.00001
    for d in 1:D
        intermidiate_ps = (L1 = ps[k].L1, L2 = ps[k].L2)
        r₀[:,d] = AbstractNeuralNetworks.Chain(NN.layers[1:end-1]...)([0.0],intermidiate_ps)
        r₁[:,d] = AbstractNeuralNetworks.Chain(NN.layers[1:end-1]...)([1.0],intermidiate_ps)
        # for i in 1:S
        gs0 = Zygote.gradient(p->NN([0.0],p)[1],ps[d])[1]
        dqdWr₀[:,:,d] = gs0[end-1].W[:]
        dqdbr₀[:,d] = gs0[end-1].b[:]

        gs1 = Zygote.gradient(p->NN([1.0],p)[1],ps[d])[1]
        dqdWr₁[:,:,d] = gs1[end-1].W[:]
        dqdbr₁[:,d] = gs1[end-1].b[:]

        for j in eachindex(quad_nodes)
            m[j,:,d] =  AbstractNeuralNetworks.Chain(NN.layers[1:end-1]...)([quad_nodes[j]],intermidiate_ps)
            a[j,:,d] = basis_first_order_central_difference(NN,ps[d],quad_nodes[j])
            
            gsc = Zygote.gradient(p->NN([quad_nodes[j]],p)[1],ps[d])[1]
            dqdWc[j,:,:,d] = gsc[end-1].W[:]
            dqdbc[j,:,d] = gsc[end-1].b[:]

            vgs_f =Zygote.gradient(p->NN([quad_nodes[j]+ϵ],p)[1],ps[d])[1]
            vgs_b =Zygote.gradient(p->NN([quad_nodes[j]-ϵ],p)[1],ps[d])[1]

            dvdWc[j,:,:,d] = (vgs_f[end-1].W[:] .- vgs_b[end-1].W[:])/(2*ϵ)
            dvdbc[j,:,d] = (vgs_f[end-1].b[:] .- vgs_b[end-1].b[:])/(2*ϵ)

        end
    end

    # compute Q q at quaadurature points
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

    # compute b = - [(P-AF)]
    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in 1:R
                z += method(int).b[j] * F[j][k] * m[j,i,k] * timestep(int)
                z += method(int).b[j] * P[j][k] * a[j,i,k] 
            end
            b[D*(i-1)+k] = (r₁[i,k] * p̃[k] - r₀[i,k] * p̄[k]) - z
        end
    end # the residual in actual action, vatiation with respect to Q_{n,i}

    for k in 1:D
        y = zero(ST)
        for j in eachindex(X)
            y += r₀[j,k] * X[j][k]
        end
        b[D*S+k] = q̄[k] - y # the continue constraint from hamilton pontryagin principle
    end

    for k in 1:D
        for i in 1:S
            for kk in 1:S₁
                z = zero(ST)
                for j in 1:R
                    z += timestep(int) * method(int).b[j] * F[j][k] * dqdWc[j,i,kk,k] #zeros(ST, R, S, S₁, D)
                    z += method(int).b[j] * P[j][k] * dvdWc[j,i,kk,k] 
                end
                b[D*(S+1)+D*(i-1)*S₁+D*(kk-1)+k] = (dqdWr₁[i,kk,k] * p̃[k]- dqdWr₀[i,kk,k]* p̄[k])  - z #(ST, S, S₁, D)
            end
        end
    end 

    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in 1:R
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdbc[j,i,k]#dqdbc=zeros(ST, R, S, D)
                z += method(int).b[j] * P[j][k] * dvdbc[j,i,k]
            end
            b[D*(S+1+S₁*S)+D*(i-1)+k] = (dqdbr₁[i,k] * p̃[k] - dqdbr₀[i,k]* p̄[k]) - z 
        end
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

function stages_compute!(int::GeometricIntegrator{<:NonLinear_DenseNet_GML})
    local x = nlsolution(int)
    local stage_values = cache(int).stage_values
    local network_inputs = method(int).network_inputs
    local D = ndims(int)
    local S = int.method.basis.S
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local show_status = method(int).show_status
    local S₁ = method(int).basis.S₁

    if show_status
        print("\n solution x after solving by Newton \n")
        print(x)
    end

    for d in 1:D
        for i in 1:S
            ps[d][end].W[i]= x[D*(i-1)+d]
            for kk in 1:S₁
                ps[d][end-1].W[i,kk] = x[D*(S+1)+D*(i-1)*S₁+D*(kk-1)+d]
            end
            ps[d][end-1].b[i]= x[D*(S+1+S₁*S)+D*(i-1)+d]
        end
        stage_values[:,d] = NN(network_inputs,ps[d])[2:end]
    end

    if show_status
        print("\n stages prediction after solving \n")
        print(stage_values)
    end

end