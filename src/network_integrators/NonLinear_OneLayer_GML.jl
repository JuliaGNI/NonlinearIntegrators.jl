using CompactBasisFunctions
using GeometricIntegrators

struct NonLinear_OneLayer_GML{T, NBASIS, NNODES, basisType <: Basis{T}} <: OneLayerMethod
    basis::basisType
    quadrature::QuadratureRule{T,NNODES}

    b::SVector{NNODES,T}
    c::SVector{NNODES,T}

    nstages::Int
    show_status::Bool
    network_inputs::Matrix{T}
    training_epochs::Int
    function NonLinear_OneLayer_GML(basis::Basis{T}, quadrature::QuadratureRule{T};nstages::Int = 10,show_status::Bool=true,training_epochs::Int=50000) where {T}
        # get number of quadrature nodes and number of basis functions
        NNODES = QuadratureRules.nnodes(quadrature)
        NBASIS = basis.S

        # get quadrature nodes and weights
        quad_weights = QuadratureRules.weights(quadrature)
        quad_nodes = QuadratureRules.nodes(quadrature)

        network_inputs = reshape(collect(0:1/nstages:1),1,nstages+1)
        new{T, NBASIS, NNODES,typeof(basis)}(basis, quadrature, quad_weights, quad_nodes, nstages, show_status, network_inputs, training_epochs)
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

isexplicit(::Union{NonLinear_OneLayer_GML, Type{<:NonLinear_OneLayer_GML}}) = false
isimplicit(::Union{NonLinear_OneLayer_GML, Type{<:NonLinear_OneLayer_GML}}) = true
issymmetric(::Union{NonLinear_OneLayer_GML, Type{<:NonLinear_OneLayer_GML}}) = missing
issymplectic(::Union{NonLinear_OneLayer_GML, Type{<:NonLinear_OneLayer_GML}}) = missing

default_solver(::NonLinear_OneLayer_GML) = Newton()
# default_iguess(::NonLinear_OneLayer_GML) = HermiteExtrapolation()# HarmonicOscillator
default_iguess(::NonLinear_OneLayer_GML) = MidpointExtrapolation()#CoupledHarmonicOscillator
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

    ps::Vector{Tuple{NamedTuple{},NamedTuple{}}}

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
        ps = [((W = zeros(ST,S,1),b = zeros(ST,S)),(W = zeros(ST,1,S),))  for k in 1:D]

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

function GeometricIntegrators.Integrators.initial_guess!(int::GeometricIntegrator{<:NonLinear_OneLayer_GML}) 
    local h = int.problem.tstep
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local show_status = method(int).show_status 
    local current_step = cache(int).current_step

    show_status ? print("\n current time step: $current_step") : nothing
    current_step[1]+=1

    # choose initial guess method based on the value of h
    # if h < 0.5
    #     initial_guess_Extrapolation!(int)
    # else
    initial_guess_integrator!(int)
    # end 
    
    if show_status
        print("\n network inputs \n")
        print(network_inputs)

        print("\n network labels from initial guess methods \n")
        print(network_labels)
    end

    initial_guess_networktraining!(int)

end

function initial_guess_Extrapolation!(int::GeometricIntegrator{<:NonLinear_OneLayer_GML})
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

function initial_guess_integrator!(int::GeometricIntegrator{<:NonLinear_OneLayer_GML})
    local network_labels = cache(int).network_labels
    local integrator = default_iguess_integrator(method(int))
    local h = int.problem.tstep
    local N = method(int).nstages
    local D = ndims(int)
    local problem = int.problem
    local S = nbasis(method(int))   
    local x = nlsolution(int)

    tem_ode=similar(problem,[0.,h],h/N,(q = StateVariable(int.solstep.q[:]), p = StateVariable(int.solstep.p[:]), λ = AlgebraicVariable(problem.ics.λ)))
    sol = integrate(tem_ode, integrator)

    for k in 1:D
        network_labels[:,k]=sol.s.q[:,k]
        cache(int).q̃[k] = sol.s.q[:,k][end]
        cache(int).p̃[k] = sol.s.p[:,k][end]
        x[D*S+k] = cache(int).p̃[k]
    end
end 

function initial_guess_networktraining!(int::GeometricIntegrator{<:NonLinear_OneLayer_GML})
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

        if backend == CUDABackend()
            #TODO add a CUDA version
            network_inputs =  CuArray(network_inputs)
            labels = CuArray(labels)
        end

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


function GeometricIntegrators.Integrators.components!(x::AbstractVector{ST}, int::GeometricIntegrator{<:NonLinear_OneLayer_GML}) where {ST}
    local D = ndims(int)
    local S = nbasis(method(int))
    local σ = int.method.basis.activation

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
            a[j,:,d] = basis_first_order_central_difference(NN,ps[d],quad_nodes[j])
        end
    end

    # compute the derivatives of the coefficients on the quadrature nodes and at the boundaries
    ϵ=0.00001
    for d in 1:D 
        for j in eachindex(quad_nodes)
            g= Zygote.gradient(p->NN([quad_nodes[j]],p)[1],ps[d])[1]
            dqdWc[j,:,d] = g[1].W[:]
            dqdbc[j,:,d] = g[1].b[:]

            gvf= Zygote.gradient(p->NN([quad_nodes[j]+ϵ],p)[1],ps[d])[1]
            gvb= Zygote.gradient(p->NN([quad_nodes[j]-ϵ],p)[1],ps[d])[1]
            dvdWc[j,:,d] = (gvf[1].W[:] .- gvb[1].W[:])/(2*ϵ)
            dvdbc[j,:,d] = (gvf[1].b[:] .- gvb[1].b[:])/(2*ϵ)
        end

        g0= Zygote.gradient(p->NN([0.0],p)[1],ps[d])[1]
        dqdWr₀[:,d] = g0[1].W[:]
        dqdbr₀[:,d] = g0[1].b[:]

        g1 = Zygote.gradient(p->NN([1.0],p)[1],ps[d])[1]
        dqdWr₁[:,d] = g1[1].W[:]
        dqdbr₁[:,d] = g1[1].b[:]
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

    # compute P=ϑ(Q,V) pl/pv and F=f(Q,V) pl/px
    for i in eachindex(Q,V,P,F)
        tᵢ = solstep(int).t + timestep(int) * method(int).c[i]
        equations(int).ϑ(P[i], tᵢ, Q[i], V[i], parameters(solstep(int))) # P[i] : momentum at t_n+h* c_i
        equations(int).f(F[i], tᵢ, Q[i], V[i], parameters(solstep(int))) # F[i] : Force at t_n + h* cᵢ
    end

end


function GeometricIntegrators.Integrators.residual!(b::Vector{ST}, int::GeometricIntegrator{<:NonLinear_OneLayer_GML}) where {ST}
    local D = ndims(int)
    local S = nbasis(method(int))
    local q̄ = cache(int, ST).q̄ 
    local p̄ = cache(int, ST).p̄ 
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

function GeometricIntegrators.Integrators.residual!(b::AbstractVector{ST}, x::AbstractVector{ST}, int::GeometricIntegrator{<:NonLinear_OneLayer_GML}) where {ST}
    @assert axes(x) == axes(b)

    # copy previous solution from solstep to cache
    reset!(cache(int, ST), current(solstep(int))...)

    # compute stages from nonlinear solver solution x
    GeometricIntegrators.Integrators.components!(x, int)

    # compute residual vector
    GeometricIntegrators.Integrators.residual!(b, int)
end

function GeometricIntegrators.Integrators.update!(x::AbstractVector{DT}, int::GeometricIntegrator{<:NonLinear_OneLayer_GML}) where {DT}
    # copy previous solution from solstep to cache
    GeometricIntegrators.Integrators.reset!(cache(int, DT), current(solstep(int))...)

    # compute vector field at internal stages
    GeometricIntegrators.Integrators.components!(x, int)

    # compute final update
    solstep(int).q .= cache(int, DT).q̃
    solstep(int).p .= cache(int, DT).p̃
end


function GeometricIntegrators.Integrators.integrate_step!(int::GeometricIntegrator{<:NonLinear_OneLayer_GML, <:AbstractProblemIODE})
    # copy previous solution from solstep to cache
    reset!(cache(int), current(solstep(int))...)

    # call nonlinear solver
    solve!(nlsolution(int), (b,x) -> GeometricIntegrators.Integrators.residual!(b, x, int), solver(int)) # nlsoution : initialguess from HermiteExtrapolation

    # print solver status
    # print_solver_status(int.solver.status, int.solver.params)

    # check if solution contains NaNs or error bounds are violated
    # check_solver_status(int.solver.status, int.solver.params)

    # compute final update
    update!(nlsolution(int), int)

    #compute the trajectory after solving by newton method
    stages_compute!(int)

    #check for NaNs
    # if sum(isnan.(cache(int).q̃[:])) > 0 
    #     error("NaN value encountered, terminating program.")
    # end
end

function stages_compute!(int::GeometricIntegrator{<:NonLinear_OneLayer_GML})
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
        print("q:",solstep(int).q,"\n")
        print("p:",solstep(int).p,"\n")

    end

end