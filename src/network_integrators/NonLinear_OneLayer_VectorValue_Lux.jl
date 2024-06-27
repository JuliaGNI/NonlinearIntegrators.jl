using CompactBasisFunctions
using GeometricIntegrators

struct NonLinear_OneLayer_VectorValue_Lux{T, NBASIS, NNODES, basisType <: Basis{T}} <: OneLayerMethod
    basis::basisType
    quadrature::QuadratureRule{T,NNODES}

    b::SVector{NNODES,T}
    c::SVector{NNODES,T}

    nstages::Int
    show_status::Bool
    network_inputs::Matrix{T}
    training_epochs::Int

    problem_module::Module
    problem_initial_hamitltonian::Float64
    use_hamiltonian_loss::Bool
    function NonLinear_OneLayer_VectorValue_Lux(basis::Basis{T}, quadrature::QuadratureRule{T},problem_module;
        nstages::Int = 10,show_status::Bool=true,training_epochs::Int=50000,problem_initial_hamitltonian = 0.0,use_hamiltonian_loss=true) where {T}
        # get number of quadrature nodes and number of basis functions
        NNODES = QuadratureRules.nnodes(quadrature)
        NBASIS = basis.S

        # get quadrature nodes and weights
        quad_weights = QuadratureRules.weights(quadrature)
        quad_nodes = QuadratureRules.nodes(quadrature)

        network_inputs = reshape(collect(0:1/nstages:1),1,nstages+1)
        new{T, NBASIS, NNODES,typeof(basis)}(basis, quadrature, quad_weights, quad_nodes, nstages, show_status, network_inputs, 
        training_epochs,problem_module,problem_initial_hamitltonian,use_hamiltonian_loss)
    end
end

CompactBasisFunctions.basis(method::NonLinear_OneLayer_VectorValue_Lux) = method.basis
quadrature(method::NonLinear_OneLayer_VectorValue_Lux) = method.quadrature
CompactBasisFunctions.nbasis(method::NonLinear_OneLayer_VectorValue_Lux) = method.basis.S
nnodes(method::NonLinear_OneLayer_VectorValue_Lux) = QuadratureRules.nnodes(method.quadrature)
activation(method::NonLinear_OneLayer_VectorValue_Lux) = method.basis.activation
nstages(method::NonLinear_OneLayer_VectorValue_Lux) = method.nstages
show_status(method::NonLinear_OneLayer_VectorValue_Lux) = method.show_status
training_epochs(method::NonLinear_OneLayer_VectorValue_Lux) = method.training_epochs    

isexplicit(::Union{NonLinear_OneLayer_VectorValue_Lux, Type{<:NonLinear_OneLayer_VectorValue_Lux}}) = false
isimplicit(::Union{NonLinear_OneLayer_VectorValue_Lux, Type{<:NonLinear_OneLayer_VectorValue_Lux}}) = true
issymmetric(::Union{NonLinear_OneLayer_VectorValue_Lux, Type{<:NonLinear_OneLayer_VectorValue_Lux}}) = missing
issymplectic(::Union{NonLinear_OneLayer_VectorValue_Lux, Type{<:NonLinear_OneLayer_VectorValue_Lux}}) = missing

default_solver(::NonLinear_OneLayer_VectorValue_Lux) = Newton()
default_iguess(::NonLinear_OneLayer_VectorValue_Lux) = MidpointExtrapolation()#CoupledHarmonicOscillator
# default_iguess_integrator(::NonLinear_OneLayer_VectorValue_Lux) =  CGVI(Lagrange(QuadratureRules.nodes(QuadratureRules.GaussLegendreQuadrature(4))),QuadratureRules.GaussLegendreQuadrature(4))
default_iguess_integrator(::NonLinear_OneLayer_VectorValue_Lux) =  ImplicitMidpoint()

struct NonLinear_OneLayer_VectorValue_LuxCache{ST,D,S,R,N} <: IODEIntegratorCache{ST,D}
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

    ps::@NamedTuple{layer_1::@NamedTuple{weight::Matrix{ST}, bias::Matrix{ST}}, layer_2::@NamedTuple{weight::Matrix{ST}}}
    st::@NamedTuple{layer_1::@NamedTuple{}, layer_2::@NamedTuple{}}

    r₀::Vector{ST}
    r₁::Vector{ST}
    m::VecOrMat{ST}
    a::VecOrMat{ST}

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
    function NonLinear_OneLayer_VectorValue_LuxCache{ST,D,S,R,N}() where {ST,D,S,R,N}
        x = zeros(ST,D*(S+1)+2*S) # Last layer Weight S (no bias for now) + P + hidden layer W (S*S₁) + hidden layer bias S

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

        # create first layer parameter vectors
        ps = (layer_1 = (weight = zeros(ST,S,1), bias = zeros(ST,S,1)), layer_2 = (weight = zeros(ST,D,S),))
        st = (layer_1 = NamedTuple(), layer_2 = NamedTuple())

        r₀ = zeros(ST, S)
        r₁ = zeros(ST, S)
        m  = zeros(ST, R, S)
        a  = zeros(ST, R, S)

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

        return new(x, q̄, p̄, q̃, p̃, ṽ, f̃, s̃, X, Q, P, V, F, ps, st, r₀, r₁, m, a, 
            dqdWc, dqdbc, dvdWc, dvdbc, dqdWr₁, dqdWr₀, dqdbr₁, dqdbr₀,
            current_step,stage_values,network_labels)
    end
end

GeometricIntegrators.Integrators.nlsolution(cache::NonLinear_OneLayer_VectorValue_LuxCache) = cache.x

function GeometricIntegrators.Integrators.Cache{ST}(problem::AbstractProblemIODE, method::NonLinear_OneLayer_VectorValue_Lux; kwargs...) where {ST}
    NonLinear_OneLayer_VectorValue_LuxCache{ST, ndims(problem), nbasis(method), nnodes(method),nstages(method)}(; kwargs...)
end

@inline GeometricIntegrators.Integrators.CacheType(ST, problem::AbstractProblemIODE, method::NonLinear_OneLayer_VectorValue_Lux) = NonLinear_OneLayer_VectorValue_LuxCache{ST, ndims(problem), nbasis(method), nnodes(method),nstages(method)}

@inline function Base.getindex(c::NonLinear_OneLayer_VectorValue_LuxCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end

function GeometricIntegrators.Integrators.reset!(cache::NonLinear_OneLayer_VectorValue_LuxCache, t, q, p)
    copyto!(cache.q̄, q)
    copyto!(cache.p̄, p)
end

function GeometricIntegrators.Integrators.initial_guess!(int::GeometricIntegrator{<:NonLinear_OneLayer_VectorValue_Lux})
    local h = int.problem.tstep
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local show_status = method(int).show_status 
    local current_step = cache(int).current_step

    show_status ? print("\n current time step: $current_step") : nothing
    current_step[1]+=1

    # choose initial guess method based on the value of h
    initial_guess_integrator!(int)

    if show_status
        print("\n network inputs \n")
        print(network_inputs)

        print("\n network labels from initial guess methods \n")
        print(network_labels)
    end

    initial_guess_networktraining!(int)

end

function initial_guess_Extrapolation!(int::GeometricIntegrator{<:NonLinear_OneLayer_VectorValue_Lux})
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

function initial_guess_integrator!(int::GeometricIntegrator{<:NonLinear_OneLayer_VectorValue_Lux})
    local network_labels = cache(int).network_labels
    local integrator = default_iguess_integrator(method(int))
    local h = int.problem.tstep
    local nstages = method(int).nstages
    local D = ndims(int)
    local problem = int.problem
    local S = nbasis(method(int))   
    local x = nlsolution(int)

    tem_ode=similar(problem,[0.,h],h/nstages,(q = StateVariable(int.solstep.q[:]), p = StateVariable(int.solstep.p[:]), λ = AlgebraicVariable(problem.ics.λ)))
    sol = integrate(tem_ode, integrator)

    for k in 1:D
        network_labels[:,k]=sol.s.q[:,k]
        cache(int).q̃[k] = sol.s.q[:,k][end]
        cache(int).p̃[k] = sol.s.p[:,k][end]
        x[D*S+k] = cache(int).p̃[k]
    end
end 

function initial_guess_networktraining!(int::GeometricIntegrator{<:NonLinear_OneLayer_VectorValue_Lux})
    local D = ndims(int)
    local show_status = method(int).show_status 
    local x = nlsolution(int)
    local S = nbasis(method(int))  

    local nepochs = method(int).training_epochs
    local NN = method(int).basis.NN
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local problem_module = method(int).problem_module 
    local initial_hamiltonian = method(int).problem_initial_hamitltonian
    local use_hamiltonian_loss = method(int).use_hamiltonian_loss
    local problem_params = method(int).problem_module.default_parameters

    if show_status
        print("\n network lables \n")
        print(network_labels')
    end

    ps,st=Lux.setup(Random.seed!(10),NN) #Random.seed!(1), create a temperoary ps,st 
    opt = Optimisers.Adam()
    st_opt = Optimisers.setup(opt, ps)
    err = 0
    for ep in 1:nepochs
        if use_hamiltonian_loss
            gs = Zygote.gradient(p -> vector_mse_energy_loss(network_inputs,network_labels',NN,p,st,problem_module,problem_params,initial_hamiltonian)[1],ps)[1]
            st_opt, ps = Optimisers.update(st_opt, ps, gs)
            err = vector_mse_energy_loss(network_inputs,network_labels',NN,ps,st,problem_module,problem_params,initial_hamiltonian)[1]
        else
            gs = Zygote.gradient(p -> vector_mse_loss(network_inputs,network_labels',NN,p,st)[1],ps)[1]
            st_opt, ps = Optimisers.update(st_opt, ps, gs)
            err = vector_mse_loss(network_inputs,network_labels',NN,ps,st)[1]
        end
    end
    show_status ? print("\n final loss: $err by $nepochs epochs") : nothing

    for k in 1:D
        for i in 1:S
            x[D*(i-1)+k] = ps[2].weight[k,i]
            x[D*(S+1)+i] = ps[1].weight[i]
            x[D*(S+1)+S+i] = ps[1].bias[i]
        end
    end

    if show_status
        print("\n network prediction \n")
        pre = NN(network_inputs,ps,st)[1]
        print(pre')

        # print("\n network parameters \n")
        # print(ps)

        print("\n initial guess x from network training \n")
        print(x)
    end
end


function GeometricIntegrators.Integrators.components!(x::AbstractVector{ST}, int::GeometricIntegrator{<:NonLinear_OneLayer_VectorValue_Lux}) where {ST}
    local D = ndims(int)
    local S = nbasis(method(int))
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
    local st = cache(int, ST).st

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

    for k in 1:D
        for i in 1:S
            ps[2].weight[k,i] = x[D*(i-1)+k] 
            ps[1].weight[i] = x[D*(S+1)+i] 
            ps[1].bias[i] = x[D*(S+1)+S+i] 
        end
    end

    # compute coefficients
    r₀[:] = NN[1]([0.0],ps[1],st[1])[1]
    r₁[:] = NN[1]([1.0],ps[1],st[1])[1]
    m = NN[1](quad_nodes',ps[1],st[1])[1]'
    a = vector_central_difference(method(int).basis,ps,st,quad_nodes')'

    # compute the derivatives of the coefficients on the quadrature nodes and at the boundaries
    ϵ=0.00001
    for d in 1:D 
        for j in eachindex(quad_nodes)
            g= Zygote.gradient(p->NN([quad_nodes[j]],p,st)[1][d],ps)[1]
            dqdWc[j,:,d] = g[1].weight[:]
            dqdbc[j,:,d] = g[1].bias[:]

            gvf= Zygote.gradient(p->NN([quad_nodes[j]+ϵ],p,st)[1][d],ps)[1]
            gvb= Zygote.gradient(p->NN([quad_nodes[j]-ϵ],p,st)[1][d],ps)[1]
            dvdWc[j,:,d] = (gvf[1].weight[:] .- gvb[1].weight[:])/(2*ϵ)
            dvdbc[j,:,d] = (gvf[1].bias[:] .- gvb[1].bias[:])/(2*ϵ)
        end

        g0= Zygote.gradient(p->NN([0.0],p,st)[1][d],ps)[1]
        dqdWr₀[:,d] = g0[1].weight[:]
        dqdbr₀[:,d] = g0[1].bias[:]

        g1 = Zygote.gradient(p->NN([1.0],p,st)[1][d],ps)[1]
        dqdWr₁[:,d] = g1[1].weight[:]
        dqdbr₁[:,d] = g1[1].bias[:]
    end

    # compute Q : q at quaadurature points
    for i in eachindex(Q)
        for d in eachindex(Q[i])
            y = zero(ST)
            for j in eachindex(X)
                y += m[i,j] * X[j][d]
            end
            Q[i][d] = y
        end
    end

    # compute q[t_{n+1}]
    for d in eachindex(q)
        y = zero(ST)
        for i in eachindex(X)
            y += r₁[i] * X[i][d]
        end
        q[d] = y
    end

    # compute V volicity at quadrature points
    for i in eachindex(V)
        for k in eachindex(V[i])
            y = zero(ST)
            for j in eachindex(X)
                y += a[i,j] * X[j][k]
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


function GeometricIntegrators.Integrators.residual!(b::Vector{ST}, int::GeometricIntegrator{<:NonLinear_OneLayer_VectorValue_Lux}) where {ST}
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
                z += method(int).b[j] * m[j,i] * F[j][k] * timestep(int)
                z += method(int).b[j] * a[j,i] * P[j][k]
            end
            b[D*(i-1)+k] = (r₁[i] * p̃[k] - r₀[i] * p̄[k]) - z
        end
    end 

    # the continue constraint from hamilton pontryagin principle
    for k in eachindex(q̄)
        y = zero(ST)
        for j in eachindex(X)
            y += r₀[j] * X[j][k]
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
            b[D*(S+1)+i] = dqdWr₁[i,k] * p̃[k]  - z
        end
    end 

    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P,F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdbc[j,i,k] 
                z += method(int).b[j] * P[j][k] * dvdbc[j,i,k]
            end
            b[D*(S+1)+S+i] = (dqdbr₁[i,k] * p̃[k] - dqdbr₀[i,k] * p̄[k]) - z
        end
    end 

end

function GeometricIntegrators.Integrators.residual!(b::AbstractVector{ST}, x::AbstractVector{ST}, int::GeometricIntegrator{<:NonLinear_OneLayer_VectorValue_Lux}) where {ST}
    @assert axes(x) == axes(b)

    # copy previous solution from solstep to cache
    reset!(cache(int, ST), current(solstep(int))...)

    # compute stages from nonlinear solver solution x
    GeometricIntegrators.Integrators.components!(x, int)

    # compute residual vector
    GeometricIntegrators.Integrators.residual!(b, int)
end

function GeometricIntegrators.Integrators.update!(x::AbstractVector{DT}, int::GeometricIntegrator{<:NonLinear_OneLayer_VectorValue_Lux}) where {DT}
    # copy previous solution from solstep to cache
    GeometricIntegrators.Integrators.reset!(cache(int, DT), current(solstep(int))...)

    # compute vector field at internal stages
    GeometricIntegrators.Integrators.components!(x, int)

    # compute final update
    solstep(int).q .= cache(int, DT).q̃
    solstep(int).p .= cache(int, DT).p̃
end


function GeometricIntegrators.Integrators.integrate_step!(int::GeometricIntegrator{<:NonLinear_OneLayer_VectorValue_Lux, <:AbstractProblemIODE})
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
end

function stages_compute!(int::GeometricIntegrator{<:NonLinear_OneLayer_VectorValue_Lux})
    local x = nlsolution(int)
    local stage_values = cache(int).stage_values
    local network_inputs = method(int).network_inputs
    local D = ndims(int)
    local S = nbasis(method(int))
    local σ = method(int).basis.activation
    local show_status = method(int).show_status
    local ps = cache(int).ps
    local st = cache(int).st
    local NN = method(int).basis.NN

    if show_status
        print("\n solution x after solving by Newton \n")
        print(x)
    end

    for k in 1:D
        for i in 1:S
            ps[2].weight[k,i] =  x[D*(i-1)+k] 
            ps[1].weight[i] = x[D*(S+1)+i] 
            ps[1].bias[i] = x[D*(S+1)+S+i] 
        end
    end
    stage_values = NN(network_inputs,ps,st)[1][:,2:end]

    if show_status
        print("\n stages prediction after solving \n")
        print(stage_values)
    end
end