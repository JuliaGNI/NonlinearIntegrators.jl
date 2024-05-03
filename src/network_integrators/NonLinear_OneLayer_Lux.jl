using CompactBasisFunctions
using GeometricIntegrators

struct NonLinear_OneLayer_Lux{T, NBASIS, NNODES, basisType <: Basis{T}} <: LODEMethod
    basis::basisType
    quadrature::QuadratureRule{T,NNODES}

    b::SVector{NNODES,T}
    c::SVector{NNODES,T}

    nstages::Int
    show_status::Bool
    network_inputs::Matrix{T}
    training_epochs::Int
    function NonLinear_OneLayer_Lux(basis::Basis{T}, quadrature::QuadratureRule{T};nstages::Int = 10,show_status::Bool=true,training_epochs::Int=50000) where {T}
        # get number of quadrature nodes and number of basis functions
        NNODES = QuadratureRules.nnodes(quadrature)
        NBASIS = CompactBasisFunctions.nbasis(basis)

        # get quadrature nodes and weights
        quad_weights = QuadratureRules.weights(quadrature)
        quad_nodes = QuadratureRules.nodes(quadrature)

        network_inputs = reshape(collect(0:1/nstages:1),1,nstages+1)
        new{T, NBASIS, NNODES,typeof(basis)}(basis, quadrature, quad_weights, quad_nodes, nstages, show_status, network_inputs, training_epochs)
    end
end

CompactBasisFunctions.basis(method::NonLinear_OneLayer_Lux) = method.basis
quadrature(method::NonLinear_OneLayer_Lux) = method.quadrature
CompactBasisFunctions.nbasis(method::NonLinear_OneLayer_Lux) = method.basis.S
nnodes(method::NonLinear_OneLayer_Lux) = QuadratureRules.nnodes(method.quadrature)
activation(method::NonLinear_OneLayer_Lux) = method.basis.activation
nstages(method::NonLinear_OneLayer_Lux) = method.nstages
show_status(method::NonLinear_OneLayer_Lux) = method.show_status
training_epochs(method::NonLinear_OneLayer_Lux) = method.training_epochs    

isexplicit(::Union{NonLinear_OneLayer_Lux, Type{<:NonLinear_OneLayer_Lux}}) = false
isimplicit(::Union{NonLinear_OneLayer_Lux, Type{<:NonLinear_OneLayer_Lux}}) = true
issymmetric(::Union{NonLinear_OneLayer_Lux, Type{<:NonLinear_OneLayer_Lux}}) = missing
issymplectic(::Union{NonLinear_OneLayer_Lux, Type{<:NonLinear_OneLayer_Lux}}) = missing

default_solver(::NonLinear_OneLayer_Lux) = Newton()
# default_iguess(::NonLinear_OneLayer_Lux) = HermiteExtrapolation()# HarmonicOscillator
default_iguess(::NonLinear_OneLayer_Lux) = MidpointExtrapolation()#CoupledHarmonicOscillator
default_iguess_integrator(::NonLinear_OneLayer_Lux) = ImplicitMidpoint()

struct NonLinear_OneLayer_LuxCache{ST,D,S,R} <: IODEIntegratorCache{ST,D}
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

    W ::Vector{Vector{ST}}
    bias::Vector{Vector{ST}}

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

    stage_values::Vector{ST}
    network_labels::VecOrMat{ST}
    function NonLinear_OneLayer_LuxCache{ST,D,S,R,N}() where {ST,D,S,R,N}
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

        # create first layer parameter vectors
        W = create_internal_stage_vector(ST, D, S)
        bias = create_internal_stage_vector(ST, D, S)

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

        return new(x, q̄, p̄, q̃, p̃, ṽ, f̃, s̃, X, Q, P, V, F, W, bias, r₀, r₁, m, a, 
            dqdWc, dqdbc, dvdWc, dvdbc, dqdWr₁, dqdWr₀, dqdbr₁, dqdbr₀,
            current_step,stage_values,network_labels)
    end
end

GeometricIntegrators.Integrators.nlsolution(cache::NonLinear_OneLayer_LuxCache) = cache.x

function GeometricIntegrators.Integrators.Cache{ST}(problem::AbstractProblemIODE, method::NonLinear_OneLayer_Lux; kwargs...) where {ST}
    NonLinear_OneLayer_LuxCache{ST, ndims(problem), nbasis(method), nnodes(method),nstages(method)}(; kwargs...)
end

@inline GeometricIntegrators.Integrators.CacheType(ST, problem::AbstractProblemIODE, method::NonLinear_OneLayer_Lux) = NonLinear_OneLayer_LuxCache{ST, ndims(problem), nbasis(method), nnodes(method),nstages(method)}

@inline function Base.getindex(c::NonLinear_OneLayer_LuxCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end

function GeometricIntegrators.Integrators.reset!(cache::NonLinear_OneLayer_LuxCache, t, q, p)
    copyto!(cache.q̄, q)
    copyto!(cache.p̄, p)
end

function GeometricIntegrators.Integrators.initial_guess!(int::GeometricIntegrator{<:NonLinear_OneLayer_Lux})
    local h = int.problem.tstep
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local show_status = method(int).show_status 
    local current_step = cache(int).current_step

    show_status ? print("\n current time step: $current_step") : nothing
    current_step+=1

    # choose initial guess method based on the value of h
    if h > 0.5
        initial_guess_Extrapolation!(int)
    else
        initial_guess_integrator!(int)
    end 
    
    if show_status
        print("\n network inputs")
        print(network_inputs)

        print("\n network labels from initial guess methods")
        print(network_labels)
    end

    initial_guess_networktraining!(int)

end

function initial_guess_Extrapolation!(int::GeometricIntegrator{<:NonLinear_OneLayer_Lux})
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local D = ndims(int)

    for i in eachindex(network_inputs)
        initialguess!(solstep(int).t̄+network_inputs[i]*h, cache(int).q̃, cache(int).p̃, solstep(int), int.problem, int.iguess)
        for k in 1:D
            network_labels[i,k] = cache(int).q̃[k]
        end
    end
    network_labels[1,:] = solstep(int).q #safe check for MidpointExtrapolation
end

function initial_guess_integrator!(int::GeometricIntegrator{<:NonLinear_OneLayer_Lux})
    local network_labels = cache(int).network_labels
    local integrator = method(int).default_iguess_integrator

    tem_ode = odeproblem([int.solstep.q[1],int.solstep.p[1]],tstep = h/ñ,tspan=(0,h))
    #TODO use similar method from GeometricEquations.jl
    sol = integrate(tem_ode, integrator)

    for k in 1:D
        network_labels[:,k]=sol.s.q[:,k]
        cache(int).q̃[k] = sol.s.q[:,k][end]
        cache(int).p̃[k] = sol.s.p[:,k][end]
        x[D*S+k] = cache(int).p̃[k]
    end
end 

function initial_guess_networktraining!(int)
    local D = ndims(int)
    local show_status = method(int).show_status 
    local x = nlsolution(int)
    local NN = method(int).basis.NN
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local nepochs = method(int).training_epochs

    for k in 1:D
        if show_status
            print("\n network lables for dimension $k")
            print(network_labels[:,k])
        end

        ps,st=Lux.setup(Random.default_rng(),NN) #Random.seed!(1)
        opt = Optimisers.Adam()
        st_opt = Optimisers.setup(opt, ps)
        err = 0
        for ep in 1:nepochs
            gs = Zygote.gradient(p -> mse_loss2(network_inputs',network_labels[:,k]',NN,p,st)[1],ps)[1]
            st_opt, ps = Optimisers.update(st_opt, ps, gs)
            err = mse_loss2(network_inputs',network_labels[:,k]',NN,ps,st)[1]

            if err < 5e-5
                show_status ? print("\n dimension $k,final loss: $err by $ep epochs") : nothing
                break
            elseif ep == nepochs
                show_status ? print("\n dimension $k,final loss: $err by $ep epochs") : nothing
            end
        end

        for i in 1:S
            x[D*(i-1)+k] = ps.layer_2.weight[i]
            x[D*(S+1)+D*(i-1)+k] = ps.layer_1.weight[i]
            x[D*(S+1 + S)+D*(i-1)+k] = ps.layer_1.bias[i]
        end

        if show_status
            print("\n network parameters for dimension $k")
            print(ps)
        end
    end

    if show_status
        print("\n initial guess x from network training")
        print(x)
    end

end

function GeometricIntegrators.Integrators.components!(x::AbstractVector{ST}, int::GeometricIntegrator{<:NonLinear_OneLayer_Lux}) where {ST}
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
end
