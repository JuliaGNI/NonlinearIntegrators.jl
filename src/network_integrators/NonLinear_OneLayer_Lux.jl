using CompactBasisFunctions
using GeometricIntegrators

struct NonLinear_OneLayer_Lux{T, NBASIS, NNODES, basisType <: Basis{T}} <: LODEMethod
    basis::basisType
    quadrature::QuadratureRule{T,NNODES}

    b::SVector{NNODES,T}
    c::SVector{NNODES,T}

    function NonLinear_OneLayer_Lux(basis::Basis{T}, quadrature::QuadratureRule{T}) where {T}
        # get number of quadrature nodes and number of basis functions
        NNODES = QuadratureRules.nnodes(quadrature)
        NBASIS = CompactBasisFunctions.nbasis(basis)

        # get quadrature nodes and weights
        quad_weights = QuadratureRules.weights(quadrature)
        quad_nodes = QuadratureRules.nodes(quadrature)

        new{T, NBASIS, NNODES,typeof(basis)}(basis, quadrature, quad_weights, quad_nodes)
    end
end

CompactBasisFunctions.basis(method::NonLinear_OneLayer_Lux) = method.basis
quadrature(method::NonLinear_OneLayer_Lux) = method.quadrature
CompactBasisFunctions.nbasis(method::NonLinear_OneLayer_Lux) = method.basis.S
nnodes(method::NonLinear_OneLayer_Lux) = QuadratureRules.nnodes(method.quadrature)
activation(method::NonLinear_OneLayer_Lux) = method.basis.activation

isexplicit(::Union{NonLinear_OneLayer_Lux, Type{<:NonLinear_OneLayer_Lux}}) = false
isimplicit(::Union{NonLinear_OneLayer_Lux, Type{<:NonLinear_OneLayer_Lux}}) = true
issymmetric(::Union{NonLinear_OneLayer_Lux, Type{<:NonLinear_OneLayer_Lux}}) = missing
issymplectic(::Union{NonLinear_OneLayer_Lux, Type{<:NonLinear_OneLayer_Lux}}) = missing

default_solver(::NonLinear_OneLayer_Lux) = Newton()
# default_iguess(::NonLinear_OneLayer_Lux) = HermiteExtrapolation()# HarmonicOscillator
default_iguess(::NonLinear_OneLayer_Lux) = MidpointExtrapolation()#CoupledHarmonicOscillator


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

    inte_step_n::Vector{ST}
    sub_values::Vector{ST}

    function NonLinear_OneLayer_LuxCache{ST,D,S,R}() where {ST,D,S,R}
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

        inte_step_n = zeros(ST, 1)
        sub_values = []
        new(x, q̄, p̄, q̃, p̃, ṽ, f̃, s̃, X, Q, P, V, F, W, bias, r₀, r₁, m, a, dqdWc, dqdbc, dvdWc, dvdbc, dqdWr₁, dqdWr₀, dqdbr₁, dqdbr₀,inte_step_n,sub_values)#da, dr₀, dr₁
    end
end

function GeometricIntegrators.Integrators.reset!(cache::NonLinear_OneLayer_LuxCache, t, q, p)
    copyto!(cache.q̄, q)
    copyto!(cache.p̄, p)
end

GeometricIntegrators.Integrators.nlsolution(cache::NonLinear_OneLayer_LuxCache) = cache.x

function GeometricIntegrators.Integrators.Cache{ST}(problem::AbstractProblemIODE, method::NonLinear_OneLayer_Lux; kwargs...) where {ST}
    NonLinear_OneLayer_LuxCache{ST, ndims(problem), nbasis(method), nnodes(method)}(; kwargs...)
end

@inline GeometricIntegrators.Integrators.CacheType(ST, problem::AbstractProblemIODE, method::NonLinear_OneLayer_Lux) = NonLinear_OneLayer_LuxCache{ST, ndims(problem), nbasis(method), nnodes(method)}
