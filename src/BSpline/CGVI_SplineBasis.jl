@doc raw"""
Continuous Galerkin Variational Integrator.

* `b`: weights of the quadrature rule
* `c`: nodes of the quadrature rule
* `x`: nodes of the basis
* `m`: mass matrix
* `a`: derivative matrix
* `r₀`: reconstruction coefficients at the beginning of the interval
* `r₁`: reconstruction coefficients at the end of the interval

"""
struct CGVI_BSpline{T, NBASIS, NNODES, NDOF, basisType <: Basis{T}} <: LODEMethod
    basis::basisType
    quadrature::QuadratureRule{T,NNODES}

    b::SVector{NNODES,T}
    c::SVector{NNODES,T}

    m::SMatrix{NNODES, NBASIS, T}
    a::SMatrix{NNODES, NBASIS, T}

    r₀::SVector{NBASIS,T}
    r₁::SVector{NBASIS,T}

    function CGVI_BSpline(basis::Basis{T}, quadrature::QuadratureRule{T}) where {T}
        # get number of quadrature nodes and number of basis functions
        NNODES = QuadratureRules.nnodes(quadrature)
        NBASIS = length(basis)

        # get quadrature nodes and weights
        quad_weights = QuadratureRules.weights(quadrature)
        quad_nodes = QuadratureRules.nodes(quadrature)

        # compute coefficients
        r₀ = zeros(T, NBASIS)
        r₁ = zeros(T, NBASIS)
        m  = zeros(T, NNODES, NBASIS)
        a  = zeros(T, NNODES, NBASIS)

        for i in eachindex(basis.B)
            r₀[i] = basis.B[i](zero(T))
            r₁[i] = basis.B[i](one(T))
            for j in eachindex(quad_nodes)
                m[j,i] = basis.B[i](quad_nodes[j])
                a[j,i] = basis.Der_B[i](quad_nodes[j])
            end
        end

        new{T, NBASIS, NNODES, NBASIS * NNODES, typeof(basis)}(basis, quadrature, quad_weights, quad_nodes, m, a, r₀, r₁)
    end
end

basis(method::CGVI_BSpline) = method.basis
quadrature(method::CGVI_BSpline) = method.quadrature

nbasis(::CGVI_BSpline{T,NB,NN}) where {T,NB,NN} = NB
nnodes(::CGVI_BSpline{T,NB,NN}) where {T,NB,NN} = NN

isexplicit(::Union{CGVI_BSpline, Type{<:CGVI_BSpline}}) = false
isimplicit(::Union{CGVI_BSpline, Type{<:CGVI_BSpline}}) = true
issymmetric(::Union{CGVI_BSpline, Type{<:CGVI_BSpline}}) = missing
issymplectic(::Union{CGVI_BSpline, Type{<:CGVI_BSpline}}) = true

isiodemethod(::Union{CGVI_BSpline, Type{<:CGVI_BSpline}}) = true

default_solver(::CGVI_BSpline) = Newton()
default_iguess(::CGVI_BSpline) = HermiteExtrapolation()

function Base.show(io::IO, method::CGVI_BSpline)
    print(io, "\n")
    print(io, "  Continuous Galerkin Variational Integrator", "\n")
    print(io, "  ==========================================", "\n")
    print(io, "\n")
    print(io, "    c  = ", method.c, "\n")
    print(io, "    b  = ", method.b, "\n")
    print(io, "    m  = ", method.m, "\n")
    print(io, "    a  = ", method.a, "\n")
    print(io, "    r₀ = ", method.r₀, "\n")
    print(io, "    r₁ = ", method.r₁, "\n")
    print(io, "\n")
end


struct CGVI_BSplineCache{ST,D,S,R} <: IODEIntegratorCache{ST,D}
    x::Vector{ST}

    q̃::Vector{ST}
    p̃::Vector{ST}
    ṽ::Vector{ST}
    f̃::Vector{ST}
    s̃::Vector{ST}

    X::Vector{Vector{ST}}
    Q::Vector{Vector{ST}}
    P::Vector{Vector{ST}}
    V::Vector{Vector{ST}}
    F::Vector{Vector{ST}}


    function CGVI_BSplineCache{ST,D,S,R}() where {ST,D,S,R}
        x = zeros(ST, D*(S+1))
        
        # create temporary vectors
        q̃ = zeros(ST,D)
        p̃ = zeros(ST,D)
        ṽ = zeros(ST,D)
        f̃ = zeros(ST,D)
        s̃ = zeros(ST,D)

        # create internal stage vectors
        X = create_internal_stage_vector(ST,D,S)
        Q = create_internal_stage_vector(ST,D,R)
        P = create_internal_stage_vector(ST,D,R)
        V = create_internal_stage_vector(ST,D,R)
        F = create_internal_stage_vector(ST,D,R)

        new(x, q̃, p̃, ṽ, f̃, s̃, X, Q, P, V, F)
    end
end

GeometricIntegrators.Integrators.nlsolution(cache::CGVI_BSplineCache) = cache.x

function GeometricIntegrators.Integrators.Cache{ST}(problem::AbstractProblemIODE, method::CGVI_BSpline; kwargs...) where {ST}
    CGVI_BSplineCache{ST, ndims(problem), nbasis(method), nnodes(method)}(; kwargs...)
end

@inline GeometricIntegrators.Integrators.CacheType(ST, problem::AbstractProblemIODE, method::CGVI_BSpline) = CGVI_BSplineCache{ST, ndims(problem), nbasis(method), nnodes(method)}

@inline function Base.getindex(c::CGVI_BSplineCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end

function GeometricIntegrators.Integrators.reset!(cache::CGVI_BSplineCache, t, q, p)
    copyto!(cache.q̄, q)
    copyto!(cache.p̄, p)
end


function GeometricIntegrators.Integrators.initial_guess!(sol, history, params, int::GeometricIntegrator{<:CGVI_BSpline})
    # set some local variables for convenience
    local D = ndims(int)
    local S = nbasis(method(int))
    local x = nlsolution(int)
    local h = int.problem.tstep
    local problem = int.problem
    local k = method(int).basis.k

    tem_ode = similar(problem, [0.0, h], h / (S-1), (q=StateVariable(sol.q[:]), p=StateVariable(sol.p[:])))
    tem_sol = integrate(tem_ode, ImplicitMidpoint())

    for d in 1:D
        xx = 0:1/(S-1):1
        yy = collect(tem_sol.q[:, d])
        interpolation_function = interpolate(xx, yy, BSplineOrder(k))
        for i in 1:S
            x[D*(i-1)+d] = interpolation_function.spline.coefs[i]
        end

        cache(int).p̃[d] = tem_sol.p[:, d][end]
        x[D*S+d] = cache(int).p̃[d]   
    end
end


function GeometricIntegrators.Integrators.components!(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:CGVI_BSpline}) where {ST}
    # set some local variables for convenience and clarity
    local C = cache(int, ST)
    local D = ndims(int)
    local S = nbasis(method(int))

    # copy x to X
    for i in eachindex(C.X)
        for k in eachindex(C.X[i])
            C.X[i][k] = x[D*(i-1)+k]
        end
    end

    # copy x to p
    for k in eachindex(C.p̃)
        C.p̃[k] = x[D*S+k]
    end

    # compute Q
    for i in eachindex(C.Q)
        for k in eachindex(C.Q[i])
            y = zero(ST)
            for j in eachindex(C.X)
                y += method(int).m[i,j] * C.X[j][k]
            end
            C.Q[i][k] = y
        end
    end

    # compute q
    for k in eachindex(C.q̃)
        y = zero(ST)
        for i in eachindex(C.X)
            y += method(int).r₁[i] * C.X[i][k]
        end
        C.q̃[k] = y
    end

    # compute V
    for i in eachindex(C.V)
        for k in eachindex(C.V[i])
            y = zero(ST)
            for j in eachindex(C.X)
                y += method(int).a[i,j] * C.X[j][k]
            end
            C.V[i][k] = y / timestep(int)
        end
    end

    # compute P=ϑ(Q,V) and F=f(Q,V)
    for i in eachindex(C.Q, C.V, C.P, C.F)
        tᵢ = sol.t + timestep(int) * (method(int).c[i] - 1)
        equations(int).ϑ(C.P[i], tᵢ, C.Q[i], C.V[i], params)
        equations(int).f(C.F[i], tᵢ, C.Q[i], C.V[i], params)
    end
end


function GeometricIntegrators.Integrators.residual!(b::Vector{ST}, sol, params, int::GeometricIntegrator{<:CGVI_BSpline}) where {ST}
    # set some local variables for convenience and clarity
    local C = cache(int, ST)
    local D = ndims(int)
    local S = nbasis(method(int))

    # compute b = - [(P-AF)]
    for i in eachindex(method(int).r₀, method(int).r₁)
        for k in eachindex(C.p̃)#, sol.p # TODO
            z = zero(ST)
            for j in eachindex(C.P, C.F)
                z += method(int).b[j] * method(int).m[j,i] * C.F[j][k] * timestep(int)
                z += method(int).b[j] * method(int).a[j,i] * C.P[j][k]
            end
            b[D*(i-1)+k] = (method(int).r₁[i] * C.p̃[k] - method(int).r₀[i] * sol.p[k]) - z
        end
    end

    # compute b = - [(q-r₀Q)]
    for k in eachindex(sol.q)
        y = zero(ST)
        for j in eachindex(C.X)
            y += method(int).r₀[j] * C.X[j][k]
        end
        b[D*S+k] = sol.q[k] - y
    end
end


# Compute stages of Variational Partitioned Runge-Kutta methods.
function GeometricIntegrators.Integrators.residual!(b::AbstractVector{ST}, x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:CGVI_BSpline}) where {ST}
    # check that x and b are compatible
    @assert axes(x) == axes(b)

    # compute stages from nonlinear solver solution x
    GeometricIntegrators.Integrators.components!(x, sol, params, int)

    # compute residual vector
    GeometricIntegrators.Integrators.residual!(b, sol, params, int)
end


function GeometricIntegrators.Integrators.update!(sol, params, int::GeometricIntegrator{<:CGVI_BSpline}, DT)
    sol.q .= cache(int, DT).q̃
    sol.p .= cache(int, DT).p̃
end

function GeometricIntegrators.Integrators.update!(sol, params, x::AbstractVector{DT}, int::GeometricIntegrator{<:CGVI_BSpline}) where {DT}
    # compute vector field at internal stages
    GeometricIntegrators.Integrators.components!(x, sol, params, int)

    # compute final update
    GeometricIntegrators.Integrators.update!(sol, params, int, DT)
end


function GeometricIntegrators.Integrators.integrate_step!(sol, history, params, int::GeometricIntegrator{<:CGVI_BSpline, <:AbstractProblemIODE})
    # call nonlinear solver
    solve!(nlsolution(int), (b,x) -> GeometricIntegrators.Integrators.residual!(b, x, sol, params, int), solver(int))

    # print solver status
    # print_solver_status(int.solver.status, int.solver.params)

    # check if solution contains NaNs or error bounds are violated
    # check_solver_status(int.solver.status, int.solver.params)

    # compute final update
    GeometricIntegrators.Integrators.update!(sol, params, nlsolution(int), int)
end
