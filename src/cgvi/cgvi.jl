@doc raw"""
    CGVINodal <: LODEMethod

Continuous Galerkin Variational Integrator on a *nodal* basis. This is the linear
reference integrator of this package: the same formulation the network integrators use,
with an interpolatory basis in place of a network ansatz.

It differs from `GeometricIntegrators.CGVI`, which solves for all `S` basis coefficients
plus the endpoint momentum (`D*(S+1)` unknowns). Here the basis must be interpolatory with
nodes at both ends of the interval, so the coefficients *are* nodal values: the first is
pinned to the known `q̄`, the last *is* the new `q`, and the momentum is computed explicitly
rather than solved for. That leaves `D*(S-1)` unknowns in the nonlinear system. A Lagrange
basis on Lobatto nodes satisfies the requirement.

* `b`: weights of the quadrature rule
* `c`: nodes of the quadrature rule
* `x`: nodes of the basis
* `m`: mass matrix
* `a`: derivative matrix
* `r₀`: reconstruction coefficients at the beginning of the interval
* `r₁`: reconstruction coefficients at the end of the interval

"""
struct CGVINodal{T,NBASIS,NNODES,NDOF,basisType<:Basis{T}} <: LODEMethod
    basis::basisType
    quadrature::QuadratureRule{T,NNODES}

    b::SVector{NNODES,T}
    c::SVector{NNODES,T}

    x::SVector{NBASIS,T}

    m::SMatrix{NNODES,NBASIS,T,NDOF}
    a::SMatrix{NNODES,NBASIS,T,NDOF}

    r₀::SVector{NBASIS,T}
    r₁::SVector{NBASIS,T}

    function CGVINodal(basis::Basis{T}, quadrature::QuadratureRule{T}) where {T}
        # get number of quadrature nodes and number of basis functions
        NNODES = nnodes(quadrature)
        NBASIS = CompactBasisFunctions.nbasis(basis)

        # get quadrature nodes and weights
        quad_weights = QuadratureRules.weights(quadrature)
        quad_nodes = QuadratureRules.nodes(quadrature)

        # compute coefficients
        r₀ = zeros(T, NBASIS)
        r₁ = zeros(T, NBASIS)
        m = zeros(T, NNODES, NBASIS)
        a = zeros(T, NNODES, NBASIS)

        for i in eachindex(basis)
            r₀[i] = basis[zero(T), i]
            r₁[i] = basis[one(T), i]
            for j in eachindex(quad_nodes)
                m[j, i] = basis[quad_nodes[j], i]
                a[j, i] = basis'[quad_nodes[j], i]
            end
        end

        new{T,NBASIS,NNODES,NBASIS * NNODES,typeof(basis)}(basis, quadrature, quad_weights, quad_nodes, CompactBasisFunctions.grid(basis), m, a, r₀, r₁)
    end
end

basis(method::CGVINodal) = method.basis
quadrature(method::CGVINodal) = method.quadrature

nbasis(::CGVINodal{T,NB,NN}) where {T,NB,NN} = NB
nnodes(::CGVINodal{T,NB,NN}) where {T,NB,NN} = NN

# Qualified with `GeometricIntegratorsBase.` — see the note on the same four traits in
# `src/vise/vise.jl`. Unqualified, these created a shadowing `NonlinearIntegrators.isexplicit`
# and the framework kept answering `missing`. It matters most here: `issymplectic = true` is a
# real claim about the continuous-Galerkin construction on a linear basis, and it was the one
# property of this integrator that downstream code could have selected on.
GeometricIntegratorsBase.isexplicit(::Union{CGVINodal,Type{<:CGVINodal}}) = false
GeometricIntegratorsBase.isimplicit(::Union{CGVINodal,Type{<:CGVINodal}}) = true
GeometricIntegratorsBase.issymmetric(::Union{CGVINodal,Type{<:CGVINodal}}) = missing
GeometricIntegratorsBase.issymplectic(::Union{CGVINodal,Type{<:CGVINodal}}) = true

GeometricIntegratorsBase.isiodemethod(::Union{CGVINodal,Type{<:CGVINodal}}) = true

default_solver(::CGVINodal) = Newton()
default_iguess(::CGVINodal) = HermiteExtrapolation()

function Base.show(io::IO, method::CGVINodal)
    print(io, "\n")
    print(io, "  Continuous Galerkin Variational Integrator", "\n")
    print(io, "  ==========================================", "\n")
    print(io, "\n")
    print(io, "    c  = ", method.c, "\n")
    print(io, "    b  = ", method.b, "\n")
    print(io, "    x  = ", method.x, "\n")
    print(io, "    m  = ", method.m, "\n")
    print(io, "    a  = ", method.a, "\n")
    print(io, "    r₀ = ", method.r₀, "\n")
    print(io, "    r₁ = ", method.r₁, "\n")
    print(io, "\n")
end


# `{ST}` only — see the note on `SymbolicShallowNetCache` in `nvi/network_integrator_core.jl`.
# Unlike the network methods, `CGVINodal` already carried `NBASIS`/`NNODES` as type parameters,
# so `CacheType` did fold here; dropping the phantom parameters keeps the seven caches uniform
# and removes the dependency on that.
struct CGVINodalCache{ST} <: IODEIntegratorCache{ST}
    x::Vector{ST}

    q̃::Vector{ST}
    p̃::Vector{ST}
    ṽ::Vector{ST}
    f̃::Vector{ST}

    X::Vector{Vector{ST}}
    Q::Vector{Vector{ST}}
    P::Vector{Vector{ST}}
    V::Vector{Vector{ST}}
    F::Vector{Vector{ST}}


    function CGVINodalCache{ST}(ics, S::Int, R::Int) where {ST}
        D = length(vec(ics.q))
        x = zeros(ST, D * (S-1))

        # create temporary vectors
        q̃ = zeros(ST, D)
        p̃ = zeros(ST, D)
        ṽ = zeros(ST, D)
        f̃ = zeros(ST, D)

        # create internal stage vectors
        X = create_internal_stage_vector(ST, D, S)
        Q = create_internal_stage_vector(ST, D, R)
        P = create_internal_stage_vector(ST, D, R)
        V = create_internal_stage_vector(ST, D, R)
        F = create_internal_stage_vector(ST, D, R)

        new(x, q̃, p̃, ṽ, f̃, X, Q, P, V, F)
    end
end

GeometricIntegratorsBase.nlsolution(cache::CGVINodalCache) = cache.x

function GeometricIntegratorsBase.Cache{ST}(problem::AbstractProblemIODE, method::CGVINodal; kwargs...) where {ST}
    CGVINodalCache{ST}(initial_conditions(problem), nbasis(method), nnodes(method); kwargs...)
end

@inline GeometricIntegratorsBase.CacheType(ST, problem::AbstractProblemIODE, method::CGVINodal) = CGVINodalCache{ST}


function GeometricIntegratorsBase.initial_guess!(sol, history, params, int::GeometricIntegrator{<:CGVINodal})
    # set some local variables for convenience
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local x = nlsolution(int)

    # TODO: here we should not initialise with the solution q but with the degree of freedom x,
    # obtained e.g. from an L2 projection of q onto the basis

    for i in 1:length(method(int).x)-1
        soltmp = (
            t=sol.t + timestep(int) * (method(int).x[i+1] - 1),
            q=cache(int).q̃,
            p=cache(int).p̃,
            q̇=cache(int).ṽ,
            ṗ=cache(int).f̃,
        )
        solutionstep!(soltmp, history, problem(int), iguess(int))

        for k in 1:D
            x[D*(i-1)+k] = cache(int).q̃[k]
        end
    end
end


function GeometricIntegratorsBase.components!(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:CGVINodal}) where {ST}
    # set some local variables for convenience and clarity
    local C = cache(int, ST)
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local q̄ = sol.q    

    for d in 1:D
        C.X[1][d] = q̄[d]
    end

    # Copy x to X. The nonlinear solution vector holds the `S-1` free basis coefficients
    # `X[2] … X[S]` with the degree of freedom running fastest: coefficient `s+1` of degree of
    # freedom `d` sits at `x[D*(s-1)+d]`. That is the layout `initial_guess!` writes and the one
    # `residual!` assumes for `b`; all three have to agree or the Jacobian picks up a zero column.
    for s in 1:S-1
        for d in 1:D
            C.X[s+1][d] = x[D*(s-1)+d]
        end
    end

    # compute Q
    for i in eachindex(C.Q)
        for k in eachindex(C.Q[i])
            y = zero(ST)
            for j in eachindex(C.X)
                y += method(int).m[i, j] * C.X[j][k]
            end
            C.Q[i][k] = y
        end
    end

    # compute V
    for i in eachindex(C.V)
        for k in eachindex(C.V[i])
            y = zero(ST)
            for j in eachindex(C.X)
                y += method(int).a[i, j] * C.X[j][k]
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


function GeometricIntegratorsBase.residual!(b::Vector{ST}, sol, params, int::GeometricIntegrator{<:CGVINodal}) where {ST}
    # set some local variables for convenience and clarity
    local C = cache(int, ST)
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local p̄ = sol.p

    for k in eachindex(p̄)
        z = zero(ST)
        for j in eachindex(C.P, C.F)
            z += method(int).b[j] * C.F[j][k] * method(int).m[j, 1] * timestep(int)
            z += method(int).b[j] * C.P[j][k] * method(int).a[j, 1]
        end
        b[k] = p̄[k] + z
    end

    # compute b = - [(P-AF)]
    for i in 1:S-2  
        for k in 1:D 
            z = zero(ST)
            for j in eachindex(C.P, C.F) # quad_nodes index 
                z += method(int).b[j] * method(int).m[j, i+1] * C.F[j][k] * timestep(int)
                z += method(int).b[j] * method(int).a[j, i+1] * C.P[j][k]
            end
            b[D + D*(i-1)+k] = z
        end
    end
end


# Compute stages of Variational Partitioned Runge-Kutta methods.
function GeometricIntegratorsBase.residual!(b::AbstractVector{ST}, x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:CGVINodal}) where {ST}
    # check that x and b are compatible
    @assert axes(x) == axes(b)

    # compute stages from nonlinear solver solution x
    GeometricIntegratorsBase.components!(x, sol, params, int)

    # compute residual vector
    GeometricIntegratorsBase.residual!(b, sol, params, int)
end


function update_solution!(sol, params, int::GeometricIntegrator{<:CGVINodal}, ::Type{DT}) where {DT}
   local C = cache(int, DT)
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local h = timestep(int)

    # The basis is interpolatory with a node at the end of the interval, so the last coefficient
    # *is* the new position — one value per degree of freedom. Read it off `C.X`, which
    # `components!` has just filled, rather than indexing into the flat solution vector.
    for k in 1:D
        sol.q[k] = C.X[S][k]
    end

    for k in 1:D
        z = zero(DT)
        for j in 1:nnodes(method(int))
            z += method(int).b[j] * C.F[j][k] * method(int).m[j, S] * h
            z += method(int).b[j] * C.P[j][k] * method(int).a[j, S]
        end
        sol.p[k] = z
    end
end

function GeometricIntegratorsBase.update!(sol, params, x::AbstractVector{DT}, int::GeometricIntegrator{<:CGVINodal}) where {DT}
    # compute vector field at internal stages
    GeometricIntegratorsBase.components!(x, sol, params, int)

    # compute final update
    update_solution!(sol, params, int, DT)
end


function GeometricIntegratorsBase.integrate_step!(sol, history, params, int::GeometricIntegrator{<:CGVINodal,<:AbstractProblemIODE})
    # Call the nonlinear solver and act on the outcome it reports. The state-taking form: this is
    # a `GeometricIntegrator` and so carries a persistent `solverstate`, which the three-argument
    # form used here before ignored, allocating a fresh `NonlinearSolverState` on every time step.
    solverstatus = solve_with_status!(nlsolution(int), solver(int), solverstate(int), (sol, params, int))
    check_solver_status(solverstatus, int)

    # compute final update
    GeometricIntegratorsBase.update!(sol, params, nlsolution(int), int)
end