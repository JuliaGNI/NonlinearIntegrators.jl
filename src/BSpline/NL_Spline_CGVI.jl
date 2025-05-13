struct Nonlinear_BSpline_Integrator{T,NBASIS,NNODES,basisType<:Basis{T}} <: LODEMethod
    basis::basisType
    quadrature::QuadratureRule{T,NNODES}

    b::SVector{NNODES,T}
    c::SVector{NNODES,T}

    function Nonlinear_BSpline_Integrator(basis::Basis{T}, quadrature::QuadratureRule{T}) where {T}
        NNODES = QuadratureRules.nnodes(quadrature)
        NBASIS = basis.S

        # get quadrature nodes and weights
        quad_weights = QuadratureRules.weights(quadrature)
        quad_nodes = QuadratureRules.nodes(quadrature)
        return new{T,NBASIS,NNODES,typeof(basis)}(basis, quadrature, quad_weights, quad_nodes)
    end
end

CompactBasisFunctions.basis(method::Nonlinear_BSpline_Integrator) = method.basis
quadrature(method::Nonlinear_BSpline_Integrator) = method.quadrature
CompactBasisFunctions.nbasis(method::Nonlinear_BSpline_Integrator) = method.basis.S
nnodes(method::Nonlinear_BSpline_Integrator) = QuadratureRules.nnodes(method.quadrature)

nbasis(::Nonlinear_BSpline_Integrator{T,NB,NN,TB}) where {T,NB,NN,TB} = NB
nnodes(::Nonlinear_BSpline_Integrator{T,NB,NN,TB}) where {T,NB,NN,TB} = NN
isexplicit(::Union{Nonlinear_BSpline_Integrator, Type{<:Nonlinear_BSpline_Integrator}}) = false
isimplicit(::Union{Nonlinear_BSpline_Integrator, Type{<:Nonlinear_BSpline_Integrator}}) = true
issymmetric(::Union{Nonlinear_BSpline_Integrator, Type{<:Nonlinear_BSpline_Integrator}}) = missing
issymplectic(::Union{Nonlinear_BSpline_Integrator, Type{<:Nonlinear_BSpline_Integrator}}) = true

isiodemethod(::Union{Nonlinear_BSpline_Integrator, Type{<:Nonlinear_BSpline_Integrator}}) = true

default_solver(::Nonlinear_BSpline_Integrator) = Newton()
default_iguess(::Nonlinear_BSpline_Integrator) = HermiteExtrapolation()

struct Nonlinear_BSpline_IntegratorCache{ST,D,S,R,N} <: IODEIntegratorCache{ST,D}
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

    internal_knots::Matrix{ST}
    coefficients::Matrix{ST}
    r₀::Matrix{ST}
    r₁::Matrix{ST}
    m::Array{ST,3}
    a::Array{ST,3}

    dqdWc::Array{ST,3}
    dvdWc::Array{ST,3}
    dqdWr₁::Matrix{ST}
    dqdWr₀::Matrix{ST}

    x_abstol_final::Vector{ST}
    x_reltol_final::Vector{ST}
    x_suctol_final::Vector{ST}
    f_abstol_final::Vector{ST}
    f_reltol_final::Vector{ST}
    f_suctol_final::Vector{ST}
    converge_status::Vector{Bool}

    function Nonlinear_BSpline_IntegratorCache{ST,D,S,R,N}() where {ST,D,S,R,N}
        x = zeros(ST, D * (S + 1 + N))

        q̄ = zeros(ST, D)
        p̄ = zeros(ST, D)

        # create temporary vectors
        q̃ = zeros(ST, D)
        p̃ = zeros(ST, D)
        ṽ = zeros(ST, D)
        f̃ = zeros(ST, D)
        s̃ = zeros(ST, D)

        # create internal stage vectors
        X = create_internal_stage_vector(ST, D, S)
        Q = create_internal_stage_vector(ST, D, R)
        P = create_internal_stage_vector(ST, D, R)
        V = create_internal_stage_vector(ST, D, R)
        F = create_internal_stage_vector(ST, D, R)

        internal_knots = zeros(ST, N, D)
        coefficients = zeros(ST, S, D)
        # create parameter vectors
        r₀ = zeros(ST, S, D)
        r₁ = zeros(ST, S, D)
        m = zeros(ST, R, S, D)
        a = zeros(ST, R, S, D)

        dqdWc = zeros(ST, R, N, D)
        dvdWc = zeros(ST, R, N, D)
        dqdWr₁ = zeros(ST, S, D)
        dqdWr₀ = zeros(ST, S, D)

        x_abstol_final = zeros(ST, 1)
        x_reltol_final = zeros(ST, 1)
        x_suctol_final = zeros(ST, 1)
        f_abstol_final = zeros(ST, 1)
        f_reltol_final = zeros(ST, 1)
        f_suctol_final = zeros(ST, 1)
        converge_status = [false]

        new{ST,D,S,R,N}(x, q̄, p̄, q̃, p̃, ṽ, f̃, s̃, X, Q, P, V, F,
            internal_knots, coefficients,
            r₀, r₁, m, a,
            dqdWc, dvdWc, dqdWr₁, dqdWr₀,
            x_abstol_final, x_reltol_final, x_suctol_final,
            f_abstol_final, f_reltol_final, f_suctol_final,
            converge_status)
    end
end

GeometricIntegrators.Integrators.nlsolution(cache::Nonlinear_BSpline_IntegratorCache) = cache.x

function GeometricIntegrators.Integrators.Cache{ST}(problem::AbstractProblemIODE, method::Nonlinear_BSpline_Integrator; kwargs...) where {ST}
    Nonlinear_BSpline_IntegratorCache{ST,ndims(problem),method.basis.S,nnodes(method),method.basis.N_internal_knots}(; kwargs...)
end

@inline GeometricIntegrators.Integrators.CacheType(ST, problem::AbstractProblemIODE, method::Nonlinear_BSpline_Integrator) = Nonlinear_BSpline_IntegratorCache{ST,ndims(problem),method.basis.S,nnodes(method),method.basis.N_internal_knots}

@inline function Base.getindex(c::Nonlinear_BSpline_IntegratorCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key) 
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end

function GeometricIntegrators.Integrators.reset!(cache::Nonlinear_BSpline_IntegratorCache, t, q, p)
    copyto!(cache.q̄, q)
    copyto!(cache.p̄, p)
end

function GeometricIntegrators.Integrators.initial_guess!(sol, history, params, int::GeometricIntegrator{<:Nonlinear_BSpline_Integrator})
    # set some local variables for convenience
    local D = ndims(int)
    local S = nbasis(method(int))
    local x = nlsolution(int)
    local h = int.problem.tstep
    local problem = int.problem
    local K = method(int).basis.K
    local N = method(int).basis.N_internal_knots

    tem_ode = similar(problem, [0.0, h], h / (S - 1), (q=StateVariable(sol.q[:]), p=StateVariable(sol.p[:])))
    tem_sol = integrate(tem_ode, ImplicitMidpoint())

    for d in 1:D
        xx = 0:1/(S-1):1 
        yy = collect(tem_sol.q[:, d])
        interpolation_function = interpolate(xx, yy, BSplineOrder(K))
        for i in 1:S
            x[D*(i-1)+d] = interpolation_function.spline.coefs[i]
        end

        cache(int).p̃[d] = tem_sol.p[:, d][end]
        x[D*S+d] = cache(int).p̃[d]
        for i in 1:N
            x[D*(S+1)+D*(i-1)+d] = method(int).basis.internal_knots[i]
        end
    end
end

function GeometricIntegrators.Integrators.components!(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:Nonlinear_BSpline_Integrator}) where {ST}
    local D = ndims(int)
    local S = nbasis(method(int))
    local C = cache(int, ST)
    local N = method(int).basis.N_internal_knots
    local quad_nodes = QuadratureRules.nodes(int.method.quadrature)

    local K = method(int).basis.K # Order of the BSpline basis
    local q = cache(int, ST).q̃
    local p = cache(int, ST).p̃
    local Q = cache(int, ST).Q
    local V = cache(int, ST).V
    local P = cache(int, ST).P
    local F = cache(int, ST).F
    local X = cache(int, ST).X

    local r₀ = cache(int, ST).r₀
    local r₁ = cache(int, ST).r₁
    local m = cache(int, ST).m
    local a = cache(int, ST).a
    local dqdWc = cache(int, ST).dqdWc
    local dvdWc = cache(int, ST).dvdWc
    local dqdWr₁ = cache(int, ST).dqdWr₁
    local dqdWr₀ = cache(int, ST).dqdWr₀
    local internal_knots = cache(int, ST).internal_knots
    local coefficients = cache(int, ST).coefficients
    # copy x to X
    for i in eachindex(X)
        for d in eachindex(X[i])
            X[i][d] = x[D*(i-1)+d]
            coefficients[i, d] = x[D*(i-1)+d]
        end
    end

    # copy x to p # momenta
    for k in eachindex(p)
        p[k] = x[D*S+k]
    end

    for d in 1:D
        for i in 1:N
            internal_knots[i, d] = x[D*(S+1)+D*(i-1)+d]
        end
    end

    # compute coefficients
    for d in 1:D
        knots = [0.0; internal_knots[:, d]; 1.0]
        B = BSplineBasis(BSplineOrder(K), knots)
        r₀[:, d] = [B[i](0.0) for i in 1:S]
        r₁[:, d] = [B[i](1.0) for i in 1:S]
        for j in eachindex(quad_nodes)
            m[j, :, d] = [B[i](quad_nodes[j]) for i in 1:S]
            for s in eachindex(B)
                tem_coes = zeros(S)
                tem_coes[s] = 1
                interpolation_function = BSplineKit.Spline(B, tem_coes)
                DB = BSplineKit.Derivative(1) * interpolation_function
                a[j, s, d] = DB(quad_nodes[j])
            end
        end
    end


    # compute the derivatives of the knots on the quadrature nodes and at the boundaries
    for d in 1:D
        for j in eachindex(quad_nodes)
            dqdWc[j, :, d] = ForwardDiff.gradient(t -> spline_value_at_knots(t, quad_nodes[j], K, coefficients[:, d]), internal_knots[:, d])
            dvdWc[j, :, d] = ForwardDiff.gradient(t -> velocity_value_at_knots(t, quad_nodes[j], K, coefficients[:, d]), internal_knots[:, d])
        end
        dqdWr₀[:, d] = ForwardDiff.gradient(t -> spline_value_at_knots(t, 0.0, K, coefficients[:, d]), internal_knots[:, d])
        dqdWr₁[:, d] = ForwardDiff.gradient(t -> spline_value_at_knots(t, 1.0, K, coefficients[:, d]), internal_knots[:, d])
    end

    # compute Q : q at quaadurature points
    for i in eachindex(Q)
        for d in eachindex(Q[i])
            y = zero(ST)
            for j in eachindex(X)
                y += m[i, j, d] * X[j][d]
            end
            Q[i][d] = y
        end
    end

    # compute q[t_{n+1}]
    for d in eachindex(q)
        y = zero(ST)
        for i in eachindex(X)
            y += r₁[i, d] * X[i][d]
        end
        q[d] = y
    end

    # compute V volicity at quadrature points
    for i in eachindex(V)
        for k in eachindex(V[i])
            y = zero(ST)
            for j in eachindex(X)
                y += a[i, j, k] * X[j][k]
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

function spline_value_at_knots(t0, eval_point, k, coes)
    knots = [0.0; t0; 1.0]
    B = BSplineBasis(BSplineOrder(k), knots)
    S = BSplineKit.Spline(B, coes)
    return S(eval_point)
end

function velocity_value_at_knots(t0, eval_point, k, coes)
    knots = [0.0; t0; 1.0]
    B = BSplineBasis(BSplineOrder(k), knots)
    S = BSplineKit.Spline(B, coes)
    DB = BSplineKit.Derivative(1) * S
    return DB(eval_point)
end

function GeometricIntegrators.Integrators.residual!(b::Vector{ST}, sol, params, int::GeometricIntegrator{<:Nonlinear_BSpline_Integrator}) where {ST}
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
    local m = cache(int, ST).m
    local a = cache(int, ST).a

    local dqdWc = cache(int, ST).dqdWc
    local dvdWc = cache(int, ST).dvdWc
    local dqdWr₁ = cache(int, ST).dqdWr₁
    local dqdWr₀ = cache(int, ST).dqdWr₀

    # compute b = - [(P-AF)], the residual in actual action, vatiation with respect to Q_{n,i}
    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += method(int).b[j] * m[j, i, k] * F[j][k] * timestep(int)
                z += method(int).b[j] * a[j, i, k] * P[j][k]
            end
            b[D*(i-1)+k] = (r₁[i, k] * p̃[k] - r₀[i, k] * p̄[k]) - z
        end
    end

    # the continue constraint from hamilton pontryagin principle
    for k in eachindex(q̄)
        y = zero(ST)
        for j in eachindex(X)
            y += r₀[j, k] * X[j][k]
        end
        b[D*S+k] = q̄[k] - y
    end

    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdWc[j, i, k]
                z += method(int).b[j] * P[j][k] * dvdWc[j, i, k]
            end
            b[D*(S+1+S)+D*(i-1)+k] = (dqdWr₁[i, k] * p̃[k] - dqdWr₀[i, k] * p̄[k]) - z
        end
    end
end

# Compute stages of Variational Partitioned Runge-Kutta methods.
function GeometricIntegrators.Integrators.residual!(b::AbstractVector{ST}, x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:Nonlinear_BSpline_Integrator}) where {ST}
    # check that x and b are compatible
    @assert axes(x) == axes(b)

    # compute stages from nonlinear solver solution x
    GeometricIntegrators.Integrators.components!(x, sol, params, int)

    # compute residual vector
    GeometricIntegrators.Integrators.residual!(b, sol, params, int)
end

function GeometricIntegrators.Integrators.update!(sol, params, int::GeometricIntegrator{<:Nonlinear_BSpline_Integrator}, DT)
    sol.q .= cache(int, DT).q̃
    sol.p .= cache(int, DT).p̃
end

function GeometricIntegrators.Integrators.update!(sol, params, x::AbstractVector{DT}, int::GeometricIntegrator{<:Nonlinear_BSpline_Integrator}) where {DT}
    # compute vector field at internal stages
    GeometricIntegrators.Integrators.components!(x, sol, params, int)

    # compute final update
    GeometricIntegrators.Integrators.update!(sol, params, int, DT)
end


function GeometricIntegrators.Integrators.integrate_step!(sol, history, params, int::GeometricIntegrator{<:Nonlinear_BSpline_Integrator,<:AbstractProblemIODE})
    # call nonlinear solver
    solve!(nlsolution(int), (b, x) -> GeometricIntegrators.Integrators.residual!(b, x, sol, params, int), solver(int))

    # print solver status
    # print_solver_status(int.solver.status, int.solver.params)

    # check if solution contains NaNs or error bounds are violated
    # check_solver_status(int.solver.status, int.solver.params)

    # copy solution status
    cache(int).x_abstol_final[1] = solver(int).status.rxₐ
    cache(int).x_reltol_final[1] = solver(int).status.rxᵣ
    cache(int).x_suctol_final[1] = solver(int).status.rxₛ
    cache(int).f_abstol_final[1] = solver(int).status.rfₐ
    cache(int).f_reltol_final[1] = solver(int).status.rfᵣ
    cache(int).f_suctol_final[1] = solver(int).status.rfₛ
    cache(int).converge_status[1] = solver(int).status.x_converged || solver(int).status.f_converged

    # compute final update
    GeometricIntegrators.Integrators.update!(sol, params, nlsolution(int), int)

    #compute the trajectory after solving by newton method
    stages_compute!(sol, int)

end