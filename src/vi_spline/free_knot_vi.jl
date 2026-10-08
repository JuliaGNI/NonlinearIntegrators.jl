# VISplineFree: the free-knot B-spline variational integrator — Theorem 3 of
# benchmark/theory/relu_cgvi_equivalence_en.md, as prototyped (one dimension, truncated powers) in
# benchmark/theory/free_knot_vi.jl (C7).
#
# One step [tₙ, tₙ + h], τ ∈ [0, 1]. The trial space S_k(Z) is spanned by the clamped B-splines of
# degree k on the mesh 0 < z₁ < … < z_m < 1, the knots Z being shared by all D dimensions, and
# the quadrature is the base rule on every cell of that mesh, so that it is not cut by a kink
# (Remark 5.4) and all dimensions are evaluated at the same times.
#
#   inner (Z fixed): the Galerkin VI on S_k(Z), i.e. the CGVI equations, for the unknowns
#          x[D(i−1)+d] = c[i,d] (i = 1..N, N = k+m+1) and x[DN+d] = p̃[d],
#          solved by the integrator's nonlinear solver (Theorem 2).
#   outer (Z):       g(Z) = dL_d^Z(qₙ, qₙ₊₁)/dZ = 0. By the envelope formula (proof of
#          Theorem 3, step 4), g = ∂A/∂Z at fixed c less the boundary terms p̃·∂q(1)/∂Z and
#          p̄·∂q(0)/∂Z. A clamped basis has q(0) = c₁ and q(1) = c_N for every Z, so these
#          vanish and g = ∇_Z A(c, Z), which ForwardDiff takes through SimpleSplines, including
#          the motion of the quadrature nodes and weights with the knots. Z is updated by a
#          safeguarded Newton with a forward-difference Jacobian (m more inner solves): knots in
#          [gap, 1 − gap] with spacing at least gap, every move at most maxmove, each step
#          starting from the knots of the previous step. A step whose knot equation does not
#          converge takes the knots of smallest |g|, is no longer variational, and is counted
#          in the cache's `fallbacks`.

"""
    VISplineFree(degree, nknots, quadrature; extrapolation_substep = 10, gtol = 1e-12,
                 maxouter = 30, dz = 1e-6, maxmove = 0.1, gap = 0.02)

Free-knot B-spline variational integrator: B-splines of `degree` k with `nknots` interior
knots, shared by all dimensions and found by the variational principle, and the rule
`quadrature` on every cell of the knot mesh. `nknots = 0` is CGVI(P_k) on `quadrature`. The problem has to be an `LODEProblem`: the knot
equation differentiates the discrete action, which needs the Lagrangian.

The knot equation is solved to |g|∞ ≤ `gtol`·max(1, |A|), A the discrete action, in at most
`maxouter` Newton iterations with forward-difference step `dz`; `maxouter = 0` freezes the
knots at their uniform seed. See the head of the file for `maxmove` and `gap`.
"""
struct VISplineFree{T, QT <: QuadratureRule{T}} <: LODEMethod
    degree::Int
    nknots::Int
    quadrature::QT
    extrapolation_substep::Int
    gtol::T
    maxouter::Int
    dz::T
    maxmove::T
    gap::T
end

function VISplineFree(degree::Int, nknots::Int, quadrature::QuadratureRule{T};
        extrapolation_substep::Int = 10, gtol = 1e-12, maxouter::Int = 30, dz = 1e-6,
        maxmove = 0.1, gap = 0.02) where {T}
    degree ≥ 1 || throw(ArgumentError("the degree must be at least 1, got $degree"))
    nknots ≥ 0 || throw(ArgumentError("the number of knots must be non-negative, got $nknots"))
    (nknots + 1) * gap < 1 || throw(ArgumentError("$nknots knots do not fit with gap = $gap"))
    VISplineFree{T, typeof(quadrature)}(degree, nknots, quadrature, extrapolation_substep,
        T(gtol), maxouter, T(dz), T(maxmove), T(gap))
end

nbasis(m::VISplineFree) = m.degree + m.nknots + 1
nnodes(m::VISplineFree) = (m.nknots + 1) * QuadratureRules.nnodes(m.quadrature)

GeometricIntegratorsBase.isexplicit(::Union{VISplineFree, Type{<:VISplineFree}}) = false
GeometricIntegratorsBase.isimplicit(::Union{VISplineFree, Type{<:VISplineFree}}) = true
GeometricIntegratorsBase.issymmetric(::Union{VISplineFree, Type{<:VISplineFree}}) = missing
GeometricIntegratorsBase.issymplectic(::Union{VISplineFree, Type{<:VISplineFree}}) = true

default_solver(::VISplineFree) = Newton()
default_iguess(::VISplineFree) = GeometricIntegratorsBase.NoInitialGuess()

"""
The basis of S_k(Z) and the rule of `method` on every cell of the mesh `[0, Z…, 1]`, as
`(basis, nodes, weights)` on [0, 1]. Generic in the element type of `Z`, so that ForwardDiff
can differentiate through it.
"""
function knot_space(method::VISplineFree, Z::AbstractVector{T}) where {T}
    y = [zero(T); Z; one(T)]
    c, b = QuadratureRules.nodes(method.quadrature), QuadratureRules.weights(method.quadrature)
    nodes = [y[l] + (y[l + 1] - y[l]) * c[r] for l in 1:(length(y) - 1) for r in eachindex(c)]
    weights = [(y[l + 1] - y[l]) * b[r] for l in 1:(length(y) - 1) for r in eachindex(b)]
    BSplineBasis(GeneralMesh(y), method.degree), nodes, weights
end

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

struct VISplineFreeCache{ST} <: IODEIntegratorCache{ST}
    x::Vector{ST}           # Newton unknowns, length D(N+1)

    q̄::Vector{ST}
    p̄::Vector{ST}

    q̃::Vector{ST}
    p̃::Vector{ST}
    ṽ::Vector{ST}
    f̃::Vector{ST}

    X::Vector{Vector{ST}}   # coefficients c[i, :]
    Q::Vector{Vector{ST}}   # q, ϑ, q̇ and f at the quadrature nodes
    P::Vector{Vector{ST}}
    V::Vector{Vector{ST}}
    F::Vector{Vector{ST}}

    # the knots and the tables of the current S_k(Z); only the cache of the problem's datatype
    # holds them, the inner solve reads them from there whatever its ST
    Z::Vector{ST}
    c::Vector{ST}           # quadrature nodes and weights on [0, 1]
    b::Vector{ST}
    m::Matrix{ST}           # m[j, i] = Bᵢ(cⱼ)
    a::Matrix{ST}           # a[j, i] = Bᵢ′(cⱼ)
    r₀::Vector{ST}          # Bᵢ(0)
    r₁::Vector{ST}          # Bᵢ(1)

    network_labels::Matrix{ST}
    fallbacks::Vector{Int}  # steps whose knot equation did not converge

    function VISplineFreeCache{ST}(ics, N::Int, m::Int, R::Int, n::Int) where {ST}
        D = length(vec(ics.q))
        new(zeros(ST, D * (N + 1)), zeros(ST, D), zeros(ST, D), zeros(ST, D), zeros(ST, D),
            zeros(ST, D), zeros(ST, D), [zeros(ST, D) for _ in 1:N], [zeros(ST, D) for _ in 1:R],
            [zeros(ST, D) for _ in 1:R], [zeros(ST, D) for _ in 1:R], [zeros(ST, D) for _ in 1:R],
            ST[ST(l) / ST(m + 1) for l in 1:m], zeros(ST, R), zeros(ST, R), zeros(ST, R, N),
            zeros(ST, R, N), zeros(ST, N), zeros(ST, N), zeros(ST, n + 1, D), [0])
    end
end

function GeometricIntegratorsBase.Cache{ST}(
        problem::AbstractProblemIODE, method::VISplineFree; kwargs...) where {ST}
    # the knot equation differentiates the discrete action, so it needs L itself
    problem isa LODEProblem || throw(ArgumentError(
        "VISplineFree needs the Lagrangian of an LODEProblem, got a $(nameof(typeof(problem)))"))
    VISplineFreeCache{ST}(initial_conditions(problem), nbasis(method), method.nknots,
        nnodes(method), method.extrapolation_substep)
end

GeometricIntegratorsBase.CacheType(ST, ::AbstractProblemIODE, ::VISplineFree) =
    VISplineFreeCache{ST}

GeometricIntegratorsBase.nlsolution(c::VISplineFreeCache) = c.x

function GeometricIntegratorsBase.reset!(cache::VISplineFreeCache, _, q, p)
    copyto!(cache.q̄, q)
    copyto!(cache.p̄, p)
end

"Set the knots of the step to `Z` and tabulate the basis of S_k(Z) on its rule."
function set_knots!(int::GeometricIntegrator{<:VISplineFree}, Z)
    C = cache(int)
    basis, c, b = knot_space(method(int), Z)
    C.Z .= Z
    C.c .= c
    C.b .= b
    for i in 1:nbasis(method(int))
        C.r₀[i] = SimpleSplines.evaluate(basis, i, 0.0)
        C.r₁[i] = SimpleSplines.evaluate(basis, i, 1.0)
        for j in eachindex(c)
            C.m[j, i] = SimpleSplines.evaluate(basis, i, c[j])
            C.a[j, i] = SimpleSplines.evaluate(basis, i, c[j], 1)
        end
    end
end

# ---------------------------------------------------------------------------
# initial_guess!: the coefficients by a least-squares fit of S_k(Z) to an implicit-midpoint
# trajectory over the step, Z the knots of the previous step; p̃ from that trajectory
# ---------------------------------------------------------------------------

function GeometricIntegratorsBase.initial_guess!(
        sol, history, params, int::GeometricIntegrator{<:VISplineFree})
    n = method(int).extrapolation_substep
    N = nbasis(method(int))
    D = length(cache(int).q̃)
    h = timestep(int)
    x = nlsolution(int)

    tem_ode = similar(int.problem, [zero(h), h], h / n,
        (q = StateVariable(sol.q[:]), p = StateVariable(sol.p[:])))
    tem_sol = integrate(tem_ode, GeometricIntegratorsBase.ImplicitMidpoint())

    basis = first(knot_space(method(int), cache(int).Z))
    B = qr(basis[collect(range(0, 1; length = n + 1)), :])
    for d in 1:D
        cache(int).network_labels[:, d] = tem_sol.q[:, d]
        c = B \ cache(int).network_labels[:, d]
        for i in 1:N
            x[D * (i - 1) + d] = c[i]
        end
        x[D * N + d] = tem_sol.p[:, d][end]
    end
end

# ---------------------------------------------------------------------------
# inner problem: the CGVI equations on the tables of the current knots
# ---------------------------------------------------------------------------

function GeometricIntegratorsBase.components!(x::AbstractVector{ST}, sol, params,
        int::GeometricIntegrator{<:VISplineFree}) where {ST}
    N = nbasis(method(int))
    D = length(cache(int).q̃)
    C = cache(int, ST)
    K = cache(int)          # the tables
    h = timestep(int)

    for i in 1:N, d in 1:D
        C.X[i][d] = x[D * (i - 1) + d]
    end
    for d in 1:D
        C.p̃[d] = x[D * N + d]
        C.q̃[d] = sum(K.r₁[i] * C.X[i][d] for i in 1:N)
    end
    for j in eachindex(C.Q), d in 1:D
        C.Q[j][d] = sum(K.m[j, i] * C.X[i][d] for i in 1:N)
        C.V[j][d] = sum(K.a[j, i] * C.X[i][d] for i in 1:N) / h
    end
    for j in eachindex(C.Q)
        tⱼ = sol.t + h * (K.c[j] - 1)
        equations(int).ϑ(C.P[j], tⱼ, C.Q[j], C.V[j], params)
        equations(int).f(C.F[j], tⱼ, C.Q[j], C.V[j], params)
    end
end

function GeometricIntegratorsBase.residual!(b::AbstractVector{ST}, x::AbstractVector{ST}, sol,
        params, int::GeometricIntegrator{<:VISplineFree}) where {ST}
    GeometricIntegratorsBase.components!(x, sol, params, int)
    N = nbasis(method(int))
    D = length(cache(int).q̃)
    C = cache(int, ST)
    K = cache(int)
    h = timestep(int)

    # δA/δc[i, d] = 0
    for i in 1:N, d in 1:D
        z = sum(K.b[j] * (h * K.m[j, i] * C.F[j][d] + K.a[j, i] * C.P[j][d]) for j in eachindex(C.P))
        b[D * (i - 1) + d] = K.r₁[i] * C.p̃[d] - K.r₀[i] * sol.p[d] - z
    end
    # q_h(0) = q̄
    for d in 1:D
        b[D * N + d] = sol.q[d] - sum(K.r₀[i] * C.X[i][d] for i in 1:N)
    end
end

function GeometricIntegratorsBase.update!(sol, params, x::AbstractVector{DT},
        int::GeometricIntegrator{<:VISplineFree}) where {DT}
    GeometricIntegratorsBase.components!(x, sol, params, int)
    sol.q .= cache(int, DT).q̃
    sol.p .= cache(int, DT).p̃
end

# ---------------------------------------------------------------------------
# outer problem: the knots
# ---------------------------------------------------------------------------

"""
The discrete action A(c, Z) = h Σⱼ bⱼ L(tⱼ, q(cⱼ), q′(cⱼ)/h) of the coefficients `X`
(`X[i][d]`) on S_k(Z), its rule moving with `Z`.
"""
function knot_action(int::GeometricIntegrator{<:VISplineFree}, Z, X, sol, params)
    basis, c, b = knot_space(method(int), Z)
    N, D, h = nbasis(method(int)), length(first(X)), timestep(int)
    sum(eachindex(c)) do j
        φ = [SimpleSplines.evaluate(basis, i, c[j]) for i in 1:N]
        φ′ = [SimpleSplines.evaluate(basis, i, c[j], 1) for i in 1:N]
        Q = [sum(X[i][d] * φ[i] for i in 1:N) for d in 1:D]
        V = [sum(X[i][d] * φ′[i] for i in 1:N) for d in 1:D] ./ h
        h * b[j] * equations(int).l(sol.t + h * (c[j] - 1), Q, V, params)
    end
end

"Solve the inner problem on the knots `Z` from the start value `x₀`; returns the solver status."
function inner_solve!(int::GeometricIntegrator{<:VISplineFree}, Z, x₀, sol, params)
    set_knots!(int, Z)
    nlsolution(int) .= x₀
    solve_with_status!(nlsolution(int), solver(int), solverstate(int), (sol, params, int))
end

"`(g, tolerance)` at the inner solution in `nlsolution(int)` for the knots of the cache."
function knot_gradient(int::GeometricIntegrator{<:VISplineFree}, sol, params)
    GeometricIntegratorsBase.components!(nlsolution(int), sol, params, int)
    X, Z = deepcopy(cache(int).X), copy(cache(int).Z)
    A = knot_action(int, Z, X, sol, params)
    g = ForwardDiff.gradient(z -> knot_action(int, z, X, sol, params), Z)
    g, method(int).gtol * max(1, abs(A))
end

"The knots `Z` sorted, clamped into [gap, 1 − gap] and at least `gap` apart."
function sanitize_knots(Z, gap)
    Z = sort(clamp.(Z, gap, 1 - gap))
    for i in 2:length(Z)
        Z[i] = max(Z[i], Z[i - 1] + gap)
    end
    isempty(Z) || (Z[end] = min(Z[end], 1 - gap))
    for i in (length(Z) - 1):-1:1
        Z[i] = min(Z[i], Z[i + 1] - gap)
    end
    Z
end

function GeometricIntegratorsBase.integrate_step!(
        sol, history, params, int::GeometricIntegrator{<:VISplineFree, <:AbstractProblemIODE})
    M = method(int)
    Z = sanitize_knots(copy(cache(int).Z), M.gap)
    status = inner_solve!(int, Z, copy(nlsolution(int)), sol, params)
    x = copy(nlsolution(int))
    best = (Z = copy(Z), x = copy(x), status = status, gn = Inf)
    # no knots, or knots frozen (maxouter = 0): the inner solve is the step
    converged = M.nknots == 0 || M.maxouter == 0
    for it in 0:(converged ? -1 : M.maxouter)
        SimpleSolvers.isconverged(status) || break
        g, gtol = knot_gradient(int, sol, params)
        norm(g, Inf) < best.gn && (best = (Z = copy(Z), x = copy(x), status = status, gn = norm(g, Inf)))
        (converged = norm(g, Inf) ≤ gtol) && break
        it == M.maxouter && break
        J = fill(oftype(first(g), NaN), length(g), length(g))   # NaN if a shifted solve fails
        for l in eachindex(Z)
            Zₗ = copy(Z)
            Zₗ[l] += M.dz
            SimpleSolvers.isconverged(inner_solve!(int, Zₗ, x, sol, params)) || break
            J[:, l] = (first(knot_gradient(int, sol, params)) .- g) ./ M.dz
        end
        ΔZ = try
            -(J \ g)
        catch e
            e isa SingularException || rethrow()
            -sign.(g) .* M.maxmove / 10     # a singular knot Jacobian: a small descent step
        end
        all(isfinite, ΔZ) || break          # an inner solve on the shifted knots failed
        Z = sanitize_knots(Z .+ clamp.(ΔZ, -M.maxmove, M.maxmove), M.gap)
        status = inner_solve!(int, Z, x, sol, params)
        x = copy(nlsolution(int))
    end
    if !converged && isfinite(best.gn)
        cache(int).fallbacks[1] += 1
        Z, x, status = best.Z, best.x, best.status
        set_knots!(int, Z)
        nlsolution(int) .= x
    end
    check_solver_status(status, int)
    GeometricIntegratorsBase.update!(sol, params, nlsolution(int), int)
end
