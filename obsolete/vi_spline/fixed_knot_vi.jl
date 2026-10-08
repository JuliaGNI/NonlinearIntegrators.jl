# VISplineFixed: fixed-knot B-spline variational integrator.
#
# Method struct, cache, and all integrator functions in one file.
#
# Unknown vector layout (per dimension k, total D*(S+1)):
#   x[D*(i-1)+k]   i=1..S   →  B-spline coefficient c[i,k]
#   x[D*S+k]                →  end-of-step momentum p̃[k]
#
# Residual blocks:
#   Block 1  (δSd/δc[i,k] = 0):
#       r₁[i]*p̃[k] − r₀[i]*p̄[k] − Σⱼ bⱼ(h·m[j,i]·F[j][k] + a[j,i]·P[j][k]) = 0
#   Block 2  (continuation q_h(0) = q̄):
#       q̄[k] − Σᵢ r₀[i]·c[i,k] = 0

# ---------------------------------------------------------------------------
# Method struct
# ---------------------------------------------------------------------------

"""
    VISplineFixed{T, NNODES, NBASIS, NDOF} <: VISplineMethod

Fixed-knot B-spline variational integrator.  The knot vector is set at construction;
the matrices `m`, `a`, `r₀`, `r₁` (CGVI notation) are precomputed once.

    VISplineFixed(basis, quadrature; extrapolation_substep=10)
    VISplineFixed(order, n_internal, quadrature; extrapolation_substep=10)
"""
struct VISplineFixed{T, NNODES, NBASIS, NDOF} <: VISplineMethod
    basis::NativeSplineBasis{T}
    b::SVector{NNODES, T}                    # quadrature weights
    c::SVector{NNODES, T}                    # quadrature nodes
    m::SMatrix{NNODES, NBASIS, T, NDOF}      # m[j,i] = B_i(c_j)
    a::SMatrix{NNODES, NBASIS, T, NDOF}      # a[j,i] = B'_i(c_j)
    r₀::SVector{NBASIS, T}                   # B_i(0)
    r₁::SVector{NBASIS, T}                   # B_i(1)
    extrapolation_substep::Int
end

function VISplineFixed(basis::NativeSplineBasis{T}, quadrature::QuadratureRule{T};
        extrapolation_substep::Int = 10) where {T}
    NNODES = QuadratureRules.nnodes(quadrature)
    NBASIS = nbasis(basis)
    bw = SVector{NNODES, T}(QuadratureRules.weights(quadrature))
    cn = SVector{NNODES, T}(QuadratureRules.nodes(quadrature))

    m_buf = zeros(T, NNODES, NBASIS)
    a_buf = zeros(T, NNODES, NBASIS)
    r0    = zeros(T, NBASIS)
    r1    = zeros(T, NBASIS)

    db = basis'
    for i in eachindex(basis)
        r0[i] = basis[T(0), i]
        r1[i] = basis[T(1), i]
        for j in 1:NNODES
            m_buf[j, i] = basis[cn[j], i]
            a_buf[j, i] = db[cn[j], i]
        end
    end

    VISplineFixed{T, NNODES, NBASIS, NNODES * NBASIS}(
        basis, bw, cn,
        SMatrix{NNODES, NBASIS, T, NNODES * NBASIS}(m_buf),
        SMatrix{NNODES, NBASIS, T, NNODES * NBASIS}(a_buf),
        SVector{NBASIS, T}(r0),
        SVector{NBASIS, T}(r1),
        extrapolation_substep)
end

function VISplineFixed(order::Int, n_internal::Int, quadrature::QuadratureRule{T};
        kwargs...) where {T}
    VISplineFixed(
        NativeSplineBasis(order, uniform_internal_knots(n_internal; T); T),
        quadrature; kwargs...)
end

nbasis(m::VISplineFixed) = CompactBasisFunctions.nbasis(m.basis)
nnodes(m::VISplineFixed) = length(m.c)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

struct VISplineFixedCache{ST} <: VISplineCache{ST}
    x::Vector{ST}           # Newton unknowns, length D*(S+1)

    q̄::Vector{ST}
    p̄::Vector{ST}

    q̃::Vector{ST}
    p̃::Vector{ST}
    ṽ::Vector{ST}
    f̃::Vector{ST}

    X::Vector{Vector{ST}}   # B-spline coefficients c[i]
    Q::Vector{Vector{ST}}   # position at quadrature nodes
    P::Vector{Vector{ST}}   # ϑ(Q,V) at quadrature nodes
    V::Vector{Vector{ST}}   # velocity at quadrature nodes
    F::Vector{Vector{ST}}   # f(Q,V) at quadrature nodes

    network_labels::Matrix{ST}
    solver_converged::Vector{Bool}

    function VISplineFixedCache{ST}(ics, S::Int, R::Int, N::Int) where {ST}
        D  = length(vec(ics.q))
        x  = zeros(ST, D * (S + 1))
        q̄  = zeros(ST, D); p̄ = zeros(ST, D)
        q̃  = zeros(ST, D); p̃ = zeros(ST, D); ṽ = zeros(ST, D); f̃ = zeros(ST, D)
        X  = [zeros(ST, D) for _ in 1:S]
        Q  = [zeros(ST, D) for _ in 1:R]
        P  = [zeros(ST, D) for _ in 1:R]
        V  = [zeros(ST, D) for _ in 1:R]
        F  = [zeros(ST, D) for _ in 1:R]
        network_labels   = zeros(ST, N + 1, D)
        solver_converged = [true]
        new(x, q̄, p̄, q̃, p̃, ṽ, f̃, X, Q, P, V, F, network_labels, solver_converged)
    end
end

function GeometricIntegratorsBase.Cache{ST}(
        problem::AbstractProblemIODE, method::VISplineFixed; kwargs...) where {ST}
    VISplineFixedCache{ST}(initial_conditions(problem),
        nbasis(method), nnodes(method), method.extrapolation_substep)
end

GeometricIntegratorsBase.CacheType(ST, ::AbstractProblemIODE, ::VISplineFixed) =
    VISplineFixedCache{ST}

# ---------------------------------------------------------------------------
# initial_guess!
# ---------------------------------------------------------------------------

function GeometricIntegratorsBase.initial_guess!(
        sol, history, params, int::GeometricIntegrator{<:VISplineFixed})
    _vi_spline_initial_trajectory!(sol, history, params, int)
    S = nbasis(method(int))
    _fit_bspline_to_labels!(nlsolution(int), cache(int),
        method(int).basis.order, method(int).basis.knots,
        length(cache(int).q̃), S)
end

# ---------------------------------------------------------------------------
# components!
# ---------------------------------------------------------------------------

function GeometricIntegratorsBase.components!(x::AbstractVector{ST}, sol, params,
        int::GeometricIntegrator{<:VISplineFixed}) where {ST}
    S  = nbasis(method(int))
    D  = length(cache(int).q̃)
    C  = cache(int, ST)
    h  = timestep(int)

    # Unpack coefficients c[i,k] and momentum p̃[k]
    for i in 1:S
        for k in 1:D
            C.X[i][k] = x[D * (i - 1) + k]
        end
    end
    for k in 1:D
        C.p̃[k] = x[D * S + k]
    end

    # Reconstruct q̃ = Σᵢ r₁[i] · c[i,k]  and  Q[j][k] = Σᵢ m[j,i] · c[i,k]
    for k in 1:D
        y = zero(ST)
        for i in 1:S
            y += ST(method(int).r₁[i]) * C.X[i][k]
        end
        C.q̃[k] = y
    end
    for j in 1:nnodes(method(int))
        for k in 1:D
            y = zero(ST)
            for i in 1:S
                y += ST(method(int).m[j, i]) * C.X[i][k]
            end
            C.Q[j][k] = y
        end
    end

    # Reconstruct V[j][k] = (1/h) Σᵢ a[j,i] · c[i,k]
    for j in 1:nnodes(method(int))
        for k in 1:D
            y = zero(ST)
            for i in 1:S
                y += ST(method(int).a[j, i]) * C.X[i][k]
            end
            C.V[j][k] = y / h
        end
    end

    # Evaluate ϑ(Q,V) → P  and  f(Q,V) → F  at each quadrature node
    for j in eachindex(C.Q, C.V, C.P, C.F)
        tⱼ = sol.t + h * (method(int).c[j] - one(ST))
        equations(int).ϑ(C.P[j], tⱼ, C.Q[j], C.V[j], params)
        equations(int).f(C.F[j], tⱼ, C.Q[j], C.V[j], params)
    end
end

# ---------------------------------------------------------------------------
# residual!
# ---------------------------------------------------------------------------

function GeometricIntegratorsBase.residual!(b::AbstractVector{ST}, sol, params,
        int::GeometricIntegrator{<:VISplineFixed}) where {ST}
    S  = nbasis(method(int))
    D  = length(cache(int).q̃)
    C  = cache(int, ST)
    h  = timestep(int)
    bw = method(int).b    # quadrature weights

    q̄ = sol.q
    p̄ = sol.p

    # Block 1: δSd/δc[i,k] = 0
    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(C.P, C.F)
                z += bw[j] * (h * ST(method(int).m[j, i]) * C.F[j][k] +
                              ST(method(int).a[j, i]) * C.P[j][k])
            end
            b[D * (i - 1) + k] = (ST(method(int).r₁[i]) * C.p̃[k] -
                                   ST(method(int).r₀[i]) * ST(p̄[k])) - z
        end
    end

    # Block 2: continuation q_h(0) = q̄
    for k in 1:D
        y = zero(ST)
        for i in 1:S
            y += ST(method(int).r₀[i]) * C.X[i][k]
        end
        b[D * S + k] = ST(q̄[k]) - y
    end
end
