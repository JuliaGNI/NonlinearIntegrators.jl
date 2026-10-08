# VISplineFree: free-knot B-spline variational integrator.
#
# Internal knot positions are additional Newton unknowns alongside B-spline coefficients
# and end-of-step momentum. Analogous to ShallowNet (biases ≡ knot positions).
#
# Unknown vector layout per dimension k, total D*(S+1+K):
#   x[D*(i-1)+k]          i=1..S   →  B-spline coefficient c[i,k]
#   x[D*S+k]                       →  end-of-step momentum p̃[k]
#   x[D*(S+1)+D*(l-1)+k]  l=1..K   →  internal knot τ[l,k]
#
# ForwardDiff differentiates through `bspline_value`/`bspline_deriv` (pure Cox-de Boor
# arithmetic) to obtain all knot-position Jacobian columns automatically.
#
# Residual blocks:
#   Block 1  (δSd/δc[i,k] = 0):
#       r₁[i,k]*p̃[k] − r₀[i,k]*p̄[k] − Σⱼ bⱼ(h·m[j,i,k]·F[j][k] + a[j,i,k]·P[j][k]) = 0
#   Block 2  (continuation q_h(0) = q̄):
#       q̄[k] − Σᵢ r₀[i,k]·c[i,k] = 0
#   Block 3  (δSd/δτ[l,k] = 0):
#       ∂q̃[k]/∂τ[l,k]·p̃[k] − ∂q̄[k]/∂τ[l,k]·p̄[k]
#       − Σⱼ bⱼ(h·∂Q[j][k]/∂τ[l,k]·F[j][k] + ∂V[j][k]/∂τ[l,k]·P[j][k]) = 0
#
# Block 3 is computed analytically using `bspline_knot_deriv` and `bspline_mixed_deriv`.

# ---------------------------------------------------------------------------
# Method struct
# ---------------------------------------------------------------------------

"""
    VISplineFree{T, NNODES, NBASIS, NDOF} <: VISplineMethod

Free-knot B-spline variational integrator.  Internal knot positions `τ[l,k]` are
additional Newton unknowns alongside B-spline coefficients `c[i,k]` and end-of-step
momentum `p̃[k]`.

    VISplineFree(order, n_internal, quadrature; extrapolation_substep=10)
"""
struct VISplineFree{T, NNODES, NBASIS, NDOF} <: VISplineMethod
    order::Int
    n_internal::Int          # K internal knots per dimension
    init_knots::Vector{T}    # uniform initial knots — seed for every step
    b::SVector{NNODES, T}
    c::SVector{NNODES, T}
    extrapolation_substep::Int
end

function VISplineFree(order::Int, n_internal::Int, quadrature::QuadratureRule{T};
        extrapolation_substep::Int = 10) where {T}
    NNODES = QuadratureRules.nnodes(quadrature)
    NBASIS = n_internal + order
    bw     = SVector{NNODES, T}(QuadratureRules.weights(quadrature))
    cn     = SVector{NNODES, T}(QuadratureRules.nodes(quadrature))
    ik     = uniform_internal_knots(n_internal; T)
    VISplineFree{T, NNODES, NBASIS, NNODES * NBASIS}(
        order, n_internal, ik, bw, cn, extrapolation_substep)
end

nbasis(m::VISplineFree) = m.n_internal + m.order
nnodes(m::VISplineFree) = length(m.c)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

struct VISplineFreeCache{ST} <: VISplineCache{ST}
    x::Vector{ST}           # Newton unknowns, length D*(S+1+K)

    q̄::Vector{ST}
    p̄::Vector{ST}

    q̃::Vector{ST}
    p̃::Vector{ST}
    ṽ::Vector{ST}
    f̃::Vector{ST}

    X::Vector{Vector{ST}}
    Q::Vector{Vector{ST}}
    P::Vector{Vector{ST}}
    V::Vector{Vector{ST}}
    F::Vector{Vector{ST}}

    # Per-step B-spline matrices (recomputed in components! from current knots).
    # Stored as plain arrays because they hold ST (= Dual during ForwardDiff).
    r₀::Matrix{ST}       # S × D
    r₁::Matrix{ST}       # S × D
    m::Array{ST, 3}      # R × S × D
    a::Array{ST, 3}      # R × S × D

    network_labels::Matrix{ST}
    solver_converged::Vector{Bool}

    function VISplineFreeCache{ST}(ics, S::Int, K::Int, R::Int, N::Int) where {ST}
        D  = length(vec(ics.q))
        x  = zeros(ST, D * (S + 1 + K))
        q̄  = zeros(ST, D); p̄ = zeros(ST, D)
        q̃  = zeros(ST, D); p̃ = zeros(ST, D); ṽ = zeros(ST, D); f̃ = zeros(ST, D)
        X  = [zeros(ST, D) for _ in 1:S]
        Q  = [zeros(ST, D) for _ in 1:R]
        P  = [zeros(ST, D) for _ in 1:R]
        V  = [zeros(ST, D) for _ in 1:R]
        F  = [zeros(ST, D) for _ in 1:R]
        r₀ = zeros(ST, S, D)
        r₁ = zeros(ST, S, D)
        m  = zeros(ST, R, S, D)
        a  = zeros(ST, R, S, D)
        network_labels   = zeros(ST, N + 1, D)
        solver_converged = [true]
        new(x, q̄, p̄, q̃, p̃, ṽ, f̃, X, Q, P, V, F, r₀, r₁, m, a,
            network_labels, solver_converged)
    end
end

function GeometricIntegratorsBase.Cache{ST}(
        problem::AbstractProblemIODE, method::VISplineFree; kwargs...) where {ST}
    VISplineFreeCache{ST}(initial_conditions(problem),
        nbasis(method), method.n_internal, nnodes(method), method.extrapolation_substep)
end

GeometricIntegratorsBase.CacheType(ST, ::AbstractProblemIODE, ::VISplineFree) =
    VISplineFreeCache{ST}

# ---------------------------------------------------------------------------
# initial_guess!
# ---------------------------------------------------------------------------

function GeometricIntegratorsBase.initial_guess!(
        sol, history, params, int::GeometricIntegrator{<:VISplineFree})
    _vi_spline_initial_trajectory!(sol, history, params, int)

    S = nbasis(method(int))
    K = method(int).n_internal
    D = length(cache(int).q̃)
    x = nlsolution(int)

    # Fit B-spline coefficients using the uniform initial knots
    init_knots_full = build_knot_vector(method(int).order, method(int).init_knots;
        T = eltype(method(int).init_knots))
    _fit_bspline_to_labels!(x, cache(int), method(int).order, init_knots_full, D, S)

    # Seed knot positions uniformly (same seed every step)
    for l in 1:K
        for k in 1:D
            x[D * (S + 1) + D * (l - 1) + k] = method(int).init_knots[l]
        end
    end
end

# ---------------------------------------------------------------------------
# components!
# ---------------------------------------------------------------------------

function GeometricIntegratorsBase.components!(x::AbstractVector{ST}, sol, params,
        int::GeometricIntegrator{<:VISplineFree}) where {ST}
    S  = nbasis(method(int))
    K  = method(int).n_internal
    D  = length(cache(int).q̃)
    C  = cache(int, ST)
    h  = timestep(int)
    ord = method(int).order

    # Unpack coefficients, momentum, and knot positions from x
    for i in 1:S
        for k in 1:D
            C.X[i][k] = x[D * (i - 1) + k]
        end
    end
    for k in 1:D
        C.p̃[k] = x[D * S + k]
    end

    # Evaluate per-dimension B-spline matrices from the current knot positions.
    # Knots are ST (Dual during ForwardDiff), so the recursion propagates ∂B/∂τ.
    for k in 1:D
        τ_k = [x[D * (S + 1) + D * (l - 1) + k] for l in 1:K]
        knots_k = build_knot_vector(ord, τ_k; T = ST)

        for i in 1:S
            C.r₀[i, k] = bspline_value(i, ord, ST(0), knots_k; T = ST)
            C.r₁[i, k] = bspline_value(i, ord, ST(1), knots_k; T = ST)
        end
        for j in 1:nnodes(method(int))
            cⱼ = method(int).c[j]
            for i in 1:S
                C.m[j, i, k] = bspline_value(i, ord, ST(cⱼ), knots_k; T = ST)
                C.a[j, i, k] = bspline_deriv(i, ord, ST(cⱼ), knots_k; T = ST)
            end
        end
    end

    # Reconstruct q̃ and Q
    for k in 1:D
        y = zero(ST)
        for i in 1:S
            y += C.r₁[i, k] * C.X[i][k]
        end
        C.q̃[k] = y
    end
    for j in 1:nnodes(method(int))
        for k in 1:D
            y = zero(ST)
            for i in 1:S
                y += C.m[j, i, k] * C.X[i][k]
            end
            C.Q[j][k] = y
        end
    end

    # Reconstruct V[j][k] = (1/h) Σᵢ a[j,i,k] · c[i,k]
    for j in 1:nnodes(method(int))
        for k in 1:D
            y = zero(ST)
            for i in 1:S
                y += C.a[j, i, k] * C.X[i][k]
            end
            C.V[j][k] = y / h
        end
    end

    # Evaluate ϑ(Q,V) → P  and  f(Q,V) → F
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
        int::GeometricIntegrator{<:VISplineFree}) where {ST}
    S   = nbasis(method(int))
    K   = method(int).n_internal
    D   = length(cache(int).q̃)
    C   = cache(int, ST)
    h   = timestep(int)
    bw  = method(int).b
    ord = method(int).order

    q̄ = sol.q
    p̄ = sol.p
    x  = nlsolution(int)

    # Block 1: δSd/δc[i,k] = 0
    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(C.P, C.F)
                z += bw[j] * (h * C.m[j, i, k] * C.F[j][k] +
                              C.a[j, i, k] * C.P[j][k])
            end
            b[D * (i - 1) + k] = (C.r₁[i, k] * C.p̃[k] -
                                   C.r₀[i, k] * ST(p̄[k])) - z
        end
    end

    # Block 2: continuation q_h(0) = q̄
    for k in 1:D
        y = zero(ST)
        for i in 1:S
            y += C.r₀[i, k] * C.X[i][k]
        end
        b[D * S + k] = ST(q̄[k]) - y
    end

    # Block 3: δSd/δτ[l,k] = 0, using analytic knot-position derivatives
    for l in 1:K
        for k in 1:D
            τ_k    = [x[D * (S + 1) + D * (ll - 1) + k] for ll in 1:K]
            knots_k = build_knot_vector(ord, τ_k; T = ST)
            jknot   = ord + l    # index of τ[l] in the full knot vector

            dq̃_dτ = zero(ST)
            dq̄_dτ = zero(ST)
            for i in 1:S
                dB1 = bspline_knot_deriv(i, ord, ST(1), knots_k, jknot; T = ST)
                dB0 = bspline_knot_deriv(i, ord, ST(0), knots_k, jknot; T = ST)
                dq̃_dτ += dB1 * C.X[i][k]
                dq̄_dτ += dB0 * C.X[i][k]
            end

            z = zero(ST)
            for j in eachindex(C.P, C.F)
                cⱼ = method(int).c[j]
                dQ_dτ = zero(ST)
                dV_dτ = zero(ST)
                for i in 1:S
                    dBv = bspline_knot_deriv(i, ord, ST(cⱼ), knots_k, jknot; T = ST)
                    dBd = bspline_mixed_deriv(i, ord, ST(cⱼ), knots_k, jknot; T = ST)
                    dQ_dτ += dBv * C.X[i][k]
                    dV_dτ += dBd * C.X[i][k]
                end
                z += bw[j] * (h * dQ_dτ * C.F[j][k] + dV_dτ * C.P[j][k])
            end

            b[D * (S + 1) + D * (l - 1) + k] = (dq̃_dτ * C.p̃[k] -
                                                  dq̄_dτ * ST(p̄[k])) - z
        end
    end
end
