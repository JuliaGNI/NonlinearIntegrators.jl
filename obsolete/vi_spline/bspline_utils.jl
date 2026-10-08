# Native B-spline primitives — no external library required.
#
# Indexing convention (1-based throughout):
#   - Knot vector:      knots[1..m],  where m = 2*order + K  (K internal knots)
#   - Basis functions:  i ∈ 1..S,    where S = m - order = K + order
#
# Every function accepts a keyword argument `T` (default: eltype of the knot vector)
# that controls the element type of the computation.  All scalar constants are
# constructed as T(value) so no implicit Float64 promotion occurs at Float32 or other
# precisions.  Recursive calls forward T explicitly.

# ---------------------------------------------------------------------------
# B-spline value  B_{i,k}(t)
# ---------------------------------------------------------------------------

"""
    bspline_value(i, k, t, knots; T=eltype(knots)) -> T

Cox-de Boor recursion for the i-th B-spline basis function of order k at point t.
`T` pins the element type; all arithmetic is performed at precision T.

Right-endpoint convention: the last non-degenerate sub-interval includes its right
boundary so that `Σᵢ B_{i,k}(t) = T(1)` holds for all t in the full domain.
"""
function bspline_value(i::Int, k::Int, t, knots::AbstractVector; T = promote_type(eltype(knots), typeof(t)))
    if k == 1
        tL = T(knots[i])
        tR = T(knots[i + 1])
        iszero(tR - tL) && return T(0)
        in_support = tL ≤ t < tR || (t == T(knots[end]) && tL ≤ t ≤ tR)
        return in_support ? T(1) : T(0)
    end

    result = T(0)

    d1 = T(knots[i + k - 1]) - T(knots[i])
    if !iszero(d1)
        result += (t - T(knots[i])) / d1 * bspline_value(i, k - 1, t, knots; T)
    end

    d2 = T(knots[i + k]) - T(knots[i + 1])
    if !iszero(d2)
        result += (T(knots[i + k]) - t) / d2 * bspline_value(i + 1, k - 1, t, knots; T)
    end

    return result
end

# ---------------------------------------------------------------------------
# B-spline time derivative  ∂B_{i,k}(t)/∂t
# ---------------------------------------------------------------------------

"""
    bspline_deriv(i, k, t, knots; T=eltype(knots)) -> T

Analytic time derivative of the i-th B-spline basis function of order k:

    B'_{i,k}(t) = (k-1)/d₁ · B_{i,k-1}(t)  −  (k-1)/d₂ · B_{i+1,k-1}(t)

where d₁ = knots[i+k-1] − knots[i],  d₂ = knots[i+k] − knots[i+1].
Degenerate intervals (dₗ = 0) contribute T(0).
"""
function bspline_deriv(i::Int, k::Int, t, knots::AbstractVector; T = promote_type(eltype(knots), typeof(t)))
    k == 1 && return T(0)

    result = T(0)

    d1 = T(knots[i + k - 1]) - T(knots[i])
    if !iszero(d1)
        result += T(k - 1) / d1 * bspline_value(i, k - 1, t, knots; T)
    end

    d2 = T(knots[i + k]) - T(knots[i + 1])
    if !iszero(d2)
        result -= T(k - 1) / d2 * bspline_value(i + 1, k - 1, t, knots; T)
    end

    return result
end

# ---------------------------------------------------------------------------
# B-spline knot-position derivative  ∂B_{i,k}(t)/∂knots[j]
# ---------------------------------------------------------------------------

"""
    bspline_knot_deriv(i, k, t, knots, j; T=eltype(knots)) -> T

Analytic derivative of `B_{i,k}(t)` with respect to the j-th knot position `knots[j]`,
computed via the differentiated Cox-de Boor recursion.

Only knots j ∈ {i, …, i+k} can affect `B_{i,k}(t)`; returns T(0) immediately outside
that range.  Base case: `∂B_{i,1}/∂knots[j] = T(0)` for generic t (not at a knot boundary).

Derivation of the weight derivatives (a = knots[i], b = knots[i+k-1], c = knots[i+1], d = knots[i+k]):
    α₁ = (t−a)/(b−a):  ∂α₁/∂a = −(b−t)/(b−a)²,  ∂α₁/∂b = −(t−a)/(b−a)²
    α₂ = (d−t)/(d−c):  ∂α₂/∂c = +(d−t)/(d−c)²,  ∂α₂/∂d = +(t−c)/(d−c)²
"""
function bspline_knot_deriv(
        i::Int, k::Int, t, knots::AbstractVector, j::Int;
        T = promote_type(eltype(knots), typeof(t)))
    (j < i || j > i + k) && return T(0)
    k == 1 && return T(0)

    result = T(0)

    d1 = T(knots[i + k - 1]) - T(knots[i])
    if !iszero(d1)
        α1      = (t - T(knots[i])) / d1
        val_ik1 = bspline_value(i, k - 1, t, knots; T)
        dα1 = if j == i
            -(T(knots[i + k - 1]) - t) / d1^2
        elseif j == i + k - 1
            -(t - T(knots[i])) / d1^2
        else
            T(0)
        end
        result += dα1 * val_ik1 + α1 * bspline_knot_deriv(i, k - 1, t, knots, j; T)
    end

    d2 = T(knots[i + k]) - T(knots[i + 1])
    if !iszero(d2)
        α2        = (T(knots[i + k]) - t) / d2
        val_ip1k1 = bspline_value(i + 1, k - 1, t, knots; T)
        dα2 = if j == i + 1
            (T(knots[i + k]) - t) / d2^2
        elseif j == i + k
            (t - T(knots[i + 1])) / d2^2
        else
            T(0)
        end
        result += dα2 * val_ip1k1 + α2 * bspline_knot_deriv(i + 1, k - 1, t, knots, j; T)
    end

    return result
end

# ---------------------------------------------------------------------------
# Mixed derivative  ∂²B_{i,k}/(∂t ∂knots[j])
# ---------------------------------------------------------------------------

"""
    bspline_mixed_deriv(i, k, t, knots, j; T=eltype(knots)) -> T

Mixed partial `∂(∂B_{i,k}/∂t)/∂knots[j]`, obtained by differentiating the
time-derivative recurrence with respect to `knots[j]`.  Used in the free-knot residual
to compute `∂v_h/∂τ` (velocity variation with respect to internal knot positions).

Derivation of ∂(1/d)/∂knots[j] used below:
    d₁ = knots[i+k-1] − knots[i]:  ∂(1/d₁)/∂knots[i] = +1/d₁²,  ∂(1/d₁)/∂knots[i+k-1] = −1/d₁²
    d₂ = knots[i+k]   − knots[i+1]: ∂(1/d₂)/∂knots[i+1] = +1/d₂², ∂(1/d₂)/∂knots[i+k]  = −1/d₂²
"""
function bspline_mixed_deriv(
        i::Int, k::Int, t, knots::AbstractVector, j::Int;
        T = promote_type(eltype(knots), typeof(t)))
    (j < i || j > i + k) && return T(0)
    k == 1 && return T(0)

    result = T(0)

    d1 = T(knots[i + k - 1]) - T(knots[i])
    if !iszero(d1)
        val_ik1  = bspline_value(i, k - 1, t, knots; T)
        dval_ik1 = bspline_knot_deriv(i, k - 1, t, knots, j; T)
        dfac1 = if j == i
            T(k - 1) / d1^2
        elseif j == i + k - 1
            -T(k - 1) / d1^2
        else
            T(0)
        end
        result += dfac1 * val_ik1 + T(k - 1) / d1 * dval_ik1
    end

    d2 = T(knots[i + k]) - T(knots[i + 1])
    if !iszero(d2)
        val_ip1k1  = bspline_value(i + 1, k - 1, t, knots; T)
        dval_ip1k1 = bspline_knot_deriv(i + 1, k - 1, t, knots, j; T)
        dfac2 = if j == i + 1
            T(k - 1) / d2^2
        elseif j == i + k
            -T(k - 1) / d2^2
        else
            T(0)
        end
        result -= dfac2 * val_ip1k1 + T(k - 1) / d2 * dval_ip1k1
    end

    return result
end

# ---------------------------------------------------------------------------
# Knot vector construction
# ---------------------------------------------------------------------------

"""
    build_knot_vector(order, internal_knots; T=eltype(internal_knots)) -> Vector{T}

Builds the full B-spline knot vector on [0, 1] with `order`-fold repeated boundary knots:

    [0, 0, …, 0,  τ₁, τ₂, …, τ_K,  1, 1, …, 1]
      order times                      order times

Length = 2*order + K;  supports S = K + order basis functions.
"""
function build_knot_vector(
        order::Int, internal_knots::AbstractVector; T = eltype(internal_knots))
    K = length(internal_knots)
    knots = Vector{T}(undef, 2 * order + K)
    knots[1:order] .= T(0)
    knots[(order + 1):(order + K)] .= T.(internal_knots)
    knots[(order + K + 1):end] .= T(1)
    return knots
end

"""
    uniform_internal_knots(K; T=Float64) -> Vector{T}

Returns K uniformly-spaced internal knot positions in the open interval (0, 1):

    τᵢ = T(i) / T(K + 1),  i = 1, …, K
"""
function uniform_internal_knots(K::Int; T = Float64)
    [T(i) / T(K + 1) for i in 1:K]
end

"""
    bspline_eval_matrix(order, knots, eval_points; T=eltype(knots)) -> Matrix{T}

Returns an (N × S) matrix `B[r, i] = bspline_value(i, order, eval_points[r], knots; T)`.
S = length(knots) - order,  N = length(eval_points).
"""
function bspline_eval_matrix(
        order::Int, knots::AbstractVector, eval_points::AbstractVector; T = eltype(knots))
    S = length(knots) - order
    N = length(eval_points)
    B = zeros(T, N, S)
    for (r, t) in enumerate(eval_points)
        for i in 1:S
            B[r, i] = bspline_value(i, order, t, knots; T)
        end
    end
    return B
end
