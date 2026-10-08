# Network-to-spline equivalence conversion.
#
# Implements the constructive direction of the Curry-Schoenberg equivalence theorem:
# a shallow network with ReLU^(k-1) activation and positive input weights is exactly
# representable as a free-knot B-spline of order k, and vice versa.
#
# Reference: see docs/src/relu_spline_equivalence.md for the full proof.

"""
    NNSplineEquiv{T}

Lightweight struct holding a B-spline representation equivalent to a shallow
ReLU^(k-1) network.  Evaluate with `evaluate_nn_equiv`.
"""
struct NNSplineEquiv{T}
    order::Int           # spline order k  (k=2 ↔ ReLU, k=3 ↔ ReLU², k=4 ↔ ReLU³)
    knots::Vector{T}     # full knot vector [0…0, τ₁,…,τ_K, 1…1]
    coefs::Vector{T}     # B-spline coefficients c[1..S]
end

"""
    evaluate_nn_equiv(sp::NNSplineEquiv{T}, t; T=T) -> T

Evaluate the B-spline equivalent of a shallow network at point t.
"""
function evaluate_nn_equiv(sp::NNSplineEquiv{T}, t; kw_T = T) where {T}
    S = length(sp.coefs)
    val = kw_T(0)
    for i in 1:S
        val += kw_T(sp.coefs[i]) * bspline_value(i, sp.order, t, sp.knots; T = kw_T)
    end
    return val
end

"""
    nn_spline_max_error(W2, W1, b1, sp::NNSplineEquiv{T}, t_grid; T=eltype(W2)) -> T

Compute `max |q_NN(t) - q_spline(t)|` over `t_grid` for numerical verification.
The NN forward pass is `Σᵢ W2[i] * relu(W1[i]*t + b1[i])^(sp.order-1)`.
"""
function nn_spline_max_error(
        W2::AbstractVector, W1::AbstractVector, b1::AbstractVector,
        sp::NNSplineEquiv{T}, t_grid::AbstractVector; kw_T = eltype(W2)) where {T}
    k = sp.order
    max_err = kw_T(0)
    for t in t_grid
        q_nn = kw_T(0)
        for i in eachindex(W2)
            pre = kw_T(W1[i]) * kw_T(t) + kw_T(b1[i])
            if pre > kw_T(0)
                q_nn += kw_T(W2[i]) * pre^(k - 1)
            end
        end
        q_sp = evaluate_nn_equiv(sp, t; kw_T)
        max_err = max(max_err, abs(q_nn - q_sp))
    end
    return max_err
end

# ---------------------------------------------------------------------------
# Change-of-basis: truncated powers → B-splines
# ---------------------------------------------------------------------------
#
# The truncated power function of degree p = k-1 is:
#     TP_j(t) = (t - τ_j)₊^p
#
# Each TP_j can be expressed as a linear combination of B-splines via divided differences
# (Curry-Schoenberg).  For practical conversion we evaluate both bases at S+1 collocation
# points and solve the S×S linear system.

"""
    _truncated_power_to_bspline_matrix(order, knots_sorted_internal; T) -> Matrix{T}

Returns the (S × K) change-of-basis matrix C such that
    TP_j(t) = Σᵢ C[i, j] · B_{i,k}(t)
for the K truncated power functions and S B-spline basis functions.

Computed by collocating both bases at S distinct points and solving the resulting
linear system using the B-spline Vandermonde matrix.
"""
function _truncated_power_to_bspline_matrix(
        order::Int, τ_sorted::AbstractVector; T = Float64)
    K = length(τ_sorted)
    knots = build_knot_vector(order, τ_sorted; T)
    S = length(knots) - order

    # Collocation points: S points distributed across [0,1], avoiding knot positions
    cols = [T(i) / T(S + 1) for i in 1:S]

    # B-spline Vandermonde matrix  (S × S)
    B_vandermonde = bspline_eval_matrix(order, knots, cols; T)   # S × S

    # Truncated power matrix  (S × K)
    p = order - 1
    TP = zeros(T, S, K)
    for (r, t) in enumerate(cols), (j, τ) in enumerate(τ_sorted)
        pre = T(t) - T(τ)
        TP[r, j] = pre > T(0) ? pre^p : T(0)
    end

    # Solve: B_vandermonde * C = TP  →  C = B_vandermonde \ TP
    C = B_vandermonde \ TP    # (S × K)
    return C
end

# ---------------------------------------------------------------------------
# Public conversion API
# ---------------------------------------------------------------------------

"""
    relu_network_to_spline(W2, W1, b1; order=2, T=eltype(W2)) -> NNSplineEquiv{T}

Convert a shallow ReLU^(order-1) network to an equivalent free-knot B-spline.

Arguments:
- `W2`: output-layer weights (length S)
- `W1`: hidden-layer input weights (length S)
- `b1`: hidden-layer biases (length S)
- `order`: spline order k (2 = ReLU/linear, 3 = ReLU²/quadratic, 4 = ReLU³/cubic)

Steps:
1. Compute raw knot positions τᵢ = −b1ᵢ / W1ᵢ for neurons with W1ᵢ > 0.
2. Filter to τᵢ ∈ (0, 1) and sort.
3. Compute truncated-power coefficients αᵢ = W2ᵢ · W1ᵢ^(order-1).
4. Convert αᵢ to B-spline coefficients via the Curry-Schoenberg basis change.

Neurons with W1 ≤ 0 are silently skipped; they contribute (τᵢ − t)₊^(k−1) terms
(reflected truncated powers) which lie outside the standard free-knot B-spline space.
Include them by negating W1 and b1 before calling if desired.
"""
function relu_network_to_spline(
        W2::AbstractVector, W1::AbstractVector, b1::AbstractVector;
        order::Int = 2, T = eltype(W2))
    S_total = length(W2)
    @assert length(W1) == S_total && length(b1) == S_total

    # Step 1-2: extract, filter, sort knots
    knot_pairs = Tuple{T, T}[]   # (τ, α) pairs
    for i in 1:S_total
        W1i = T(W1[i])
        W1i ≤ T(0) && continue
        τ = -T(b1[i]) / W1i
        (T(0) < τ < T(1)) || continue
        α = T(W2[i]) * W1i^(order - 1)
        push!(knot_pairs, (τ, α))
    end

    if isempty(knot_pairs)
        # Degenerate case: no interior knots — return zero spline
        knots = build_knot_vector(order, T[]; T)
        S = length(knots) - order
        return NNSplineEquiv{T}(order, knots, zeros(T, S))
    end

    sort!(knot_pairs; by = first)
    τ_sorted = first.(knot_pairs)
    α        = last.(knot_pairs)

    # Step 3: build knot vector and the change-of-basis matrix
    knots = build_knot_vector(order, τ_sorted; T)
    C = _truncated_power_to_bspline_matrix(order, τ_sorted; T)   # S × K

    # Step 4: B-spline coefficients  c = C * α
    coefs = C * α

    return NNSplineEquiv{T}(order, knots, coefs)
end

"""
    shallownet_to_spline(x, S, D, k; order=2, T=Float64) -> NNSplineEquiv{T}

Convenience wrapper that unpacks the ShallowNet solution vector `x`
(layout: `[W2₁..W2_S | p̃ | W1₁..W1_S | b1₁..b1_S]` per dimension)
for dimension `k` and calls `relu_network_to_spline`.

Arguments:
- `x`: the flattened Newton-solve solution vector (length D*(3S+1))
- `S`: number of hidden neurons
- `D`: problem dimension
- `k`: dimension index (1-based)
"""
function shallownet_to_spline(
        x::AbstractVector, S::Int, D::Int, k::Int;
        order::Int = 2, T = eltype(x))
    W2 = [T(x[D * (i - 1) + k]) for i in 1:S]
    W1 = [T(x[D * (S + 1) + D * (i - 1) + k]) for i in 1:S]
    b1 = [T(x[D * (S + 1 + S) + D * (i - 1) + k]) for i in 1:S]
    return relu_network_to_spline(W2, W1, b1; order, T)
end
