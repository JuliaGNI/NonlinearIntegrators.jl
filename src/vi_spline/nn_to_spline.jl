# Network-to-spline conversion.
#
# On [0, 1] a shallow network q(t) = Σᵢ aᵢ relu(wᵢ t + bᵢ)ᵏ is a spline of degree k whose knots
# are the interior kinks τᵢ = −bᵢ/wᵢ ∈ (0, 1), with C^{k−1} continuity there (Proposition 5.2 of
# benchmark/theory/relu_cgvi_equivalence_en.md). Every neuron counts, whatever the signs: a kink
# outside [0, 1] contributes a polynomial piece, and wᵢ < 0 a reflected truncated power
# (Lemma 5.1). The B-spline coefficients follow by interpolation at the Greville points of the
# basis, which is exact, since q lies in the spline space.

"""
    NNSplineEquiv{T, BT}

The B-spline form of a shallow ReLUᵏ network on [0, 1]: `basis` is the clamped `BSplineBasis`
of degree k on the mesh `[0, kinks…, 1]`, and `coefs` holds its coefficients. Evaluate with
`evaluate_nn_equiv`.
"""
struct NNSplineEquiv{T, BT <: BSplineBasis{T}}
    basis::BT
    coefs::Vector{T}
end

"Evaluate the B-spline form `sp` of a network at `t ∈ [0, 1]`."
evaluate_nn_equiv(sp::NNSplineEquiv, t) = SimpleSplines.evaluate(sp.basis, sp.coefs, t)

"The network Σᵢ aᵢ relu(wᵢ t + bᵢ)ᵏ at `t`."
relu_network(a, w, b, k, t) = sum(a[i] * max(w[i] * t + b[i], zero(t))^k for i in eachindex(a, w, b))

"""
    relu_network_to_spline(a, w, b; degree) -> NNSplineEquiv

Convert the shallow ReLU^`degree` network with output weights `a`, input weights `w` and
biases `b` to its B-spline form on [0, 1]. Neurons whose kinks coincide share one knot.
"""
function relu_network_to_spline(a::AbstractVector{T}, w::AbstractVector{T}, b::AbstractVector{T};
        degree::Int) where {T}
    kinks = sort(unique(-b[i] / w[i] for i in eachindex(w, b) if !iszero(w[i]) && 0 < -b[i] / w[i] < 1))
    basis = BSplineBasis(GeneralMesh([zero(T); kinks; one(T)]), degree)
    τ = SimpleSplines.nodes(basis)
    coefs = basis[τ, :] \ [relu_network(a, w, b, degree, t) for t in τ]
    NNSplineEquiv(basis, coefs)
end

"""
    nn_spline_max_error(a, w, b, sp::NNSplineEquiv, t_grid)

`max |q_network(t) − q_spline(t)|` over `t_grid`, the check of the conversion.
"""
function nn_spline_max_error(a, w, b, sp::NNSplineEquiv, t_grid::AbstractVector)
    k = SimpleSplines.degree(sp.basis)
    maximum(abs(relu_network(a, w, b, k, t) - evaluate_nn_equiv(sp, t)) for t in t_grid)
end

"""
    shallownet_to_spline(x, S, D, d; degree) -> NNSplineEquiv

The B-spline form of dimension `d` of the `ShallowNet` unknowns `x`, laid out per dimension
as `[a₁…a_S | p̃ | w₁…w_S | b₁…b_S]` with the dimensions interleaved (length D(3S+1)).
"""
function shallownet_to_spline(x::AbstractVector, S::Int, D::Int, d::Int; degree::Int)
    a = [x[D * (i - 1) + d] for i in 1:S]
    w = [x[D * (S + 1) + D * (i - 1) + d] for i in 1:S]
    b = [x[D * (2S + 1) + D * (i - 1) + d] for i in 1:S]
    relu_network_to_spline(a, w, b; degree)
end
