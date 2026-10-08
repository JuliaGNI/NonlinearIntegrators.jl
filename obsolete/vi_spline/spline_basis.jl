# NativeSplineBasis{T} <: Basis{T}
#
# Implements the CompactBasisFunctions.Basis{T} interface for the native Cox-de Boor
# B-spline backed by the primitives in bspline_utils.jl.

"""
    NativeSplineBasis{T} <: Basis{T}

B-spline basis on [0,1] of polynomial order `order` (degree = order-1) with a fixed
full knot vector.  Backed by the native Cox-de Boor recursion in `bspline_utils.jl`.

Construction:
    NativeSplineBasis(order, internal_knots; T = eltype(internal_knots))
    NativeSplineBasis{T}(order, internal_knots)

The knot vector is `[0^order, τ₁..τ_K, 1^order]`; the number of basis functions is
`S = K + order`.  The adjoint `b'` returns a `NativeSplineBasisDerivative` whose
`getindex(d, t, i)` evaluates `B'_i(t)` via `bspline_deriv`.
"""
struct NativeSplineBasis{T} <: Basis{T}
    order::Int          # polynomial order k  (degree = k-1)
    knots::Vector{T}    # full knot vector, length = 2*order + K

    # Inner constructor: takes the pre-built full knot vector directly.
    NativeSplineBasis{T}(order::Int, knots::Vector{T}) where {T} = new{T}(order, knots)
end

# Outer constructor: builds the full knot vector from internal knots.
function NativeSplineBasis(order::Int, internal_knots::AbstractVector; T = eltype(internal_knots))
    NativeSplineBasis{T}(order, build_knot_vector(order, internal_knots; T))
end

# --- CompactBasisFunctions.Basis interface ---

CompactBasisFunctions.nbasis(b::NativeSplineBasis)  = length(b.knots) - b.order
CompactBasisFunctions.order(b::NativeSplineBasis)   = b.order
CompactBasisFunctions.degree(b::NativeSplineBasis)  = b.order - 1

Base.eltype(::NativeSplineBasis{T}) where {T} = T
Base.eachindex(b::NativeSplineBasis) = 1:nbasis(b)
Base.length(b::NativeSplineBasis)   = nbasis(b)

# Evaluation: b[t, i] = B_{i,k}(t)
(b::NativeSplineBasis{T})(t, i::Integer) where {T} =
    bspline_value(i, b.order, t, b.knots; T)

Base.getindex(b::NativeSplineBasis, t, i::Integer)     = b(t, i)
Base.getindex(b::NativeSplineBasis{T}, t, ::Colon) where {T} =
    [bspline_value(i, b.order, t, b.knots; T) for i in eachindex(b)]
Base.getindex(b::NativeSplineBasis{T}, ts::AbstractVector, i::Integer) where {T} =
    [bspline_value(i, b.order, t, b.knots; T) for t in ts]
Base.getindex(b::NativeSplineBasis{T}, ts::AbstractVector, ::Colon) where {T} =
    [bspline_value(i, b.order, t, b.knots; T) for t in ts, i in eachindex(b)]

# --- Time-derivative object ---

"""
    NativeSplineBasisDerivative{T}

Lazy time-derivative of a `NativeSplineBasis`, obtained by `b'`.
Indexing `d[t, i]` returns `B'_{i,k}(t)` via `bspline_deriv`.
"""
struct NativeSplineBasisDerivative{T}
    basis::NativeSplineBasis{T}
end

Base.adjoint(b::NativeSplineBasis)              = NativeSplineBasisDerivative(b)
Base.eachindex(d::NativeSplineBasisDerivative)  = eachindex(d.basis)

(d::NativeSplineBasisDerivative{T})(t, i::Integer) where {T} =
    bspline_deriv(i, d.basis.order, t, d.basis.knots; T)

Base.getindex(d::NativeSplineBasisDerivative, t, i::Integer)         = d(t, i)
Base.getindex(d::NativeSplineBasisDerivative{T}, t, ::Colon) where {T} =
    [bspline_deriv(i, d.basis.order, t, d.basis.knots; T) for i in eachindex(d.basis)]
