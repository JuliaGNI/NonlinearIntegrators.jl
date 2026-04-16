struct Nonlinear_BSpline_Basis{T} <: Basis{T}
    K::Int # Order of the BSpline basis
    B # BSpline basis functions from BSplineKit

    internal_knots::AbstractVector{T} # Internal knots
    N_internal_knots::Int # Number of internal knots

    knot_seq # Knots sequence expanded from internal_knots to satisfy the boundary conditions at a given order 
    S::Int # Number of basis functions
    function Nonlinear_BSpline_Basis(K::Int, internal_knots::AbstractVector{T}) where T
        N_internal_knots = length(internal_knots)
        knots = [0.0; internal_knots; 1.0]
        B = BSplineBasis(BSplineOrder(K), knots)
        return new{T}(K, B, internal_knots, N_internal_knots, B.t, length(B))
    end
end

Base.getindex(basis::Nonlinear_BSpline_Basis, j::Integer) = basis.B[j]
Base.getindex(basis::Nonlinear_BSpline_Basis, x::Real, j::Integer) = basis.B[j](x)
Base.length(basis::Nonlinear_BSpline_Basis) = Base.length(basis.B)

function Base.show(io::IO, basis::Nonlinear_BSpline_Basis{T}) where {T}
    println(io, " Nonlinear BSpline basis of order ", basis.K)
    println(io, " Knots Sequence: ", basis.knot_seq)
end