struct BSplineDirichlet{T} <:Basis{T}
    k::Int # order
    t::AbstractVector{T} # vector to generate knots

    knot_seq
    B #  basis functions 
    Der_B  # Derivative of basis functions
    function BSplineDirichlet(k::Int,t::AbstractVector{T}) where T
        B = BSplineBasis(BSplineOrder(k), t)
        knot_seq = B.t
        basis_fct = []
        for i in eachindex(B)
            push!(basis_fct, B[i])
        end

        tem_coes = zeros(length(B))
        Der_B = []
        # create a temporary coefficient vector with only one non-zero entry, which is 1
        # for each basis function, create a spline and then take the derivative
        # to achieve the derivative of the basis function
        for i in eachindex(B)
            tem_coes[i] = 1
            S = BSplineKit.Spline(B, tem_coes)
            DB = BSplineKit.Derivative(1) * S
            push!(Der_B, DB)
        end

        return new{T}(k, t, knot_seq, basis_fct, Der_B)
    end
end

Base.length(Basis::BSplineDirichlet) = Base.length(Basis.B)

getindex(B::BSplineDirichlet, j::Integer) =  B.B[j]
getindex(B::BSplineDirichlet, x::Real, j::Integer) = B.B[j](x)
getindex(B::BSplineDirichlet, x::Real, j::Colon) = [B.B[i](x) for i in 1:length(B.B)]
getindex(B::BSplineDirichlet, x::AbstractVector, j::Integer) = B.B[j].(x)
getindex(B::BSplineDirichlet, x::AbstractVector, j::Colon) = [B.B[i](x[j]) for j in 1:length(x), i in 1:length(B.B)]

function Base.show(io::IO, Basis::BSplineDirichlet{T}) where {T}
    println(io, T, " Dirichlet BSpline basis of order ", Basis.k, "\n")
    println(io, T, " Knots Sequence: ", Basis.knot_seq)
    nothing
end

