using CompactBasisFunctions
struct BSplineDirichlet{T}
    t::AbstractVector{T} # vector to generate knots
    interval::AbstractVector{T} # domain 
    k::Int # order

    knots::AbstractVector{T} # knots
    B::Vector{Function} #  basis functions 

    Der_knots::AbstractVector{T} # knots for the derivative
    Der_B::Vector{Function}  # Derivative of basis functions
    function BSplineDirichlet(k::Int,t::AbstractVector{T}; interval = [0,1]) where T
        knots = ArbitraryKnotSet(k,t)
        B = BSpline(knots)
        basis_fct = []

        for i in 1:size(B)[2]
            push!(basis_fct, x -> B[x, i])
        end

        Der_k = k - 1
        Der_knots = ArbitraryKnotSet(Der_k,t)
        D_B = BSpline(Der_knots)
        Der_basis_fct = []
        for i in 1:size(D_B)[2]
            push!(Der_basis_fct, x -> k/(Der_knots[i+k] - Der_knots[i]) * D_B[x, i])
        end

        return new{T}(t, interval, k, knots, basis_fct, Der_knots, Der_basis_fct)
    end
end



getindex(B::BSplineDirichlet, x::Real, j::Integer) = B.B[j](x)
getindex(B::BSplineDirichlet, x::Real, j::Colon) = [B.B[i](x) for i in 1:length(B.B)]
getindex(B::BSplineDirichlet, x::AbstractVector, j::Integer) = B.B[j].(x)
getindex(B::BSplineDirichlet, x::AbstractVector, j::Colon) = [B.B[i](x[j]) for j in 1:length(x), i in 1:length(B.B)]

function Base.show(io::IO, B::BSplineDirichlet{T}) where {T}
    println(io, T, " Dirichlet BSpline basis of order ", B.k, " on interval ", B.interval, " with knots", B.t)
    nothing
end
