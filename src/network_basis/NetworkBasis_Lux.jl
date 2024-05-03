using Lux
using CompactBasisFunctions
using ContinuumArrays

struct NetworkBasis_Lux{T,NT}<:Basis{T}
    activation
    S₁::Int
    S::Int
    NN::NT

    function NetworkBasis_Lux{T}(activation,S₁,S) where {T}
        NN = Lux.Chain(Lux.Dense(1,S₁,activation),
                            Lux.Dense(S₁,S₁,activation),
                            Lux.Dense(S₁,S,activation),
                            Lux.Dense(S₁,S,identity))
        new{T,typeof(NN)}(activation,S₁,S,NN)
    end
end



function Base.show(io::IO,basis::NetworkBasis_Lux)
    print(io, "\n")
    print(io, "  =========================================", "\n")
    print(io, "  =======4 Layer NetworkBasis by Lux=======", "\n")
    print(io, "  =========================================", "\n")
    print(io, "\n")
    print(io, "    Activation function σ  = ", basis.activation, "\n")
    print(io, "    Hidden Nodes S₁ = ", basis.S₁, "\n")
    print(io, "    Last Layer Nodes, Number of Basis S  = ", basis.S, "\n")
    print(io, "\n")
end



(L::NetworkBasis_Lux)(x::Number, j::Integer) = L.b[j](x)
Base.eachindex(L::NetworkBasis_Lux) = Base.eachindex(L.b)
Base.axes(L::NetworkBasis_Lux) = (Inclusion(0..1), eachindex(L))

Base.getindex(L::NetworkBasis_Lux, j::Integer)  = L.b[j]
Base.getindex(L::NetworkBasis_Lux, x::Number, j::Integer) = L(x,j)
Base.getindex(L::NetworkBasis_Lux, x::Number,  ::Colon) = [b(x) for b in L.b]
Base.getindex(L::NetworkBasis_Lux, X::AbstractVector, j::Integer) = L.(X,j)
Base.getindex(L::NetworkBasis_Lux, X::AbstractVector,  ::Colon) = [b(x) for x in X, b in L.b]

CompactBasisFunctions.basis(L::NetworkBasis_Lux) = L.b
CompactBasisFunctions.nbasis(L::NetworkBasis_Lux) = L.S

ContinuumArrays.grid(L::NetworkBasis_Lux) = L.x