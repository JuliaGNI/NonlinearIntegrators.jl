using Lux
using CompactBasisFunctions
using ContinuumArrays

struct DenseNet_Lux{T,NT}<:Basis{T}
    activation
    S₁::Int
    S::Int
    NN::NT

    function DenseNet_Lux{T}(activation,S₁,S) where {T}
        NN = Lux.Chain(Lux.Dense(1,S₁,activation),
                            Lux.Dense(S₁,S₁,activation),
                            Lux.Dense(S₁,S,activation),
                            Lux.Dense(S₁,S,identity))
        new{T,typeof(NN)}(activation,S₁,S,NN)
    end
end



function Base.show(io::IO,basis::DenseNet_Lux)
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



(L::DenseNet_Lux)(x::Number, j::Integer) = L.b[j](x)
Base.eachindex(L::DenseNet_Lux) = Base.eachindex(L.b)
Base.axes(L::DenseNet_Lux) = (Inclusion(0..1), eachindex(L))

Base.getindex(L::DenseNet_Lux, j::Integer)  = L.b[j]
Base.getindex(L::DenseNet_Lux, x::Number, j::Integer) = L(x,j)
Base.getindex(L::DenseNet_Lux, x::Number,  ::Colon) = [b(x) for b in L.b]
Base.getindex(L::DenseNet_Lux, X::AbstractVector, j::Integer) = L.(X,j)
Base.getindex(L::DenseNet_Lux, X::AbstractVector,  ::Colon) = [b(x) for x in X, b in L.b]

CompactBasisFunctions.basis(L::DenseNet_Lux) = L.b
CompactBasisFunctions.nbasis(L::DenseNet_Lux) = L.S

ContinuumArrays.grid(L::DenseNet_Lux) = L.x