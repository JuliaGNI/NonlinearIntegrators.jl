using CompactBasisFunctions
using AbstractNeuralNetworks
using ContinuumArrays

struct NetworkBasis_GML{T,NT}<:Basis{T}
    activation
    S₁::Int
    S::Int
    NN::NT
    
    function NetworkBasis_GML{T}(activation,S₁,S) where {T}
        NN = AbstractNeuralNetworks.Chain(AbstractNeuralNetworks.Dense(1,S₁,activation),
                                        AbstractNeuralNetworks.Dense(S₁,S₁,activation),
                                        AbstractNeuralNetworks.Dense(S₁,S,activation),
                                        AbstractNeuralNetworks.Dense(S,1,identity,use_bias= false))
        new{T,typeof(NN)}(activation,S₁,S,NN)
    end
end


function Base.show(io::IO,basis::NetworkBasis_GML)
    print(io, "\n")
    print(io, "  =========================================", "\n")
    print(io, "  =======4 Layer NetworkBasis by GML=======", "\n")
    print(io, "  =========================================", "\n")
    print(io, "\n")
    print(io, "    Activation function σ  = ", basis.activation, "\n")
    print(io, "    Hidden Nodes S₁ = ", basis.S₁, "\n")
    print(io, "    Last Layer Nodes, Number of Basis S  = ", basis.S, "\n")
    print(io, "\n")
end

(L::NetworkBasis_GML)(x::Number, j::Integer) = L.b[j](x)
Base.eachindex(L::NetworkBasis_GML) = Base.eachindex(L.b)
Base.axes(L::NetworkBasis_GML) = (Inclusion(0..1), eachindex(L))

Base.getindex(L::NetworkBasis_GML, j::Integer)  = L.b[j]
Base.getindex(L::NetworkBasis_GML, x::Number, j::Integer) = L(x,j)
Base.getindex(L::NetworkBasis_GML, x::Number,  ::Colon) = [b(x) for b in L.b]
Base.getindex(L::NetworkBasis_GML, X::AbstractVector, j::Integer) = L.(X,j)
Base.getindex(L::NetworkBasis_GML, X::AbstractVector,  ::Colon) = [b(x) for x in X, b in L.b]

CompactBasisFunctions.basis(L::NetworkBasis_GML) = L.b
CompactBasisFunctions.nbasis(L::NetworkBasis_GML) = L.S

ContinuumArrays.grid(L::NetworkBasis_GML) = L.x