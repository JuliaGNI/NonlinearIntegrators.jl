using CompactBasisFunctions
using AbstractNeuralNetworks
using ContinuumArrays

struct DenseNet_GML{T,NT,BT}<:DenseNetBasis{T}
    activation
    S₁::Int
    S::Int
    layers::Int
    NN::NT
    backend::BT
    function DenseNet_GML{T}(activation,S₁,S;layers=4,backend=CPU()) where {T}
        NN = AbstractNeuralNetworks.Chain(AbstractNeuralNetworks.Dense(1,S₁,activation),
                                        AbstractNeuralNetworks.Dense(S₁,S₁,activation),
                                        AbstractNeuralNetworks.Dense(S₁,S,activation),
                                        AbstractNeuralNetworks.Dense(S,1,identity,use_bias= false))
        new{T,typeof(NN),typeof(backend)}(activation,S₁,S,layers,NN,backend)
    end
end


function Base.show(io::IO,basis::DenseNet_GML)
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

(L::DenseNet_GML)(x::Number, j::Integer) = L.b[j](x)
Base.eachindex(L::DenseNet_GML) = Base.eachindex(L.b)
Base.axes(L::DenseNet_GML) = (Inclusion(0..1), eachindex(L))

Base.getindex(L::DenseNet_GML, j::Integer)  = L.b[j]
Base.getindex(L::DenseNet_GML, x::Number, j::Integer) = L(x,j)
Base.getindex(L::DenseNet_GML, x::Number,  ::Colon) = [b(x) for b in L.b]
Base.getindex(L::DenseNet_GML, X::AbstractVector, j::Integer) = L.(X,j)
Base.getindex(L::DenseNet_GML, X::AbstractVector,  ::Colon) = [b(x) for x in X, b in L.b]

CompactBasisFunctions.basis(L::DenseNet_GML) = L.b
CompactBasisFunctions.nbasis(L::DenseNet_GML) = L.S

ContinuumArrays.grid(L::DenseNet_GML) = L.x