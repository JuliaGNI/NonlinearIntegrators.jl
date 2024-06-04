using AbstractNeuralNetworks
struct DenseNet_GML{T,NT,BT}<:DenseNetBasis{T}
    activation
    S₁::Int
    S::Int
    layers::Int
    NN::NT
    backend::BT
    function DenseNet_GML{T}(activation,S₁,S;layers=3,backend=CPU()) where {T}
        NN = AbstractNeuralNetworks.Chain(AbstractNeuralNetworks.Dense(1,S₁,activation),
                                        AbstractNeuralNetworks.Dense(S₁,S,activation),
                                        AbstractNeuralNetworks.Dense(S,1,identity,use_bias= false))
        new{T,typeof(NN),typeof(backend)}(activation,S₁,S,layers,NN,backend)
    end
end


function Base.show(io::IO,basis::DenseNet_GML)
    print(io, "\n")
    print(io, "  =========================================", "\n")
    print(io, "  =======3 Layer NetworkBasis by GML=======", "\n")
    print(io, "  =========================================", "\n")
    print(io, "\n")
    print(io, "    Activation function σ  = ", basis.activation, "\n")
    print(io, "    Hidden Nodes S₁ = ", basis.S₁, "\n")
    print(io, "    Last Layer Nodes, Number of Basis S  = ", basis.S, "\n")
    print(io, "\n")
end
