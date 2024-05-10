using AbstractNeuralNetworks

struct OneLayerNetwork_GML{T,NT}<:OneLayerNetBasis{T}
    activation
    S::Int
    NN::NT
    function OneLayerNetwork_GML{T}(activation,S) where {T}
        NN = AbstractNeuralNetworks.Chain(AbstractNeuralNetworks.Dense(1,S,activation),
            AbstractNeuralNetworks.Dense(S,1,identity,use_bias= false))
        new{T,typeof(NN)}(activation,S,NN)
    end
end

function Base.show(io::IO,basis::OneLayerNetwork_GML)
    print(io, "\n")
    print(io, "  =====================================", "\n")
    print(io, "  ======One Layer Network by GML======", "\n")
    print(io, "  =====================================", "\n")
    print(io, "\n")
    print(io, "    Activation function σ  = ", basis.activation, "\n")
    print(io, "    Last Layer Nodes, Number of Basis S  = ", basis.S, "\n")
    print(io, "    Trainable NN Parameters Amount  = ",3*basis.S, "\n")
    print(io, "\n")
end