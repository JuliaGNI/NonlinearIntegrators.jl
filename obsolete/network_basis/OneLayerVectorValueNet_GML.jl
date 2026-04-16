using AbstractNeuralNetworks
struct OneLayerVectorValueNet_GML{T,NT}<:OneLayerNetBasis{T}
    activation
    S::Int
    D::Int
    NN::NT

    function OneLayerVectorValueNet_GML{T}(S,activation,D) where {T}
        NN = AbstractNeuralNetworks.Chain(AbstractNeuralNetworks.Dense(1,S,activation),
            AbstractNeuralNetworks.Dense(S,D,identity,use_bias= false))
        new{T,typeof(NN)}(activation,S,D,NN)
    end
end

function Base.show(io::IO,basis::OneLayerVectorValueNet_GML)
    print(io, "\n")
    print(io, "  =====================================", "\n")
    print(io, "  ======One Layer Network by GML======", "\n")
    print(io, "  =====================================", "\n")
    print(io, "\n")
    print(io, "    Activation function σ  = ", basis.activation, "\n")
    print(io, "    Hidden Layer Nodes, Number of Basis S  = ", basis.S, "\n")
    print(io, "    Last Layer Nodes, Problem Dimension D  = ", basis.D, "\n")
    print(io, "\n")
end