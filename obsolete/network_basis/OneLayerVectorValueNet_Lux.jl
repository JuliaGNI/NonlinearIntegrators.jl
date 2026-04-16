using Lux
struct OneLayerVectorValueNet_Lux{T,NT}<:OneLayerNetBasis{T}
    activation
    S::Int
    D::Int
    NN::NT

    function OneLayerVectorValueNet_Lux{T}(S,activation,D) where {T}
        NN = Lux.Chain(Lux.Dense(1,S,activation),Lux.Dense(S,D,use_bias = false))
        new{T,typeof(NN)}(activation,S,D,NN)
    end
end

function Base.show(io::IO,basis::OneLayerVectorValueNet_Lux)
    print(io, "\n")
    print(io, "  =====================================", "\n")
    print(io, "  ======One Layer Network by Lux======", "\n")
    print(io, "  =====================================", "\n")
    print(io, "\n")
    print(io, "    Activation function σ  = ", basis.activation, "\n")
    print(io, "    Hidden Layer Nodes, Number of Basis S  = ", basis.S, "\n")
    print(io, "    Last Layer Nodes, Problem Dimension D  = ", basis.D, "\n")
    print(io, "\n")
end