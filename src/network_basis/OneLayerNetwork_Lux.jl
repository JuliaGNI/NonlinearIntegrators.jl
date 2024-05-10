using Lux
struct OneLayerNetwork_Lux{T,NT}<:OneLayerNetBasis{T}
    activation
    S::Int
    NN::NT

    function OneLayerNetwork_Lux{T}(S,activation) where {T}
        NN = Lux.Chain(Lux.Dense(1,S,activation),Lux.Dense(S,1,use_bias = false))
        new{T,typeof(NN)}(activation,S,NN)
    end
end

function Base.show(io::IO,basis::OneLayerNetwork_Lux)
    print(io, "\n")
    print(io, "  =====================================", "\n")
    print(io, "  ======One Layer Network by Lux======", "\n")
    print(io, "  =====================================", "\n")
    print(io, "\n")
    print(io, "    Activation function σ  = ", basis.activation, "\n")
    print(io, "    Last Layer Nodes, Number of Basis S  = ", basis.S, "\n")
    print(io, "    Trainable NN Parameters Amount  = ",3*basis.S, "\n")
    print(io, "\n")
end