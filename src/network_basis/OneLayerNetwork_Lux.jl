using Lux
struct OneLayerNetwork_Lux{T,SNT,NT}<:OneLayerNetBasis{T}
    activation
    S::Int
    SubNN::SNT
    NN::NT
    function OneLayerNetwork_Lux{T}(S,activation,D) where {T}
        SubNN = Lux.Chain(Lux.Dense(1,S,activation,init_weight=rand64,init_bias = rand64),Lux.Dense(S,1,use_bias = false))
        layers = [SubNN for _ in 1:D]
        connection(outputs...) = vcat(outputs...)
        NN = Lux.Parallel(connection, layers...)
        new{T,typeof(SubNN),typeof(NN)}(activation,S,SubNN,NN)
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