using Lux
struct DenseNet_Lux{T,NT}<:DenseNetBasis{T}
    activation
    S₁::Int
    S::Int
    layers::Int
    NN::NT

    function DenseNet_Lux{T}(activation,S₁,S;layers=3) where {T}
        NN = Lux.Chain(Lux.Dense(1,S₁,activation),
                            Lux.Dense(S₁,S,activation),
                            Lux.Dense(S,1,identity,use_bias=false))
        new{T,typeof(NN)}(activation,S₁,S,layers,NN)
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

