using AbstractNeuralNetworks

struct OneLayerNetwork_GML{T,NT,BT,SNNT,QWFT,VWFT}<:OneLayerNetBasis{T}
    activation
    S::Int
    NN::NT
    backend::BT

    SNN::SNNT
    dqdθ::QWFT
    dvdθ::VWFT

    function OneLayerNetwork_GML{T}(activation,S;backend=CPU()) where {T}
        NN = AbstractNeuralNetworks.Chain(AbstractNeuralNetworks.Dense(1,S,activation),
            AbstractNeuralNetworks.Dense(S,1,identity,use_bias= false))
        SNN = SymbolicNeuralNetwork(NN)

        dqdθ = SymbolicNeuralNetworks.derivative(SymbolicNeuralNetworks.Gradient(SNN)) #[1]
        dqdθ_built_function = build_nn_function(dqdθ, SNN.params, SNN.input)

        jac = SymbolicNeuralNetworks.derivative(SymbolicNeuralNetworks.Jacobian(SNN))
        g = SymbolicNeuralNetworks.Gradient(jac,SNN)
        dvdθ =SymbolicNeuralNetworks.derivative(g)
        dvdθ_built_function = build_nn_function(dvdθ, SNN.params, SNN.input)

        new{T,typeof(NN),typeof(backend),typeof(SNN),typeof(dqdθ_built_function),typeof(dvdθ_built_function)}(activation,S,NN,backend,SNN,
        dqdθ_built_function,dvdθ_built_function)
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