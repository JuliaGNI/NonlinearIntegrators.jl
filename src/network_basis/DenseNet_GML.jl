struct DenseNet_GML{T, NT, BT, SNNT, QWFT, VT, VWFT} <: DenseNetBasis{T}
    S      :: Int
    S₁     :: Int
    NP     :: Int
    common :: NetworkBasisCore{NT, BT, SNNT, QWFT, VT, VWFT}

    function DenseNet_GML{T}(activation, S₁, S; backend=CPU()) where T
        NN = AbstractNeuralNetworks.Chain(
            AbstractNeuralNetworks.Dense(1, S₁, activation),
            AbstractNeuralNetworks.Dense(S₁, S, activation),
            AbstractNeuralNetworks.Dense(S, 1, identity, use_bias=false))
        NP = parameterlength(NN)
        SNN = SymbolicNeuralNetwork(NN)

        soutput = SNN.model(SNN.input, SNN.params)
        dqdθ_sym = SymbolicNeuralNetworks.symbolic_pullback(soutput, SNN)[1]
        dqdθ_built = build_nn_function(dqdθ_sym, SNN.params, SNN.input)

        VNN = SymbolicNeuralNetworks.derivative(SymbolicNeuralNetworks.Jacobian(SNN))
        V_built = build_nn_function(VNN, SNN.params, SNN.input)

        dvdθ_sym = SymbolicNeuralNetworks.symbolic_pullback(VNN, SNN)
        dvdθ_built = build_nn_function(dvdθ_sym, SNN.params, SNN.input)

        core = NetworkBasisCore(activation, NN, backend, SNN, dqdθ_built, V_built, dvdθ_built)
        new{T, typeof(NN), typeof(backend), typeof(SNN),
            typeof(dqdθ_built), typeof(V_built), typeof(dvdθ_built)}(S, S₁, NP, core)
    end
end

function Base.show(io::IO, basis::DenseNet_GML)
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
