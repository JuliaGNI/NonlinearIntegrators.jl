using AbstractNeuralNetworks

"""
    OneLayerNetwork_GML{T} <: OneLayerNetBasis{T}

Single-hidden-layer network basis with neural network construction by GeometricMachineLearning.jl (GML).
The network maps a scalar time input to a scalar position: `Dense(1, S, σ) → Dense(S, 1)`.
Symbolic derivatives (`dqdθ`, `V_func`, `dvdθ`) are compiled once at construction time
via `SymbolicNeuralNetworks.jl`.

# Constructor

    OneLayerNetwork_GML{T}(activation, S; backend = CPU())

- `activation`: any elementwise activation function (e.g. `tanh`, `relu_k(3)`)
- `S::Int`: number of hidden neurons (= number of basis functions)

# Example

```julia
basis = OneLayerNetwork_GML{Float64}(tanh, 8)
```
"""
struct OneLayerNetwork_GML{T, AT, NT, BT, SNNT, QWFT, VT, VWFT} <: OneLayerNetBasis{T}
    S      :: Int
    common :: NetworkBasisCore{AT,NT, BT, SNNT, QWFT, VT, VWFT}

    function OneLayerNetwork_GML{T}(activation, S; backend=CPU()) where T
        NN = AbstractNeuralNetworks.Chain(
            AbstractNeuralNetworks.Dense(1, S, activation),
            AbstractNeuralNetworks.Dense(S, 1, identity, use_bias=false))
        SNN = SymbolicNeuralNetwork(NN)

        soutput = SNN.model(SNN.input, SNN.params)
        dqdθ_sym = SymbolicNeuralNetworks.symbolic_pullback(soutput, SNN)[1,1]
        dqdθ_built = build_nn_function(dqdθ_sym, SNN.params, SNN.input)

        VNN = SymbolicNeuralNetworks.derivative(SymbolicNeuralNetworks.Jacobian(SNN))
        V_built = build_nn_function(VNN, SNN.params, SNN.input)

        jac = VNN[1,1]
        dvdθ_sym = SymbolicNeuralNetworks.symbolic_pullback(jac, SNN)[1]
        dvdθ_built = build_nn_function(dvdθ_sym, SNN.params, SNN.input)

        core = NetworkBasisCore(activation, NN, backend, SNN, dqdθ_built, V_built, dvdθ_built)
        new{T, typeof(activation), typeof(NN), typeof(backend), typeof(SNN),
            typeof(dqdθ_built), typeof(V_built), typeof(dvdθ_built)}(S, core)
    end
end

function Base.show(io::IO, basis::OneLayerNetwork_GML)
    print(io, "\n")
    print(io, "  =====================================", "\n")
    print(io, "  ======One Layer Network by GML======", "\n")
    print(io, "  =====================================", "\n")
    print(io, "\n")
    print(io, "    Activation function σ  = ", basis.activation, "\n")
    print(io, "    Last Layer Nodes, Number of Basis S  = ", basis.S, "\n")
    print(io, "    Trainable NN Parameters Amount  = ", 3*basis.S, "\n")
    print(io, "\n")
end
