# struct DenseNet_GML{T,NT,BT}<:DenseNetBasis{T}
#     activation
#     S₁::Int
#     S::Int
#     layers::Int
#     NN::NT
#     backend::BT
#     function DenseNet_GML{T}(activation,S₁,S;layers=3,backend=CPU()) where {T}
#         NN = AbstractNeuralNetworks.Chain(AbstractNeuralNetworks.Dense(1,S₁,activation),
#                                         AbstractNeuralNetworks.Dense(S₁,S,activation),
#                                         AbstractNeuralNetworks.Dense(S,1,identity,use_bias= false))
#         new{T,typeof(NN),typeof(backend)}(activation,S₁,S,layers,NN,backend)
#     end
# end

struct DenseNet_GML{T,NT,BT,SNNT,QWFT,VT,VWFT} <: DenseNetBasis{T}
    activation
    S::Int
    S₁::Int
    NN::NT
    backend::BT
    NP::Int

    SNN::SNNT
    dqdθ::QWFT

    V_func::VT
    dvdθ::VWFT
 
    function DenseNet_GML{T}(activation, S₁, S; backend=CPU()) where {T}
        NN = AbstractNeuralNetworks.Chain(AbstractNeuralNetworks.Dense(1, S₁, activation),
            AbstractNeuralNetworks.Dense(S₁, S, activation),
            AbstractNeuralNetworks.Dense(S, 1, identity, use_bias=false))
        NP = parameterlength(NN)
        SNN = SymbolicNeuralNetwork(NN)


        dqdθ = SymbolicNeuralNetworks.derivative(SymbolicNeuralNetworks.Gradient(SNN))[1]
        dqdθ_built_function = build_nn_function(dqdθ, SNN.params, SNN.input)

        VNN = SymbolicNeuralNetworks.derivative(SymbolicNeuralNetworks.Jacobian(SNN))
        V_built_function = build_nn_function(VNN, SNN.params, SNN.input)

        g = SymbolicNeuralNetworks.Gradient(VNN, SNN)
        dvdθ = SymbolicNeuralNetworks.derivative(g)
        dvdθ_built_function = build_nn_function(dvdθ, SNN.params, SNN.input)

        new{T,typeof(NN),typeof(backend),typeof(SNN),typeof(dqdθ_built_function),typeof(V_built_function),typeof(dvdθ_built_function)}(activation,S,S₁,NN, backend, NP, SNN,
            dqdθ_built_function, V_built_function, dvdθ_built_function)
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

