"""
    NetworkBasisCore{AT,NT,BT,SNNT,QWFT,VT,VWFT}

Common sub-struct shared by all `NetworkBasis` concrete types. Bundles the neural
network together with its symbolically-compiled derivatives, which are built once at
construction time and reused at every integration step.

`SNN`, `dqdθ`, `V_func` and `dvdθ` are all `nothing` for a
[`ShallowNetBasis`](@ref) built with `symbolic = false` — the form the `ForwardDiff`-based
integrators want, since they differentiate their ansatz at run time and never read these
fields. Test for it with [`has_symbolic_derivatives`](@ref) rather than by comparing a
field against `nothing`; the four type parameters are `Nothing` in that case, so the
distinction is visible to dispatch.

# Fields
- `activation`: activation function used in the hidden layer.
- `NN`: neural-network model (the forward map q(t; θ)).
- `backend`: computation backend (e.g. CPU array backend).
- `SNN`: `SymbolicNeuralNetwork` wrapping `NN`, used to derive symbolic expressions
  for all required derivatives.
- `dqdθ`: compiled function returning ∂q/∂θ — the Jacobian of the network output
  (position) with respect to the network parameters θ.
- `V_func`: compiled function returning the velocity v = dq/dt — the time-derivative
  of the network output, obtained from the symbolic Jacobian with respect to the
  time input.
- `dvdθ`: compiled function returning ∂v/∂θ — the Jacobian of the velocity with
  respect to the network parameters θ.
"""
struct NetworkBasisCore{AT, NT, BT, SNNT, QWFT, VT, VWFT}
    activation::AT
    NN::NT
    backend::BT
    SNN::SNNT
    dqdθ::QWFT
    V_func::VT
    dvdθ::VWFT
end

"""
    build_network_derivatives(NN; cse = true, inplace = true) -> (SNN, dqdθ, V_func, dvdθ)

Compile the four symbolic slots of a [`NetworkBasisCore`](@ref) for the network `NN`: the
gradient of the output with respect to the parameters, the time derivative of the output,
and the gradient of *that* with respect to the parameters. `cse` and `inplace` go straight
to `SymbolicNeuralNetworks.build_nn_function`; see [`ShallowNetBasis`](@ref).

Shared by [`ShallowNetBasis`](@ref) and [`DenseNetBasis`](@ref), which differ only in the
`NN` they hand over — keeping the two in one place rather than in two copies that drifted
apart once already. It also lets `ShallowNetBasis`'s `symbolic = false` branch read as the
single expression it is.
"""
function build_network_derivatives(NN; cse::Bool = true, inplace::Bool = true)
    SNN = SymbolicNeuralNetwork(NN)
    build(eq) = build_nn_function(eq, SNN.params, SNN.input; cse = cse, inplace = inplace)

    # The network maps a scalar to a scalar, so its output and its Jacobian are both
    # one-element arrays. Differentiating the scalar entry rather than the array is what makes
    # `symbolic_parameter_gradient` return the parameter-shaped gradient itself instead of an
    # array holding one of them, which is the shape `components!` reads.
    soutput = SNN.model(SNN.input, SNN.params)
    dqdθ_built = build(SymbolicNeuralNetworks.symbolic_parameter_gradient(soutput[1], SNN))

    VNN = SymbolicNeuralNetworks.derivative(SymbolicNeuralNetworks.Jacobian(SNN))
    V_built = build(VNN)

    dvdθ_built = build(SymbolicNeuralNetworks.symbolic_parameter_gradient(VNN[1, 1], SNN))

    return SNN, dqdθ_built, V_built, dvdθ_built
end
