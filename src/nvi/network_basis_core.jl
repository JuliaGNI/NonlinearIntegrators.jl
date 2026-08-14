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
struct NetworkBasisCore{AT,NT, BT, SNNT, QWFT, VT, VWFT}
    activation:: AT
    NN        :: NT
    backend   :: BT
    SNN       :: SNNT
    dqdθ      :: QWFT
    V_func    :: VT
    dvdθ      :: VWFT
end
