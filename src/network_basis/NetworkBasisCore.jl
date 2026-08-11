"""
    NetworkBasisCore{AT,NT,BT,SNNT,QWFT,VT,VWFT}

Common sub-struct shared by all `NetworkBasis` concrete types. Bundles the neural
network together with its symbolically-compiled derivatives, which are built once at
construction time and reused at every integration step.

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
