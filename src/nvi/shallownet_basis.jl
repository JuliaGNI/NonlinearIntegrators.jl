using AbstractNeuralNetworks

"""
    ShallowNetBasis{T} <: AbstractShallowNetBasis{T}

Single-hidden-layer network basis, built with `AbstractNeuralNetworks`.
The network maps a scalar time input to a scalar position: `Dense(1, S, σ) → Dense(S, 1)`.
Symbolic derivatives (`dqdθ`, `V_func`, `dvdθ`) are compiled once at construction time
via `SymbolicNeuralNetworks.jl`, unless `symbolic = false`.

# Constructor

    ShallowNetBasis{T}(activation, S; backend = CPU(), symbolic = true,
                       cse = true, inplace = true)

- `activation`: any elementwise activation function (e.g. `tanh`, `relu_k(3)`)
- `S::Int`: number of hidden neurons (= number of basis functions)
- `symbolic`: compile the symbolic derivatives (default `true`). Pass `false` to build the
  network only and leave `SNN`, `dqdθ`, `V_func` and `dvdθ` as `nothing`.
- `cse`, `inplace`: forwarded to `SymbolicNeuralNetworks.build_nn_function`; see below.

`symbolic = false` exists for [`ShallowNetAutodiff`](@ref) and
[`ShallowNetAutodiffReversible`](@ref): those two differentiate their ansatz with
`ForwardDiff` at run time and never read the compiled derivatives, so building them is
pure overhead — 0.095 s against 0.003 s for `tanh` at `S = 8`, Float64. The integrators that
*do* read them ([`ShallowNet`](@ref), [`ShallowNetReversible`](@ref), [`DenseNet`](@ref))
reject such a basis in their constructor; see [`has_symbolic_derivatives`](@ref).

`cse` (common-subexpression elimination during code generation) and `inplace` (evaluate a
batch through a kernel writing into one preallocated array) default to whatever
`SymbolicNeuralNetworks` defaults to — currently both `true`, which is what you want.
They are exposed to be turned *off*: `cse = false, inplace = false` is the code generation
`SymbolicNeuralNetworks` performed before 0.4.0, which is how
`benchmark/compare_derivative_backends.jl` measures what that release bought. Note that
`inplace = true` mutates its output and so cannot be differentiated with `Zygote`; nothing
in this package does that, but a caller who wants to needs `inplace = false`.

# Example

```julia
basis = ShallowNetBasis{Float64}(tanh, 8)
autodiff_basis = ShallowNetBasis{Float64}(tanh, 8; symbolic = false)
legacy_codegen = ShallowNetBasis{Float64}(tanh, 8; cse = false, inplace = false)
```
"""
struct ShallowNetBasis{T, AT, NT, BT, SNNT, QWFT, VT, VWFT} <: AbstractShallowNetBasis{T}
    S      :: Int
    common :: NetworkBasisCore{AT,NT, BT, SNNT, QWFT, VT, VWFT}

    function ShallowNetBasis{T}(activation, S; backend=CPU(), symbolic::Bool=true,
                                cse::Bool=true, inplace::Bool=true) where T
        NN = AbstractNeuralNetworks.Chain(
            AbstractNeuralNetworks.Dense(1, S, activation),
            AbstractNeuralNetworks.Dense(S, 1, identity, use_bias=false))

        # `nothing` in all four slots rather than a separate type: `NetworkBasisCore` is
        # parametric over them, so the derivative-free basis is just the one whose
        # `SNNT`/`QWFT`/`VT`/`VWFT` are `Nothing`, and every call site that only wants
        # `NN`, `activation` or `S` keeps working unchanged.
        SNN, dqdθ_built, V_built, dvdθ_built =
            symbolic ? build_shallownet_derivatives(NN; cse = cse, inplace = inplace) :
                       (nothing, nothing, nothing, nothing)

        core = NetworkBasisCore(activation, NN, backend, SNN, dqdθ_built, V_built, dvdθ_built)
        new{T, typeof(activation), typeof(NN), typeof(backend), typeof(SNN),
            typeof(dqdθ_built), typeof(V_built), typeof(dvdθ_built)}(S, core)
    end
end

"""
    build_shallownet_derivatives(NN; cse = true, inplace = true) -> (SNN, dqdθ, V_func, dvdθ)

Compile the symbolic derivatives of the shallow network `NN`: the pullback of the output
with respect to the parameters, the time derivative of the output, and the pullback of
*that* with respect to the parameters. `cse` and `inplace` go straight to
`SymbolicNeuralNetworks.build_nn_function`; see [`ShallowNetBasis`](@ref).

Split out of the [`ShallowNetBasis`](@ref) constructor so the `symbolic = false` branch
reads as the single expression it is.
"""
function build_shallownet_derivatives(NN; cse::Bool = true, inplace::Bool = true)
    SNN = SymbolicNeuralNetwork(NN)
    build(eq) = build_nn_function(eq, SNN.params, SNN.input; cse = cse, inplace = inplace)

    soutput = SNN.model(SNN.input, SNN.params)
    dqdθ_built = build(SymbolicNeuralNetworks.symbolic_pullback(soutput, SNN)[1,1])

    VNN = SymbolicNeuralNetworks.derivative(SymbolicNeuralNetworks.Jacobian(SNN))
    V_built = build(VNN)

    dvdθ_built = build(SymbolicNeuralNetworks.symbolic_pullback(VNN[1,1], SNN)[1])

    return SNN, dqdθ_built, V_built, dvdθ_built
end

function Base.show(io::IO, basis::ShallowNetBasis)
    print(io, "\n")
    print(io, "  =====================================", "\n")
    print(io, "  ========Shallow Network Basis========", "\n")
    print(io, "  =====================================", "\n")
    print(io, "\n")
    print(io, "    Activation function σ  = ", basis.activation, "\n")
    print(io, "    Last Layer Nodes, Number of Basis S  = ", basis.S, "\n")
    print(io, "    Trainable NN Parameters Amount  = ", 3*basis.S, "\n")
    print(io, "    Symbolic derivatives  = ",
          has_symbolic_derivatives(basis) ? "compiled" : "none (symbolic = false)", "\n")
    print(io, "\n")
end
