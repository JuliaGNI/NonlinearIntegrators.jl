"""
    NetworkBasis{T} <: Basis{T}

Abstract supertype for all neural-network basis types in this package. The type
parameter `T` is the floating-point element type (e.g. `Float64`, `Float32`).
Common fields (`NN`, `activation`, `SNN`, `dqdθ`, `V_func`, `dvdθ`) live in a
`NetworkBasisCore` sub-struct and are forwarded through `getproperty`, so call
sites can write `basis.NN`, `basis.activation`, etc.
"""
abstract type NetworkBasis{T} <: Basis{T} end

"""
    AbstractDenseNetBasis{T} <: NetworkBasis{T}

Abstract supertype for three-layer dense-network bases. The concrete implementation
is `DenseNetBasis{T}`.
"""
abstract type AbstractDenseNetBasis{T} <: NetworkBasis{T} end

"""
    AbstractShallowNetBasis{T} <: NetworkBasis{T}

Abstract supertype for single-hidden-layer network bases. The concrete implementation
is `ShallowNetBasis{T}`.
"""
abstract type AbstractShallowNetBasis{T} <: NetworkBasis{T} end

# Forward common-core field access so call sites like basis.NN, basis.activation, etc. keep working.
@inline function Base.getproperty(b::NetworkBasis, s::Symbol)
    s in (:activation, :NN, :backend, :SNN, :dqdθ, :V_func, :dvdθ) &&
        return getfield(getfield(b, :common), s)
    return getfield(b, s)
end

activation(b::NetworkBasis) = b.common.activation
backend(b::NetworkBasis) = b.common.backend
nbasis(b::NetworkBasis) = b.S

"""
    has_symbolic_derivatives(basis) -> Bool

Whether `basis` carries the derivatives (`dqdθ`, `V_func`, `dvdθ`) that
`SymbolicNeuralNetworks.jl` compiles at construction time.

`false` only for a [`ShallowNetBasis`](@ref) built with `symbolic = false`, which is the
form the `ForwardDiff`-based integrators want — they differentiate their ansatz at run
time and never read these fields. Every other basis builds them unconditionally.
"""
has_symbolic_derivatives(b::NetworkBasis) = b.dqdθ !== nothing

"""
    require_symbolic_derivatives(basis, method_name)

Throw an `ArgumentError` unless `basis` carries compiled symbolic derivatives.

Called from the constructors of the integrators whose `components!` evaluates them, so a
basis built with `symbolic = false` is rejected where the mistake was made rather than
several call levels down as a `nothing` being called on the first Newton iteration.
"""
function require_symbolic_derivatives(b::NetworkBasis, method_name::AbstractString)
    has_symbolic_derivatives(b) || throw(ArgumentError(
        "$(method_name) evaluates the symbolically compiled derivatives of its basis, but " *
        "the basis was built with `symbolic = false`. Rebuild it without that keyword. " *
        "`symbolic = false` is for ShallowNetAutodiff and ShallowNetAutodiffReversible, " *
        "which differentiate their ansatz with ForwardDiff and never read these fields."))
    return nothing
end

# The integrator constructors accept any `Basis`, not only a `NetworkBasis`. A basis that
# has no notion of compiled derivatives has nothing to check, so it passes — the guard is
# about the `symbolic = false` opt-out, not about narrowing the accepted basis types.
require_symbolic_derivatives(::Basis, ::AbstractString) = nothing
