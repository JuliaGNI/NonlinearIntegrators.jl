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
backend(b::NetworkBasis)    = b.common.backend
nbasis(b::NetworkBasis)     = b.S
