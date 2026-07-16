using CompactBasisFunctions: Basis

abstract type NetworkBasis{T} <: Basis{T} end

abstract type DenseNetBasis{T} <: NetworkBasis{T} end
abstract type OneLayerNetBasis{T} <: NetworkBasis{T} end

# Forward common-core field access so call sites like basis.NN, basis.activation, etc. keep working.
@inline function Base.getproperty(b::NetworkBasis, s::Symbol)
    s in (:activation, :NN, :backend, :SNN, :dqdθ, :V_func, :dvdθ) &&
        return getfield(getfield(b, :common), s)
    return getfield(b, s)
end

activation(b::NetworkBasis) = b.common.activation
backend(b::NetworkBasis)    = b.common.backend
nbasis(b::NetworkBasis)     = b.S
