using CompactBasisFunctions: Basis

abstract type NetworkBasis{T} <: Basis{T} end

abstract type DenseNetBasis{T} <: NetworkBasis{T} end
abstract type OneLayerNetBasis{T} <: NetworkBasis{T} end