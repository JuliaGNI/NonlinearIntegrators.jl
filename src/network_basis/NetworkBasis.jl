using CompactBasisFunctions: Basis

abstract type NetworkBasis{T} <: Basis{T} end

abstract type DenseNet{T} <: NetworkBasis{T} end
abstract type OneLayerNet{T} <: NetworkBasis{T} end