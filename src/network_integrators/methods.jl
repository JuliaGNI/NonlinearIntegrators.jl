using GeometricIntegrators:LODEMethod

abstract type NetworkIntegratorMethod<:LODEMethod end

abstract type OneLayerMethod <: NetworkIntegratorMethod end
abstract type DenseNetMethod <: NetworkIntegratorMethod end
