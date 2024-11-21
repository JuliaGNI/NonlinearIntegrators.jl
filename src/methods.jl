
using GeometricIntegrators:LODEMethod

abstract type NetworkIntegratorMethod<:LODEMethod end

abstract type OneLayerMethod <: NetworkIntegratorMethod end
abstract type DenseNetMethod <: NetworkIntegratorMethod end


using GeometricIntegrators:GeometricMethod
abstract type TimeDependentPDEMethod <: GeometricMethod end
abstract type NinePointStencil <: TimeDependentPDEMethod end

# using GeometricIntegrators:IntegratorCache
# abstract type TimeDependentPDEMethodCache{ST,N} <: IntegratorCache{ST} end
# abstract type NinePointStencilCache{ST,N} <: TimeDependentPDEMethodCache{ST,N} end
