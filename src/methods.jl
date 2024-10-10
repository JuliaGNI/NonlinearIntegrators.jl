
using GeometricIntegrators:LODEMethod

abstract type NetworkIntegratorMethod<:LODEMethod end

abstract type OneLayerMethod <: NetworkIntegratorMethod end
abstract type DenseNetMethod <: NetworkIntegratorMethod end


# using GeometricIntegrators:GeometricMethod
# abstract type TimeDependentPDEMethod <: GeometricMethod end
# abstract type FiniteDifference <: TimeDependentPDEMethod end

# using GeometricIntegrators:IntegratorCache
# abstract type TimeDependentPDEMethodCache <: IntegratorCache end