
using GeometricIntegrators: LODEMethod

abstract type NetworkIntegratorMethod <: LODEMethod end

abstract type OneLayerMethod <: NetworkIntegratorMethod end
abstract type DenseNetMethod <: NetworkIntegratorMethod end


using GeometricIntegrators: GeometricMethod
abstract type TimeDependentPDEMethod <: GeometricMethod end
abstract type NinePointStencil <: TimeDependentPDEMethod end

using GeometricIntegrators: Extrapolation
struct IntegratorExtrapolation <: Extrapolation end

abstract type InitialParametersMethod end
struct TrainingMethod <: InitialParametersMethod end
struct OGA1d <: InitialParametersMethod end
