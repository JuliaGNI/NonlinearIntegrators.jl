
using GeometricIntegrators: LODEMethod

abstract type NetworkIntegratorMethod <: LODEMethod end

abstract type OneLayerMethod <: NetworkIntegratorMethod end
abstract type DenseNetMethod <: NetworkIntegratorMethod end

using GeometricIntegrators: Extrapolation
struct IntegratorExtrapolation <: Extrapolation end

abstract type InitialParametersMethod end
struct TrainingMethod <: InitialParametersMethod end
struct OGA1d <: InitialParametersMethod end
# Legacy OGA: the pre-refactor Float64 normal-equations variant, kept as a
# selectable alternative to OGA1d for comparison (see NonLinear_OneLayer_GML).
struct OGA1d_Legacy <: InitialParametersMethod end
struct LSGD <: InitialParametersMethod end