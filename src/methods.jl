
using GeometricIntegrators: LODEMethod

abstract type NetworkIntegratorMethod <: LODEMethod end

abstract type OneLayerMethod <: NetworkIntegratorMethod end
abstract type DenseNetMethod <: NetworkIntegratorMethod end

using GeometricIntegrators: Extrapolation
struct IntegratorExtrapolation <: Extrapolation end

abstract type InitialParametersMethod end
struct TrainingMethod <: InitialParametersMethod end
struct LSGD <: InitialParametersMethod end
# The Orthogonal Greedy Algorithm seeds (`OGA`, its presets `OGA1d`/`OGA2d`/…, and the
# `OGA1dNormalEquations` reference) are also `InitialParametersMethod`s; they live in
# `src/oga/` because they carry a whole subsystem's worth of configuration.