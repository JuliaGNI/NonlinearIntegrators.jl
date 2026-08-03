
using GeometricIntegrators: LODEMethod

"""
    NetworkIntegratorMethod <: LODEMethod

Abstract supertype for all neural-network-based variational integrators in this package.
Every concrete subtype wraps a `NetworkIntegratorCore` under the field `common` and
exposes its fields (basis, quadrature, extrapolation settings, …) via `getproperty`
forwarding, so call sites can write `method.basis`, `method.record_grid_points`, etc.
"""
abstract type NetworkIntegratorMethod <: LODEMethod end

"""
    OneLayerMethod <: NetworkIntegratorMethod

Abstract supertype for integrators whose ansatz is a single-hidden-layer network:
`NonLinear_OneLayer_GML`, `Hardcode_int`, `Time_reversible_OneLayer`, and
`Time_Reversible_Hardcode`. All accept `OGA1d` (default) or `OGA1d_Legacy` as their
`initial_guess_method`.
"""
abstract type OneLayerMethod <: NetworkIntegratorMethod end

"""
    DenseNetMethod <: NetworkIntegratorMethod

Abstract supertype for integrators whose ansatz is a three-layer dense network:
`NonLinear_DenseNet_GML`. Accepts `LSGD` (default) or `TrainingMethod` as its
`initial_guess_method`.
"""
abstract type DenseNetMethod <: NetworkIntegratorMethod end

using GeometricIntegrators: Extrapolation

"""
    IntegratorExtrapolation <: Extrapolation

Initial-trajectory method that seeds the per-step Newton solve by integrating a
short sub-problem with `ImplicitMidpoint` over `extrapolation_substep` sub-steps.
This is the default `initial_trajectory_method` for all network integrators and
usually gives the best convergence.

See also: `NoExtrapolation`, `HermiteExtrapolation` (from `GeometricIntegratorsBase`).
"""
struct IntegratorExtrapolation <: Extrapolation end

"""
    InitialParametersMethod

Abstract supertype for the strategy used to initialise the network weights (parameter
vector) at the start of each time step before the Newton solve.
"""
abstract type InitialParametersMethod end

"""
    TrainingMethod <: InitialParametersMethod

Initialise network parameters by gradient descent (via `Optimisers.jl`) against an
MSE target built from the extrapolated trajectory. Applies to both
`NonLinear_OneLayer_GML` and `NonLinear_DenseNet_GML`; controlled by the
`training_epochs` constructor kwarg.
"""
struct TrainingMethod <: InitialParametersMethod end

"""
    OGA1d <: InitialParametersMethod

Initialise single-layer network parameters using the Orthogonal Greedy Algorithm (OGA).
Atoms are drawn from a uniform dictionary over `bias_interval`; the best atom is
selected by maximising the (quadrature-weighted) inner product with the residual.
This is the default `initial_guess_method` for `OneLayerMethod` integrators.

The implementation uses QR-based least squares (`weighted_lstsq`) for numerical
robustness at reduced precision. See also `OGA1d_Legacy`.
"""
struct OGA1d <: InitialParametersMethod end

"""
    OGA1d_Legacy <: InitialParametersMethod

Pre-refactor OGA variant: builds the Gram matrix and solves the normal equations in
Float64 regardless of the working precision. Kept as a selectable alternative to
`OGA1d` for reproducibility comparisons. Prefer `OGA1d` for new work.
"""
struct OGA1d_Legacy <: InitialParametersMethod end

"""
    LSGD <: InitialParametersMethod

Initialise dense-network parameters with a Lipschitz-SGD step (LSGD). Used as the
default `initial_guess_method` for `NonLinear_DenseNet_GML`.
"""
struct LSGD <: InitialParametersMethod end