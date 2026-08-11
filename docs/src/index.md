
# NonlinearIntegrators.jl

**NonlinearIntegrators.jl** provides structure-preserving variational integrators for
Lagrangian mechanical systems whose ansatz is a neural network rather than a polynomial.
The package is built on top of [GeometricIntegrators.jl](https://github.com/JuliaGNI/GeometricIntegrators.jl)
and accepts any `AbstractProblemIODE` (implicit ODE in Lagrangian form).

## Abstract type hierarchy

Every integrator is a subtype of `NetworkIntegratorMethod <: LODEMethod`. Two families
are provided:

- **`OneLayerMethod`** — single-hidden-layer ansatz. Includes `NonLinear_OneLayer_GML`,
  `Hardcode_int`, `Time_reversible_OneLayer`, and `Time_Reversible_Hardcode`. All use the
  Orthogonal Greedy Algorithm to initialise the network weights at each step, configured
  through `initial_guess_method` — `OGA1d()` by default, `OGA1dNormalized()` for
  `Hardcode_int`. See [`OGA`](@ref) and the *Orthogonal Greedy Algorithm* section of this
  manual for the dictionary, selection and fit variants and when to reach for each.

- **`DenseNetMethod`** — three-layer dense-network ansatz. The only implementation is
  `NonLinear_DenseNet_GML`, which uses `LSGD` (Least Square Gradient Descent) for parameter initialisation.

The initial trajectory used to warm-start the Newton solve is controlled by
`initial_trajectory_method`: `IntegratorExtrapolation` (default — integrates a
sub-problem with `ImplicitMidpoint`), `HermiteExtrapolation`, or `NoExtrapolation`.

## Getting started

For most applications, `NonLinear_OneLayer_GML` with the default settings is the best
starting point. Below is a minimal example using the built-in Harmonic Oscillator
problem from [GeometricProblems.jl](https://github.com/JuliaGNI/GeometricProblems.jl).

```julia
using NonlinearIntegrators
using QuadratureRules
using GeometricProblems.HarmonicOscillator

# 1. Define the problem
prob = HarmonicOscillator.lodeproblem(
    [1.0], [0.0];
    timespan = (0.0, 10.0), timestep = 0.1)

# 2. Build the basis and quadrature rule
basis  = OneLayerNetwork_GML{Float64}(tanh, 8)
quad   = GaussLegendreQuadrature(Float64, 8)

# 3. Construct the integrator
method = NonLinear_OneLayer_GML(basis, quad;
    bias_interval = [-π, π],
    dict_amount   = 400)

# 4. Integrate
sol, _ = integrate(prob, method;
    regularization_factor = 1e-5,
    max_iterations        = 10000)
```

## API reference

```@autodocs
Modules = [NonlinearIntegrators]
```

```@meta
CurrentModule = NonlinearIntegrators
```
