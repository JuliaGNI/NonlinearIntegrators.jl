# Shared setup and helpers for the NonlinearIntegrators test suite.
#
# The suite is parametrized over `TEST_TYPES` so that every integrator and basis
# is exercised at more than one floating point precision. The central invariant
# is *no silent upcast*: a run started at `Float32` must stay at `Float32` all the
# way to the final state (see `assert_no_upcast`). Float16 is intentionally
# excluded here — the OGA/Newton path stays ill-conditioned at half precision;
# that case is covered by a dedicated regression test in test/integration.

using Test
using NonlinearIntegrators
# `import`, not `using`: only `NeuralNetworkParameters`, `NeuralNetwork` and `params` are
# needed (to call the compiled derivative kernels directly in dispatch_variants_unit.jl), and
# importing the module keeps the rest of its exports out of the way of the Geometric* ones.
import AbstractNeuralNetworks
# `import`, not `using`: `ForwardDiff.derivative` and `ForwardDiff.gradient` are the
# independent reference the compiled kernels are checked against in
# dispatch_variants_unit.jl, and both names are too generic to bring into scope unqualified.
import ForwardDiff
# The reference check draws one random parameter set per basis; seeded so a failure is
# reproducible rather than a different point every run.
import Random
using QuadratureRules
using CompactBasisFunctions
using GeometricIntegratorsBase
using GeometricProblems.HarmonicOscillator
# `import`, not `using`: this only binds the module name, so the two-degree-of-freedom problem in
# `cgvi_unit.jl` can be reached as `CoupledHarmonicOscillator.lodeproblem` without its exported
# `lodeproblem`/`podeproblem` colliding with the `HarmonicOscillator` ones that
# `integration/shallownet_accuracy.jl` calls unqualified.
import GeometricProblems.CoupledHarmonicOscillator
using GeometricSolutions: relative_maximum_error
using LinearAlgebra: SingularException
using SimpleSolvers: NonlinearSolverException
using Symbolics: @variables

const TEST_TYPES = (Float64, Float32)

# Shorthand for reaching internals that are deliberately not exported (the OGA kernels,
# the quadrature helper). Defined here so every test file can rely on it.
const NI = NonlinearIntegrators

# Type-generic ReLU^k activation: `max(zero(x), x)^k`, never `max(0.0, x)`, so the
# network is evaluated at the working precision rather than silently upcasting.
relu_k(k::Int = 3) = x -> max(zero(x), x)^k

# ---- basis / quadrature builders -------------------------------------------

build_shallownet_basis(::Type{T}; S = 4, k = 3) where {T} =
    ShallowNetBasis{T}(relu_k(k), S)

build_densenet_basis(::Type{T}; S₁ = 3, S = 3) where {T} =
    DenseNetBasis{T}(tanh, S₁, S)

function build_vise_basis(::Type{T}) where {T}
    @variables tvar
    @variables Wv[1:3]
    q_expr = Wv[1] * cos(Wv[2] * tvar + Wv[3])
    VISEBasis{T}([q_expr], [Wv], tvar, 1)
end

gauss(::Type{T}, R = 8) where {T} = QuadratureRules.GaussLegendreQuadrature(T, R)
lobatto(::Type{T}, R = 4) where {T} = QuadratureRules.LobattoLegendreQuadrature(T, R)

# ---- problem builder --------------------------------------------------------

# Minimal Harmonic Oscillator LODE problem at precision `T`. A short time span and
# a single/couple of steps keep the network solves fast; smoke/unit tests only
# check type propagation, not long-time accuracy.
function ho_problem(::Type{T}; timespan = (T(0.0), T(0.2)), timestep = T(0.1)) where {T}
    params = HarmonicOscillator.default_parameters(T)
    HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)];
        timespan = timespan, timestep = timestep, parameters = params)
end

# ---- assertions -------------------------------------------------------------

# The single "no silent upcast" gate. `q` is a solution state variable (indexable
# per time step); its final entry must retain the working element type `T`.
assert_no_upcast(q, ::Type{T}) where {T} = @test eltype(q[end]) == T

# The two ways a nonlinear solve is allowed to give up: `SingularException` (a factorisation
# hit a zero pivot) and `NonlinearSolverException` (the Newton direction came back
# non-finite). Half precision usually reaches the latter, since `oga_solve` guarantees the
# seed itself is finite at every precision.
#
# Used only by the `Float16` regression test, which asserts a failure there is one of these
# rather than a new class. `TEST_TYPES` stops at `Float32`, where these problems are well
# conditioned on every platform, so every other integration calls `integrate` directly and a
# give-up is a real failure.
const SOLVER_GAVE_UP = Union{SingularException,NonlinearSolverException}

# `initial_trajectory_method` selects which `initial_trajectory!` method runs; `iguess` is the
# extrapolation `GeometricIntegratorsBase.solutionstep!` actually applies, and its default
# (`NoInitialGuess`) makes that call a no-op. So a Hermite row that passes only
# `initial_trajectory_method = HermiteExtrapolation()` takes the Hermite code path but
# extrapolates nothing. Passing both is what `benchmark/shallownet_benchmark_common.jl` does, and it is
# what makes these rows measure something.
hermite_kw(extrap) = extrap isa HermiteExtrapolation ? (; initialguess = HermiteExtrapolation()) : (;)

# Newton iteration cap for the unit tests. Measured on the ShallowNetAutodiff accuracy guard: the solve
# converges in 88 iterations, but with a 10000 cap some *earlier* step burns the whole budget,
# costing 30s per case against 0.7s at 100 — for identical accuracy (2.6e-6 vs 3.1e-6 against a
# 1e-4 threshold). The shallow-net guards converge in 2–6 iterations and are unaffected.
#
# The extrapolation cross-products do not converge within two steps at all: they exhaust
# whatever cap they are given. They assert dispatch and a finite result at the working type,
# which is what `integrate` returns after exhausting the budget — deliberately not a
# convergence claim (see `benchmark/shallownet_benchmark_common.jl`, which records the same situation
# as a distinct `maxiter` status rather than as `ok`).
const MAX_NEWTON_ITERATIONS = 100

