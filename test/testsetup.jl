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
# `import`, not `using`: only `NeuralNetwork` and `params` are needed (to call the compiled
# derivative kernels directly in dispatch_variants_unit.jl), and importing the module keeps the rest
# of its exports out of the way of the Geometric* ones. `NetworkParameters` comes from its own
# package, for the same reason.
import AbstractNeuralNetworks
import NeuralNetworkParameters
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
# `network_integrators_unit.jl` can be reached as `CoupledHarmonicOscillator.lodeproblem` without
# its exported `lodeproblem`/`podeproblem` colliding with the `HarmonicOscillator` ones that
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

function build_vise_basis(::Type{T}) where {T}
    @variables tvar
    @variables Wv[1:3]
    q_expr = Wv[1] * cos(Wv[2] * tvar + Wv[3])
    VISEBasis{T}([q_expr], [Wv], tvar, 1)
end

gauss(::Type{T}, R = 8) where {T} = QuadratureRules.GaussLegendreQuadrature(T, R)

# Memoised basis builder.
#
# Building a `ShallowNetBasis` runs SymbolicNeuralNetworks' code generation, which is the
# single largest cost in this suite: the tests used to perform ~72 of these builds to obtain
# about two distinct objects per element type, and `.githooks/pre-push` pays that on every
# push. A basis is immutable and stateless once built, so one instance can back every testset
# that asks for the same `(kind, T, sizes, options)`.
#
# Anything that varies the *construction* (`symbolic = false`, `cse`/`inplace`) is part of the
# key, so `dispatch_variants_unit.jl` and `bases_smoke.jl` still get their own objects.
const _BASIS_CACHE = Dict{Any, Any}()

function cached_shallownet_basis(::Type{T}; S = 4, k = 3, kwargs...) where {T}
    key = (:shallow, T, S, k, NamedTuple(kwargs))
    get!(_BASIS_CACHE, key) do
        ShallowNetBasis{T}(relu_k(k), S; kwargs...)
    end
end

function cached_densenet_basis(::Type{T}; S₁ = 3, S = 3, kwargs...) where {T}
    key = (:dense, T, S₁, S, NamedTuple(kwargs))
    get!(_BASIS_CACHE, key) do
        DenseNetBasis{T}(tanh, S₁, S; kwargs...)
    end
end

# The plain names route through the memoised builders above. Only `bases_smoke.jl`, which tests
# *construction*, calls the concrete constructors directly and gets fresh objects.
function build_shallownet_basis(::Type{T}; kwargs...) where {T}
    cached_shallownet_basis(T; kwargs...)
end
build_densenet_basis(::Type{T}; kwargs...) where {T} = cached_densenet_basis(T; kwargs...)

# A smooth target on the unit interval and the Simpson weights the integrators use — the
# fixture the OGA kernels are exercised on. Lives here rather than in oga_kernels.jl because
# the allocation gates in test/quality use it too.
function oga_testcase(::Type{T}; n = 10) where {T}
    nodes = T.((0:n) ./ n)
    weights = NI.simpson_quadrature(n, T)
    y = T.(cos.(3 .* Float64.(nodes)))
    return (nodes, weights, y)
end

# ---- problem builders -------------------------------------------------------

# Minimal Harmonic Oscillator LODE problem at precision `T`. A short time span and
# a single/couple of steps keep the network solves fast; smoke/unit tests only
# check type propagation, not long-time accuracy.
function ho_problem(::Type{T}; timespan = (T(0.0), T(0.2)), timestep = T(0.1)) where {T}
    params = HarmonicOscillator.default_parameters(T)
    HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)];
        timespan = timespan, timestep = timestep, parameters = params)
end

# The ten-step problem the accuracy guards run on, plus its analytic endpoint. Five test
# files used to spell this out by hand — build `default_parameters`, build the problem, call
# `exact_solution_q` with the same six literals — and the copies could drift apart silently.
function ho_accuracy_problem(::Type{T}; tend = T(1.0), timestep = T(0.1)) where {T}
    params = HarmonicOscillator.default_parameters(T)
    prob = HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)];
        timespan = (T(0.0), tend), timestep = timestep, parameters = params)
    qref = HarmonicOscillator.exact_solution_q(tend, T(0.5), T(0.0), T(0.0), params)
    return prob, qref
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
const SOLVER_GAVE_UP = Union{SingularException, NonlinearSolverException}

# `initial_trajectory_method` selects which `initial_trajectory!` method runs; `iguess` is the
# extrapolation `GeometricIntegratorsBase.solutionstep!` actually applies, and its default
# (`NoInitialGuess`) makes that call a no-op. So a Hermite row that passes only
# `initial_trajectory_method = HermiteExtrapolation()` takes the Hermite code path but
# extrapolates nothing. Passing both is what `benchmark/shallownet_benchmark_common.jl` does, and it is
# what makes these rows measure something.
function hermite_kw(extrap)
    extrap isa HermiteExtrapolation ? (; initialguess = HermiteExtrapolation()) : (;)
end

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

# The three extrapolation variants every network integrator is driven over. This literal used
# to be repeated, identically, in five separate unit files.
const EXTRAPOLATIONS = [
    (NoExtrapolation(), "NoExtrapolation"),
    (IntegratorExtrapolation(), "IntegratorExtrapolation"),
    (HermiteExtrapolation(), "HermiteExtrapolation")
]

# ---- the integrator table ---------------------------------------------------
#
# The five network integrators and what genuinely differs between them. Lives here, not in
# `unit/network_integrators_unit.jl`, because `quality/inference_and_allocations.jl` drives the
# same rows: keeping it in a unit file made `quality/` silently depend on `unit/` having been
# included first.
#
#   name   — used in testset names and to look up the allocation budget
#   make   — the constructor, taking `T` and forwarding keywords
#   seeds  — the `initial_guess_method`s to drive. `ShallowNet` supports all four OGA
#            variants; the reversible and autodiff ones are only meaningful with a single
#            seed; `DenseNet` takes the two gradient-descent seeds instead.
#   tol    — endpoint tolerance for the accuracy guard, `nothing` to skip it. The symbolic
#            pair reaches 1e-8; the autodiff pair is limited to 1e-4 by the Newton floor of
#            the hand-written ansatz. `DenseNet` has no guard at all: its Training/LSGD seeds
#            are not stable enough for one, so its rows assert dispatch and finiteness only
#            (this is deliberate — see the note in runtests.jl).

function shallow_kw(::Type{T}) where {T}
    (; show_status = false, bias_interval = [-T(pi), T(pi)],
        dict_amount = 400)
end

const NETWORK_INTEGRATORS = [
    (name = "ShallowNet",
        make = (T; kw...) -> ShallowNet(cached_shallownet_basis(T; S = 4), gauss(T, 8);
            shallow_kw(T)..., kw...),
        seeds = [(OGA1d(), "OGA1d"),
            (OGA1dNormalized(), "OGA1dNormalized"),
            (OGA1dStable(), "OGA1dStable"),
            (OGA1dNormalEquations(), "OGA1dNormalEquations")],
        tol = (Float64 = 1e-8, Float32 = 1e-3)),
    (name = "ShallowNetReversible",
        make = (T; kw...) -> ShallowNetReversible(
            cached_shallownet_basis(T; S = 4), gauss(T, 8);
            shallow_kw(T)..., kw...),
        seeds = [(OGA1d(), "OGA1d")],
        tol = (Float64 = 1e-8, Float32 = 1e-3)),

    # `symbolic = false`: the autodiff integrators differentiate their own ansatz with
    # ForwardDiff and never read the compiled derivative slots, so building them is wasted
    # work. This also keeps the cached basis distinct from the symbolic one above.
    (name = "ShallowNetAutodiff",
        make = (T; kw...) -> ShallowNetAutodiff(
            cached_shallownet_basis(T; S = 4, symbolic = false), gauss(T, 8);
            shallow_kw(T)..., kw...),
        seeds = [(OGA1dNormalized(), "OGA1dNormalized")],
        tol = (Float64 = 1e-4, Float32 = 1e-3)),
    (name = "ShallowNetAutodiffReversible",
        make = (T; kw...) -> ShallowNetAutodiffReversible(
            cached_shallownet_basis(T; S = 4, symbolic = false), gauss(T, 8);
            shallow_kw(T)..., kw...),
        seeds = [(OGA1d(), "OGA1d")],
        tol = (Float64 = 1e-4, Float32 = 1e-3)),
    (name = "DenseNet",
        make = (T; kw...) -> DenseNet(cached_densenet_basis(T; S₁ = 3, S = 3), gauss(T, 8);
            show_status = false, training_epochs = 3, kw...),
        seeds = [(TrainingMethod(), "TrainingMethod"), (LSGD(), "LSGD")],
        tol = nothing)
]

# The endpoint of a run must be finite *and* still at the working element type. Spelled out
# in seven places before.
function assert_finite_endpoint(sol, ::Type{T}) where {T}
    assert_no_upcast(sol.q, T)
    @test all(isfinite, collect(sol.q[:, 1])[end])
end

# ---- the two shared run shapes ----------------------------------------------
#
# `make(T; kwargs...)` builds the method under test. Both helpers take it rather than a
# constructed method so that each case gets a fresh method (and hence a fresh cache), which is
# what the per-case testsets used to do by hand.

"""
    accuracy_guard(name, make, T; tol, kwargs...)

Ten-step Harmonic Oscillator run against the analytic solution: asserts no silent upcast and
an absolute endpoint error below `tol`. This is the block that was copied, with only the
constructor and the tolerance varying, into five unit files.
"""
function accuracy_guard(name, make, ::Type{T}; tol, kwargs...) where {T}
    @testset "$name accuracy ($T)" begin
        prob, qref = ho_accuracy_problem(T)
        sol, _ = integrate(prob, make(T; kwargs...);
            regularization_factor = T(1e-5), max_iterations = MAX_NEWTON_ITERATIONS)
        assert_no_upcast(sol.q, T)
        qend = collect(sol.q[:, 1])[end]
        err = abs(Float64(qend) - Float64(qref))
        @debug "$name ($T)" q_end=Float64(qend) q_ref=Float64(qref) abs_err=err
        @test err < tol
    end
end

"""
    dispatch_case(name, make, T, extrap; kwargs...)

Two-step run asserting dispatch, element type and finiteness — deliberately **not**
convergence. These combinations exhaust whatever Newton budget they are given (see the note on
`MAX_NEWTON_ITERATIONS`), so the claim is that every (seed, extrapolation) pair reaches a
finite state at the working precision, not that it converges.

See `hermite_kw` for why the Hermite rows also pass `initialguess`.
"""
function dispatch_case(name, make, ::Type{T}, extrap; kwargs...) where {T}
    prob = ho_problem(T)
    sol, _ = integrate(prob, make(T; initial_trajectory_method = extrap, kwargs...);
        regularization_factor = T(1e-5), max_iterations = MAX_NEWTON_ITERATIONS,
        hermite_kw(extrap)...)
    assert_finite_endpoint(sol, T)
end
