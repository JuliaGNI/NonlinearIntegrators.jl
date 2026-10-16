# Per-precision unit tests for all five network integrators, driven from the
# `NETWORK_INTEGRATORS` table in testsetup.jl.
#
# This replaces shallownet_unit.jl, shallownet_reversible_unit.jl, shallownet_autodiff_unit.jl,
# shallownet_autodiff_reversible_unit.jl and densenet_unit.jl, which were structurally
# identical: each declared its own copy of the same three-element extrapolation list, then
# repeated the same seven-line accuracy block and the same cross-product loop, varying only the
# constructor and (for the accuracy block) one tolerance. Two of those files differed in 27 of
# 44 lines, every one of them a type-name substitution plus a single number.
#
# The `Float16` OGA dictionary regression below is *not* part of the cross-product and keeps
# its own testset.
#
# NOTE: DenseNet is exercised end to end here, but only on two-step runs with
# `training_epochs = 3`. Its Training/LSGD initial-guess methods are not stable enough for an
# accuracy guard, so its rows carry `tol = nothing` and assert dispatch, element type and
# finiteness only — a converged DenseNet solve is not something CI can rely on.

using Test
using NonlinearIntegrators
include(joinpath(@__DIR__, "..", "helpers", "testsetup.jl"))

# ---- the linear solver every network integrator gets ------------------------
#
# `initsolver` hands these methods `SimpleSolvers.PivotedQR()`, because their Newton
# Jacobian is *exactly* rank deficient — see the docstring in `network_integrator_core.jl` and
# `scripts/newton_jacobian_rank.jl`, which measures rank 5 of 13 unknowns at `S = 4` with a gap
# of fourteen orders in the spectrum. An LU raises `SingularException` on such a matrix, and
# which pivot it reaches first is decided by the BLAS build rather than by the problem, which is
# issue #98.
#
# These assertions are here because the failure they guard against is *invisible on macOS*: the
# cross product below raises nothing locally on any BLAS tested here, so "no exception was
# thrown" is not evidence that the plumbing works. What can be checked everywhere is that the
# solver the integrator actually holds is the rank-revealing one.
@testset "the Newton solve uses a rank-revealing linear solver" begin
    function lsm(int)
        SimpleSolvers.method(
            SimpleSolvers.linearsolver(GeometricIntegratorsBase.solver(int)))
    end

    for row in NETWORK_INTEGRATORS, T in TEST_TYPES

        int = GeometricIntegrator(ho_problem(T), row.make(T))
        @test lsm(int) isa SimpleSolvers.PivotedQR
    end

    # `Float16` gets the same default: `SimpleSolvers.PivotedQR` is a pure-Julia QR, not a
    # LAPACK call, so it accepts a half-precision matrix.
    int16 = GeometricIntegrator(
        HarmonicOscillator.lodeproblem([Float16(0.5)], [Float16(0.0)];
            timespan = (Float16(0.0), Float16(0.2)), timestep = Float16(0.1)),
        NETWORK_INTEGRATORS[1].make(Float16))
    @test lsm(int16) isa SimpleSolvers.PivotedQR

    # It is a default, not a decision taken away from the caller: a `linear_solver_method` passed
    # to `GeometricIntegrator` wins, so one keyword restores an LU.
    T = Float64
    m = NETWORK_INTEGRATORS[1].make(T)
    @test lsm(GeometricIntegrator(ho_problem(T), m;
        linear_solver_method = SimpleSolvers.LapackLU())) isa SimpleSolvers.LapackLU
    @test lsm(GeometricIntegrator(ho_problem(T), m;
        linear_solver_method = SimpleSolvers.SVDSolver())) isa SimpleSolvers.SVDSolver

    # Only the solvers that solve a linear system get the default. `Picard` takes no
    # `linear_solver_method`, so handing it one raises a `MethodError` at construction.
    @test GeometricIntegrator(ho_problem(T), m; solver = SimpleSolvers.Picard()) isa
          GeometricIntegrator
    @test lsm(GeometricIntegrator(ho_problem(T), m; solver = SimpleSolvers.QuasiNewton())) isa
          SimpleSolvers.PivotedQR
    @test lsm(GeometricIntegrator(ho_problem(T), m; solver = SimpleSolvers.DogLeg())) isa
          SimpleSolvers.PivotedQR

    # and the framework's own solver options still reach the solver beside it
    int = GeometricIntegrator(ho_problem(T), m)
    opts = GeometricIntegratorsBase.default_options(
        GeometricIntegratorsBase.method(int), ho_problem(T))
    cfg = SimpleSolvers.config(GeometricIntegratorsBase.solver(int))
    for k in (:min_iterations, :f_abstol, :f_stall_window)
        @test getfield(cfg, k) == opts[k]
    end

    # `PivotedQR` is ambiguous in this package: the exported one is the OGA fit, a different
    # type at a different layer. Pinned so that a future `using SimpleSolvers` here cannot change
    # what the unqualified name means. Two exporting modules do not silently rebind it — Julia
    # makes the name ambiguous, so every unqualified use raises `UndefVarError` — and that is
    # exactly what these two assertions turn into a named failure.
    @test PivotedQR() isa NonlinearIntegrators.OGAFit
    @test PivotedQR !== SimpleSolvers.PivotedQR
end

# ---- accuracy guards: default seed, ten steps, analytic reference ------------
for row in NETWORK_INTEGRATORS, T in TEST_TYPES

    row.tol === nothing && continue
    accuracy_guard(row.name, row.make, T; tol = getfield(row.tol, Symbol(T)),
        initial_guess_method = first(first(row.seeds)))
end

# ---- cross product: seeds × extrapolations, two steps, finiteness only -------
#
# `OGA1dNormalEquations` is included for `ShallowNet`. It used to raise `SingularException`
# under Hermite at both element types, which looked like the κ(Φ)² conditioning of its Gram
# solve — but the real cause was the Hermite path leaving `network_labels` at zero, which made
# the Gram matrix rank-deficient for *any* fit. With that fixed it behaves like the rest.
#
# Every cell asserts, and any exception fails the run. The Newton Jacobian of these methods is
# exactly rank deficient (issue #98), but the `SimpleSolvers.PivotedQR` the first testset pins
# solves it in the minimum-norm sense and has no throwing path, so a `SingularException` from
# the Newton solve cannot occur. One that does comes from somewhere else, such as an OGA Gram
# solve, and is a real failure.
for row in NETWORK_INTEGRATORS,
    T in TEST_TYPES,
    (seed, seed_name) in row.seeds,
    (extrap, extrap_name) in EXTRAPOLATIONS
    @testset "$(row.name) $seed_name × $extrap_name ($T)" begin
        dispatch_case(row.name, row.make, T, extrap; initial_guess_method = seed)
    end
end

# ---- Float16 OGA dictionary regression --------------------------------------
#
# A `dict_amount` above the finite range of Float16 (max ≈ 65504) makes the bias-interval step
# evaluate to zero (`Float16(70000) == Inf`), which a range built in `T` rejects with
# `ArgumentError: range step cannot be zero` before the solve is reached; the dictionary range
# is built in Float64 to keep a 70000-atom dictionary constructible at Float16.
#
# The assertion is split deliberately. What this file can state as a *fact* is that the seed
# runs and returns a finite Float16 fit — checked directly, without an integrator, so it holds
# independently of rounding. Whether the subsequent Newton solve converges at half precision is
# not a contract: the Jacobian is ill-conditioned there, and which side of the divergence a
# machine lands on is decided by rounding (measured: the same configuration converges at some
# initial conditions and raises `NonlinearSolverException` at others a few percent away). So
# the end-to-end run guards only against a *new class* of failure.
@testset "Float16 OGA dictionary construction is robust (dict_amount = 70000)" begin
    # ---- the seed, directly: deterministic and rounding-independent -----------
    nodes = Float16.((0:10) ./ 10)                      # the method's `network_inputs`
    weights = NI.simpson_quadrature(10, Float16)
    y = Float16.(cos.(3 .* Float64.(nodes)))

    r = oga_fit(OGA1d(), relu_k(3), nodes, weights, y, 4;
        bias_interval = [-Float16(pi), Float16(pi)], dict_amount = 70000)

    @test eltype(r.W) === Float16 && eltype(r.b) === Float16 && eltype(r.c) === Float16
    @test all(isfinite, r.W) && all(isfinite, r.b) && all(isfinite, r.c)
    @test isfinite(r.residual)

    # ---- end to end: only that the failure mode is one of the documented ones --
    prob = HarmonicOscillator.lodeproblem([Float16(0.5)], [Float16(0.0)];
        timespan = (Float16(0.0), Float16(0.2)), timestep = Float16(0.1))
    method = ShallowNet(cached_shallownet_basis(Float16; S = 4), gauss(Float16, 8);
        show_status = false, bias_interval = [-Float16(pi), Float16(pi)], dict_amount = 70000)

    err = try
        integrate(prob, method; regularization_factor = Float16(1e-3), max_iterations = 100)
        nothing
    catch e
        e
    end
    @test !(err isa ArgumentError)                 # the range-step regression is fixed
    # Written as `typeof(err) <: Union{...}` rather than `err isa ...` so that a failure names
    # the offending exception type in the CI log — the earlier form printed only the
    # expression, which is why an added error class took a local repro to identify.
    @test typeof(err) <: Union{Nothing, SOLVER_GAVE_UP}
end

# ---- D = 2 layout guard ------------------------------------------------------
#
# `components!`, `residual!`, `initial_guess!` and `update!` all index one flat vector of
# `D × (parameters per dimension)` unknowns, and *every* layout mistake between them collapses
# to the identity at `D = 1` — which is all the rest of this file covers. The guard exists here
# because that class of bug was first found in `CGVINodal`, the linear reference integrator this
# package used to carry; the guard went upstream with it (JuliaGNI/GeometricIntegrators.jl#219),
# where `test/integrators/galerkin_integrators_tests.jl` runs it for both CGVI variants.
#
# `CoupledHarmonicOscillator` with coupling `k = 0` is two *independent* oscillators with
# different masses, spring constants and initial conditions, so each degree of freedom has its
# own closed-form solution and its own frequency: any layout bug that duplicates, swaps or drops
# a component shows up as a wrong number rather than a slightly worse one.
#
# Float64 only — the layout is precision-independent, and each `lodeproblem` call runs
# EulerLagrange's symbolic code generation.
@testset "D = 2 layout (Float64)" begin
    params = (m₁ = 2.0, m₂ = 1.0, k₁ = 1.5, k₂ = 0.3, k = 0.0)
    q₀ = [0.5, -0.3]
    p₀ = [0.0, 0.4]
    tend = 0.3

    m = [params.m₁, params.m₂]
    ω = sqrt.([params.k₁, params.k₂] ./ m)
    qref = q₀ .* cos.(ω .* tend) .+ p₀ ./ (m .* ω) .* sin.(ω .* tend)

    # Tolerances follow the same split as the accuracy guards above: the symbolic-derivative
    # pair is exact to round-off, the autodiff pair is limited by the Newton floor of the
    # hand-written ansatz.
    for row in NETWORK_INTEGRATORS
        row.tol === nothing && continue      # DenseNet: no accuracy claim anywhere
        @testset "$(row.name)" begin
            prob = CoupledHarmonicOscillator.lodeproblem(q₀, p₀;
                timespan = (0.0, tend), timestep = 0.1, parameters = params)
            sol, _ = integrate(prob, row.make(Float64);
                regularization_factor = 1e-5, max_iterations = MAX_NEWTON_ITERATIONS)
            qend = [collect(sol.q[:, d])[end] for d in 1:2]
            @debug "$(row.name) D=2" q_end=qend q_ref=qref
            @test maximum(abs.(qend .- qref)) < row.tol.Float64
        end
    end
end
