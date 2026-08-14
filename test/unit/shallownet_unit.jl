# Per-precision unit tests for ShallowNet. The accuracy guard uses the
# default OGA1d × IntegratorExtrapolation combination over a full second. The cross-
# product loop covers all combinations of OGA init methods × extrapolation variants
# on a short two-step run (finite-state check only). The Float16 dictionary regression
# test is kept separate. The tight-accuracy Float64 guard lives in test/integration.

build_ol_method(::Type{T}; R = 8, S = 4, k = 3, dict_amount = 400,
        init_method  = OGA1d(),
        extrap = IntegratorExtrapolation()) where {T} =
    ShallowNet(build_shallownet_basis(T; S = S, k = k), gauss(T, R);
        show_status = false,
        bias_interval = [-T(pi), T(pi)], dict_amount = dict_amount,
        initial_guess_method      = init_method,
        initial_trajectory_method = extrap)

# Accuracy guard: default combination, long run, precision-appropriate error bound.
@testset "ShallowNet accuracy ($T)" for T in TEST_TYPES
    @debug "ShallowNet unit: element type = $T"
    params = HarmonicOscillator.default_parameters(T)
    prob = HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)];
        timespan = (T(0.0), T(1.0)), timestep = T(0.1), parameters = params)

    sol, _ = integrate(prob, build_ol_method(T); regularization_factor = T(1e-5), max_iterations = MAX_NEWTON_ITERATIONS)

    assert_no_upcast(sol.q, T)
    qend = collect(sol.q[:, 1])[end]
    ref = HarmonicOscillator.exact_solution_q(T(1.0), T(0.5), T(0.0), T(0.0), params)
    err = abs(Float64(qend) - Float64(ref))
    @debug "ShallowNet ($T)" q_end=Float64(qend) q_ref=Float64(ref) abs_err=err
    @test err < (T == Float64 ? 1e-8 : 1e-3)
end

# Cross-product: OGA init methods × extrapolation variants, short run, finite check.
# See `hermite_kw` in testsetup.jl for why the Hermite rows also pass `initialguess`.
#
# `OGA1dNormalEquations` is included. It used to raise `SingularException` under Hermite at
# both element types, which looked like the κ(Φ)² conditioning of its Gram solve — but the
# real cause was the Hermite path leaving `network_labels` at zero, which made the Gram
# matrix rank-deficient for *any* fit. With that fixed it behaves like the rest.
const OL_INIT_METHODS = [
    (OGA1d(),                "OGA1d"),
    (OGA1dNormalized(),      "OGA1dNormalized"),
    (OGA1dStable(),          "OGA1dStable"),
    (OGA1dNormalEquations(), "OGA1dNormalEquations"),
]

const OL_EXTRAPOLATIONS = [
    (NoExtrapolation(),          "NoExtrapolation"),
    (IntegratorExtrapolation(),  "IntegratorExtrapolation"),
    (HermiteExtrapolation(),     "HermiteExtrapolation"),

]

for T in TEST_TYPES,
    (init_method, init_name) in OL_INIT_METHODS,
    (extrap, extrap_name) in OL_EXTRAPOLATIONS

    @testset "ShallowNet $init_name × $extrap_name ($T)" begin
        prob = ho_problem(T; timespan = (T(0.0), T(0.2)), timestep = T(0.1))
        # `dict_amount` stays at the builder's 400: this loop tests that every
        # (seed, extrapolation) pair dispatches and produces a finite state, not how
        # accurate the seed is. The accuracy guards above and in test/integration use
        # the larger dictionaries.
        sol, _ = integrate(prob,
            build_ol_method(T; init_method = init_method, extrap = extrap);
            regularization_factor = T(1e-5), max_iterations = MAX_NEWTON_ITERATIONS, hermite_kw(extrap)...)
        assert_no_upcast(sol.q, T)
        @test all(isfinite, collect(sol.q[:, 1])[end])
    end
end

# Regression test for OGA dictionary construction at half precision. A `dict_amount` above
# the finite range of Float16 (max ≈ 65504) makes the bias-interval step evaluate to zero
# (`Float16(70000) == Inf`), which a range built in `T` rejects with `ArgumentError: range
# step cannot be zero` before the solve is reached; the dictionary range is built in Float64
# to keep a 70000-atom dictionary constructible at Float16.
#
# The assertion is split deliberately. What this file can state as a *fact* is that the seed
# runs and returns a finite Float16 fit — checked directly, without an integrator, so it
# holds independently of rounding. Whether the subsequent Newton solve converges at half
# precision is not a contract: the Jacobian is ill-conditioned there, and which side of the
# divergence a machine lands on is decided by rounding (measured: the same configuration
# converges at some initial conditions and raises `NonlinearSolverException` at others a few
# percent away). So the end-to-end run guards only against a *new class* of failure.
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
    method = build_ol_method(Float16; dict_amount = 70000)

    err = nothing
    try
        integrate(prob, method; regularization_factor = Float16(1e-3), max_iterations = 100)
    catch e
        err = e
    end
    @test !(err isa ArgumentError)                 # the range-step regression is fixed
    # Written as `typeof(err) <: Union{...}` rather than `err isa ...` so that a failure
    # names the offending exception type in the CI log — the earlier form printed only the
    # expression, which is why an added error class took a local repro to identify.
    @test typeof(err) <: Union{Nothing,SOLVER_GAVE_UP}
end
