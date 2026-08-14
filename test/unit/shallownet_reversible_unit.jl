# Per-precision unit tests for ShallowNetReversible. The accuracy guard uses the
# default OGA1d × IntegratorExtrapolation combination. The cross-product loop covers
# all three extrapolation variants (OGA1d is the only supported init method).

# Accuracy guard: default combination, long run, precision-appropriate error bound.
@testset "ShallowNetReversible accuracy ($T)" for T in TEST_TYPES
    @debug "ShallowNetReversible unit: element type = $T"
    params = HarmonicOscillator.default_parameters(T)
    prob = HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)];
        timespan = (T(0.0), T(1.0)), timestep = T(0.1), parameters = params)

    method = ShallowNetReversible(build_shallownet_basis(T; S = 4), gauss(T, 8);
        show_status = false, bias_interval = [-T(pi), T(pi)], dict_amount = 400)

    sol, _ = integrate(prob, method; regularization_factor = T(1e-5), max_iterations = MAX_NEWTON_ITERATIONS)

    assert_no_upcast(sol.q, T)

    qend = collect(sol.q[:, 1])[end]
    ref = HarmonicOscillator.exact_solution_q(T(1.0), T(0.5), T(0.0), T(0.0), params)
    err = abs(Float64(qend) - Float64(ref))
    @debug "ShallowNetReversible ($T)" q_end=Float64(qend) q_ref=Float64(ref) abs_err=err
    @test err < (T == Float64 ? 1e-8 : 1e-3)
end

# Cross-product: OGA1d × extrapolation variants, short run, finite check.
const TROL_EXTRAPOLATIONS = [
    (NoExtrapolation(),          "NoExtrapolation"),
    (IntegratorExtrapolation(),  "IntegratorExtrapolation"),
    (HermiteExtrapolation(),     "HermiteExtrapolation"),
]

for T in TEST_TYPES, (extrap, extrap_name) in TROL_EXTRAPOLATIONS
    @testset "ShallowNetReversible OGA1d × $extrap_name ($T)" begin
        prob = ho_problem(T; timespan = (T(0.0), T(0.2)), timestep = T(0.1))
        method = ShallowNetReversible(build_shallownet_basis(T; S = 4), gauss(T, 8);
            initial_trajectory_method = extrap,
            show_status = false, bias_interval = [-T(pi), T(pi)], dict_amount = 400)
        sol, _ = integrate(prob, method; regularization_factor = T(1e-5), max_iterations = MAX_NEWTON_ITERATIONS,
            hermite_kw(extrap)...)
        assert_no_upcast(sol.q, T)
        @test all(isfinite, collect(sol.q[:, 1])[end])
    end
end
