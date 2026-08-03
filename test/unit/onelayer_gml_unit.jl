# Per-precision unit tests for NonLinear_OneLayer_GML. The accuracy guard uses the
# default OGA1d × IntegratorExtrapolation combination over a full second. The cross-
# product loop covers all combinations of OGA init methods × extrapolation variants
# on a short two-step run (finite-state check only). The Float16 dictionary regression
# test is kept separate. The tight-accuracy Float64 guard lives in test/integration.

build_ol_method(::Type{T}; R = 8, S = 4, k = 3, dict_amount = 400,
        init_method  = OGA1d(),
        extrap = IntegratorExtrapolation()) where {T} =
    NonLinear_OneLayer_GML(build_onelayer_basis(T; S = S, k = k), gauss(T, R);
        bias_interval = [-T(pi), T(pi)], dict_amount = dict_amount,
        initial_guess_method      = init_method,
        initial_trajectory_method = extrap)

# Accuracy guard: default combination, long run, precision-appropriate error bound.
@testset "NonLinear_OneLayer_GML accuracy ($T)" for T in TEST_TYPES
    @debug "NonLinear_OneLayer_GML unit: element type = $T"
    params = HarmonicOscillator.default_parameters(T)
    prob = HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)];
        timespan = (T(0.0), T(1.0)), timestep = T(0.1), parameters = params)

    sol, _ = integrate(prob, build_ol_method(T); regularization_factor = T(1e-5), max_iterations = 10000)

    assert_no_upcast(sol.q, T)
    qend = collect(sol.q[:, 1])[end]
    ref = HarmonicOscillator.exact_solution_q(T(1.0), T(0.5), T(0.0), T(0.0), params)
    err = abs(Float64(qend) - Float64(ref))
    @debug "NonLinear_OneLayer_GML ($T)" q_end=Float64(qend) q_ref=Float64(ref) abs_err=err
    @test err < (T == Float64 ? 1e-8 : 1e-3)
end

# Cross-product: OGA init methods × extrapolation variants, short run, finite check.
const OL_INIT_METHODS = [
    (OGA1d(),        "OGA1d"),
    (OGA1d_Legacy(), "OGA1d_Legacy"),
]

const OL_EXTRAPOLATIONS = [
    (NoExtrapolation(),          "NoExtrapolation"),
    (IntegratorExtrapolation(),  "IntegratorExtrapolation"),
    (HermiteExtrapolation(),     "HermiteExtrapolation"),
]

for T in TEST_TYPES,
    (init_method, init_name) in OL_INIT_METHODS,
    (extrap, extrap_name) in OL_EXTRAPOLATIONS

    @testset "NonLinear_OneLayer_GML $init_name × $extrap_name ($T)" begin
        prob = ho_problem(T; timespan = (T(0.0), T(0.2)), timestep = T(0.1))
        sol, _ = integrate(prob,
            build_ol_method(T; dict_amount = 400, init_method = init_method, extrap = extrap);
            regularization_factor = T(1e-5), max_iterations = 10000)
        assert_no_upcast(sol.q, T)
        @test all(isfinite, collect(sol.q[:, 1])[end])
    end
end

# Regression test for OGA dictionary construction at half precision. A `dict_amount`
# above the finite range of Float16 (max ≈ 65504) previously made the bias-interval
# step evaluate to zero (`Float16(70000) == Inf`), throwing `ArgumentError: range
# step cannot be zero` before the solve was reached. The dictionary range is now
# built in Float64, so the run proceeds to the (still ill-conditioned) Float16 solve.
@testset "Float16 OGA dictionary construction is robust (dict_amount = 70000)" begin
    prob = HarmonicOscillator.lodeproblem([Float16(0.5)], [Float16(0.0)];
        timespan = (Float16(0.0), Float16(1.0)), timestep = Float16(0.1))
    method = build_ol_method(Float16; dict_amount = 70000)

    err = nothing
    try
        integrate(prob, method; regularization_factor = Float16(1e-3), max_iterations = 100)
    catch e
        err = e
    end
    @test !(err isa ArgumentError)                 # the range-step regression is fixed
    @test err === nothing || err isa SingularException
end
