# Per-precision unit tests for NonLinear_OneLayer_GML: assert the run stays at the
# working element type (no silent upcast) and tracks the analytic harmonic
# oscillator to a precision-appropriate level. Solver options are passed through
# `integrate(...)` (no `default_options` override). A small `dict_amount` keeps the
# OGA seed fast; the tight-accuracy Float64 guard lives in test/integration.

build_ol_method(::Type{T}; R = 8, S = 4, k = 3, dict_amount = 400) where {T} =
    NonLinear_OneLayer_GML(build_onelayer_basis(T; S = S, k = k), gauss(T, R);
        bias_interval = [-T(pi), T(pi)], dict_amount = dict_amount)

@testset "NonLinear_OneLayer_GML ($T)" for T in TEST_TYPES
    params = HarmonicOscillator.default_parameters(T)
    prob = HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)];
        timespan = (T(0.0), T(1.0)), timestep = T(0.1), parameters = params)

    res = integrate(prob, build_ol_method(T); regularization_factor = T(1e-5), max_iterations = 10000)
    q = res.sol.q

    assert_no_upcast(q, T)

    qend = collect(q[:, 1])[end]
    ref = HarmonicOscillator.exact_solution_q(T(1.0), T(0.5), T(0.0), T(0.0), params)
    @test abs(Float64(qend) - Float64(ref)) < (T == Float64 ? 1e-8 : 1e-3)
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
