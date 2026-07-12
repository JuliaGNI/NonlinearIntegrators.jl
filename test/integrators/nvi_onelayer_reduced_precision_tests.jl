using GeometricIntegratorsBase
using NonlinearIntegrators
using QuadratureRules
using GeometricProblems.HarmonicOscillator
using LinearAlgebra: SingularException
using Test

# Build the one-layer network integrator at an arbitrary floating point type `T`.
# The activation is type-generic (`max(zero(x), x)^k`, not `max(0.0, x)`) and the
# quadrature is constructed at `T`, so `basis` and `quadrature` share the same
# element type as required by the `NonLinear_OneLayer_GML` constructor.
function build_method(::Type{T}; R = 8, S = 4, k = 3, dict_amount = 400) where {T}
    act = x -> max(zero(x), x)^k
    net = OneLayerNetwork_GML{T}(act, S)
    quad = QuadratureRules.GaussLegendreQuadrature(T, R)
    NonLinear_OneLayer_GML(net, quad; bias_interval = [-T(pi), T(pi)], dict_amount = dict_amount)
end

# The network solve is near-singular, so a small regularisation factor is required
# for the Newton iteration to converge. It is passed through `integrate` as a
# solver option (rather than by overriding `default_options`, which would clash
# with the definition in `nvi_onelayer_integrators_tests.jl`).
@testset "reduced precision ($T)" for T in (Float64, Float32)
    params = HarmonicOscillator.default_parameters(T)
    prob = HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)], T;
        timespan = (T(0.0), T(1.0)), timestep = T(0.1), parameters = params)

    sol = integrate(prob, build_method(T); regularization_factor = T(1e-5), max_iterations = 10000)
    q = sol.sol.q

    # the integration runs genuinely at precision T (no silent upcast to Float64)
    @test eltype(q[end]) == T

    # the solution tracks the analytic harmonic oscillator to a precision-appropriate level
    qend = collect(q[:, 1])[end]
    ref  = HarmonicOscillator.exact_solution_q(T(1.0), T(0.5), T(0.0), T(0.0), params)
    @test abs(Float64(qend) - Float64(ref)) < (T == Float64 ? 1e-8 : 1e-3)
end

# Regression test for the OGA dictionary construction at low precision. A
# `dict_amount` above the finite range of `Float16` (max ≈ 65504) previously made
# `(bias_interval[2] - bias_interval[1]) / dict_amount` evaluate to a zero step
# (`Float16(70000) == Inf`), throwing `ArgumentError: range step cannot be zero`
# before the solve was even reached. The dictionary range is now built in Float64,
# so the run proceeds to the (still ill-conditioned) Float16 solve instead. We only
# assert that the range-construction bug does not recur — the half-precision Newton
# solve itself is expected to remain singular.
@testset "Float16 OGA dictionary construction is robust (dict_amount = 70000)" begin
    prob = HarmonicOscillator.lodeproblem([Float16(0.5)], [Float16(0.0)], Float16;
        timespan = (Float16(0.0), Float16(1.0)), timestep = Float16(0.1))
    method = build_method(Float16; dict_amount = 70000)

    err = nothing
    try
        integrate(prob, method; regularization_factor = Float16(1e-3), max_iterations = 100)
    catch e
        err = e
    end
    @test !(err isa ArgumentError)                 # the range-step regression is fixed
    @test err === nothing || err isa SingularException
end
