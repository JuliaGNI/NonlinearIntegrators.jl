using GeometricIntegratorsBase
using NonlinearIntegrators
using QuadratureRules
using GeometricProblems.HarmonicOscillator
using Test

# The nonlinear network solve is near-singular without regularisation, so a small
# regularisation factor is required for the Newton iteration to converge.
GeometricIntegratorsBase.default_options(method::NonLinear_OneLayer_GML) = (
    max_iterations = 10000,
    regularization_factor = 1e-5,
    linesearch = GeometricIntegratorsBase.default_linesearch(method),
)

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

@testset "reduced precision ($T)" for T in (Float64, Float32)
    params = HarmonicOscillator.default_parameters(T)
    prob = HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)], T;
        timespan = (T(0.0), T(1.0)), timestep = T(0.1), parameters = params)

    sol = integrate(prob, build_method(T))
    q = sol.sol.q

    # the integration runs genuinely at precision T (no silent upcast to Float64)
    @test eltype(q[end]) == T

    # the solution tracks the analytic harmonic oscillator to a precision-appropriate level
    qend = collect(q[:, 1])[end]
    ref  = HarmonicOscillator.exact_solution_q(T(1.0), T(0.5), T(0.0), T(0.0), params)
    @test abs(Float64(qend) - Float64(ref)) < (T == Float64 ? 1e-8 : 1e-3)
end
