# Per-precision unit test for the standard Continuous Galerkin Variational
# Integrator (CGVI_standard). This is the linear reference integrator (Lagrange
# basis + Lobatto quadrature, no neural-network stack), so it is well-conditioned
# and runs at reduced precision without regularization. Asserts no silent upcast
# plus tight accuracy against the analytic harmonic oscillator.
@testset "CGVI_standard ($T)" for T in TEST_TYPES
    params = HarmonicOscillator.default_parameters(T)
    prob = HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)], T;
        timespan = (T(0.0), T(1.0)), timestep = T(0.1), parameters = params)

    qlob = lobatto(T, 4)
    blob = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(qlob))
    method = CGVI_standard(blob, qlob)

    sol = integrate(prob, method)

    assert_no_upcast(sol.q, T)

    qend = collect(sol.q[:, 1])[end]
    ref = HarmonicOscillator.exact_solution_q(T(1.0), T(0.5), T(0.0), T(0.0), params)
    @test abs(Float64(qend) - Float64(ref)) < (T == Float64 ? 1e-8 : 1e-3)
end
