# Per-precision unit test for the standard Continuous Galerkin Variational
# Integrator (CGVINodal). This is the linear reference integrator (Lagrange
# basis + Lobatto quadrature, no neural-network stack), so it is well-conditioned
# and runs at reduced precision without regularization. Asserts no silent upcast
# plus tight accuracy against the analytic harmonic oscillator.
@testset "CGVINodal ($T)" for T in TEST_TYPES
    @debug "CGVINodal unit: element type = $T"
    params = HarmonicOscillator.default_parameters(T)
    prob = HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)];
        timespan = (T(0.0), T(1.0)), timestep = T(0.1), parameters = params)

    qlob = lobatto(T, 4)
    blob = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(qlob))
    method = CGVINodal(blob, qlob)

    sol = integrate(prob, method)

    assert_no_upcast(sol.q, T)

    qend = collect(sol.q[:, 1])[end]
    ref = HarmonicOscillator.exact_solution_q(T(1.0), T(0.5), T(0.0), T(0.0), params)
    err = abs(Float64(qend) - Float64(ref))
    @debug "CGVINodal ($T)" q_end=Float64(qend) q_ref=Float64(ref) abs_err=err
    @test err < (T == Float64 ? 1e-8 : 1e-3)
end

# Regression guard for `D > 1`.
#
# `components!`, `residual!`, `initial_guess!` and `update!` all index the same flat vector of
# `D*(S-1)` basis coefficients, and every layout mistake between them collapses to the identity
# at `D = 1` — which is all the testset above covers. Two of them used to disagree: `components!`
# read `x[D*(d-1)+s]` (so at `D = 2, S = 4` it read `x[3]` twice and `x[6]` never, leaving a zero
# Jacobian column and a `SingularException: Zero pivot found at index 6` on the very first step),
# and `update!` broadcast a single scalar across all `D` components of `q`.
#
# `CoupledHarmonicOscillator` with the coupling parameter `k = 0` is two *independent* harmonic
# oscillators with different masses, different spring constants and — as set up here — different
# initial conditions, so each degree of freedom has its own closed-form solution and its own
# frequency. Any layout bug that duplicates, swaps or drops a component therefore shows up as a
# wrong number rather than merely a slightly worse one.
#
# Float64 only: the layout is precision-independent, the `D = 1` testset above already covers
# Float32, and each `lodeproblem` call runs EulerLagrange's symbolic code generation.
@testset "CGVINodal D = 2 (Float64)" begin
    params = (m₁ = 2.0, m₂ = 1.0, k₁ = 1.5, k₂ = 0.3, k = 0.0)
    q₀ = [0.5, -0.3]
    p₀ = [0.0, 0.4]
    tend = 1.0

    prob = CoupledHarmonicOscillator.lodeproblem(q₀, p₀;
        timespan = (0.0, tend), timestep = 0.1, parameters = params)

    qlob = lobatto(Float64, 4)
    blob = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(qlob))

    sol = integrate(prob, CGVINodal(blob, qlob))

    m = [params.m₁, params.m₂]
    ω = sqrt.([params.k₁, params.k₂] ./ m)
    qref = q₀ .* cos.(ω .* tend) .+ p₀ ./ (m .* ω) .* sin.(ω .* tend)
    pref = .-m .* ω .* q₀ .* sin.(ω .* tend) .+ p₀ .* cos.(ω .* tend)

    qend = [collect(sol.q[:, d])[end] for d in 1:2]
    pend = [collect(sol.p[:, d])[end] for d in 1:2]

    @debug "CGVINodal D = 2" q_end=qend q_ref=qref p_end=pend p_ref=pref
    @test maximum(abs.(qend .- qref)) < 1e-8
    @test maximum(abs.(pend .- pref)) < 1e-8
end
