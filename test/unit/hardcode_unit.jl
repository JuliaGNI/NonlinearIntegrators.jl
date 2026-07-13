# Per-precision unit tests for the hardcoded-ansatz integrators (Hardcode_int and
# its time-reversible variant Time_Reversible_Hardcode). These build the ansatz
# derivatives with ForwardDiff rather than symbolic-network derivatives, so the
# interval boundary points t=0/t=1 must stay at the (plain) quadrature element type
# rather than the solver's Dual type — the main reason these were precision-fragile.
# Asserts no silent upcast plus a precision-appropriate accuracy bound.

@testset "Hardcode_int ($T)" for T in TEST_TYPES
    params = HarmonicOscillator.default_parameters(T)
    prob = HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)], T;
        timespan = (T(0.0), T(1.0)), timestep = T(0.1), parameters = params)

    method = Hardcode_int(build_onelayer_basis(T; S = 4), gauss(T, 8);
        show_status = false, bias_interval = [-T(pi), T(pi)], dict_amount = 400)

    sol, _ = integrate(prob, method; regularization_factor = T(1e-5), max_iterations = 10000)

    assert_no_upcast(sol.q, T)

    qend = collect(sol.q[:, 1])[end]
    ref = HarmonicOscillator.exact_solution_q(T(1.0), T(0.5), T(0.0), T(0.0), params)
    @test abs(Float64(qend) - Float64(ref)) < (T == Float64 ? 1e-4 : 1e-3)
end

@testset "Time_Reversible_Hardcode ($T)" for T in TEST_TYPES
    params = HarmonicOscillator.default_parameters(T)
    prob = HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)], T;
        timespan = (T(0.0), T(1.0)), timestep = T(0.1), parameters = params)

    method = Time_Reversible_Hardcode(build_onelayer_basis(T; S = 4), gauss(T, 8);
        show_status = false, bias_interval = [-T(pi), T(pi)], dict_amount = 400)

    sol, _ = integrate(prob, method; regularization_factor = T(1e-5), max_iterations = 10000)

    assert_no_upcast(sol.q, T)
    @test all(isfinite, collect(sol.q[:, 1])[end])
end
