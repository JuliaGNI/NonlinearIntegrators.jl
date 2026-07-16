# Per-precision unit test for the Time_reversible_OneLayer integrator. Same
# OGA-seeded one-layer structure as NonLinear_OneLayer_GML, with a time-reversible
# discretisation. Asserts no silent upcast plus precision-appropriate accuracy.
@testset "Time_reversible_OneLayer ($T)" for T in TEST_TYPES
    @debug "Time_reversible_OneLayer unit: element type = $T"
    params = HarmonicOscillator.default_parameters(T)
    prob = HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)];
        timespan = (T(0.0), T(1.0)), timestep = T(0.1), parameters = params)

    method = Time_reversible_OneLayer(build_onelayer_basis(T; S = 4), gauss(T, 8);
        show_status = false, bias_interval = [-T(pi), T(pi)], dict_amount = 400)

    sol, _ = integrate(prob, method; regularization_factor = T(1e-5), max_iterations = 10000)

    assert_no_upcast(sol.q, T)

    qend = collect(sol.q[:, 1])[end]
    ref = HarmonicOscillator.exact_solution_q(T(1.0), T(0.5), T(0.0), T(0.0), params)
    err = abs(Float64(qend) - Float64(ref))
    @debug "Time_reversible_OneLayer ($T)" q_end=Float64(qend) q_ref=Float64(ref) abs_err=err
    @test err < (T == Float64 ? 1e-8 : 1e-3)
end
