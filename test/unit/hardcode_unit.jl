# Per-precision unit tests for Hardcode_int: the hardcoded-ansatz integrator that
# builds derivatives with ForwardDiff rather than symbolic-network derivatives.
# The interval boundary points t=0/t=1 must stay at the (plain) quadrature element
# type rather than the solver's Dual type — the main reason this was precision-fragile.
# Time_Reversible_Hardcode tests live in time_reversible_hardcode_unit.jl.

# Accuracy guard: default combination (OGA1d × IntegratorExtrapolation), long run.
@testset "Hardcode_int accuracy ($T)" for T in TEST_TYPES
    @debug "Hardcode_int unit: element type = $T"
    params = HarmonicOscillator.default_parameters(T)
    prob = HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)];
        timespan = (T(0.0), T(1.0)), timestep = T(0.1), parameters = params)

    method = Hardcode_int(build_onelayer_basis(T; S = 4), gauss(T, 8);
        show_status = false, bias_interval = [-T(pi), T(pi)], dict_amount = 400)

    sol, _ = integrate(prob, method; regularization_factor = T(1e-5), max_iterations = 10000)

    assert_no_upcast(sol.q, T)

    qend = collect(sol.q[:, 1])[end]
    ref = HarmonicOscillator.exact_solution_q(T(1.0), T(0.5), T(0.0), T(0.0), params)
    err = abs(Float64(qend) - Float64(ref))
    @debug "Hardcode_int ($T)" q_end=Float64(qend) q_ref=Float64(ref) abs_err=err
    @test err < (T == Float64 ? 1e-4 : 1e-3)
end

# Cross-product: OGA1d × extrapolation variants, short run, finite check.
const HC_EXTRAPOLATIONS = [
    (NoExtrapolation(),          "NoExtrapolation"),
    (IntegratorExtrapolation(),  "IntegratorExtrapolation"),
    (HermiteExtrapolation(),     "HermiteExtrapolation"),
]

for T in TEST_TYPES, (extrap, extrap_name) in HC_EXTRAPOLATIONS
    @testset "Hardcode_int OGA1d × $extrap_name ($T)" begin
        prob = ho_problem(T; timespan = (T(0.0), T(0.2)), timestep = T(0.1))
        method = Hardcode_int(build_onelayer_basis(T; S = 4), gauss(T, 8);
            initial_trajectory_method = extrap,
            show_status = false, bias_interval = [-T(pi), T(pi)], dict_amount = 400)
        sol, _ = integrate(prob, method; regularization_factor = T(1e-5), max_iterations = 10000)
        assert_no_upcast(sol.q, T)
        @test all(isfinite, collect(sol.q[:, 1])[end])
    end
end
