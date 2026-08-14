# Per-precision unit tests for ShallowNetAutodiff: the integrator that differentiates a
# hand-written ansatz with ForwardDiff rather than using symbolic-network derivatives.
# The interval boundary points t=0/t=1 must stay at the (plain) quadrature element
# type rather than the solver's Dual type — the main reason this was precision-fragile.
# ShallowNetAutodiffReversible tests live in shallownet_autodiff_reversible_unit.jl.

# Accuracy guard: default combination (OGA1d × IntegratorExtrapolation), long run.
@testset "ShallowNetAutodiff accuracy ($T)" for T in TEST_TYPES
    @debug "ShallowNetAutodiff unit: element type = $T"
    params = HarmonicOscillator.default_parameters(T)
    prob = HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)];
        timespan = (T(0.0), T(1.0)), timestep = T(0.1), parameters = params)

    method = ShallowNetAutodiff(build_shallownet_basis(T; S = 4), gauss(T, 8);
        show_status = false, bias_interval = [-T(pi), T(pi)], dict_amount = 400)

    sol, _ = integrate(prob, method; regularization_factor = T(1e-5), max_iterations = MAX_NEWTON_ITERATIONS)

    assert_no_upcast(sol.q, T)

    qend = collect(sol.q[:, 1])[end]
    ref = HarmonicOscillator.exact_solution_q(T(1.0), T(0.5), T(0.0), T(0.0), params)
    err = abs(Float64(qend) - Float64(ref))
    @debug "ShallowNetAutodiff ($T)" q_end=Float64(qend) q_ref=Float64(ref) abs_err=err
    @test err < (T == Float64 ? 1e-4 : 1e-3)
end

# Cross-product: the default seed × extrapolation variants, short run, finite check.
#
# See `hermite_kw` in testsetup.jl for why the Hermite row also passes `initialguess`.
const AUTODIFF_EXTRAPOLATIONS = [
    (NoExtrapolation(),          "NoExtrapolation"),
    (IntegratorExtrapolation(),  "IntegratorExtrapolation"),
    (HermiteExtrapolation(),     "HermiteExtrapolation"),
]

for T in TEST_TYPES, (extrap, extrap_name) in AUTODIFF_EXTRAPOLATIONS
    @testset "ShallowNetAutodiff OGA1dNormalized × $extrap_name ($T)" begin
        prob = ho_problem(T; timespan = (T(0.0), T(0.2)), timestep = T(0.1))
        method = ShallowNetAutodiff(build_shallownet_basis(T; S = 4), gauss(T, 8);
            initial_trajectory_method = extrap,
            show_status = false, bias_interval = [-T(pi), T(pi)], dict_amount = 400)
        sol, _ = integrate(prob, method; regularization_factor = T(1e-5), max_iterations = MAX_NEWTON_ITERATIONS,
            hermite_kw(extrap)...)
        assert_no_upcast(sol.q, T)
        @test all(isfinite, collect(sol.q[:, 1])[end])
    end
end
