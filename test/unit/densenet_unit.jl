# Per-precision unit tests for DenseNet: the cross product of initialization
# methods (TrainingMethod, LSGD) × extrapolation variants (NoExtrapolation,
# IntegratorExtrapolation, HermiteExtrapolation). Training is capped at a few epochs so the
# tests run in CI time.
#
# Its `HermiteExtrapolation` rows used to raise `NaN detected in direction vector!`, which
# looked like the documented instability of the gradient-descent seeds. It was not: the shared
# Hermite path was extrapolating nothing, because `iguess` defaults to `NoInitialGuess` and
# `solutionstep!` is then a no-op. With `initialguess` passed (see `hermite_kw` in
# testsetup.jl) every combination converges, so convergence is asserted outright rather than
# tolerated — a test that accepts failure cannot report a regression.
const DENSENET_INIT_METHODS = [
    (TrainingMethod(), "TrainingMethod"),
    (LSGD(),           "LSGD"),
]

const DENSENET_EXTRAPOLATIONS = [
    (NoExtrapolation(),          "NoExtrapolation"),
    (IntegratorExtrapolation(),  "IntegratorExtrapolation"),
    (HermiteExtrapolation(),     "HermiteExtrapolation"),
]

for T in TEST_TYPES,
    (init_method, init_name) in DENSENET_INIT_METHODS,
    (extrap, extrap_name) in DENSENET_EXTRAPOLATIONS

    @testset "DenseNet $init_name × $extrap_name ($T)" begin
        prob = ho_problem(T; timespan = (T(0.0), T(0.2)), timestep = T(0.1))
        method = DenseNet(build_densenet_basis(T; S₁ = 3, S = 3), gauss(T, 8);
            show_status = false,
            initial_guess_method = init_method,
            initial_trajectory_method = extrap,
            training_epochs = 3)

        sol, _ = integrate(prob, method; regularization_factor = T(1e-5),
            max_iterations = MAX_NEWTON_ITERATIONS, hermite_kw(extrap)...)
        assert_no_upcast(sol.q, T)
        @test all(isfinite, collect(sol.q[:, 1])[end])
    end
end
