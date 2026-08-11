# Per-precision unit tests for NonLinear_DenseNet_GML. Tests the full cross product
# of initialization methods (TrainingMethod, LSGD) × extrapolation variants
# (NoExtrapolation, IntegratorExtrapolation, HermiteExtrapolation).
# Training is capped at few epochs so the tests run in CI time.
# Checks no silent upcast and a finite final state in every case.

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

    @testset "NonLinear_DenseNet_GML $init_name × $extrap_name ($T)" begin
        prob = ho_problem(T; timespan = (T(0.0), T(0.2)), timestep = T(0.1))
        method = NonLinear_DenseNet_GML(build_densenet_basis(T; S₁ = 3, S = 3), gauss(T, 8);
            initial_guess_method = init_method,
            initial_trajectory_method = extrap,
            training_epochs = 3)
        sol, _ = integrate(prob, method; regularization_factor = T(1e-5), max_iterations = 10000)
        assert_no_upcast(sol.q, T)
        @test all(isfinite, collect(sol.q[:, 1])[end])
    end
end
