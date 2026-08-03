# Unit tests for initial-parameter-method dispatch variants not exercised by the
# default OGA1d path. NonLinear_OneLayer_GML variants (OGA1d_Legacy, TrainingMethod)
# are covered here. DenseNet dispatch variants and extrapolation cross-products live
# in densenet_gml_unit.jl. Each testset checks:
#   (a) the run stays at the working element type (no silent upcast), and
#   (b) the final position is finite.

@testset "OGA1d_Legacy ($T)" for T in TEST_TYPES
    prob = ho_problem(T; timespan = (T(0.0), T(0.2)), timestep = T(0.1))
    method = NonLinear_OneLayer_GML(
        build_onelayer_basis(T; S = 4), gauss(T, 8);
        initial_guess_method = OGA1d_Legacy(),
        bias_interval = [-T(pi), T(pi)], dict_amount = 400)

    sol, _ = integrate(prob, method; regularization_factor = T(1e-5), max_iterations = 10000)

    assert_no_upcast(sol.q, T)
    qend = collect(sol.q[:, 1])[end]
    @test all(isfinite, qend)
end

@testset "TrainingMethod OneLayer ($T)" for T in TEST_TYPES
    prob = ho_problem(T; timespan = (T(0.0), T(0.2)), timestep = T(0.1))
    method = NonLinear_OneLayer_GML(
        build_onelayer_basis(T; S = 4), gauss(T, 8);
        initial_guess_method = TrainingMethod(),
        training_epochs = 3)

    sol, _ = integrate(prob, method; regularization_factor = T(1e-5), max_iterations = 10000)

    assert_no_upcast(sol.q, T)
    qend = collect(sol.q[:, 1])[end]
    @test all(isfinite, qend)
end

