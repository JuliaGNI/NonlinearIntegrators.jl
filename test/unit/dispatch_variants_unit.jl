# Unit tests for initial-parameter-method dispatch variants not exercised by the default
# OGA1d path. The 1-D seeds are covered by the cross-product in shallownet_unit.jl; what is
# left here is `TrainingMethod` and the two-dimensional dictionaries — the latter have unit
# coverage in oga_kernels.jl but are otherwise never driven through an integrator.
# DenseNet dispatch variants and extrapolation cross-products live in densenet_unit.jl.
# Each testset checks:
#   (a) the run stays at the working element type (no silent upcast), and
#   (b) the final position is finite.

# The 2-D dictionaries cross weight magnitudes with the bias grid, so `dict_amount` counts
# bias points per weight magnitude and the atom count is a multiple of it. Kept small: this
# asserts that the adapter, the symmetry mapping and the incremental QR compose end to end,
# not that the seed is accurate.
@testset "$name ShallowNet ($T)" for (seed, name) in [(OGA2d(), "OGA2d"), (OGASphere(), "OGASphere")],
                                   T in TEST_TYPES
    prob = ho_problem(T; timespan = (T(0.0), T(0.2)), timestep = T(0.1))
    method = ShallowNet(
        build_shallownet_basis(T; S = 4), gauss(T, 8);
        initial_guess_method = seed,
        show_status = false,
        bias_interval = [-T(pi), T(pi)], dict_amount = 200)

    sol, _ = integrate(prob, method; regularization_factor = T(1e-5), max_iterations = MAX_NEWTON_ITERATIONS)

    assert_no_upcast(sol.q, T)
    qend = collect(sol.q[:, 1])[end]
    @test all(isfinite, qend)
end

@testset "TrainingMethod ShallowNet ($T)" for T in TEST_TYPES
    prob = ho_problem(T; timespan = (T(0.0), T(0.2)), timestep = T(0.1))
    method = ShallowNet(
        build_shallownet_basis(T; S = 4), gauss(T, 8);
        initial_guess_method = TrainingMethod(),
        show_status = false,
        training_epochs = 3)

    sol, _ = integrate(prob, method; regularization_factor = T(1e-5), max_iterations = MAX_NEWTON_ITERATIONS)

    assert_no_upcast(sol.q, T)
    qend = collect(sol.q[:, 1])[end]
    @test all(isfinite, qend)
end
