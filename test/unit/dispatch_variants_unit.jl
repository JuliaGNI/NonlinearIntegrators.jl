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

# The `symbolic = false` basis. The three integrators that read the compiled derivatives
# must refuse it in their constructor rather than fail on a `nothing` call several levels
# down; the two that differentiate with ForwardDiff must not merely accept it but produce
# *exactly* the same trajectory as with a full basis — the build they skip has no other
# effect on the run, and anything else would mean the opt-out changed the numerics.
@testset "symbolic = false basis ($T)" for T in TEST_TYPES
    nosym = ShallowNetBasis{T}(relu_k(3), 4; symbolic = false)
    full  = build_shallownet_basis(T; S = 4)
    quad  = gauss(T, 8)

    @test_throws ArgumentError ShallowNet(nosym, quad)
    @test_throws ArgumentError ShallowNetReversible(nosym, quad)
    @test_throws ArgumentError DenseNet(nosym, quad)

    prob = ho_problem(T; timespan = (T(0.0), T(0.2)), timestep = T(0.1))
    for ctor in (ShallowNetAutodiff, ShallowNetAutodiffReversible)
        kw = (; show_status = false, bias_interval = [-T(pi), T(pi)], dict_amount = 400)
        sol_nosym, _ = integrate(prob, ctor(nosym, quad; kw...);
            regularization_factor = T(1e-5), max_iterations = MAX_NEWTON_ITERATIONS)
        sol_full, _ = integrate(prob, ctor(full, quad; kw...);
            regularization_factor = T(1e-5), max_iterations = MAX_NEWTON_ITERATIONS)

        assert_no_upcast(sol_nosym.q, T)
        @test collect(sol_nosym.q[:, 1])[end] == collect(sol_full.q[:, 1])[end]
        @test collect(sol_nosym.p[:, 1])[end] == collect(sol_full.p[:, 1])[end]
    end
end

# `cse` and `inplace` are forwarded to `SymbolicNeuralNetworks.build_nn_function`. They
# change the emitted code, not the mathematics, so the two settings have to evaluate to the
# same derivative — which is checked here, on the kernels themselves. It is deliberately
# *not* checked end to end: the Newton solve stalls near the round-off floor, so a last-bit
# difference decides which iterate is accepted and the integrated results legitimately
# differ by orders of magnitude (see `benchmark/compare_derivative_backends.jl`).
@testset "cse/inplace code generation ($T)" for T in TEST_TYPES
    default = build_shallownet_basis(T; S = 4)
    plain   = ShallowNetBasis{T}(relu_k(3), 4; cse = false, inplace = false)
    @test has_symbolic_derivatives(plain)

    input = [T(0.3)]
    ps = AbstractNeuralNetworks.NeuralNetworkParameters((
        L1 = (W = reshape(T[0.7, -0.4, 0.2, 0.9], 4, 1), b = T[0.1, -0.2, 0.3, -0.4]),
        L2 = (W = reshape(T[0.5, -0.6, 0.7, -0.8], 1, 4),)))

    # `dqdθ` and `dvdθ` return a parameter tree; `V_func` returns the velocity itself.
    for field in (:dqdθ, :dvdθ)
        a = NI.flatten_params(getproperty(default, field)(input, ps))
        b = NI.flatten_params(getproperty(plain, field)(input, ps))
        @test length(a) == length(b)
        @test a ≈ b rtol = 1024 * eps(T)
    end
    @test vec(default.V_func(input, ps)) ≈ vec(plain.V_func(input, ps)) rtol = 1024 * eps(T)

    # And the plain-codegen basis still drives an integrator.
    prob = ho_problem(T; timespan = (T(0.0), T(0.2)), timestep = T(0.1))
    sol, _ = integrate(prob, ShallowNet(plain, gauss(T, 8);
            show_status = false, bias_interval = [-T(pi), T(pi)], dict_amount = 400);
        regularization_factor = T(1e-5), max_iterations = MAX_NEWTON_ITERATIONS)
    assert_no_upcast(sol.q, T)
    @test all(isfinite, collect(sol.q[:, 1])[end])
end
