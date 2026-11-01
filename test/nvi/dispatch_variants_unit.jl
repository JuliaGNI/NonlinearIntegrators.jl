# Unit tests for the construction-time variants not exercised by the default path — the
# initial-parameter methods other than OGA1d, and the basis keywords that change what a basis
# carries. The 1-D seeds are covered by the cross-product in shallownet_unit.jl; what is left
# here is `TrainingMethod` and the two-dimensional dictionaries (the latter have unit coverage
# in oga_kernels.jl but are otherwise never driven through an integrator), plus
# `ShallowNetBasis`'s `symbolic = false` and `cse`/`inplace`. DenseNet dispatch variants and
# extrapolation cross-products live in densenet_unit.jl.
# Each testset that integrates checks:
#   (a) the run stays at the working element type (no silent upcast), and
#   (b) the final position is finite.
# The last two testsets check the compiled derivatives, and the coefficients that `DenseNet`'s
# `components!` builds from them, against `ForwardDiff`.

using Test
using NonlinearIntegrators
include(joinpath(@__DIR__, "..", "helpers", "testsetup.jl"))

# The 2-D dictionaries cross weight magnitudes with the bias grid, so `dict_amount` counts
# bias points per weight magnitude and the atom count is a multiple of it. Kept small: this
# asserts that the adapter, the symmetry mapping and the incremental QR compose end to end,
# not that the seed is accurate.
@testset "$name ShallowNet ($T)" for (seed, name) in [
        (OGA2d(), "OGA2d"), (OGASphere(), "OGASphere")],
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
    full = build_shallownet_basis(T; S = 4)
    quad = gauss(T, 8)

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
    plain = ShallowNetBasis{T}(relu_k(3), 4; cse = false, inplace = false)
    @test has_symbolic_derivatives(plain)

    input = [T(0.3)]
    ps = NeuralNetworkParameters.NetworkParameters((
        L1 = (W = reshape(T[0.7, -0.4, 0.2, 0.9], 4, 1), b = T[0.1, -0.2, 0.3, -0.4]),
        L2 = (W = reshape(T[0.5, -0.6, 0.7, -0.8], 1, 4),)))

    # `dqdθ` and `dvdθ` return a parameter tree; `V_func` returns the velocity itself.
    for field in (:dqdθ, :dvdθ)
        a, _ = NeuralNetworkParameters.flatten(getproperty(default, field)(input, ps))
        b, _ = NeuralNetworkParameters.flatten(getproperty(plain, field)(input, ps))
        @test length(a) == length(b)
        @test a ≈ b rtol = 1024 * eps(T)
    end
    @test vec(default.V_func(input, ps)) ≈ vec(plain.V_func(input, ps)) rtol = 1024 * eps(T)

    # And the plain-codegen basis still drives an integrator.
    prob = ho_problem(T; timespan = (T(0.0), T(0.2)), timestep = T(0.1))
    sol, _ = integrate(prob,
        ShallowNet(plain, gauss(T, 8);
            show_status = false, bias_interval = [-T(pi), T(pi)], dict_amount = 400);
        regularization_factor = T(1e-5), max_iterations = MAX_NEWTON_ITERATIONS)
    assert_no_upcast(sol.q, T)
    @test all(isfinite, collect(sol.q[:, 1])[end])
end

# The inverse of `NeuralNetworkParameters.flatten`: walk the layers and their fields in the same
# order.
# That order is not incidental — `components!` writes the flattened gradient into
# `dqdθc[j, :, d]` and indexes the Newton unknowns by the same layout — so a reference
# reconstructed this way pins the layout as well as the values.
function rebuild_params(v, template)
    offset = 0
    layers = ()
    for lname in keys(template)
        layer = template[lname]
        fields = ()
        for fname in keys(layer)
            a = layer[fname]
            fields = (fields..., reshape(v[(offset + 1):(offset + length(a))], size(a)))
            offset += length(a)
        end
        layers = (layers..., NamedTuple{keys(layer)}(fields))
    end
    NeuralNetworkParameters.NetworkParameters(NamedTuple{keys(template)}(layers))
end

# The compiled kernels against an *independent* reference. The testset above compares
# `cse+inplace` against `plain`, which is one symbolic expression under two code-generation
# settings: it catches a wrong code path but not a wrong expression. Nothing else in the suite
# differentiates the network by another route, so until this testset existed the only guard
# against a mis-shaped or mis-differentiated `dqdθ`/`dvdθ` was an integration test failing
# somewhere downstream.
#
# That gap is not hypothetical. `SymbolicNeuralNetworks` 0.5 made `symbolic_parameter_gradient`
# return the parameter-shaped gradient itself for a *scalar* expression, and indexing that
# return value with `[1]` — which is what the 0.4 call sites did to unwrap the one-element
# array around it — yields the first parameter *leaf* rather than an error. Carried over
# unchanged, it would have compiled a silently truncated gradient.
#
# `ForwardDiff` over a flattened parameter vector is the reference: it reaches the network
# through `basis.NN`, so it shares no code with the symbolic path beyond the network itself,
# and unlike `NI.∂NN_ansatz_∂params` it does not require the hand-written ansatz to match.
# Only the default codegen is checked here — the testset above ties `plain` to it.
@testset "compiled derivatives vs ForwardDiff ($T)" for T in TEST_TYPES
    Random.seed!(1234)
    shallow = build_shallownet_basis(T; S = 4)
    dense = build_densenet_basis(T)

    @testset "$label" for (label, basis, np) in ((
        "ShallowNetBasis", shallow, 3 * shallow.S), ("DenseNetBasis", dense, dense.NP))
        NN = basis.NN
        ps = AbstractNeuralNetworks.params(AbstractNeuralNetworks.NeuralNetwork(NN, T))
        θ, _ = NeuralNetworkParameters.flatten(ps)
        t = T(0.37)

        # The position and the velocity at `t`, both as functions of the flat parameter
        # vector. `v` differentiates `q` in time under whatever number type it is handed, so
        # `ForwardDiff.gradient(v, ·)` nests one dual inside the other.
        q(w, s) = NN([s], rebuild_params(w, ps))[1]
        v(w) = ForwardDiff.derivative(s -> q(w, s), t)

        @test length(θ) == np
        @test eltype(θ) == T

        dqdθ, _ = NeuralNetworkParameters.flatten(basis.dqdθ([t], ps))
        @test length(dqdθ) == np
        @test dqdθ ≈ ForwardDiff.gradient(w -> q(w, t), θ) rtol = 1024 * eps(T)

        dvdθ, _ = NeuralNetworkParameters.flatten(basis.dvdθ([t], ps))
        @test length(dvdθ) == np
        @test dvdθ ≈ ForwardDiff.gradient(v, θ) rtol = 1024 * eps(T)

        @test basis.V_func([t], ps)[1] ≈ v(θ) rtol = 1024 * eps(T)
    end
end

allocations(f::F, args...) where {F} = (f(args...); @allocated f(args...))

# `DenseNet`'s `components!` flattens the compiled gradients into the Newton coefficients with the
# `ParameterLayout` that its cache keeps. Each row of `g0_params`, `g1_params`, `dqdθc` and `dvdθc`
# is checked against `ForwardDiff` over the dimension's stretch of the Newton unknowns `x`, so the
# test pins the values and the order in which `densenet_pack!` lays out the unknowns. The `D = 2`
# case reaches the second column of each array. It is Float64 only, because each
# `CoupledHarmonicOscillator.lodeproblem` call runs EulerLagrange's symbolic code generation.
@testset "DenseNet components! coefficients ($T, D = $D)" for (T, D) in (
    (Float64, 1), (Float32, 1), (Float64, 2))
    Random.seed!(4321)
    prob = D == 1 ? ho_problem(T) :
           CoupledHarmonicOscillator.lodeproblem([0.5, -0.3], [0.0, 0.4];
        timespan = (0.0, 0.2), timestep = 0.1,
        parameters = (m₁ = 2.0, m₂ = 1.0, k₁ = 1.5, k₂ = 0.3, k = 0.0))
    basis = build_densenet_basis(T)
    int = GeometricIntegrator(prob, DenseNet(basis, gauss(T, 8); show_status = false))
    C = GeometricIntegratorsBase.cache(int, T)
    NP = basis.NP
    x = randn(T, D * (NP + 1)) ./ 2
    sol = (t = zero(T), q = zeros(T, D), p = zeros(T, D))
    GeometricIntegratorsBase.components!(x, sol, parameters(prob), int)

    template = AbstractNeuralNetworks.params(AbstractNeuralNetworks.NeuralNetwork(basis.NN, T))
    q(w, s) = basis.NN([s], rebuild_params(w, template))[1]
    v(w, s) = ForwardDiff.derivative(r -> q(w, r), s)
    nodes = QuadratureRules.nodes(gauss(T, 8))

    @test eltype(C.g0_params) == eltype(C.dqdθc) == eltype(C.dvdθc) == T
    for d in 1:D
        θ = x[((d - 1) * NP + 1):(d * NP)]
        @test C.g0_params[:, d] ≈ ForwardDiff.gradient(w -> q(w, zero(T)), θ) rtol = 1024 *
                                                                                     eps(T)
        @test C.g1_params[:, d] ≈ ForwardDiff.gradient(w -> q(w, one(T)), θ) rtol = 1024 *
                                                                                    eps(T)
        for j in eachindex(nodes)
            @test C.dqdθc[j, :, d] ≈ ForwardDiff.gradient(w -> q(w, nodes[j]), θ) rtol = 1024 *
                                                                                         eps(T)
            @test C.dvdθc[j, :, d] ≈ ForwardDiff.gradient(w -> v(w, nodes[j]), θ) rtol = 1024 *
                                                                                         eps(T)
        end
    end

    # The gradient is a `NetworkParameters` of the type of the parameters, so the one layout
    # serves both; flattening with it is allocation-free and inferred.
    g = basis.dqdθ([nodes[1]], NeuralNetworkParameters.NetworkParameters(C.ps[1]))
    dest = view(C.dqdθc, 1, :, 1)
    @test C.layout == NeuralNetworkParameters.parameterlayout(g)
    @test length(C.layout) == NP
    @test allocations(NeuralNetworkParameters.flatten!, dest, g, C.layout) == 0
    @test allocations(collect, nodes) > 0
    @test @inferred(NeuralNetworkParameters.flatten!(dest, g, C.layout)) isa
          AbstractVector{T}
end
