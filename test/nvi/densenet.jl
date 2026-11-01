# The two seeding loops of `DenseNet` drive a GeometricOptimizers optimizer by hand:
# `increase_iteration_number!`, `solver_step!` and `update!` once per pass. The optimizer state
# counts the passes only through the first of the three calls, so after the loop its iteration
# number equals the number of passes the loop ran. Adam reads that number for its bias
# correction, and `DecayingStatic` for its learning rate.

using Test
using NonlinearIntegrators
include(joinpath(@__DIR__, "..", "helpers", "testsetup.jl"))

const GO = NI.GeometricOptimizers

# Labels that a fit of a few passes does not reach to the early-exit thresholds of either loop
# (5e-8 and 5e-5), so each loop runs its full budget.
function densenet_fit_data(::Type{T}) where {T}
    network_inputs = reshape(collect(range(zero(T), one(T), 11)), 1, 11)
    labels = cos.(3 .* network_inputs)
    network_inputs, labels
end

# Both fits build their network at the element type of `network_inputs`.
test_fit_eltype(params, ::Type{T}) where {T} = @test eltype(params.L1.W) == T

# A network of the layer shapes of `DenseNetBasis` whose hidden activations are `zero`. Its
# output is zero whatever its parameters, so the gradient of the loss is zero and `Adam` does not
# move them: the loss is the mean of `abs.(labels)` at every pass.
function zero_output_network(S₁, S)
    AbstractNeuralNetworks.Chain(
        AbstractNeuralNetworks.Dense(1, S₁, zero),
        AbstractNeuralNetworks.Dense(S₁, S, zero),
        AbstractNeuralNetworks.Dense(S, 1, identity, use_bias = false))
end

@testset "DenseNet seeding loops count their passes ($T, $nepochs passes)" for T in TEST_TYPES,
    nepochs in (3, 5)

    Random.seed!(42)
    basis = build_densenet_basis(T)
    network_inputs, labels = densenet_fit_data(T)

    @testset "TrainingMethod (Adam, DecayingStatic)" begin
        params, state, passes = NI.densenet_training_fit(
            basis.NN, network_inputs, labels, nepochs, 1)
        @test passes == nepochs
        @test GO.iteration_number(state) == passes
        test_fit_eltype(params, T)
    end

    # An exactly fitting network of the basis does not stop the fit: `Adam` moves each parameter
    # by about its learning rate whatever the size of the gradient, so the loss after the first
    # step is far above 5e-8. A network with zero output and labels of size 1e-9 keeps the loss at
    # 1e-9, below that threshold, so the loop stops after its first pass.
    @testset "TrainingMethod early exit" begin
        params, state, passes = NI.densenet_training_fit(
            zero_output_network(basis.S₁, basis.S), network_inputs,
            fill(T(1e-9), size(labels)), nepochs, 1)
        @test passes == 1
        @test GO.iteration_number(state) == passes
        test_fit_eltype(params, T)
    end

    @testset "LSGD (GradientMethod, Static)" begin
        params, state, passes = NI.densenet_lsgd_fit(
            basis.NN, network_inputs, labels, nepochs, basis.S₁, basis.S, 1)
        @test passes == nepochs
        @test GO.iteration_number(state) == passes
        test_fit_eltype(params, T)
    end

    # Labels of size 1e-6 leave a loss of about that size after the least-squares solve of the
    # last layer, below the threshold of 5e-5, so the loop stops after its first pass.
    @testset "LSGD early exit" begin
        params, state, passes = NI.densenet_lsgd_fit(
            basis.NN, network_inputs, fill(T(1e-6), size(labels)), nepochs, basis.S₁, basis.S, 1)
        @test passes == 1
        @test GO.iteration_number(state) == passes
        test_fit_eltype(params, T)
    end
end

# `components!` flattens the compiled gradients with the `ParameterLayout` that `DenseNetCache`
# keeps. A layout built in its place has the same value, so no coefficient tells the two apart.
# The testset counts the layouts built: a method of `parameterlayout` for the exact type of the
# cached parameters, which are also the type of the gradients, counts its calls and forwards to the
# generic method. Both a `parameterlayout` call and the two-argument `flatten!` reach it. The
# method is defined in a newer world, so the calls go through `invokelatest`, and it is deleted
# after the testset.
const LAYOUT_BUILDS = Ref(0)

@testset "DenseNet components! builds no ParameterLayout ($T)" for T in TEST_TYPES
    Random.seed!(4321)
    prob = ho_problem(T)
    basis = build_densenet_basis(T)
    int = GeometricIntegrator(prob, DenseNet(basis, gauss(T, 8); show_status = false))
    C = GeometricIntegratorsBase.cache(int, T)
    x = randn(T, basis.NP + 1) ./ 2
    sol = (t = zero(T), q = zeros(T, 1), p = zeros(T, 1))
    ps = NeuralNetworkParameters.NetworkParameters(C.ps[1])

    @eval function NeuralNetworkParameters.parameterlayout(ps::$(typeof(ps)))
        $LAYOUT_BUILDS[] += 1
        invoke(NeuralNetworkParameters.parameterlayout, Tuple{Any}, ps)
    end
    counter = which(NeuralNetworkParameters.parameterlayout, (typeof(ps),))
    try
        LAYOUT_BUILDS[] = 0
        @test Base.invokelatest(NeuralNetworkParameters.parameterlayout, ps) == C.layout
        @test LAYOUT_BUILDS[] == 1

        LAYOUT_BUILDS[] = 0
        Base.invokelatest(
            GeometricIntegratorsBase.components!, x, sol, parameters(prob), int)
        @test LAYOUT_BUILDS[] == 0
        @test C.dqdθc[1, :, 1] != zeros(T, basis.NP)
    finally
        Base.delete_method(counter)
    end
end
