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

# Both fits build their network with `NeuralNetwork(NN)`, which names no element type, so the
# parameters they return are `Float64` at every precision.
function test_fit_eltype(params, ::Type{T}) where {T}
    if T == Float32
        @test_broken eltype(params.L1.W) == T  # issue #135, KNOWN_ISSUES.md K37
    else
        @test eltype(params.L1.W) == T
    end
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
