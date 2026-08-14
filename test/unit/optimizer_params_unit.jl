# Unit tests for the flat ↔ nested parameter conversion that carries the training loops.
#
# `GeometricOptimizers.Optimizer` accepts an `AbstractVector`, a `Manifold`, or a *flat*
# `NamedTuple` of arrays, whereas `AbstractNeuralNetworks` parameters are one level deeper.
# `optimizer_params` flattens the layer nesting into `L1_W`-style keys and `network_params`
# puts it back. The property the training loops actually depend on is that neither copies:
# the optimizer updates `ps_flat` in place and the network has to see the result.

using AbstractNeuralNetworks: NeuralNetworkParameters

nested_params() = (L1 = (W = [1.0 2.0; 3.0 4.0], b = [5.0, 6.0]),
                   L2 = (W = [7.0 8.0],))

@testset "optimizer_params / network_params" begin
    ps = nested_params()

    @testset "flattening" begin
        flat = NI.optimizer_params(ps)
        @test keys(flat) == (:L1_W, :L1_b, :L2_W)
        @test flat.L1_W == ps.L1.W
        @test flat.L1_b == ps.L1.b
        @test flat.L2_W == ps.L2.W
    end

    # This is the load-bearing one. `optimizer_params` is documented as aliasing, and
    # `initial_params!` relies on it: it hands `ps_flat` to the optimizer and then reads the
    # trained weights back out of `PNN.params`, never out of the flat view.
    @testset "aliases rather than copies" begin
        flat = NI.optimizer_params(ps)
        @test flat.L1_W === ps.L1.W
        @test flat.L1_b === ps.L1.b
        @test flat.L2_W === ps.L2.W

        flat.L1_W[1, 1] = -99.0
        @test ps.L1.W[1, 1] == -99.0
    end

    @testset "round trip" begin
        flat = NI.optimizer_params(ps)
        back = NI.network_params(flat, ps)
        @test keys(back) == keys(ps)
        @test keys(back.L1) == keys(ps.L1)
        @test back.L1.W === ps.L1.W
        @test back.L1.b === ps.L1.b
        @test back.L2.W === ps.L2.W
    end

    # `NonLinear_DenseNet_GML`'s LSGD path optimises L1 and L2 only, re-solving L3 by least
    # squares each epoch, so both helpers have to work on a subset of the layers.
    @testset "subset of layers" begin
        subset = (L1 = ps.L1, L2 = ps.L2)
        flat = NI.optimizer_params(subset)
        @test keys(flat) == (:L1_W, :L1_b, :L2_W)
        @test NI.network_params(flat, subset).L1.W === ps.L1.W
    end

    # `PNN.params` is a `NeuralNetworkParameters`, not a bare `NamedTuple`, and the loss has
    # to hand the network back the same wrapper it expects.
    @testset "NeuralNetworkParameters" begin
        nnp = NeuralNetworkParameters(nested_params())
        flat = NI.optimizer_params(nnp)
        @test keys(flat) == (:L1_W, :L1_b, :L2_W)
        @test flat.L1_W === nnp.L1.W

        back = NI.network_params(flat, nnp)
        @test back isa NeuralNetworkParameters
        @test back.L1.W === nnp.L1.W
        @test back.L2.W === nnp.L2.W
    end

    # Float32 has to survive the round trip: the whole suite asserts no silent upcast, and a
    # conversion here would put the optimizer to work at the wrong precision.
    @testset "preserves element type" begin
        ps32 = (L1 = (W = Float32[1 2; 3 4], b = Float32[5, 6]), L2 = (W = Float32[7 8],))
        flat = NI.optimizer_params(ps32)
        @test eltype(flat.L1_W) == Float32
        @test eltype(NI.network_params(flat, ps32).L2.W) == Float32
    end
end
