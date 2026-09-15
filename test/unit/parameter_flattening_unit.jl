# Unit tests for the flat ↔ nested parameter conversion that carries the training loops.
#
# `GeometricOptimizers.Optimizer` works in a flat `AbstractVector`, while a network's parameters
# are a `NetworkParameters` of layers of arrays. `NeuralNetworkParameters.flatten` converts one
# to the other and hands back a `ParameterLayout`; `unflatten` rebuilds a set from a vector and
# `unflatten!` writes one back into an existing set.
#
# The helpers this replaces — `optimizer_params`/`network_params`, which built an `L1_W`-keyed
# flat `NamedTuple` aliasing the layer arrays — are gone, and with them the property those tests
# were about. What `initial_params!` depends on now is the round trip: it hands the optimizer a
# vector, the optimizer mutates it, and `unflatten!` has to put the result where the network
# reads it. These are upstream functions, so this file pins the contract this package relies on
# rather than the implementation.

using NeuralNetworkParameters: NetworkParameters, flatten, unflatten, unflatten!, flatlength

nested_params() = (L1 = (W = [1.0 2.0; 3.0 4.0], b = [5.0, 6.0]),
    L2 = (W = [7.0 8.0],))

@testset "parameter flattening" begin
    @testset "flattening" begin
        ps = nested_params()
        v, layout = flatten(ps)

        # Depth first in declaration order, which is the order `_param_arrays` walks too.
        @test v == [1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        @test length(v) == flatlength(ps) == 8
    end

    # This is the load-bearing one. `initial_params!` hands `ps_flat` to the optimizer and then
    # reads the trained weights back out of `PNN.params`, never out of the flat vector, so the
    # write-back is what makes the training loop work at all.
    @testset "mutating the vector reaches the parameters" begin
        ps = nested_params()
        v, layout = flatten(ps)

        # `flatten` copies rather than views — deliberately, so that the flat form can carry a
        # different element type from the parameters. Nothing is visible until `unflatten!`.
        v[1] = -99.0
        @test ps.L1.W[1, 1] == 1.0

        unflatten!(ps, layout, v)
        @test ps.L1.W[1, 1] == -99.0
    end

    @testset "round trip" begin
        ps = nested_params()
        v, layout = flatten(ps)
        back = unflatten(layout, v)

        @test keys(back) == keys(ps)
        @test keys(back.L1) == keys(ps.L1)
        @test back.L1.W == ps.L1.W
        @test back.L1.b == ps.L1.b
        @test back.L2.W == ps.L2.W
    end

    # `DenseNet`'s LSGD path optimises L1 and L2 only, re-solving L3 by least squares each epoch,
    # so the flattening has to work on a bare `NamedTuple` holding a subset of the layers — and
    # the write-back has to reach the network's arrays through the aliasing subset.
    @testset "subset of layers" begin
        ps = nested_params()
        subset = (L1 = ps.L1,)
        v, layout = flatten(subset)

        @test length(v) == 6
        v .= 0.0
        unflatten!(subset, layout, v)
        @test all(iszero, ps.L1.W)
        @test all(iszero, ps.L1.b)
        @test ps.L2.W == [7.0 8.0]                  # untouched
    end

    # `PNN.params` is a `NetworkParameters`, not a bare `NamedTuple`, and the loss has to hand the
    # network back the same wrapper it expects — which is what makes `unflatten` the right inverse
    # to close the loss over.
    @testset "NetworkParameters" begin
        nnp = NetworkParameters(nested_params())
        v, layout = flatten(nnp)
        back = unflatten(layout, v)

        @test back isa NetworkParameters
        @test back.L1.W == nnp.L1.W
        @test back.L2.W == nnp.L2.W

        v .= 1.0
        unflatten!(nnp, layout, v)
        @test all(isone, nnp.L1.W)
    end

    # Float32 has to survive the round trip: the whole suite asserts no silent upcast, and a
    # conversion here would put the optimizer to work at the wrong precision. `flatten` takes the
    # element type from the parameters rather than defaulting to `Float64`.
    @testset "preserves element type" begin
        ps32 = (L1 = (W = Float32[1 2; 3 4], b = Float32[5, 6]), L2 = (W = Float32[7 8],))
        v, layout = flatten(ps32)

        @test eltype(v) == Float32
        @test eltype(unflatten(layout, v).L2.W) == Float32
    end

    # The loss handed to `Optimizer` is differentiated through the flat vector, so `unflatten` is
    # called on `ForwardDiff.Dual`s. It keeps `eltype(v)` by construction; if it converted back to
    # the leaf's element type instead, every gradient in the training loops would come out zero.
    @testset "unflatten keeps the vector's element type" begin
        ps = nested_params()
        _, layout = flatten(ps)
        g = ForwardDiff.gradient(w -> sum(abs2, unflatten(layout, w).L1.W), zeros(8))

        @test g == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        @test ForwardDiff.gradient(w -> sum(unflatten(layout, w).L2.W), zeros(8))[7:8] ==
              [1.0, 1.0]
    end
end
