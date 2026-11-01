# Smoke tests for the exported integrator method structures. These construct each
# method (no integration) and check it is an `LODEMethod` of the expected family
# whose coefficient arrays carry the requested element type. The (symbolic) bases
# are built once per T and shared across the network methods to keep this fast.

using Test
using NonlinearIntegrators
include(joinpath(@__DIR__, "..", "helpers", "testsetup.jl"))

@testset "method smoke ($T)" for T in TEST_TYPES
    @debug "method smoke: element type = $T"
    net = build_shallownet_basis(T; S = 4)
    dnet = build_densenet_basis(T; S₁ = 3, S = 3)
    quad = gauss(T, 4)

    @testset "ShallowNet" begin
        m = ShallowNet(net, quad; show_status = false,
            bias_interval = [-T(pi), T(pi)], dict_amount = 400)
        @test m isa ShallowNetMethod
        @test m isa GeometricIntegratorsBase.LODEMethod
        @test eltype(m.b) == T && eltype(m.c) == T
        @test eltype(m.bias_interval) == T
        @test !GeometricIntegratorsBase.isexplicit(m)
        @debug "ShallowNet{$T} ok" extrapolation_substep=m.extrapolation_substep training_epochs=m.training_epochs
    end

    @testset "ShallowNetAutodiff" begin
        m = ShallowNetAutodiff(net, quad; show_status = false,
            bias_interval = [-T(pi), T(pi)], dict_amount = 400)
        @test m isa ShallowNetMethod
        @test m isa GeometricIntegratorsBase.LODEMethod
        @test eltype(m.b) == T && eltype(m.c) == T
        @test eltype(m.bias_interval) == T
        @debug "ShallowNetAutodiff{$T} ok" extrapolation_substep=m.extrapolation_substep
    end

    @testset "ShallowNetReversible" begin
        m = ShallowNetReversible(net, quad; show_status = false,
            bias_interval = [-T(pi), T(pi)], dict_amount = 400)
        @test m isa ShallowNetMethod
        @test m isa GeometricIntegratorsBase.LODEMethod
        @test eltype(m.b) == T && eltype(m.c) == T
        @test eltype(m.bias_interval) == T
        @test GeometricIntegratorsBase.issymmetric(m)
        @debug "ShallowNetReversible{$T} ok" extrapolation_substep=m.extrapolation_substep
    end

    @testset "ShallowNetAutodiffReversible" begin
        m = ShallowNetAutodiffReversible(net, quad; show_status = false,
            bias_interval = [-T(pi), T(pi)], dict_amount = 400)
        @test m isa ShallowNetMethod
        @test m isa GeometricIntegratorsBase.LODEMethod
        @test eltype(m.b) == T && eltype(m.c) == T
        @test eltype(m.bias_interval) == T
        @test GeometricIntegratorsBase.issymmetric(m)
        @debug "ShallowNetAutodiffReversible{$T} ok" extrapolation_substep=m.extrapolation_substep
    end

    # Both time-reversible methods store only the `S/2` independent hidden parameters, the
    # other half being the mirror image of the first, so an odd `S` is rejected where the
    # basis is handed over rather than deep in `components!`.
    @testset "odd basis size is rejected" begin
        odd = build_shallownet_basis(T; S = 5)
        @test_throws ArgumentError ShallowNetReversible(odd, quad; show_status = false,
            bias_interval = [-T(pi), T(pi)], dict_amount = 400)
        @test_throws ArgumentError ShallowNetAutodiffReversible(
            odd, quad; show_status = false,
            bias_interval = [-T(pi), T(pi)], dict_amount = 400)
    end

    @testset "DenseNet" begin
        m = DenseNet(dnet, quad; training_epochs = 100)
        @test m isa DenseNetMethod
        @test m isa GeometricIntegratorsBase.LODEMethod
        @test eltype(m.b) == T && eltype(m.c) == T
        @debug "DenseNet{$T} ok" extrapolation_substep=m.extrapolation_substep
    end

    @testset "VISE" begin
        prb = build_vise_basis(T)
        pri = VISE(prb, gauss(T, 8), [T[-0.5, 0.707, -1.57]])
        @test pri isa GeometricIntegratorsBase.LODEMethod
        @test eltype(pri.b) == T && eltype(pri.c) == T
        @test eltype(pri.init_w[1]) == T
        @debug "VISE{$T} ok" extrapolation_substep=pri.extrapolation_substep
    end
end

# `network_inputs` holds the `extrapolation_substep + 1` nodes of the warm-start trajectory, from
# 0 to 1 inclusive. A step range `zero(T):(one(T) / n):one(T)` has a length that depends on the
# round-off of its step: at `BFloat16` it is one element short, or it raises. The IEEE types
# below build such a range exactly, so they guard the node values and the shape; the test
# environment has no `BFloat16`. The expected nodes are the correctly rounded `(0:n) ./ n`,
# computed in Float64 and converted, not the way the constructor computes them.
@testset "network inputs ($T, extrapolation_substep = $n)" for T in (
        Float16, Float32, Float64),
    n in (3, 7, 10, 30)

    m = ShallowNet(build_shallownet_basis(T; S = 4), gauss(T, 4); show_status = false,
        bias_interval = [-T(pi), T(pi)], dict_amount = 400, extrapolation_substep = n)
    x = m.network_inputs
    @test x isa Matrix{T}
    @test size(x) == (1, n + 1)
    @test x[1] == zero(T) && x[end] == one(T)
    @test vec(x) == T.((0:n) ./ n)
end
