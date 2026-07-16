# Smoke tests for the exported basis data structures and option-tag singletons.
# These only *construct* the objects (no integration) and check that they carry
# the requested element type and print without error, at every TEST_TYPE.

@testset "basis smoke ($T)" for T in TEST_TYPES
    @debug "basis smoke: element type = $T"

    @testset "OneLayerNetwork_GML" begin
        net = build_onelayer_basis(T; S = 4)
        @test net isa OneLayerNetBasis{T}
        @test net isa CompactBasisFunctions.Basis{T}
        @test net.S == 4
        @test sprint(show, net) isa String
        @debug "OneLayerNetwork_GML{$T} ok" S=net.S
    end

    @testset "DenseNet_GML" begin
        dnet = build_densenet_basis(T; S₁ = 3, S = 3)
        @test dnet isa DenseNetBasis{T}
        @test dnet isa CompactBasisFunctions.Basis{T}
        @test dnet.S == 3
        @test dnet.S₁ == 3
        @test sprint(show, dnet) isa String
        @debug "DenseNet_GML{$T} ok" S=dnet.S S₁=dnet.S₁ NP=dnet.NP
    end

    @testset "PR_Basis" begin
        prb = build_pr_basis(T)
        @test prb isa CompactBasisFunctions.Basis{T}
        @test prb.problem_dim == 1
        @debug "PR_Basis{$T} ok" problem_dim=prb.problem_dim
    end
end

@testset "option-tag singletons" begin
    @test IntegratorExtrapolation() isa IntegratorExtrapolation
    @test TrainingMethod() isa InitialParametersMethod
    @test OGA1d() isa InitialParametersMethod
    @test LSGD() isa InitialParametersMethod
end
