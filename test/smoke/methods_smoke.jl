# Smoke tests for the exported integrator method structures. These construct each
# method (no integration) and check it is an `LODEMethod` of the expected family
# whose coefficient arrays carry the requested element type. The (symbolic) bases
# are built once per T and shared across the network methods to keep this fast.

@testset "method smoke ($T)" for T in TEST_TYPES
    net  = build_onelayer_basis(T; S = 4)
    dnet = build_densenet_basis(T; S₁ = 3, S = 3)
    quad = gauss(T, 4)

    @testset "NonLinear_OneLayer_GML" begin
        m = NonLinear_OneLayer_GML(net, quad; bias_interval = [-T(pi), T(pi)], dict_amount = 400)
        @test m isa OneLayerMethod
        @test m isa GeometricIntegratorsBase.LODEMethod
        @test eltype(m.b) == T && eltype(m.c) == T
        @test eltype(m.bias_interval) == T
        @test typeof(m.problem_initial_hamitltonian) == T
    end

    @testset "Hardcode_int" begin
        m = Hardcode_int(net, quad; show_status = false, bias_interval = [-T(pi), T(pi)], dict_amount = 400)
        @test m isa OneLayerMethod
        @test m isa GeometricIntegratorsBase.LODEMethod
        @test eltype(m.b) == T && eltype(m.c) == T
        @test eltype(m.bias_interval) == T
    end

    @testset "Time_reversible_OneLayer" begin
        m = Time_reversible_OneLayer(net, quad; show_status = false, bias_interval = [-T(pi), T(pi)], dict_amount = 400)
        @test m isa OneLayerMethod
        @test m isa GeometricIntegratorsBase.LODEMethod
        @test eltype(m.b) == T && eltype(m.c) == T
        @test eltype(m.bias_interval) == T
    end

    @testset "Time_Reversible_Hardcode" begin
        m = Time_Reversible_Hardcode(net, quad; show_status = false, bias_interval = [-T(pi), T(pi)], dict_amount = 400)
        @test m isa OneLayerMethod
        @test m isa GeometricIntegratorsBase.LODEMethod
        @test eltype(m.b) == T && eltype(m.c) == T
        @test eltype(m.bias_interval) == T
    end

    @testset "NonLinear_DenseNet_GML" begin
        m = NonLinear_DenseNet_GML(dnet, quad; training_epochs = 100)
        @test m isa DenseNetMethod
        @test m isa GeometricIntegratorsBase.LODEMethod
        @test eltype(m.b) == T && eltype(m.c) == T
    end

    @testset "CGVI_standard" begin
        qlob = lobatto(T, 4)
        blob = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(qlob))
        cg = CGVI_standard(blob, qlob)
        @test cg isa GeometricIntegratorsBase.LODEMethod
        @test eltype(cg.b) == T && eltype(cg.c) == T
        @test eltype(cg.x) == T
    end

    @testset "PR_Integrator" begin
        prb = build_pr_basis(T)
        pri = PR_Integrator(prb, gauss(T, 8), [T[-0.5, 0.707, -1.57]])
        @test pri isa GeometricIntegratorsBase.LODEMethod
        @test eltype(pri.b) == T && eltype(pri.c) == T
        @test eltype(pri.init_w[1]) == T
    end
end
