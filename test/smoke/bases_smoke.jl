# Smoke tests for the exported basis data structures and option-tag singletons.
# These only *construct* the objects (no integration) and check that they carry
# the requested element type and print without error, at every TEST_TYPE.

@testset "basis smoke ($T)" for T in TEST_TYPES
    @debug "basis smoke: element type = $T"

    @testset "ShallowNetBasis" begin
        net = build_shallownet_basis(T; S = 4)
        @test net isa AbstractShallowNetBasis{T}
        @test net isa CompactBasisFunctions.Basis{T}
        @test net.S == 4
        @test has_symbolic_derivatives(net)
        @test sprint(show, net) isa String
        @debug "ShallowNetBasis{$T} ok" S=net.S
    end

    # `symbolic = false` skips the SymbolicNeuralNetworks build entirely. The network
    # itself is unaffected — it is only the four derivative slots that stay `nothing`.
    @testset "ShallowNetBasis (symbolic = false)" begin
        net = ShallowNetBasis{T}(relu_k(3), 4; symbolic = false)
        @test net isa AbstractShallowNetBasis{T}
        @test net.S == 4
        @test !has_symbolic_derivatives(net)
        @test net.SNN === nothing
        @test net.dqdθ === nothing
        @test net.V_func === nothing
        @test net.dvdθ === nothing
        @test sprint(show, net) isa String
    end

    @testset "DenseNetBasis" begin
        dnet = build_densenet_basis(T; S₁ = 3, S = 3)
        @test dnet isa AbstractDenseNetBasis{T}
        @test dnet isa CompactBasisFunctions.Basis{T}
        @test dnet.S == 3
        @test dnet.S₁ == 3
        @test sprint(show, dnet) isa String
        @test has_symbolic_derivatives(dnet)
        @debug "DenseNetBasis{$T} ok" S=dnet.S S₁=dnet.S₁ NP=dnet.NP
    end

    # `cse = false, inplace = false` is the pre-0.4 code generation. It changes the emitted
    # code, not what the basis carries, so the only thing to check here is that the build
    # still goes through — this is the basis whose `cse = false` build the CHANGELOG measures
    # at 3.22 s, hence the minimum width. The numerical agreement between the two settings is
    # checked on `ShallowNetBasis` in dispatch_variants_unit.jl.
    @testset "DenseNetBasis (plain codegen)" begin
        dnet = DenseNetBasis{T}(tanh, 2, 2; cse = false, inplace = false)
        @test dnet isa AbstractDenseNetBasis{T}
        @test dnet.S == 2
        @test dnet.S₁ == 2
        @test has_symbolic_derivatives(dnet)
    end

    @testset "VISEBasis" begin
        prb = build_vise_basis(T)
        @test prb isa CompactBasisFunctions.Basis{T}
        @test prb.problem_dim == 1
        @debug "VISEBasis{$T} ok" problem_dim=prb.problem_dim
    end
end

@testset "option-tag singletons" begin
    @test IntegratorExtrapolation() isa IntegratorExtrapolation
    @test TrainingMethod() isa InitialParametersMethod
    @test LSGD() isa InitialParametersMethod
end

@testset "OGA seeds" begin
    for seed in (OGA1d(), OGA1dNormalized(), OGA1dStable(), OGA2d(), OGASphere(),
                 OGA1dNormalEquations())
        @test seed isa InitialParametersMethod
    end
    # The presets are named corners of one composable type, so a hand-built configuration
    # is just as valid a seed. Being isbits keeps the method struct's type parameter
    # concrete.
    custom = OGA(BiasGrid1d(), OrthogonalProjection(), TruncatedSVD())
    @test custom isa InitialParametersMethod
    @test isbits(custom)
    @test oga_label(OGA1d()) == "grid1d/raw/qr"
end
