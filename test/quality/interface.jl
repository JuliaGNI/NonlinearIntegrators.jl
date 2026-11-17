# The contract of `GeometricBase.AbstractMethod`, checked on one instance of each method type
# this package owns. `GeometricBase.interface_violations` calls every property of the contract
# (`isexplicit`, …, `order`, `name`, …) and, through the method for `GeometricMethod` in
# `GeometricIntegratorsBase`, the defaults that take no problem; it returns an empty vector for a
# method that keeps the contract. The instances are built as `integration/methods_smoke.jl` builds
# them, at `Float64`, with no integration.

using Test
using NonlinearIntegrators
using GeometricBase
include(joinpath(@__DIR__, "..", "helpers", "testsetup.jl"))

using InteractiveUtils: subtypes

# Every owned method type that is not abstract, found by walking the subtypes of `AbstractMethod`.
function owned_method_types()
    found = Type[]
    queue = Any[GeometricBase.AbstractMethod]
    while !isempty(queue)
        for S in subtypes(pop!(queue))
            push!(queue, S)
            if !isabstracttype(S) &&
               Base.moduleroot(parentmodule(S)) === NonlinearIntegrators
                push!(found, S)
            end
        end
    end
    return found
end

@testset "interface conformance" begin
    net = build_shallownet_basis(Float64; S = 4)
    dnet = build_densenet_basis(Float64; S₁ = 3, S = 3)
    quad = gauss(Float64, 4)

    cases = [
        ShallowNet(net, quad; shallow_kw(Float64)...),
        ShallowNetAutodiff(net, quad; shallow_kw(Float64)...),
        ShallowNetReversible(net, quad; shallow_kw(Float64)...),
        ShallowNetAutodiffReversible(net, quad; shallow_kw(Float64)...),
        DenseNet(dnet, quad; training_epochs = 100),
        VISE(build_vise_basis(Float64), gauss(Float64, 8), [[-0.5, 0.707, -1.57]]),
        FreeKnotSpline(4)
    ]

    # A type with a constructor of no argument is checked on that instance too.
    owned = owned_method_types()
    for T in owned
        hasmethod(T, Tuple{}) && push!(cases, T())
    end

    @testset "$(nameof(typeof(m)))" for m in cases
        @test isempty(GeometricBase.interface_violations(m))
    end

    # A new method type without an instance above fails here.
    @testset "every owned method type is covered" begin
        @test !isempty(owned)
        @testset "$(nameof(T))" for T in owned
            @test any(m -> m isa T, cases)
        end
    end
end
