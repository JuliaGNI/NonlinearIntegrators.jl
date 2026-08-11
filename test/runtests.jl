using Test
using Logging

# The network-based integrators run many nonlinear solves, often with very tight
# solver tolerances, which emit a large volume of line-search / iteration warnings.
# Silence everything below error level so the test output stays readable; failures
# are still reported through the @test machinery. Set JULIA_DEBUG=NonlinearIntegrators
# or lower the disable_logging level to see @debug output from the test files.
Logging.disable_logging(Logging.Warn)

# Shared constants (TEST_TYPES), builders and the no-upcast assertion.
include("testsetup.jl")

# The suite is ordered fastest-first: construction-only smoke tests, then short
# per-type integration unit tests (the "no silent upcast" gate), then the slow
# high-fidelity accuracy guard. Every phase is parametrized over TEST_TYPES so the
# whole package is exercised at both Float64 and Float32.
#
# NOTE: NonLinear_DenseNet_GML is exercised end to end in densenet_gml_unit.jl, but only
# on two-step runs with `training_epochs = 3`. Its Training/LSGD initial-guess methods are
# not stable enough for an accuracy guard, so those testsets assert dispatch, element type
# and finiteness only — a converged DenseNet solve is not something CI can rely on.
@testset "NonlinearIntegrators.jl" begin
    @testset "smoke" begin
        include("smoke/bases_smoke.jl")
        include("smoke/methods_smoke.jl")
    end

    @testset "unit" begin
        include("unit/oga_kernels.jl")
        include("unit/onelayer_gml_unit.jl")
        include("unit/cgvi_standard_unit.jl")
        include("unit/time_reversible_onelayer_unit.jl")
        include("unit/hardcode_unit.jl")
        include("unit/time_reversible_hardcode_unit.jl")
        include("unit/densenet_gml_unit.jl")
        include("unit/dispatch_variants_unit.jl")
        include("unit/pr_integrator_unit.jl")
    end

    @testset "integration" begin
        include("integration/onelayer_accuracy.jl")
    end
end
