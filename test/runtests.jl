using Test
using Logging

# The network-based integrators run many nonlinear solves, often with very tight
# solver tolerances, which emit a large volume of line-search / iteration warnings.
# Silence everything below error level so the test output stays readable; failures
# are still reported through the @test machinery.
Logging.disable_logging(Logging.Warn)

# Shared constants (TEST_TYPES), builders and the no-upcast assertion.
include("testsetup.jl")

# The suite is ordered fastest-first: construction-only smoke tests, then short
# per-type integration unit tests (the "no silent upcast" gate), then the slow
# high-fidelity accuracy guard. Every phase is parametrized over TEST_TYPES so the
# whole package is exercised at both Float64 and Float32.
#
# NOTE: NonLinear_DenseNet_GML gets a construction smoke test only. Its end-to-end
# solve relies on the unstable Training/LSGD initial-guess methods and runs for
# several minutes, so it is intentionally not integrated in CI (as before).
@testset "NonlinearIntegrators.jl" begin
    @testset "smoke" begin
        include("smoke/bases_smoke.jl")
        include("smoke/methods_smoke.jl")
    end

    @testset "unit" begin
        include("unit/onelayer_gml_unit.jl")
        include("unit/cgvi_standard_unit.jl")
        include("unit/time_reversible_onelayer_unit.jl")
        include("unit/hardcode_unit.jl")
    end

    @testset "integration" begin
        include("integration/onelayer_accuracy.jl")
    end
end
