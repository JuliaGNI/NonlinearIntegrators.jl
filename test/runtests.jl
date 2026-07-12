using Test
using Logging
using NonlinearIntegrators

# The network-based integrators run many nonlinear solves, often with very tight
# solver tolerances, which emit a large volume of line-search / iteration
# warnings. Silence everything below error level so the test output stays
# readable; failures are still reported through the @test machinery.
Logging.disable_logging(Logging.Warn)

@testset "NonlinearIntegrators.jl" begin
    @testset "NonLinear_OneLayer_GML integrators" begin
        include("integrators/nvi_onelayer_integrators_tests.jl")
    end

    @testset "Time_reversible_OneLayer integrator" begin
        include("integrators/nvi_onelayer_time_reversible_integrators_tests.jl")
    end

    # NOTE: the NonLinear_DenseNet_GML integrator tests
    # (test/integrators/nvi_dense_integrators_tests.jl) are intentionally omitted
    # for now: they integrate over a long time span with a large number of
    # training epochs (several minutes) and rely on the unstable training/LSGD
    # initial-guess methods.
end
