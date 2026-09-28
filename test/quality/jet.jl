# JET optimisation analysis of the Newton hot path.
#
# The entry points are the four `residual!` cases of `scripts/jet_residual.jl`, analysed at the
# concrete argument types that `probe` builds. One line per entry point, at `Float64`, the one
# element type the tests use.

using Test
using JET
using NonlinearIntegrators
using QuadratureRules
using GeometricIntegratorsBase
using GeometricProblems.HarmonicOscillator
using GeometricIntegratorsBase: solutionstep, nlsolution, residual!, initial_guess!, current
using GeometricSolutions: timesteps

const JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE

relu_k(k::Int = 3) = x -> max(zero(x), x)^k

function residual_types(make)
    prob = HarmonicOscillator.lodeproblem([0.5], [0.0]; timespan = (0.0, 0.2), timestep = 0.1)
    int = GeometricIntegrator(prob, make(); regularization_factor = 1e-5, max_iterations = 100)
    sol = GeometricIntegratorsBase.GeometricSolution(prob)
    ss = solutionstep(int, sol[0])
    GeometricIntegratorsBase.reset!(ss, timesteps(sol)[1])
    s = current(ss)
    params = GeometricIntegratorsBase.parameters(prob)
    initial_guess!(s, nothing, params, int)
    x = nlsolution(int)
    return typeof.((similar(x), x, s, params, int))
end

const QUAD = QuadratureRules.GaussLegendreQuadrature(Float64, 8)
const KW = (; show_status = false, bias_interval = [-pi, pi], dict_amount = 400)

symbolic_basis() = ShallowNetBasis{Float64}(relu_k(3), 4)
autodiff_basis() = ShallowNetBasis{Float64}(relu_k(3), 4; symbolic = false)

if !JET_WORKS
    @test_skip "JET does not work on this Julia version"  # aviatesk/JET.jl#681
else
    @testset "residual! ShallowNet" begin
        types = residual_types(() -> ShallowNet(symbolic_basis(), QUAD; KW...))
        @test isempty(JET.get_reports(JET.report_opt(residual!, types;
            target_modules = (NonlinearIntegrators,))))
    end
    @testset "residual! ShallowNetReversible" begin
        types = residual_types(() -> ShallowNetReversible(symbolic_basis(), QUAD; KW...))
        @test isempty(JET.get_reports(JET.report_opt(residual!, types;
            target_modules = (NonlinearIntegrators,))))
    end
    @testset "residual! ShallowNetAutodiff" begin
        types = residual_types(() -> ShallowNetAutodiff(autodiff_basis(), QUAD; KW...))
        @test isempty(JET.get_reports(JET.report_opt(residual!, types;
            target_modules = (NonlinearIntegrators,))))
    end
    @testset "residual! ShallowNetAutodiffReversible" begin
        types = residual_types(() -> ShallowNetAutodiffReversible(autodiff_basis(), QUAD; KW...))
        @test isempty(JET.get_reports(JET.report_opt(residual!, types;
            target_modules = (NonlinearIntegrators,))))
    end
end
