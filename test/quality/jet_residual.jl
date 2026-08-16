# JET optimisation analysis of the Newton hot path, run as a *standalone script*.
#
# Not part of the suite: `aqua_jet.jl` skips JET and points here. Analysing this in-process,
# or through `JET.test_package`, does not give a usable answer:
#
#   * In-process, this runs after ~1800 other tests have instantiated a large number of
#     specialisations of the same functions, and Julia's inference widens under that load —
#     `view(r₀, :, d)` comes back as `SubArray{Float64,N,Matrix{Float64},I,true} where {N,I}`
#     rather than the concrete type it infers in a clean session. The reports that follow are
#     artefacts of the analysis context, not of the shipped code.
#   * `JET.test_package` is not usable either: it re-analyses the package from source in the
#     *active* environment, which under `test/Project.toml` does not carry the package's own
#     dependencies, and reports 108 "Package GeometricEquations not found" toplevel errors.
#
# A fresh process analysing `residual!` at the concrete argument types is what actually answers
# the question "does the hot path dispatch dynamically". Exits 0 if clean, 1 otherwise, printing
# the reports.
#
# Run it from the repository root. Not with `--project=test`: `test/Project.toml` deliberately
# omits NonlinearIntegrators itself (see the note at the top of that file), so `using
# NonlinearIntegrators` below would fail there. A temporary environment gets the package plus
# JET without writing a `test/Manifest.toml`:
#
#     julia -e 'using Pkg; Pkg.activate(temp = true); Pkg.develop(path = "."); \
#               Pkg.add(["JET", "GeometricProblems", "QuadratureRules", \
#                        "GeometricIntegratorsBase", "GeometricSolutions"]); \
#               include("test/quality/jet_residual.jl")'

using NonlinearIntegrators
using QuadratureRules
using GeometricIntegratorsBase
using GeometricProblems.HarmonicOscillator
using GeometricIntegratorsBase: solutionstep, nlsolution, residual!, initial_guess!, current
using GeometricSolutions: timesteps
using JET

relu_k(k::Int = 3) = x -> max(zero(x), x)^k

function probe(make)
    prob = HarmonicOscillator.lodeproblem([0.5], [0.0]; timespan = (0.0, 0.2), timestep = 0.1)
    int = GeometricIntegrator(prob, make(); regularization_factor = 1e-5, max_iterations = 100)
    sol = GeometricIntegratorsBase.GeometricSolution(prob)
    ss = solutionstep(int, sol[0])
    GeometricIntegratorsBase.reset!(ss, timesteps(sol)[1])
    s = current(ss)
    params = GeometricIntegratorsBase.parameters(prob)
    initial_guess!(s, nothing, params, int)
    x = nlsolution(int)
    return (; int, s, params, x, b = similar(x))
end

const QUAD = QuadratureRules.GaussLegendreQuadrature(Float64, 8)
const KW = (; show_status = false, bias_interval = [-pi, pi], dict_amount = 400)

symbolic_basis() = ShallowNetBasis{Float64}(relu_k(3), 4)
autodiff_basis() = ShallowNetBasis{Float64}(relu_k(3), 4; symbolic = false)

const CASES = [
    ("ShallowNet",                   () -> ShallowNet(symbolic_basis(), QUAD; KW...)),
    ("ShallowNetReversible",         () -> ShallowNetReversible(symbolic_basis(), QUAD; KW...)),
    ("ShallowNetAutodiff",           () -> ShallowNetAutodiff(autodiff_basis(), QUAD; KW...)),
    ("ShallowNetAutodiffReversible", () -> ShallowNetAutodiffReversible(autodiff_basis(), QUAD; KW...)),
]

failed = String[]
for (name, make) in CASES
    p = probe(make)
    result = JET.report_opt(residual!, typeof.((p.b, p.x, p.s, p.params, p.int));
                            target_modules = (NonlinearIntegrators,))
    reports = JET.get_reports(result)
    if isempty(reports)
        println("ok   $name: no runtime dispatch in residual!")
    else
        println("FAIL $name: $(length(reports)) report(s)")
        show(stdout, result)
        println()
        push!(failed, name)
    end
end

exit(isempty(failed) ? 0 : 1)
