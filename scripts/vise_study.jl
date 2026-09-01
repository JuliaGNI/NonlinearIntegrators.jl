# VISE against the linear variational integrators it is the nonlinear counterpart of.
#
#   julia --project=scripts scripts/vise_study.jl [harmonic-oscillator|perturbed-pendulum|henon-heiles]...
#
# For each problem and time step: VISE on a closed-form ansatz, the polynomial Galerkin integrator
# `CGVI` at the same quadrature order, implicit midpoint, and a `Gauss(8)` reference. Prints a
# table; writes nothing.
#
# This replaces `test_vise.jl` and `vise_plot.jl`. `test_vise.jl` was 893 lines of which 854 were
# commented out — the surviving record of the ansätze and initial weights below, but not runnable.
# `vise_plot.jl` was 1257 lines that loaded `.jld2` files from an absolute path on a machine this
# is not, and mixed CairoMakie, Plots, MLJ and SymbolicRegression to do it. The figures they were
# reaching for are now `NonlinearIntegrators.Diagnostics.plot_solution`; see `docs/src/vise/vise.md`.
#
# The harmonic-oscillator ansatz *spans* the exact solution, so that row is a check rather than a
# measurement: anything above the residual floor there is a regression.

using Printf

using GeometricIntegrators
using GeometricProblems
using CompactBasisFunctions
using QuadratureRules
using SimpleSolvers
using Symbolics

using NonlinearIntegrators

const HO = GeometricProblems.HarmonicOscillator
const PPD = GeometricProblems.PerturbedPendulum
const HH = GeometricProblems.HenonHeilesPotential

# `Backtracking` and `iter1000` are what the original sweep used.
#
# **No `f_abstol` or `f_suctol`**, deliberately: `default_options` computes
# `max(8, solversize(method, problem)) * eps(datatype(problem))`, which scales with the size of the
# solver system because the norm of a residual on its round-off floor grows with the number of
# components it sums. The `8eps()` the original filenames record is that formula's *floor* — right
# for the smallest systems, too tight for the rest, and VISE's system grows with the weight count
# (Hénon–Heiles carries eight against the harmonic oscillator's three).
#
# `default_options` merges rather than replaces since GeometricIntegratorsBase 0.5, so naming
# `max_iterations` and `linesearch` keeps the sized `f_abstol` and the stall window — which the
# original `default_options` override silently dropped.
const SOLVER_OPTIONS = (
    max_iterations = 1000,
    linesearch = SimpleSolvers.Backtracking()
)

const REFERENCE_SUBSTEPS = 40

struct Study
    name::String
    dimension::Int
    problem::Function
    hamiltonian::Function
    weight_count::Vector{Int}
    ansatz::Function
    init_w::Vector{Vector{Float64}}
    final_time::Float64
    quadrature::Dict{Float64, Int}
end

const STUDIES = (
    Study("harmonic-oscillator", 1,
        (timespan, timestep) -> HO.lodeproblem(; timespan = timespan, timestep = timestep),
        HO.hamiltonian, [3],
        (W, t) -> [W[1][1] * sin(W[1][2] * t + W[1][3])],
        [[-0.5000433352162222, 0.705350078478666, -1.5678140333370576]],
        200.0, Dict(1.0 => 8, 2.0 => 16, 5.0 => 8)),
    Study("perturbed-pendulum", 1,
        (timespan, timestep) -> PPD.lodeproblem(; timespan = timespan, timestep = timestep),
        PPD.hamiltonian, [3],
        (W, t) -> [W[1][1] * cos(W[1][2] * t + W[1][3])],
        [[-0.51941, -0.47405, 2.8713]],
        200.0, Dict(1.0 => 8, 2.0 => 16, 5.0 => 16)),
    # Included because it is the case that does *not* work: a three-term ansatz per coordinate does
    # not span a Hénon–Heiles trajectory, and the relative error is O(1) from about t = 40 on. The
    # original sweep recorded the same thing (`HenonHeiles_hams_err = 0.94` over T = 200), so this
    # is the ansatz being too small and not a solver regression. It is here so that stays visible.
    Study("henon-heiles", 2,
        (timespan, timestep) -> HH.lodeproblem([0.1, 0.1], [0.1, 0.1];
            timespan = timespan, timestep = timestep),
        HH.hamiltonian, [4, 4],
        (W, t) -> [W[d][1] * cos(W[d][2] * t + W[d][3]) + W[d][4] for d in 1:2],
        [[0.14831, 1.0, -0.64812, -0.018712], [0.14298, -0.97215, 0.7615, -0.0013983]],
        60.0, Dict(1.0 => 16, 2.0 => 16, 5.0 => 16))
)

const STEPS = (1.0, 2.0, 5.0)

function build_method(study::Study, timestep::Float64)
    t = only(@variables tvar)
    W = [only(@variables($(Symbol(:W, d))[1:n]))
         for (d, n) in enumerate(study.weight_count)]
    basis = VISEBasis{Float64}(study.ansatz(W, t), W, t, study.dimension)
    VISE(basis, QuadratureRules.GaussLegendreQuadrature(study.quadrature[timestep]),
        study.init_w)
end

function galerkin_method(R::Int)
    quadrature = QuadratureRules.GaussLegendreQuadrature(R)
    CGVI(Lagrange(QuadratureRules.nodes(quadrature)), quadrature)
end

function hamiltonian_error(sol, hamiltonian, parameters)
    maximum(relative_invariant_error(
        [hamiltonian(sol.t[n], sol.q[n], sol.p[n], parameters) for n in 0:ntime(sol)]))
end

# The reference lives on a grid `REFERENCE_SUBSTEPS` times finer, so it is sampled at the macro
# steps rather than compared axis-for-axis. Normalised by the maximum of the *whole* reference and
# not per step: these are oscillators, and a per-step divisor vanishes at every zero crossing,
# which turns a bounded absolute error into an unbounded relative one.
function reference_error(sol, ref_sol, substeps)
    worst = 0.0
    scale = 0.0
    for n in 0:ntime(sol)
        reference = ref_sol.q[n * substeps]
        worst = max(worst, maximum(abs, sol.q[n] .- reference))
        scale = max(scale, maximum(abs, reference))
    end
    iszero(worst) ? 0.0 : worst / scale
end

function run(study::Study, timestep::Float64)
    prob = study.problem((0.0, study.final_time), timestep)
    R = study.quadrature[timestep]
    params = prob.parameters

    sol, internal_values, _ = integrate(prob, build_method(study, timestep); SOLVER_OPTIONS...)
    cgvi_sol = integrate(prob, galerkin_method(R))
    imp_sol = integrate(prob, ImplicitMidpoint())
    ref_sol = integrate(
        study.problem((0.0, study.final_time), timestep / REFERENCE_SUBSTEPS), Gauss(8))

    # The continuous solution is what a Galerkin variational integrator gives for free, so the
    # study reports that it is there and how long it is rather than dropping it.
    t_fine, _ = continuous_solution(internal_values, timestep)

    @printf("  %-20s %-5s %4d  %10.3e %10.3e  %10.3e %10.3e  %10.3e %6d\n",
        study.name, timestep, R,
        hamiltonian_error(sol, study.hamiltonian, params),
        reference_error(sol, ref_sol, REFERENCE_SUBSTEPS),
        hamiltonian_error(cgvi_sol, study.hamiltonian, params),
        reference_error(cgvi_sol, ref_sol, REFERENCE_SUBSTEPS),
        hamiltonian_error(imp_sol, study.hamiltonian, params),
        length(t_fine))
end

function main(args)
    studies = isempty(args) ? STUDIES :
              [STUDIES[findfirst(s -> s.name == n, STUDIES)] for n in args]

    @printf("  %-20s %-5s %4s  %10s %10s  %10s %10s  %10s %6s\n",
        "problem", "h", "R", "VISE ΔH", "VISE err", "CGVI ΔH", "CGVI err", "IMP ΔH",
        "n fine")
    println("  ", repeat("-", 104))
    for study in studies, timestep in STEPS

        run(study, timestep)
    end
end

main(ARGS)
