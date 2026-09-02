# The convergence matrix: maximum relative Hamiltonian error against the time step, for two
# problems × three integrators.
#
#   julia --project=scripts scripts/run_convergence.jl [problem] [integrator] \
#       [--steps h,…] [--final-time T] [--runs-dir dir]
#
# With no arguments, all six cells. With arguments, just the ones named:
#
#   julia --project=scripts scripts/run_convergence.jl perturbed-pendulum
#   julia --project=scripts scripts/run_convergence.jl harmonic-oscillator vise
#
#            │ cgvi              nvi                        vise
#   ─────────┼─────────────────────────────────────────────────────────────────
#   harmonic │ R ∈ {2,3,4}       tanh and ReLU³ families    the symbolic ansatz
#   pendulum │ R ∈ {2,3,4}       tanh and ReLU³ families    the symbolic ansatz
#
# The `nvi` and `vise` cells carry the `cgvi` family again, dashed: a convergence claim about a
# nonlinear method is only readable against the linear one at the same quadrature order.
#
# This is the most expensive script here, because the cost is in the number of *steps* and the
# smallest step is the point of a convergence plot. At `T = 100` and `h = 0.03125` one configuration
# is 3200 solves. The `nvi` cells dominate; `cgvi` and `vise` are minutes.
#
# On the final time: the original figures record it nowhere — not the filename, the script, or the
# paper. `CONVERGENCE_FINAL_TIME = 100` matches the package's own benchmarks. The plotted quantity is
# a maximum over the run, so it can only grow with a longer one, and curves computed over different
# windows are not comparable — which is why `--final-time` prints what it used and why anything but
# the default has to be said in the caption.

include(joinpath(@__DIR__, "experiments.jl"))

using Logging

"""
    max_hamiltonian_error(build_method, problem_fn, hamiltonian, timestep, final_time; options)

The maximum relative Hamiltonian error of one integrator at one step size, or `NaN` if the run
failed outright.

`options` differs between the families, which is why it is an argument rather than a constant. The
neural runs need [`NVI_SOLVER_OPTIONS`](@ref) and in particular its `regularization_factor`: without
it the greedy seed's least-squares system runs unregularised, and for ReLU³ that design matrix is
rank-deficient. The first version of this study omitted them and **22 of the 32 ReLU³ runs died with
a `SingularException`**, leaving one point per curve — while the very same configurations ran to
`T = 1000` in `run_nvi.jl`, which does pass them. A convergence curve with one point is not a curve.

`CGVI` must *not* be given them: `regularization_factor` is a keyword of this package's network
integrators, not of the solver, and a `GeometricIntegrators` method has nowhere to put it.

`NaN` rather than a rethrown error: a study over eight step sizes and ten configurations will have
entries that genuinely break, and losing the rest to the first one is worse than plotting a curve
with a hole in it. The extension drops non-finite entries, and the failure is printed here with its
exception type, so it is visible rather than silent.
"""
function max_hamiltonian_error(build_method, problem_fn, hamiltonian, timestep, final_time;
        options = NamedTuple())
    prob = problem_fn((0.0, final_time), timestep)
    try
        result = Logging.with_logger(Logging.NullLogger()) do
            integrate(prob, build_method(); options...)
        end
        sol = result isa Tuple ? first(result) : result
        return maximum(relative_invariant_error(sol, hamiltonian, prob.parameters))
    catch exception
        exception isa InterruptException && rethrow()
        @printf("      failed: %s\n", typeof(exception))
        return NaN
    end
end

# One series: a label, a line style, and the error at each step.
struct Series
    label::String
    linestyle::String
    errors::Vector{Float64}
end

"""
    sweep(label, linestyle, build_method, problem, steps, final_time; options) -> Series

One integrator over the whole step ladder, printing as it goes.
"""
function sweep(
        label, linestyle, build_method, problem::ConvergenceProblem, steps, final_time;
        options = NamedTuple())
    println("    ", label)
    errors = Float64[]
    for h in steps
        t0 = time()
        err = max_hamiltonian_error(build_method, problem.problem, problem.hamiltonian,
            h, final_time; options = options)
        push!(errors, err)
        @printf("      h = %-8g  max |ΔH/H₀| = %-12.3e  (%.1f s, %d steps)\n",
            h, err, time() - t0, round(Int, final_time / h))
    end
    Series(label, linestyle, errors)
end

# The polynomial Galerkin family. Dashed wherever it appears beside a nonlinear method, so the two
# families are separable by style as well as by colour.
function galerkin_series(problem, steps, final_time; linestyle = "dash")
    [sweep(galerkin_label(R), linestyle, () -> galerkin_method(R),
         problem, steps, final_time)
     for R in GALERKIN_ORDERS]
end

function nvi_series(problem, steps, final_time)
    series = Series[]
    for family in CONVERGENCE_NVI, (S, R) in family.configurations

        push!(series,
            sweep(convergence_label(S, R, family.name), "solid",
                () -> ShallowNet(ShallowNetBasis{Float64}(family.σ, S),
                    QuadratureRules.GaussLegendreQuadrature(R);
                    show_status = false, bias_interval = [-pi, pi],
                    dict_amount = NVI_DICT_AMOUNT),
                problem, steps, final_time; options = NVI_SOLVER_OPTIONS))
    end
    series
end

# VISE at a constant quadrature order across the ladder — see `CONVERGENCE_VISE_ORDERS`. The ansatz
# and the initial weights come from the problem's `VISEProblem`, so this is the same integrator the
# solution figures show.
function vise_series(problem, steps, final_time)
    # The problem's own ansatz, and then whatever `CONVERGENCE_VISE_EXTRA` adds for it — the
    # earlier curves are kept so the extension is readable against them.
    variants = vcat(["" => problem.vise],
        get(CONVERGENCE_VISE_EXTRA, problem.name,
            Pair{String, VISEProblem}[]))

    series = Series[]
    for (tag, experiment) in variants, R in CONVERGENCE_VISE_ORDERS

        label = isempty(tag) ? "VISE R$(R)" : "VISE $(tag) R$(R)"
        push!(series,
            sweep(label, "solid",
                () -> begin
                    t = only(@variables tvar)
                    W = [only(@variables($(Symbol(:W, d))[1:n]))
                         for (d, n) in enumerate(experiment.weight_count)]
                    VISE(
                        VISEBasis{Float64}(experiment.ansatz(W, t), W, t, experiment.dimension),
                        QuadratureRules.GaussLegendreQuadrature(R), experiment.init_w)
                end,
                problem, steps, final_time; options = VISE_SOLVER_OPTIONS))
    end
    series
end

const CELL_TITLES = Dict(
    "cgvi" => "Polynomial Galerkin variational integrators",
    "nvi" => "Neural variational integrators, against polynomial Galerkin",
    "vise" => "Symbolic-ansatz variational integrators, against polynomial Galerkin"
)

function run_cell(problem::ConvergenceProblem, integrator::String, steps, final_time)
    banner("$(problem.label) × $(integrator) — T = $(final_time), h ∈ $(collect(steps))")

    series = if integrator == "cgvi"
        # On its own axes the reference family is the subject, so it is drawn solid.
        galerkin_series(problem, steps, final_time; linestyle = "solid")
    elseif integrator == "nvi"
        vcat(nvi_series(problem, steps, final_time),
            galerkin_series(problem, steps, final_time))
    elseif integrator == "vise"
        vcat(vise_series(problem, steps, final_time),
            galerkin_series(problem, steps, final_time))
    else
        error("unknown integrator $(integrator); one of $(CONVERGENCE_INTEGRATORS)")
    end

    data = Dict{String, Any}(
        "kind" => "convergence",
        # `h²`, `h⁴`, `h⁶` — the orders the polynomial family actually has. A continuous Galerkin
        # variational integrator on `R` Gauss nodes is of order `2R - 2`, and measured between every
        # pair of successive steps on both problems, `CGVI(2)`, `CGVI(3)` and `CGVI(4)` come out at
        # `2.00`, `4.00` and `6.00` to three digits. The `h³` guide the original figures carried
        # matched none of the three. Stored rather than passed at render time, so the guides belong
        # to the study and not to whoever draws it.
        "reference_orders" => [2, 4, 6],
        "problem" => problem.name,
        "problem_label" => problem.label,
        "integrator" => integrator,
        "title" =>
            "$(CELL_TITLES[integrator]) — $(problem.label), " *
            "T = $(number_label(final_time))",
        "final_time" => final_time,
        "timesteps" => collect(steps),
        "labels" => [s.label for s in series],
        "linestyles" => [s.linestyle for s in series],
        "errors" => [s.errors for s in series]
    )

    stem = study_stem(problem.name, "convergence", integrator)
    report_path("archive", store_run!(stem, data))
    return data
end

function main(args)
    names, options = parse_arguments(args, ("--steps", "--final-time"))
    steps = option_steps(options, CONVERGENCE_STEPS)
    final_time = option_final_time(options, CONVERGENCE_FINAL_TIME)

    # Every name must select something. Without this check an unrecognised one filtered *both* axes
    # down to nothing and the two `isempty` fallbacks below then restored both to everything — so a
    # typo silently ran the entire twenty-minute sweep instead of the one cell that was asked for.
    known = union((p.name for p in CONVERGENCE_PROBLEMS), CONVERGENCE_INTEGRATORS)
    unknown = filter(∉(known), names)
    isempty(unknown) || throw(ArgumentError(
        "unknown name(s) $(join(unknown, ", ")); this study has problems " *
        join((p.name for p in CONVERGENCE_PROBLEMS), ", ") * " and integrators " *
        join(CONVERGENCE_INTEGRATORS, ", ")))

    problems = filter(p -> isempty(names) || p.name in names, collect(CONVERGENCE_PROBLEMS))
    integrators = filter(i -> isempty(names) || i in names, collect(CONVERGENCE_INTEGRATORS))
    # A bare problem name selects every integrator of it, and a bare integrator name every problem.
    isempty(problems) && (problems = collect(CONVERGENCE_PROBLEMS))
    isempty(integrators) && (integrators = collect(CONVERGENCE_INTEGRATORS))

    # The window is printed because it is a choice: the plotted quantity is a maximum over the run,
    # so it can only grow with a longer one, and two curves over different windows are not
    # comparable.
    report("final time", final_time)

    for problem in problems, integrator in integrators

        run_cell(problem, integrator, steps, final_time)
    end

    banner("done")
end

main(ARGS)
