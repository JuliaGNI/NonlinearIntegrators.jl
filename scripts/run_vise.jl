# Run the VISE (variational integrator with symbolic expression) experiments.
#
#   julia --project=scripts scripts/run_vise.jl [problem] [timestep]
#
# With no arguments, every problem at every time step. With arguments, just that one run — which
# is how a single figure gets regenerated without paying for the other eight.
#
#   julia --project=scripts scripts/run_vise.jl harmonic-oscillator 1.0
#
# Writes one archive per run into `runs/`, and prints the numbers the slides quote. It does not
# plot: `figures.jl` does that, from the archives, so a figure can be restyled without re-running
# a simulation. `--runs-dir` redirects the archives, `--results-dir` the figures.
#
# Each run computes four solutions of the same problem at the same macro step:
#
#   * VISE, on the symbolic ansatz of `experiments.jl`
#   * the polynomial Galerkin variational integrator, `CGVI`, at the same quadrature order —
#     the linear method VISE is the nonlinear counterpart of
#   * implicit midpoint, the second-order symplectic baseline
#   * a `Gauss(8)` reference at h/40, standing in for the exact solution
#
# and, for the harmonic oscillator only, the exact solution as well, since there is one.

include(joinpath(@__DIR__, "experiments.jl"))

# The harmonic-oscillator ansatz spans the exact solution, so the only error VISE can have there
# is the nonlinear solver's residual. That makes the run a check rather than a picture, and the
# threshold is the one the package's own unit test uses.
const HO_ACCURACY_TARGET = 1e-10

function run_vise(experiment::VISEProblem, timestep::Float64)
    banner("$(experiment.label), h = $(timestep)")

    T = experiment.final_time
    D = experiment.dimension
    prob = experiment.problem((0.0, T), timestep)
    R = experiment.quadrature[timestep]
    params = prob.parameters

    report("ansatz weights per degree of freedom", experiment.weight_count)
    report("Gauss-Legendre quadrature nodes R", R)
    report("final time", T)

    # --- VISE -----------------------------------------------------------------
    method = build_vise_method(experiment, timestep)
    t_start = time()
    sol, internal_values, weights = integrate(prob, method; VISE_SOLVER_OPTIONS...)
    vise_seconds = time() - t_start
    report("VISE wall clock", @sprintf("%.1f s for %d steps", vise_seconds, ntime(sol)))

    # --- comparisons ----------------------------------------------------------
    cgvi_sol = integrate(prob, galerkin_method(R))
    imp_sol = integrate(prob, ImplicitMidpoint())
    ref_sol = reference_solution(experiment.problem, T, timestep)

    # --- diagnostics ----------------------------------------------------------
    vise_error = relative_invariant_error(sol, experiment.hamiltonian, params)
    cgvi_error = relative_invariant_error(cgvi_sol, experiment.hamiltonian, params)
    imp_error = relative_invariant_error(imp_sol, experiment.hamiltonian, params)

    report_error("max |ΔH/H₀|, VISE", maximum(vise_error))
    report_error("max |ΔH/H₀|, $(galerkin_label(R))", maximum(cgvi_error))
    report_error("max |ΔH/H₀|, implicit midpoint", maximum(imp_error))

    vise_accuracy = coarse_grid_error(sol, ref_sol, REFERENCE_SUBSTEPS)
    cgvi_accuracy = coarse_grid_error(cgvi_sol, ref_sol, REFERENCE_SUBSTEPS)
    report_error("rel. max error vs reference, VISE", vise_accuracy)
    report_error("rel. max error vs reference, $(galerkin_label(R))", cgvi_accuracy)
    report_error("rel. max error vs reference, midpoint",
        coarse_grid_error(imp_sol, ref_sol, REFERENCE_SUBSTEPS))

    # The harmonic oscillator is the one case with a closed-form solution, so it is the one case
    # where the claim can be checked rather than illustrated. Asserted, not printed: a run that
    # silently stopped reproducing the exact solution would otherwise go into a figure.
    exact_sol = nothing
    if experiment.name == "harmonic-oscillator"
        exact_sol = HO.exact_solution(
            HO.podeproblem(; timespan = (0.0, T), timestep = timestep / REFERENCE_SUBSTEPS))
        accuracy = coarse_grid_error(sol, exact_sol, REFERENCE_SUBSTEPS)
        report_error("rel. max error vs exact, VISE", accuracy)
        accuracy < HO_ACCURACY_TARGET || error(
            "the harmonic-oscillator ansatz spans the exact solution, so VISE must reproduce " *
            "it to the residual floor; got $(accuracy) at h = $(timestep), target " *
            "$(HO_ACCURACY_TARGET).")
    end

    # --- archive --------------------------------------------------------------
    t, q, p = solution_data(sol, D)
    ref_t, ref_q, ref_p = solution_data(ref_sol, D)

    continuous = [continuous_solution(internal_values, timestep; dof = d) for d in 1:D]

    # The continuous solution is what a Galerkin variational integrator gives for free, so the run
    # reports that it is there and how long it is rather than only archiving it.
    report("continuous solution points", length(first(first(continuous))))

    comparisons = Dict{String, Any}()
    for (clabel, csol) in [galerkin_label(R) => cgvi_sol, "Implicit midpoint" => imp_sol]
        ct, cq, cp = solution_data(csol, D)
        comparisons[clabel] = Dict{String, Any}(
            "t" => ct, "q" => cq, "p" => cp,
            "hamiltonian_error" => relative_invariant_error(csol, experiment.hamiltonian,
                params))
    end

    data = Dict{String, Any}(
        # `kind` is what lets `figures.jl` glob `runs/` and know what shape of figure this draws,
        # without consulting a registry that can drift from the stems actually written.
        "kind" => "solution",
        "problem" => experiment.name,
        # `label` names the *curve*, `problem_label` the run. The first render of these figures
        # legended the VISE trace as "Harmonic oscillator", which says nothing — every curve in the
        # panel is the harmonic oscillator.
        "label" => "VISE",
        "problem_label" => experiment.label,
        "timestep" => timestep,
        "final_time" => T,
        "dimension" => D,
        "quadrature_nodes" => R,
        "vise_seconds" => vise_seconds,
        "t" => t,
        "q" => q,
        "p" => p,
        # One shared grid — every component is recorded on the same sub-step grid — and one
        # series per component.
        "continuous_t" => first(first(continuous)),
        "continuous_q" => [last(c) for c in continuous],
        "hamiltonian_error" => vise_error,
        "max_hamiltonian_error" => maximum(vise_error),
        "reference_error" => vise_accuracy,
        "reference_t" => ref_t,
        "reference_q" => ref_q,
        "reference_p" => ref_p,
        "comparisons" => comparisons,
        # The converged ansatz weights of every step. Not plotted, but this is what makes the run
        # inspectable: a weight that walks away from `init_w` is how a failure to extrapolate
        # shows up before it reaches the figure.
        "weights" => weights
    )

    if exact_sol !== nothing
        et, eq, ep = solution_data(exact_sol, D)
        data["exact_t"] = et
        data["exact_q"] = eq
        data["exact_p"] = ep
    end

    stem = figure_stem(experiment.name, "vise", timestep)
    report_path("archive", store_run!(stem, data))

    return (problem = experiment.name, timestep = timestep, R = R,
        vise_dh = maximum(vise_error), vise_err = vise_accuracy,
        cgvi_dh = maximum(cgvi_error), cgvi_err = cgvi_accuracy,
        fine = length(first(first(continuous))))
end

# The per-run blocks above report these same numbers, but a reader comparing VISE against the
# linear methods across time steps wants them in one grid rather than spread over nine screens.
#
# The Hénon–Heiles rows are included precisely because they do *not* work: a three-term ansatz per
# coordinate does not span that trajectory, and the relative error is O(1) well before the end of
# the run. The original sweep recorded the same thing, so this is the ansatz being too small rather
# than a regression, and the row keeps that visible.
function summary_table(rows)
    banner("Summary")
    @printf("  %-20s %-6s %4s  %10s %10s  %10s %10s  %6s\n",
        "problem", "h", "R", "VISE ΔH", "VISE err", "CGVI ΔH", "CGVI err", "n fine")
    println("  ", repeat("-", 88))
    for r in rows
        @printf("  %-20s %-6s %4d  %10.3e %10.3e  %10.3e %10.3e  %6d\n",
            r.problem, r.timestep, r.R, r.vise_dh, r.vise_err, r.cgvi_dh, r.cgvi_err,
            r.fine)
    end
end

function main(args)
    names, _ = parse_arguments(args)
    # A third positional would be silently ignored, and this script exists to be called from a
    # regeneration recipe — the same reason the shared parser rejects an unrecognised flag.
    length(names) ≤ 2 || throw(ArgumentError(
        "this script takes a problem name and a time step at most, got $(join(names, ", "))"))
    experiments = isempty(names) ? VISE_PROBLEMS : (vise_problem(names[1]),)

    rows = NamedTuple[]
    for experiment in experiments
        # Per problem, because `h = 10` is only run where it was measured to work — see
        # `VISE_EXTRA_STEPS`.
        steps = length(names) < 2 ? vise_steps(experiment) : (parse(Float64, names[2]),)
        for timestep in steps
            push!(rows, run_vise(experiment, timestep))
        end
    end

    summary_table(rows)
    banner("done")
end

main(ARGS)
