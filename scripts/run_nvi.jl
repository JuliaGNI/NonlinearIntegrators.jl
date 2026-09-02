# Run the NVI (neural variational integrator) experiments — the shallow-network runs.
#
#   julia --project=scripts scripts/run_nvi.jl [stem]... [--runs-dir dir]
#
# With no arguments, every run in `NVI_RUNS`. With arguments, just those, named by figure stem:
#
#   julia --project=scripts scripts/run_nvi.jl harmonic-oscillator-S4R8Q16relu3-h1.0
#
# Every configuration runs at every step in `VISE_STEPS` — 1, 2 and 5 — so the neural and the
# symbolic integrators are compared at the same time steps. The one exception is the double
# pendulum, which is singular at `h ≥ 0.5` for these initial conditions and keeps its own step; see
# `NVI_STEP_OVERRIDES` in `experiments.jl`.
#
# Cost scales with the number of *steps*, not the final time: the network's parameters are solved
# for afresh at every one, from an orthogonal-greedy seed. `h = 1` over `T = 1000` is 1000 Newton
# solves at roughly 0.01–0.2 s each depending on `S`.
#
# **These solves are not expected to converge**, and the run does not treat that as a failure. The
# nonlinear solver stalls with a residual around 1e-4, which is the stagnation the talk's appendix
# frame describes as an open problem — the neural variational integrators stagnate near 1e-3 in the
# Hamiltonian error while the polynomial Galerkin integrators converge at their nominal order.
#
# What is measured here is that Hamiltonian error: `max |ΔH/H₀|` is printed and archived, so the
# claim about the *figures* is measured rather than repeated. The solver's own residual is **not**
# captured — it exists only in the per-step warnings, which are silenced below. Archiving it would
# mean reading the solver status back out of `integrate`, and nothing here does that yet.

include(joinpath(@__DIR__, "experiments.jl"))

using Logging

function run_nvi(run::NVIRun)
    banner("$(nvi_label(run)), h = $(run.timestep), T = $(run.final_time)")

    D = nvi_dimension(run)
    prob = nvi_problem(run)
    hamiltonian = nvi_hamiltonian(run)
    params = prob.parameters
    nsteps = round(Int, run.final_time / run.timestep)

    report("architecture", run.architecture === :dense ?
                           "dense, $(run.S₁)→$(run.S)" : "shallow, S = $(run.S)")
    report("quadrature nodes R (order Q = 2R)", "$(run.R) (Q = $(nvi_order(run)))")
    report("activation", run.activation_name)
    report("time steps", nsteps)
    isempty(run.windows) || report("figure windows", run.windows)

    method = build_nvi_method(run)

    # The stalled-solve warnings are the expected outcome here, one per time step, and at 1000
    # steps they bury the output entirely. Silenced deliberately, and with them the only record of
    # the residual — the Hamiltonian error below is what this run measures. See the note at the top
    # of this file.
    t_start = time()
    sol, internal_values = Logging.with_logger(Logging.NullLogger()) do
        integrate(prob, method; NVI_SOLVER_OPTIONS...)
    end
    seconds = time() - t_start
    report("wall clock", @sprintf("%.1f s (%.2f s/step)", seconds, seconds / nsteps))

    # --- comparisons ----------------------------------------------------------
    #
    # The double-pendulum figure compares against a symplectic midpoint solve at h/20, which is
    # what the original did; the harmonic-oscillator figures compare against the exact solution
    # alone, since there is one and it is the honest reference.
    ref_substeps = run.problem == "double-pendulum" ? 20 : REFERENCE_SUBSTEPS
    ref_sol = reference_solution(
        (timespan, timestep) -> nvi_problem(run, timestep),
        run.final_time, run.timestep; substeps = ref_substeps)

    comparisons = Dict{String, Any}()

    # A comparison that cannot be computed is reported and skipped, not fatal. This is not defensive
    # padding: `CGVI(8)` on the double pendulum at `h = 0.5` raises a `SingularException`, for the
    # same reason `Gauss(8)` and `ImplicitMidpoint` do at that step — the LODE is singular there at
    # these initial conditions. The network run itself succeeds, and losing it because its *baseline*
    # could not be computed would be the wrong trade.
    function record_comparison!(label, build)
        try
            csol = Logging.with_logger(Logging.NullLogger()) do
                build()
            end
            ct, cq, cp = solution_data(csol, D)
            comparisons[label] = Dict{String, Any}("t" => ct, "q" => cq, "p" => cp,
                "hamiltonian_error" => relative_invariant_error(csol, hamiltonian, params))
        catch exception
            exception isa InterruptException && rethrow()
            report("comparison $(label)",
                "skipped — $(nameof(typeof(exception))): $(failure_message(exception))")
        end
    end

    # The polynomial Galerkin integrator at the *same* quadrature order, on the same problem at the
    # same step. This is the comparison that makes the neural figures readable against the symbolic
    # ones — `run_vise.jl` carries it too — and on the harmonic oscillator with ReLU³ it is the
    # whole story: `S4R8Q16relu3` and `CGVI(8)` are the same solution to round-off, because four
    # ReLU³ neurons span the cubics. See the talk's CHANGELOG.
    record_comparison!(galerkin_label(run.R), () -> integrate(prob, galerkin_method(run.R)))

    if run.problem == "double-pendulum"
        # A symplectic midpoint solve at the reference step, as the original figure had. It is not
        # run on the harmonic oscillator because there the exact solution is the honest reference
        # and already drawn.
        record_comparison!("Symplectic midpoint, Δt = $(run.timestep / ref_substeps)",
            () -> integrate(nvi_problem(run, run.timestep / ref_substeps), Gauss(1)))
    end

    # --- diagnostics ----------------------------------------------------------
    nvi_error = relative_invariant_error(sol, hamiltonian, params)
    report_error("max |ΔH/H₀|", maximum(nvi_error))
    report_error("rel. max error vs reference", coarse_grid_error(sol, ref_sol, ref_substeps))

    # --- archive --------------------------------------------------------------
    t, q, p = solution_data(sol, D)
    ref_t, ref_q, ref_p = solution_data(ref_sol, D)
    continuous = [continuous_solution(internal_values, run.timestep; dof = d) for d in 1:D]

    exact_fields = Dict{String, Any}()
    if run.problem == "harmonic-oscillator"
        exact = HO.exact_solution(HO.podeproblem(;
            timespan = (0.0, run.final_time),
            timestep = run.timestep / REFERENCE_SUBSTEPS))
        et, eq, ep = solution_data(exact, D)
        exact_fields["exact_t"] = et
        exact_fields["exact_q"] = eq
        exact_fields["exact_p"] = ep
    end

    data = Dict{String, Any}(
        "kind" => "solution",
        "problem" => run.problem,
        # `label` names the curve — here the model string, `S4R8Q16relu3`, which is exactly what
        # the original figures legended it with. `problem_label` names the run.
        "label" => nvi_label(run),
        "problem_label" => NVI_PROBLEM_LABELS[run.problem],
        "architecture" => String(run.architecture),
        "S" => run.S,
        "S1" => run.S₁,
        "R" => run.R,
        "windows" => run.windows,
        "quadrature_order" => nvi_order(run),
        "activation" => run.activation_name,
        "timestep" => run.timestep,
        "final_time" => run.final_time,
        "dimension" => D,
        "seconds" => seconds,
        "t" => t,
        "q" => q,
        "p" => p,
        "continuous_t" => first(first(continuous)),
        "continuous_q" => [last(c) for c in continuous],
        "hamiltonian_error" => nvi_error,
        "max_hamiltonian_error" => maximum(nvi_error),
        "reference_t" => ref_t,
        "reference_q" => ref_q,
        "reference_p" => ref_p,
        "reference_substeps" => ref_substeps,
        "comparisons" => comparisons,
        exact_fields...
    )

    report_path("archive", store_run!(nvi_stem(run), data))

    return data
end

# The list of valid stems, which is what a caller who mistyped one needs. It used to be printed by
# a check *after* the lookup — and the lookup indexed `NVI_RUNS` with the `nothing` that `findfirst`
# returns, so the throw happened a line early and the message was never reached.
function nvi_run(stem)
    index = findfirst(r -> nvi_stem(r) == stem, NVI_RUNS)
    index === nothing && throw(ArgumentError(
        "no run named `$(stem)`; the stems are\n  " * join(nvi_stem.(NVI_RUNS), "\n  ")))
    NVI_RUNS[index]
end

function main(args)
    stems, _ = parse_arguments(args)
    runs = isempty(stems) ? NVI_RUNS : map(nvi_run, stems)
    # One failing configuration must not cost the other eighteen. Reported with its exception and
    # counted, so a run that vanished is visible in the summary rather than only in the log — and
    # the script exits non-zero, because a sweep that archived nothing must not look like one that
    # archived everything.
    failed = String[]
    for run in runs
        try
            run_nvi(run)
        catch exception
            exception isa InterruptException && rethrow()
            push!(failed, nvi_stem(run))
            banner("$(nvi_stem(run)) FAILED — $(nameof(typeof(exception))): " *
                   failure_message(exception))
        end
    end

    isempty(failed) || begin
        banner("$(length(failed)) of $(length(runs)) runs failed")
        foreach(f -> println("  ", f), failed)
    end
    return isempty(failed)
end

ok = main(ARGS)
banner("done")
ok || exit(1)
