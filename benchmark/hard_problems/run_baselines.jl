# Linear baselines of the hard-problems benchmark (PLAN.md §2, phase 1):
#
#   L1  CGVI(P_s, R = s + 1), s = 2..6, with m = 1..4 substeps per step h
#   L2  Gauss(s), s = 1..6
#
#   julia --project=benchmark benchmark/hard_problems/run_baselines.jl            # quick
#   julia --project=benchmark benchmark/hard_problems/run_baselines.jl full
#   julia --project=benchmark benchmark/hard_problems/run_baselines.jl full --cases=FrequencyModulatedOscillator_eps0.001
#
# For every case and step h = c / Ω, c ∈ STEP_FACTORS, each method integrates (0, T) with the
# time step h / m, and is evaluated on the grid n h. Recorded per run:
#   steps       number of time steps taken
#   status      ok | unconverged (some step has a residual > RES_TOL, the run went on) |
#               failed:<exception type> | nonfinite
#   unconverged steps whose residual ∞-norm exceeds RES_TOL = 1e-10 (see integration.jl)
#   max_res     largest residual ∞-norm of all steps
#   warnings    warnings logged by the solver (no convergence criterion, see integration.jl)
#   q_err       (also for unconverged runs) max_n |q_n - q_ref(t_n)|∞ / max_n |q_ref(t_n)|∞ over t_n ≤ Terr
#   <invariant> see `invariant_errors` in setup.jl
#   iterations  Newton iterations summed over all steps
#   secs        wall time of the integration, after a warm-up of the same method; the minimum
#               of REPEATS runs for runs shorter than SHORT_RUN
#
# Writes hard_problems/results/<case>_baselines.csv and <case>_reference.txt.

include(joinpath(@__DIR__, "integration.jl"))

const METHODS = vcat(
    [(family = "CGVI", s = s, m = m, method = cgvi(s)) for s in 2:6 for m in 1:4],
    [(family = "Gauss", s = s, m = 1, method = Gauss(s)) for s in 1:6])

function run_case(case)
    println("\n── $(case.name): T = $(case.T), Terr = $(case.Terr), h Ω ∈ $(STEP_FACTORS)")
    ref = write_reference(case)
    foreach(M -> warmup(case, M.m, M.method), METHODS)

    invnames = join(string.(keys(case.invariants)), ",")
    open(joinpath(RESULTS_DIR, "$(case.name)_baselines.csv"), "w") do io
        println(io, "case,c,h,family,s,m,steps,status,unconverged,max_res,warnings,q_err,$(invnames),iterations,secs")
        for c in STEP_FACTORS, M in METHODS
            h = c / case.Ω
            r = run_one(case, ref, h, M.m, M.method)
            println(io, join((case.name, c, h, M.family, M.s, M.m, r.steps, r.status, r.unconverged,
                    csvnum(r.max_res), r.warnings, csvnum(r.q_err), csvnum.(r.inv)..., r.iterations,
                    csvnum(r.secs)), ","))
            flush(io)
            @printf("  c = %-4g %-5s s = %d m = %d  %-11s q_err = %.2e  %.2f s\n", c, M.family,
                M.s, M.m, r.status, r.q_err, r.secs)
        end
    end
end

main(args) = (mkpath(RESULTS_DIR); foreach(run_case, select_cases(args)))

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
