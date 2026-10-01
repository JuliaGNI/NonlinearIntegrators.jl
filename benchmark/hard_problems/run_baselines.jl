# Linear baselines of the hard-problems benchmark (PLAN.md §2, phase 1):
#
#   L1  CGVI(P_s, R = s + 1), s = 2..6, with m = 1..4 substeps per step h
#   L2  Gauss(s), s = 1..6
#
#   julia --project=benchmark benchmark/hard_problems/run_baselines.jl            # quick
#   julia --project=benchmark benchmark/hard_problems/run_baselines.jl full
#   julia --project=benchmark benchmark/hard_problems/run_baselines.jl full --cases=P1_eps0.001
#
# For every case and step h = c / Ω, c ∈ STEP_FACTORS, each method integrates (0, T) with the
# time step h / m, and is evaluated on the grid n h. Recorded per run:
#   steps       number of time steps taken
#   status      ok | unconverged (a solve gave up, the run went on) | failed:<exception type> |
#               nonfinite
#   floor       solves that stopped at the round-off floor of their residual (converged)
#   unconverged solves that gave up (see `WarnCounter`)
#   q_err       (also for unconverged runs) max_n |q_n - q_ref(t_n)|∞ / max_n |q_ref(t_n)|∞ over t_n ≤ Terr
#   <invariant> see `invariant_errors` in setup.jl
#   iterations  Newton iterations summed over all steps
#   secs        wall time of the integration, after a warm-up of the same method; the minimum
#               of REPEATS runs for runs shorter than SHORT_RUN
#
# Writes hard_problems/results/<case>_baselines.csv and <case>_reference.txt.

include(joinpath(@__DIR__, "setup.jl"))

import GeometricIntegratorsBase as GIB
using LinearAlgebra: SingularException
using Logging
using Printf
using QuadratureRules
using SimpleSolvers: NonlinearSolverException

const CBF = GeometricIntegrators.Integrators.CompactBasisFunctions

const RESULTS_DIR = joinpath(@__DIR__, "results")

# ---- methods -----------------------------------------------------------------------------------

function cgvi(s)
    quad = GaussLegendreQuadrature(Float64, s + 1)
    CGVI(CBF.Lagrange(QuadratureRules.nodes(quad)), quad)
end

const METHODS = vcat(
    [(family = "CGVI", s = s, m = m, method = cgvi(s)) for s in 2:6 for m in 1:4],
    [(family = "Gauss", s = s, m = 1, method = Gauss(s)) for s in 1:6])

# ---- integration -------------------------------------------------------------------------------

"""
Counts the warnings of SimpleSolvers during a run and discards all records. A solve that stops
at the round-off floor of its residual ("... achievable floor ...") has converged; every other
SimpleSolvers warning (a solve that gave up) counts as `unconverged`.
"""
mutable struct WarnCounter <: AbstractLogger
    floor::Int
    unconverged::Int
end

WarnCounter() = WarnCounter(0, 0)

Logging.min_enabled_level(::WarnCounter) = Warn
Logging.shouldlog(::WarnCounter, args...) = true
Logging.catch_exceptions(::WarnCounter) = false
function Logging.handle_message(l::WarnCounter, level, message, _module, args...; kwargs...)
    nameof(_module) === :SimpleSolvers || return nothing
    occursin("achievable floor", string(message)) ? (l.floor += 1) : (l.unconverged += 1)
    nothing
end

"""
    integrate_counting(problem, method) -> (; sol, status, steps, iterations)

Integrate step by step as `GeometricIntegratorsBase.integrate!` does, summing the Newton
iterations of every step. A step whose nonlinear or linear solve throws ends the run (status
`failed:<exception type>`), as does a non-finite state (status `nonfinite`); the solution is
valid up to the step before, and `steps` counts the steps taken.
"""
function integrate_counting(problem, method)
    int = GeometricIntegrator(problem, method)
    sol = GeometricSolution(problem)
    solstep = GIB.solutionstep(int, sol[0])
    state = GIB.current(solstep)
    iterations = 0
    for n in 1:GeometricSolutions.ntime(sol)
        GIB.reset!(solstep, GeometricSolutions.timesteps(sol)[n])
        try
            GIB.integrate!(solstep, int)
        catch e
            e isa Union{NonlinearSolverException, SingularException, DomainError} || rethrow()
            return (; sol, status = "failed:$(nameof(typeof(e)))", steps = n, iterations)
        end
        iterations += GIB.solverstate(int).iterations
        copy!(sol, state, n)
        isnan(state) && return (; sol, status = "nonfinite", steps = n, iterations)
    end
    (; sol, status = "ok", steps = GeometricSolutions.ntime(sol), iterations)
end

# Runs shorter than this are repeated and timed by the minimum over REPEATS runs.
const SHORT_RUN = 0.1
const REPEATS = 3

function run_one(case, ref, h, M)
    problem = case.build((0.0, case.T), h / M.m)
    logger = WarnCounter()
    GC.gc()
    secs = @elapsed r = with_logger(() -> integrate_counting(problem, M.method), logger)
    if secs < SHORT_RUN
        secs = minimum(1:REPEATS) do _
            @elapsed with_logger(() -> integrate_counting(problem, M.method), WarnCounter())
        end
    end
    status = r.status == "ok" && logger.unconverged > 0 ? "unconverged" : r.status

    # evaluated on the macro grid n h, n = 0..N; a run that did not finish gets NaN errors
    t = times(r.sol)[1:(M.m):end]
    Q = qmatrix(r.sol)[:, 1:(M.m):end]
    P = pmatrix(r.sol)[:, 1:(M.m):end]
    nerr = findlast(≤(case.Terr + 1E-9), t)
    done = status in ("ok", "unconverged")
    q_err = done ? relerr(Q[:, 1:nerr], ref(t[1:nerr])) : NaN
    inv = done ? invariant_errors(case, t, Q, P, problem.parameters) :
          map(_ -> NaN, keys(case.invariants))

    (; r.steps, status, floor = logger.floor,
        unconverged = logger.unconverged, q_err, inv, r.iterations, secs)
end

"""
Compile every method on two-step versions of the case with the smallest and the largest step,
so that `secs` excludes compilation, also of the warning and failure paths.
"""
function warmup(case)
    for h in extrema(step_sizes(case)), M in METHODS
        with_logger(WarnCounter()) do
            integrate_counting(case.build((0.0, 2h), h / M.m), M.method)
        end
    end
end

# ---- sweep -------------------------------------------------------------------------------------

csvnum(x) = isnan(x) ? "NaN" : @sprintf("%.6e", x)

function run_case(case)
    println("\n── $(case.name): T = $(case.T), Terr = $(case.Terr), h Ω ∈ $(STEP_FACTORS)")
    ref_secs = @elapsed ref = Reference(case)
    open(joinpath(RESULTS_DIR, "$(case.name)_reference.txt"), "w") do io
        println(io, "reference = ", ref.exact === nothing ? "Gauss(8)" : "exact")
        println(io, "dt = ", ref.dt)
        println(io, "selfcheck = ", ref.selfcheck)
        println(io, "secs = ", ref_secs)
    end
    @printf("reference: %s, dt = %g, |dt - dt/2| = %.2e (%.1f s)\n",
        ref.exact === nothing ? "Gauss(8)" : "exact", ref.dt, ref.selfcheck, ref_secs)

    warmup(case)

    invnames = join(string.(keys(case.invariants)), ",")
    open(joinpath(RESULTS_DIR, "$(case.name)_baselines.csv"), "w") do io
        println(io, "case,c,h,family,s,m,steps,status,floor,unconverged,q_err,$(invnames),iterations,secs")
        for c in STEP_FACTORS, M in METHODS
            h = c / case.Ω
            r = run_one(case, ref, h, M)
            println(io, join((case.name, c, h, M.family, M.s, M.m, r.steps, r.status, r.floor, r.unconverged,
                    csvnum(r.q_err), csvnum.(r.inv)..., r.iterations, csvnum(r.secs)), ","))
            flush(io)
            @printf("  c = %-4g %-5s s = %d m = %d  %-9s q_err = %.2e  %s  %.2f s\n", c, M.family,
                M.s, M.m, r.status, r.q_err,
                join(("$k = $(@sprintf("%.2e", v))" for (k, v) in zip(keys(case.invariants), r.inv)), "  "),
                r.secs)
        end
    end
end

function main(args)
    mode = isempty(args) || startswith(args[1], "--") ? "quick" : args[1]
    mode in ("quick", "full") || error("mode must be quick or full, got $(mode)")
    sel = findfirst(a -> startswith(a, "--cases="), args)
    cases = mode == "quick" ? filter(c -> c.quick, CASES) : CASES
    sel === nothing || (names = split(split(args[sel], "=")[2], ",");
                        cases = filter(c -> c.name in names, CASES))
    mkpath(RESULTS_DIR)
    foreach(run_case, cases)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
