# Running one method on one case of the hard-problems benchmark, shared by run_baselines.jl and
# run_nvi.jl: a step loop that counts Newton iterations and solver failures, the timing, and the
# errors of PLAN.md §3.

include(joinpath(@__DIR__, "setup.jl"))

import GeometricIntegratorsBase as GIB
using LinearAlgebra: SingularException
using Logging
using Printf
using QuadratureRules
using SimpleSolvers: NonlinearSolverException

const CBF = GeometricIntegrators.Integrators.CompactBasisFunctions

const RESULTS_DIR = joinpath(@__DIR__, "results")

"CGVI(P_s) with R = s + 1 Gauss–Legendre points, Lagrange basis on those points."
function cgvi(s)
    quad = GaussLegendreQuadrature(Float64, s + 1)
    CGVI(CBF.Lagrange(QuadratureRules.nodes(quad)), quad)
end

# ---- convergence -------------------------------------------------------------------------------

# A step converged iff the ∞-norm of the residual of its nonlinear system at the accepted iterate
# is at most RES_TOL. The solver's own warnings are no criterion: it accepts steps with residual
# O(1) through its relative tolerance without a warning (Gauss(2) on P3, e = 0.9, h = 0.3), and
# DogLeg warns about an underflowing trust region at residuals of 2e-15.
const RES_TOL = 1E-10

"Discards all log records of a run, and counts those at level ≥ Warn."
mutable struct WarnCounter <: AbstractLogger
    count::Int
end

WarnCounter() = WarnCounter(0)

Logging.min_enabled_level(::WarnCounter) = Warn
Logging.shouldlog(::WarnCounter, args...) = true
Logging.catch_exceptions(::WarnCounter) = false
Logging.handle_message(l::WarnCounter, args...; kwargs...) = (l.count += 1; nothing)

# ---- integration -------------------------------------------------------------------------------

"""
    integrate_counting(problem, method; record = false, kwargs...)
        -> (; sol, status, steps, unconverged, max_res, iterations, x)

Integrate step by step as `GeometricIntegratorsBase.integrate!` does, summing the Newton
iterations of every step, counting the steps whose residual exceeds RES_TOL and keeping the
largest residual `max_res`; `kwargs` are passed on to `GeometricIntegrator`. A step whose
nonlinear or linear solve throws ends the run (status `failed:<exception type>`), as does a
non-finite state (status `nonfinite`); the solution is valid up to the step before, and `steps`
counts the steps taken. With `record = true`, `x` holds a copy of the nonlinear solution of
every step.
"""
function integrate_counting(problem, method; record = false, kwargs...)
    int = GeometricIntegrator(problem, method; kwargs...)
    sol = GeometricSolution(problem)
    solstep = GIB.solutionstep(int, sol[0])
    state = GIB.current(solstep)
    iterations = 0
    unconverged = 0
    max_res = 0.0
    x = Vector{Float64}[]
    for n in 1:GeometricSolutions.ntime(sol)
        GIB.reset!(solstep, GeometricSolutions.timesteps(sol)[n])
        try
            GIB.integrate!(solstep, int)
        catch e
            e isa Union{NonlinearSolverException, SingularException, DomainError} || rethrow()
            return (; sol, status = "failed:$(nameof(typeof(e)))", steps = n, unconverged,
                max_res, iterations, x)
        end
        iterations += GIB.solverstate(int).iterations
        res = maximum(abs, GIB.solverstate(int).y)
        res > RES_TOL && (unconverged += 1)
        max_res = max(max_res, res)
        record && push!(x, copy(GIB.nlsolution(int)))
        copy!(sol, state, n)
        isnan(state) && return (; sol, status = "nonfinite", steps = n, unconverged, max_res,
            iterations, x)
    end
    (; sol, status = "ok", steps = GeometricSolutions.ntime(sol), unconverged, max_res,
        iterations, x)
end

# Runs shorter than this are repeated and timed by the minimum over REPEATS runs.
const SHORT_RUN = 0.1
const REPEATS = 3

"""
    run_one(case, ref, h, m, method; record = false, kwargs...)

Integrate `case` over (0, T) with the time step h / m and evaluate on the macro grid n h:
status, unconverged steps, largest step residual, solver warnings, q error against `ref` on t ≤ Terr, invariant
errors, Newton iterations, wall time; plus the macro-grid positions `Q` and, with `record`, the nonlinear
solutions `x` of every step.
"""
function run_one(case, ref, h, m, method; record = false, kwargs...)
    problem = case.build((0.0, case.T), h / m)
    logger = WarnCounter()
    GC.gc()
    secs = @elapsed r = with_logger(logger) do
        integrate_counting(problem, method; record, kwargs...)
    end
    if secs < SHORT_RUN
        secs = minimum(1:REPEATS) do _
            @elapsed with_logger(() -> integrate_counting(problem, method; kwargs...), WarnCounter())
        end
    end
    status = r.status == "ok" && r.unconverged > 0 ? "unconverged" : r.status

    # evaluated on the macro grid n h, n = 0..N; a run that did not finish gets NaN errors
    t = times(r.sol)[1:m:end]
    Q = qmatrix(r.sol)[:, 1:m:end]
    P = pmatrix(r.sol)[:, 1:m:end]
    nerr = findlast(≤(case.Terr + 1E-9), t)
    done = status in ("ok", "unconverged")
    q_err = done ? relerr(Q[:, 1:nerr], ref(t[1:nerr])) : NaN
    inv = done ? invariant_errors(case, t, Q, P, problem.parameters) :
          map(_ -> NaN, keys(case.invariants))

    (; r.steps, status, r.unconverged, r.max_res, warnings = logger.count, q_err, inv,
        r.iterations, secs, Q, r.x)
end

"""
Compile `method` on two-step versions of the case with the smallest and the largest step, so
that `secs` excludes compilation, also of the warning and failure paths.
"""
function warmup(case, m, method; kwargs...)
    for h in extrema(step_sizes(case))
        with_logger(WarnCounter()) do
            integrate_counting(case.build((0.0, 2h), h / m), method; kwargs...)
        end
    end
end

# ---- output ------------------------------------------------------------------------------------

csvnum(x) = isnan(x) ? "NaN" : @sprintf("%.6e", x)

"Reference of `case`, written to results/<case>_reference.txt."
function write_reference(case)
    secs = @elapsed ref = Reference(case)
    open(joinpath(RESULTS_DIR, "$(case.name)_reference.txt"), "w") do io
        println(io, "reference = ", ref.exact === nothing ? "Gauss(8)" : "exact")
        println(io, "dt = ", ref.dt)
        println(io, "selfcheck = ", ref.selfcheck)
        println(io, "secs = ", secs)
    end
    @printf("reference: %s, dt = %g, |dt - dt/2| = %.2e (%.1f s)\n",
        ref.exact === nothing ? "Gauss(8)" : "exact", ref.dt, ref.selfcheck, secs)
    ref
end

"Cases selected by the command line: `quick` (default) or `full`, optionally `--cases=a,b`."
function select_cases(args)
    mode = isempty(args) || startswith(args[1], "--") ? "quick" : args[1]
    mode in ("quick", "full") || error("mode must be quick or full, got $(mode)")
    sel = findfirst(a -> startswith(a, "--cases="), args)
    sel === nothing && return mode == "quick" ? filter(c -> c.quick, CASES) : CASES
    names = split(split(args[sel], "=")[2], ",")
    filter(c -> c.name in names, CASES)
end
