# Tier B: end-to-end OGA sweep on the harmonic oscillator.
#
# Where Tier A (`oga_fit_study.jl`) measures the seed in isolation, this measures what the
# integrator actually does with it: OGA variant × precision × regularization factor ×
# activation, integrated for ten steps with the Newton solve in the loop.
#
# Two stages, matching the two questions:
#
#   B1 — the reduced-precision question: ReLUᵏ for k = 1…4, where the ±1
#        dictionary is theoretically complete, so anything that goes wrong is numerical.
#   B2 — the activation question: ELU, GELU and tanh, which are not positively
#        homogeneous, against the 2-D and angular dictionaries built for them.
#
# The λ ladder is swept as multiples of `√eps(T)` (see `oga_activations.jl`): scaling the
# Jacobian-diagonal shift to the precision it protects is what makes the factors comparable
# across `Float16`/`Float32`/`Float64` at all.
#
# Run with:
#   julia --project=scripts scripts/oga_sweep.jl          # both stages
#   julia --project=scripts scripts/oga_sweep.jl relu     # B1 only
#   julia --project=scripts scripts/oga_sweep.jl smooth   # B2 only

using NonlinearIntegrators
using GeometricIntegrators
using GeometricIntegratorsBase
using GeometricProblems.HarmonicOscillator
import GeometricIntegratorsBase: solverstate
import SimpleSolvers
using QuadratureRules
using LinearAlgebra: SingularException
using Printf

include(joinpath(@__DIR__, "oga_activations.jl"))
include(joinpath(@__DIR__, "oga_report.jl"))

const TYPES = (Float16, Float32, Float64)
const S_NEURONS = 4
const R_QUAD = 8
const DT = 0.1
const NSTEPS = 10
const DICT_AMOUNT = 400
# The iteration budget dominates wall-clock: a converging case needs 2–3 iterations, so
# essentially all the runtime is spent on the cases that exhaust the budget before failing.
# 1000 matches what the nonlinear studies elsewhere use; lower it to trade sweep breadth for
# turnaround, at the cost of relabelling slow-but-eventually-converging cases as `maxiter`.
const MAXIT = 1000

# ---- the seed variants ------------------------------------------------------
#
# `bias_amount` is trimmed on the richer dictionaries so every variant scans a
# comparable number of atoms; the greedy step is linear in that count, so holding it fixed
# is what makes the comparison about the dictionary's *shape* rather than its size.
const SEEDS_1D = [
    ("reference", OGA1dNormalEquations()),
    ("oga1d", OGA1d()),
    ("oga1d-stable", OGA1dStable()),
    ("oga1d-tsvd", OGA(BiasGrid1d(), OrthogonalProjection(), TruncatedSVD())),
    ("oga1d-pivqr", OGA(BiasGrid1d(), OrthogonalProjection(), PivotedQR())),
    ("oga1d-refined", OGA(Refined(BiasGrid1d()), NormalizedProjection(), IncrementalQR()))
]

const SEEDS_2D = [
    ("oga1d", OGA1d()),
    ("oga1d-stable", OGA1dStable()),
    ("oga2d",
        OGA2d(dictionary = WeightBiasGrid2d(octaves = (-3, 3), weight_amount = 6,
            bias_amount = 56))),
    ("oga-sphere",
        OGASphere(dictionary = AngularGrid(radii = (0.25, 1.0, 4.0), amount = 266))),
    ("oga2d-refined",
        OGA(
            Refined(WeightBiasGrid2d(octaves = (-3, 3), weight_amount = 6,
                bias_amount = 56)),
            NormalizedProjection(), IncrementalQR()))
]

# ---- one case ---------------------------------------------------------------

function classify(e)
    e isa SingularException ? "singular" :
    (n = string(nameof(typeof(e)));
        occursin("NonlinearSolver", n) ? "diverged" : first(n, 14))
end

function reference_q(::Type{T}, params) where {T}
    # The analytic solution at the end of the horizon, in Float64 — the accuracy yardstick.
    return Float64(HarmonicOscillator.exact_solution_q(
        T(NSTEPS * DT), T(0.5), T(0.0), T(0.0), params))
end

function run_case(basis, ::Type{T}, seed, λ, params, prob, refq) where {T}
    method = ShallowNet(basis, QuadratureRules.GaussLegendreQuadrature(T, R_QUAD);
        show_status = false,
        bias_interval = [-T(pi), T(pi)], dict_amount = DICT_AMOUNT,
        initial_guess_method = seed)
    status, ref_err, iters, secs = "ok", NaN, NaN, NaN
    upcast = false
    # Timed outside the `try` so a *failing* case still reports its cost. Most of this
    # sweep's wall-clock is spent in runs that exhaust the iteration budget and then throw,
    # and with `@elapsed` inside the `try` that time was invisible in the report.
    t0 = time()
    try
        int = GeometricIntegrator(prob, method; regularization_factor = T(λ),
            max_iterations = MAXIT,
            f_abstol = oga_f_abstol(T))
        local sol
        sol, _ = integrate(int)
        # The iteration count of the *final* step — the integrator keeps no per-step history.
        # Enough to catch a run that stalls, which is what the status below uses it for.
        try
            iters = Float64(solverstate(int).iterations)
        catch exception
            # `iters` stays `NaN`: not every solver state carries an iteration count, and no row
            # of this sweep depends on it. An interrupt is still an interrupt, though.
            exception isa InterruptException && rethrow()
        end
        qend = collect(sol.q[:, 1])[end]
        # The precision invariant: a run started at `T` must still be at `T` at the end. A
        # silent upcast would make the reduced-precision rows meaningless, so it is recorded
        # as its own status rather than folded into "ok".
        upcast = !(eltype(sol.q[end]) === T)
        if upcast
            status = "upcast"
        elseif !isfinite(Float64(qend))
            status = "nonfinite"
        elseif !isnan(iters) && iters ≥ MAXIT
            # The Newton solve exhausted its iteration budget. It still returns a finite
            # state, so without this check the case would be recorded as converged — the
            # documented hazard of a relaxed tolerance letting a finite-but-poor result
            # through. The accuracy is still recorded, so a stalled run can be told apart
            # from a diverged one.
            status = "maxiter"
            ref_err = abs(Float64(qend) - refq)
        else
            ref_err = abs(Float64(qend) - refq)
        end
    catch e
        status = classify(e)
    finally
        secs = time() - t0
    end
    return (; status, ref_err, iters, secs)
end

const CSV_HEADER = "study,problem,T,dt,S,R,activation,seed,lambda_multiple,lambda,status," *
                   "ref_err,iterations,secs"

function run_stage(name::AbstractString, seeds, activations)
    mkpath(RUNS_DIR[])
    csvpath = joinpath(RUNS_DIR[], "$(name).csv")

    total = length(TYPES) * length(activations) * length(seeds) * (1 + 6)
    println("="^104)
    println("Tier B — $(name): $total end-to-end runs ($(NSTEPS) steps, dt=$(DT), S=$(S_NEURONS), R=$(R_QUAD))")
    println("="^104)
    @printf("%-8s %-8s %-15s %-7s %-10s | %-10s %-11s %-6s %-7s\n",
        "T", "act", "seed", "λ/√eps", "λ", "status", "ref_err", "iters", "secs")
    println("-"^104)

    open(csvpath, "w") do io
        println(io, CSV_HEADER)
        flush(io)
        for T in TYPES
            params = HarmonicOscillator.default_parameters(T)
            prob = HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)];
                timespan = (T(0), T(NSTEPS * DT)), timestep = T(DT), parameters = params)
            refq = reference_q(T, params)
            ladder = oga_reg_ladder(T)

            for (aname, σ) in activations
                # The symbolic network build is the expensive part; amortise it over every
                # seed and regularization factor for this (T, activation).
                basis = try
                    ShallowNetBasis{T}(σ, S_NEURONS)
                catch e
                    @warn "basis build failed" T aname exception = e
                    continue
                end

                for (sname, seed) in seeds, l in ladder

                    r = run_case(basis, T, seed, l.factor, params, prob, refq)
                    @printf("%-8s %-8s %-15s %-7d %-10s | %-10s %-11s %-6s %-7s\n",
                        string(T), aname, sname, l.multiple,
                        @sprintf("%.2e", Float64(l.factor)), r.status,
                        isfinite(r.ref_err) ? @sprintf("%.3e", r.ref_err) : "—",
                        isnan(r.iters) ? "—" : string(round(Int, r.iters)),
                        isnan(r.secs) ? "—" : @sprintf("%.2f", r.secs))
                    println(io,
                        join(
                            ("sweep", "harmonic_oscillator", string(T), csvnum(DT),
                                string(S_NEURONS), string(R_QUAD), aname, sname,
                                string(l.multiple), csvnum(Float64(l.factor)), r.status,
                                csvnum(r.ref_err), csvnum(r.iters), csvnum(r.secs)),
                            ","))
                    flush(io)
                end
            end
        end
    end
    println("-"^104)
    println("Wrote $(csvpath)")
    write_sweep_report(csvpath, name)
    return csvpath
end

function main(args)
    names, _ = parse_arguments(args)
    length(names) ≤ 1 || throw(ArgumentError(
        "this script takes one mode at most, got $(join(names, ", "))"))
    mode = isempty(names) ? "all" : first(names)
    mode in ("all", "relu", "smooth") ||
        error("unknown mode $(repr(mode)); use \"all\", \"relu\" or \"smooth\"")
    mode in ("all", "relu") &&
        run_stage("oga_sweep_relu", SEEDS_1D, OGA_ACTIVATIONS_RELU)
    mode in ("all", "smooth") &&
        run_stage("oga_sweep_smooth", SEEDS_2D, OGA_ACTIVATIONS_SMOOTH)
end

main(ARGS)
