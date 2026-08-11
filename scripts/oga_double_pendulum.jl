# Tier B′: the double pendulum, at a single regularization value.
#
# This is the problem the OGA seed fails hardest on — SolverBenchmark measures 0/28
# convergence at every precision below `Float64`, and even at `Float64` only 19/28. It is
# also chaotic and two-dimensional (two coupled network fits per step), so it is far more
# expensive per case than the harmonic oscillator.
#
# Hence one λ rather than the whole ladder: the harmonic-oscillator sweep already answers
# what λ does, and re-sweeping it here would spend most of the runtime re-establishing that.
# The value is taken from `oga_sweep_relu.csv` — the factor that converged most often there —
# so it is measured rather than asserted. If that CSV is absent the script falls back to
# `16√eps(T)`, the value the package documents as its default, and says so in the report.
#
# Run with:
#   julia --project=scripts scripts/oga_sweep.jl relu     # produces the CSV this reads
#   julia --project=scripts scripts/oga_double_pendulum.jl

using NonlinearIntegrators
using GeometricIntegrators
using GeometricIntegratorsBase
using GeometricProblems.DoublePendulum
import GeometricIntegratorsBase: solverstate
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
const MAXIT = 1000

const SEEDS = [
    ("reference",     OGA1dNormalEquations()),
    ("oga1d",         OGA1d()),
    ("oga1d-stable",  OGA1dStable()),
    ("oga1d-refined", OGA(Refined(BiasGrid1d()), NormalizedProjection(), IncrementalQR())),
    ("oga2d",         OGA2d(dictionary = WeightBiasGrid2d(octaves = (-3, 3), weight_amount = 6,
                                                         bias_amount = 56))),
    ("oga-sphere",    OGASphere(dictionary = AngularGrid(radii = (0.25, 1.0, 4.0), amount = 266))),
]

const ACTIVATIONS = [("relu3", relu_k(3)), ("gelu", gelu), ("tanh", tanh)]

"""
    stable_regularization(::Type{T}) -> (multiple, factor, provenance)

The regularization factor to use, taken from the harmonic-oscillator sweep: the one with the
highest convergence count at this precision, breaking ties towards the smaller factor (less
damping). `multiple` is `λ / √eps(T)`, the readable label; `factor` is the `λ` itself.

Falls back to the documented `16√eps(T)` default when the sweep has not been run.
"""
function stable_regularization(::Type{T}) where {T}
    fallback = (16, oga_reg_factor(T, 16), "documented default 16√eps(T)")
    rows = read_oga_csv(joinpath(RESULTS_DIR, "oga_sweep_relu.csv"))
    isempty(rows) && return fallback

    sub = [r for r in rows if r["T"] == string(T)]
    isempty(sub) && return (fallback[1], fallback[2],
                            "documented default 16√eps(T) (no $T rows in the sweep)")

    best, bestn = 0, -1
    for m in sort(unique(Int[inum(r, "lambda_multiple") for r in sub]))
        m == 0 && continue                        # the λ = 0 control is not a candidate
        n = count(r -> inum(r, "lambda_multiple") == m && oga_ok(r), sub)
        n > bestn && ((best, bestn) = (m, n))
    end
    best == 0 && return fallback
    total = count(r -> inum(r, "lambda_multiple") == best, sub)
    return (best, oga_reg_factor(T, best),
            "harmonic-oscillator sweep ($bestn/$total converged)")
end

classify(e) = e isa SingularException ? "singular" :
    (n = string(nameof(typeof(e))); occursin("NonlinearSolver", n) ? "diverged" : first(n, 14))

# The problem's own default initial conditions, taken once in `Float64` and converted per
# precision. Both the reference and the per-case problems must start from *these* — passing
# `p₀ = [0, 0]` while the reference used the module default `p₀ ≈ [3.33, 7.07]` silently
# compares two different trajectories, which shows up as a `ref_err` of ≈0.37 for every run
# regardless of precision, seed or status (including a `Float64` run whose Hamiltonian drift
# was 4e-7 — the two columns disagreeing is what exposed it).
const DP_DEFAULTS = DoublePendulum.lodeproblem()
dp_q0(::Type{T}) where {T} = T.(collect(DP_DEFAULTS.ics.q))
dp_p0(::Type{T}) where {T} = T.(collect(DP_DEFAULTS.ics.p))

# No closed form for the double pendulum, so accuracy is measured against a `Float64`
# Gauss(8) run over the same horizon, plus the relative drift of the Hamiltonian.
function build_reference()
    try
        prob = DoublePendulum.lodeproblem(dp_q0(Float64), dp_p0(Float64);
                                          timespan = (0.0, NSTEPS * DT), timestep = DT / 20)
        res = integrate(prob, Gauss(8))
        return Float64.(collect(res.q[:])[end])
    catch e
        @warn "reference solve failed; accuracy will be reported as missing" exception = e
        return nothing
    end
end

function ham_drift(sol, params)
    try
        qs, ps = collect(sol.q[:]), collect(sol.p[:])
        H = Float64[Float64(DoublePendulum.hamiltonian(0, q, p, params)) for (q, p) in zip(qs, ps)]
        H0 = H[1]
        (!isfinite(H0) || H0 == 0) && return NaN
        return maximum(abs.((H .- H0) ./ H0))
    catch
        return NaN
    end
end

function run_case(basis, ::Type{T}, seed, λ, prob, refq) where {T}
    method = NonLinear_OneLayer_GML(basis, QuadratureRules.GaussLegendreQuadrature(T, R_QUAD);
                                   show_status = false,
                                   bias_interval = [-T(pi), T(pi)], dict_amount = DICT_AMOUNT,
                                   initial_guess_method = seed)
    status, ref_err, drift, iters, secs = "ok", NaN, NaN, NaN, NaN
    try
        int = GeometricIntegrator(prob, method; regularization_factor = T(λ),
                                  max_iterations = MAXIT,
                                  f_abstol = oga_f_abstol(T))
        local sol
        t0 = time()
        sol, _ = integrate(int)
        secs = time() - t0
        try; iters = Float64(solverstate(int).iterations); catch; end
        qend = Float64.(collect(sol.q[:])[end])
        if !(eltype(sol.q[end]) === T)
            status = "upcast"
        elseif any(!isfinite, qend)
            status = "nonfinite"
        else
            # A run that exhausted its iteration budget returns a finite state; recording it
            # as converged would be reporting a convergence it did not achieve. Accuracy and
            # drift are still measured, so a stall is distinguishable from a divergence.
            (!isnan(iters) && iters ≥ MAXIT) && (status = "maxiter")
            if refq !== nothing && length(refq) == length(qend)
                den = maximum(abs.(refq))
                num = maximum(abs.(qend .- refq))
                ref_err = den == 0 ? num : num / den
            end
            drift = ham_drift(sol, prob.parameters)
        end
    catch e
        status = classify(e)
    end
    return (; status, ref_err, drift, iters, secs)
end

const CSV_HEADER = "study,problem,T,dt,S,R,activation,seed,lambda_multiple,lambda,status," *
                   "ref_err,ham_drift,iterations,secs"

function main()
    mkpath(RESULTS_DIR)
    csvpath = joinpath(RESULTS_DIR, "oga_double_pendulum.csv")
    refq = build_reference()

    total = length(TYPES) * length(ACTIVATIONS) * length(SEEDS)
    println("="^104)
    println("Tier B′ — double pendulum at a single λ: $total runs ($NSTEPS steps, dt=$DT)")
    for T in TYPES
        multiple, factor, why = stable_regularization(T)
        @printf("  %-8s λ = %d√eps(T) = %.3e   [%s]\n", string(T), multiple,
                Float64(factor), why)
    end
    println("="^104)
    @printf("%-8s %-8s %-15s | %-10s %-11s %-11s %-6s %-7s\n",
            "T", "act", "seed", "status", "ref_err", "ham_drift", "iters", "secs")
    println("-"^104)

    open(csvpath, "w") do io
        println(io, CSV_HEADER)
        flush(io)
        for T in TYPES
            multiple, λ, _ = stable_regularization(T)
            # The parameters must be requested at `T`: `lodeproblem`'s default set is
            # `Float64`, and Float64 lengths and masses in the Lagrangian would promote the
            # whole solve, so every reduced-precision row would come back as an upcast.
            prob = DoublePendulum.lodeproblem(dp_q0(T), dp_p0(T);
                timespan = (T(0), T(NSTEPS * DT)), timestep = T(DT),
                parameters = DoublePendulum.default_parameters(T))

            for (aname, σ) in ACTIVATIONS
                basis = try
                    OneLayerNetwork_GML{T}(σ, S_NEURONS)
                catch e
                    @warn "basis build failed" T aname exception = e
                    continue
                end
                for (sname, seed) in SEEDS
                    r = run_case(basis, T, seed, λ, prob, refq)
                    @printf("%-8s %-8s %-15s | %-10s %-11s %-11s %-6s %-7s\n",
                            string(T), aname, sname, r.status,
                            isfinite(r.ref_err) ? @sprintf("%.3e", r.ref_err) : "—",
                            isfinite(r.drift) ? @sprintf("%.3e", r.drift) : "—",
                            isnan(r.iters) ? "—" : string(round(Int, r.iters)),
                            isnan(r.secs) ? "—" : @sprintf("%.2f", r.secs))
                    println(io, join(("double_pendulum", "double_pendulum", string(T),
                                      csvnum(DT), string(S_NEURONS), string(R_QUAD), aname,
                                      sname, string(multiple), csvnum(Float64(λ)), r.status,
                                      csvnum(r.ref_err), csvnum(r.drift), csvnum(r.iters),
                                      csvnum(r.secs)), ","))
                    flush(io)
                end
            end
        end
    end
    println("-"^104)
    println("Wrote $(csvpath)")
    write_sweep_report(csvpath, "oga_double_pendulum")
end

main()
