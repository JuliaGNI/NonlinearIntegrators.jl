# Shared engine for the `NonLinear_OneLayer_GML` benchmark suite.
#
# This file defines the sweep engine, presets, builders and the per-run measurement,
# but performs NO top-level execution. Each per-problem run file (`run_*.jl`) includes
# it, supplies its problem/reference/Hamiltonian closures, and calls `run_sweep(...)`.
#
# The sweep spans, per problem: timestep × precision × quadrature order R × network
# width S × activation × solver strategy × regularization λ × initial-guess strategy,
# each integrated for exactly 10 steps. For every case we record a status
# (ok/singular/diverged/nonfinite/…), an accuracy error against a reference, the
# relative Hamiltonian drift, and timing. Rows are appended+flushed to a CSV as they
# complete, so a long `full` run leaves partial results on interruption.

using NonlinearIntegrators
using GeometricIntegrators                       # integrate, Gauss, relative_maximum_error, extrapolations
import GeometricIntegratorsBase: solverstate      # read solver iteration count after a run
using QuadratureRules
import SimpleSolvers
using LinearAlgebra: SingularException
using Printf

include(joinpath(@__DIR__, "gml_report.jl"))     # read_results / plots / markdown

const RESULTS_DIR = joinpath(@__DIR__, "results")
const DICT_AMOUNT = 4000

# Type-generic ReLU^k (never `max(0.0, x)`, which would upcast — see test/testsetup.jl).
relu_k(k::Int) = x -> max(zero(x), x)^k

# Type-generic ELU (α = 1) and tanh-approximation GELU. Kept float-generic (no bare
# Float64 literals) so Float16/Float32 sweeps do not upcast — same rule as relu_k.
# ELU is written branch-free with max/min (not a `?:` ternary) so the symbolic
# gradient build can trace it, exactly like relu_k's `max`.
elu(x)  = max(zero(x), x) + min(zero(x), exp(x) - one(x))
gelu(x) = x / 2 * (one(x) + tanh(sqrt(oftype(x, 2 / pi)) *
                                 (x + oftype(x, 0.044715) * x^3)))

# ---- axis definitions -------------------------------------------------------

const ACTIVATIONS_FULL  = [("relu2", relu_k(2)), ("relu3", relu_k(3)),
                           ("relu4", relu_k(4)), ("elu", elu),
                           ("gelu", gelu), ("tanh", tanh)]
const ACTIVATIONS_QUICK = [("elu", elu), ("tanh", tanh)]

# A solver strategy: a labelled `NonlinearSolverMethod` plus an optional linesearch
# factory (built at the working type `T`). `DogLeg` takes no linesearch.
mkstrat(solver, ls, makesolver, makels) =
    (solver = solver, linesearch = ls, makesolver = makesolver, makels = makels)

const SOLVERS_FULL = [
    mkstrat("Newton", "Static",       () -> SimpleSolvers.Newton(), T -> SimpleSolvers.Static(T)),
    mkstrat("Newton", "Backtracking", () -> SimpleSolvers.Newton(), T -> SimpleSolvers.Backtracking(T)),
    mkstrat("Newton", "StrongWolfe",  () -> SimpleSolvers.Newton(), T -> SimpleSolvers.StrongWolfe(T)),
    mkstrat("DogLeg", "-",            () -> SimpleSolvers.DogLeg(),  nothing),
]
const SOLVERS_QUICK = [SOLVERS_FULL[4]]           # DogLeg only

# An initial-guess strategy: the method's `initial_trajectory` field plus whether the
# integrator also needs `initialguess = HermiteExtrapolation()` (the Hermite branch
# delegates the actual extrapolation to `iguess(int)`).
mkig(label, extrap, hermite) = (label = label, extrap = extrap, hermite = hermite)

const IGS_FULL = [
    mkig("midpoint", IntegratorExtrapolation(), false),
    mkig("Hermite",  HermiteExtrapolation(),    true),
    mkig("previous", NoExtrapolation(),         false),
]
const IGS_QUICK = [IGS_FULL[1]]                   # midpoint only

# A regularization-λ spec: a label plus a factory mapping the working type `T` to the
# `regularization_factor` value. `16·√eps(T)` scales the Jacobian-diagonal damping with
# the precision — ≈2.4e-7 at Float64, ≈5.5e-3 at Float32, and 0.5 at Float16 (the
# Float16 value is large, so half precision is expected to over-damp; see the report).
mklam(label, f) = (label = label, f = f)
lam_scaled() = mklam("16sqrt(eps)", T -> 16 * sqrt(eps(T)))

const LAMBDAS_FULL  = [mklam("0", T -> zero(T)), mklam("1e-7", T -> T(1e-7)),
                       mklam("1e-5", T -> T(1e-5)), mklam("1e-3", T -> T(1e-3)), lam_scaled()]
const LAMBDAS_QUICK = [lam_scaled()]

function preset(mode::AbstractString)
    if mode == "full"
        return (dts = [0.01, 0.1, 1.0, 10.0], types = [Float16, Float32, Float64],
                Rs = [4, 8, 16], Ss = [4, 6, 8], activations = ACTIVATIONS_FULL,
                solvers = SOLVERS_FULL, lambdas = LAMBDAS_FULL,
                igs = IGS_FULL, maxit = 10000)
    elseif mode == "quick"
        return (dts = [0.1, 1.0, 10.0], types = [Float64, Float32, Float16],
                Rs = [8], Ss = [4], activations = ACTIVATIONS_QUICK,
                solvers = SOLVERS_QUICK, lambdas = LAMBDAS_QUICK,
                igs = IGS_QUICK, maxit = 100)
    else
        error("unknown mode $(repr(mode)); use \"quick\" or \"full\"")
    end
end

pick_mode() = get(ENV, "GML_BENCH_PRESET", isempty(ARGS) ? "quick" : ARGS[1])

# ---- metrics ----------------------------------------------------------------

function classify_error(e)
    e isa SingularException && return "singular"
    name = string(nameof(typeof(e)))
    occursin("NonlinearSolver", name) && return "diverged"
    return first(name, 14)
end

# Accuracy reference: Gauss(8) at Float64 using the smallest timestep in the sweep,
# over the same 10-step horizon (0, 10·dt). Returns the reference final-state vector
# (Float64) or `nothing` if the reference solve itself fails.
function build_gauss_reference(build_prob, dt, dt_min)
    try
        refprob = build_prob(Float64, (0.0, 10.0 * dt), dt_min)
        ref = integrate(refprob, Gauss(8))
        return Float64.(collect(ref.q[:])[end])
    catch
        return nothing
    end
end

# Relative max-norm error of the case's final state against the reference final state
# (both at t = 10·dt).
function compute_ref_err(res, qref)
    qref === nothing && return NaN
    try
        qc = Float64.(collect(res.sol.q[:])[end])
        length(qc) == length(qref) || return NaN
        num = maximum(abs.(qc .- qref))
        den = maximum(abs.(qref))
        return den == 0 ? num : num / den
    catch
        return NaN
    end
end

function compute_ham_drift(res, hamfn, params)
    hamfn === nothing && return NaN
    try
        qs = collect(res.sol.q[:]); ps = collect(res.sol.p[:])
        hams = Float64[Float64(hamfn(0, q, p, params)) for (q, p) in zip(qs, ps)]
        H0 = hams[1]
        (!isfinite(H0) || H0 == 0) && return NaN
        return maximum(abs.((hams .- H0) ./ H0))
    catch
        return NaN
    end
end

# One integration. Returns (status, ref_err, ham_drift, iters, solve_secs, total_secs).
# `iters` is the nonlinear-solver iteration count of the final step (read from the
# solver state); the integrator is built explicitly so we can query that state.
function run_case(prob, method, ::Type{T}, ig, strat, λ, maxit, refq, hamfn, params) where {T}
    kw = Pair{Symbol,Any}[:solver => strat.makesolver(),
                          :regularization_factor => T(λ),
                          :max_iterations => maxit]
    strat.makels === nothing || push!(kw, :linesearch => strat.makels(T))
    ig.hermite && push!(kw, :initialguess => HermiteExtrapolation())

    status = "ok"; ref_err = NaN; ham_drift = NaN; iters = NaN; solve_secs = NaN; total_secs = NaN
    try
        int = GeometricIntegrator(prob, method; kw...)
        local res
        total_secs = @elapsed (res = integrate(int))
        try; iters = Float64(solverstate(int).iterations); catch; end
        qend = collect(res.sol.q[:])[end]
        if any(x -> !isfinite(x), qend)
            status = "nonfinite"
        else
            solve_secs = Float64(sum(res.solving_time_list))
            ref_err    = compute_ref_err(res, refq)
            ham_drift  = compute_ham_drift(res, hamfn, params)
        end
    catch e
        status = classify_error(e)
    end
    return (; status, ref_err, ham_drift, iters, solve_secs, total_secs)
end

# ---- CSV --------------------------------------------------------------------

const CSV_HEADER = "problem,T,dt,steps,R,S,activation,solver,linesearch,initial_guess,lambda,status,ref_err,ham_drift,iterations,solve_secs,total_secs"

csvnum(x) = (x isa Integer) ? string(x) : (isfinite(x) ? @sprintf("%.8e", x) : "NaN")
csvint(x) = isnan(x) ? "NaN" : string(round(Int, x))

# ---- the sweep --------------------------------------------------------------

"""
    run_sweep(; problem_name, build_prob, hamiltonian, mode)

Run the full parameter sweep for one problem.

* `build_prob(T, timespan, timestep)` → an `AbstractProblemIODE` at element type `T`.
* `hamiltonian(t, q, p, params)` → the conserved energy (or `nothing` to skip drift).
* `mode` → `"quick"` or `"full"`.
* `Rs` / `Ss` → optional per-problem overrides of the preset's quadrature orders /
  network widths (e.g. the double pendulum and Toda lattice use `Rs=[16], Ss=[8]` in
  quick mode).

The accuracy reference for every case is a `Gauss(8)` integration at Float64 using the
*smallest* timestep in the sweep, over the same 10-step horizon `(0, 10·dt)`; the case's
final state is compared against it (computed once per `dt` and cached).

Writes `results/<problem_name>_<mode>.csv` and returns its path.
"""
function run_sweep(; problem_name, build_prob, hamiltonian, mode, Rs = nothing, Ss = nothing)
    cfg = preset(mode)
    dt_min = minimum(cfg.dts)
    Rs = Rs === nothing ? cfg.Rs : Rs
    Ss = Ss === nothing ? cfg.Ss : Ss
    mkpath(RESULTS_DIR)
    csvpath = joinpath(RESULTS_DIR, "$(problem_name)_$(mode).csv")

    total = length(cfg.types) * length(Ss) * length(cfg.activations) *
            length(Rs) * length(cfg.igs) * length(cfg.dts) *
            length(cfg.solvers) * length(cfg.lambdas)

    println("="^90)
    println("Benchmark: $(problem_name)  [mode=$(mode)]  —  $(total) cases, 10 steps each")
    mode == "full" && @warn "full mode is large ($(total) cases); expect a long run. Partial results are flushed to CSV."
    println("="^90)
    @printf("%-6s %-8s %6s %2s %2s %-6s %-11s %-9s %-11s | %-10s %-10s %-10s %-5s %-8s\n",
            "T", "dt", "", "R", "S", "act", "solver/ls", "iguess", "λ",
            "status", "ref_err", "ham_drift", "iter", "solve_s")
    println("-"^116)

    refcache  = Dict{Float64,Any}()
    probcache = Dict{Tuple{DataType,Float64},Any}()   # problem depends only on (T, dt)
    idx = 0
    open(csvpath, "w") do io
        println(io, CSV_HEADER)
        flush(io)
        for T in cfg.types
            for S in Ss, (actlabel, act) in cfg.activations
                basis = OneLayerNetwork_GML{T}(act, S)            # expensive symbolic build, amortized
                for R in Rs, ig in cfg.igs
                    method = NonLinear_OneLayer_GML(basis, QuadratureRules.GaussLegendreQuadrature(T, R);
                                bias_interval = [-T(pi), T(pi)], dict_amount = DICT_AMOUNT,
                                initial_trajectory = ig.extrap)
                    for dt in cfg.dts
                        prob   = get!(() -> build_prob(T, (T(0), T(10 * dt)), T(dt)), probcache, (T, dt))
                        params = prob.parameters
                        refq   = get!(() -> build_gauss_reference(build_prob, dt, dt_min), refcache, dt)
                        for strat in cfg.solvers, lamspec in cfg.lambdas
                            idx += 1
                            λ = lamspec.f(T)
                            r = run_case(prob, method, T, ig, strat, λ, cfg.maxit, refq, hamiltonian, params)
                            lslabel = strat.makels === nothing ? strat.solver : "$(strat.solver)/$(strat.linesearch)"
                            @printf("%-6s %-8.3g %5d/%d %2d %2d %-6s %-11s %-9s %-11s | %-10s %-10s %-10s %-5s %-8s\n",
                                    string(T), dt, idx, total, R, S, actlabel, lslabel, ig.label,
                                    "$(lamspec.label)=$(@sprintf("%.1e", Float64(λ)))",
                                    r.status,
                                    isnan(r.ref_err)   ? "—" : @sprintf("%.2e", r.ref_err),
                                    isnan(r.ham_drift) ? "—" : @sprintf("%.2e", r.ham_drift),
                                    isnan(r.iters)     ? "—" : string(round(Int, r.iters)),
                                    isnan(r.solve_secs) ? "—" : @sprintf("%.3f", r.solve_secs))
                            row = join((problem_name, string(T), csvnum(dt), "10", csvnum(R), csvnum(S),
                                        actlabel, strat.solver, strat.linesearch, ig.label, csvnum(Float64(λ)),
                                        r.status, csvnum(r.ref_err), csvnum(r.ham_drift), csvint(r.iters),
                                        csvnum(r.solve_secs), csvnum(r.total_secs)), ",")
                            println(io, row)
                            flush(io)
                        end
                    end
                end
            end
        end
    end
    println("-"^108)
    println("Wrote $(csvpath)")
    return csvpath
end
