# Derivative backends for the shallow-net variational integrator: symbolic vs ForwardDiff,
# and — within the symbolic backend — the two code-generation settings the bases expose.
#
#   julia --project=benchmark benchmark/compare_derivative_backends.jl [quick|full]
#
# The four shallow-net integrators split into two pairs by how they obtain the derivatives
# of the ansatz with respect to the network parameters:
#
#   symbolic  — `ShallowNet`, `ShallowNetReversible` read `basis.dqdθ` / `basis.dvdθ`,
#               compiled once at basis-construction time by `SymbolicNeuralNetworks.jl`;
#   autodiff  — `ShallowNetAutodiff`, `ShallowNetAutodiffReversible` call
#               `ForwardDiff.gradient` on a hand-written ansatz on every evaluation.
#
# The symbolic pair is run twice, under the two code-generation settings `ShallowNetBasis` and
# `DenseNetBasis` forward to `build_nn_function`:
#
#   cse+inplace — the defaults: common-subexpression elimination, so the forward pass shared
#                 by the gradient blocks is emitted once, and a batch evaluated by an in-place
#                 kernel writing into one preallocated array;
#   plain       — `cse = false, inplace = false`: the shared forward pass re-emitted per
#                 block, and a batch evaluated out of place, one allocation per sample.
#
# Same mathematics, different emitted code, so what this measures is the code generation.
#
# Three measurements: the one-off basis build, the end-to-end solve (cold and warm), and the
# derivative kernels in isolation.
#
# TWO CAVEATS, which the generated report repeats:
#
# 1. The two *backends* do not discretize the same thing. The symbolic pair uses the raw
#    network q(t) = NN(t; θ) with the boundary conditions imposed through the residual; the
#    autodiff pair uses the boundary-interpolating ansatz q_h(t) = (1-t)q̄ + t·q + t(1-t)·NN(t),
#    a different unknown layout and a different `update!`. Accuracy differences between the
#    pairs are therefore a property of the method, not of the differentiation backend. (The
#    two *codegen* settings, by contrast, are the same method and must agree to round-off —
#    the report checks that.)
# 2. Each integrator keeps its own default OGA seed (`ShallowNetAutodiff` selects on the
#    normalized inner product, the other three on the raw one). Those are tuned
#    per-integrator baselines, so forcing one seed on all four would measure a detuned method
#    rather than a backend. The seed in force is recorded per row.
#
# Writes results/derivative_backends_<mode>.csv, results/derivative_backends_kernels.csv,
# a markdown report and plots. It is a targeted comparison, not part of the standard sweep.

include(joinpath(@__DIR__, "shallownet_benchmark_common.jl"))

using NeuralNetworkParameters: NetworkParameters
using GeometricProblems.HarmonicOscillator
using GeometricProblems.Pendulum
using GeometricProblems.DoublePendulum
using Dates
using Random
using Statistics: median

# The ForwardDiff kernels are internal to the package: they are the *implementation* of the
# autodiff integrators, not API, and timing them is exactly the sort of thing that reaches
# past the exported surface.
const NI = NonlinearIntegrators

const NAME = "derivative_backends"

# ---- what is compared -------------------------------------------------------

# A code-generation setting for the symbolic backend. `suffix` is appended to the integrator
# name to label the variant; the default setting adds nothing, so `ShallowNet` keeps its
# plain name and only the comparison case is marked.
mkcodegen(label, suffix, cse, inplace) =
    (label = label, suffix = suffix, cse = cse, inplace = inplace)

const CG_DEFAULT = mkcodegen("cse+inplace", "",              true,  true)
const CG_PLAIN   = mkcodegen("plain",       "plain codegen", false, false)
# The autodiff pair generates no code at all; it is handed a basis built with
# `symbolic = false`, for which `cse`/`inplace` are never consulted.
const CG_NA      = mkcodegen("—",           "",              true,  true)

function mkvariant(ctor, seed; symbolic::Bool = true, codegen = CG_DEFAULT)
    base  = string(nameof(ctor))
    label = isempty(codegen.suffix) ? base : "$(base) ($(codegen.suffix))"
    return (label = label, base = base, ctor = ctor, symbolic = symbolic,
            codegen = codegen, seed = seed,
            backend = symbolic ? "symbolic" : "autodiff")
end

const VARIANTS = [
    mkvariant(ShallowNet,                   "OGA1d"),
    mkvariant(ShallowNet,                   "OGA1d"; codegen = CG_PLAIN),
    mkvariant(ShallowNetReversible,         "OGA1d"),
    mkvariant(ShallowNetReversible,         "OGA1d"; codegen = CG_PLAIN),
    mkvariant(ShallowNetAutodiff,           "OGA1dNormalized"; symbolic = false, codegen = CG_NA),
    mkvariant(ShallowNetAutodiffReversible, "OGA1d";           symbolic = false, codegen = CG_NA),
]

const VARIANT_BY_LABEL = Dict(v.label => v for v in VARIANTS)

# The variants whose accuracy is worth tabulating side by side: one per distinct *method*.
# The plain-codegen runs are the same methods again and belong in the agreement check below.
const DISTINCT_METHODS = [v.label for v in VARIANTS if v.codegen !== CG_PLAIN]

# ---- problems ---------------------------------------------------------------
#
# `R`/`S` are the measured per-problem optima the `run_*.jl` drivers use in quick mode. All
# three widths are even, which the reversible pair requires — it stores mirrored neuron pairs
# and rejects an odd `S` in its constructor.

mkproblem(name, build, ham, R, S) = (name = name, build = build, ham = ham, R = R, S = S)

harmonic_prob(::Type{T}, timespan, timestep) where {T} =
    HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)]; timespan, timestep,
        parameters = HarmonicOscillator.default_parameters(T))
harmonic_ham(t, q, p, params) = HarmonicOscillator.hamiltonian(t, q, p, params)

function pendulum_prob(::Type{T}, timespan, timestep) where {T}
    d = Pendulum.iodeproblem(); q0 = T.(d.ics.q); p0 = T.(d.ics.p)
    Pendulum.iodeproblem(q0, p0; timespan, timestep,
        parameters = Pendulum.default_parameters(T))
end
pendulum_ham(t, q, p, params) = Pendulum.hamiltonian(t, q, p, params)

function double_prob(::Type{T}, timespan, timestep) where {T}
    d = DoublePendulum.lodeproblem(); q0 = T.(d.ics.q); p0 = T.(d.ics.p)
    DoublePendulum.lodeproblem(q0, p0; timespan, timestep,
        parameters = DoublePendulum.default_parameters(T))
end
double_ham(t, q, p, params) = DoublePendulum.hamiltonian(t, q, p, params)

const PROB_HARMONIC = mkproblem("harmonic_oscillator", harmonic_prob, harmonic_ham,  8, 10)
const PROB_PENDULUM = mkproblem("pendulum",            pendulum_prob, pendulum_ham,  8,  8)
const PROB_DOUBLE   = mkproblem("double_pendulum",     double_prob,   double_ham,   16, 10)

# ---- presets ----------------------------------------------------------------
#
# Everything the sweep is not varying is pinned to the quick preset's scalar choices, so the
# variants differ only in the backend (and in the ansatz that comes with it) and the codegen.

# Selected by label, not by index: `SOLVERS_QUICK` is a `filter` over `SOLVERS_FULL`, so
# reordering the latter would silently change which strategy this file measures.
# Newton/Backtracking rather than the trust-region DogLeg because it is the one that
# converges here — measured on the harmonic oscillator at Float64, DogLeg exhausts its
# 1000-iteration budget where Newton/Backtracking converges (see `SOLVERS_QUICK`).
const STRAT = only(filter(s -> (s.solver, s.linesearch) == ("Newton", "Backtracking"), SOLVERS_QUICK))
const IG    = only(filter(g -> g.label == "midpoint", IGS_QUICK))   # IntegratorExtrapolation
const LAM   = only(filter(l -> l.label == "16sqrt(eps)", LAMBDAS_QUICK))
const STRAT_LABEL = "$(STRAT.solver)/$(STRAT.linesearch)"

function backend_preset(mode::AbstractString)
    if mode == "full"
        return (problems = [PROB_HARMONIC, PROB_PENDULUM, PROB_DOUBLE],
                types = [Float64, Float32, Float16], dts = [0.1, 1.0, 10.0],
                activations = [("tanh", tanh), ("gelu", gelu)],
                kernel_Ss = [4, 8, 12, 16], kernel_calls = 2000)
    elseif mode == "quick"
        return (problems = [PROB_HARMONIC, PROB_PENDULUM],
                types = [Float64, Float32], dts = [0.1, 1.0],
                activations = [("tanh", tanh)],
                kernel_Ss = [4, 8, 16], kernel_calls = 2000)
    else
        error("unknown mode $(repr(mode)); use \"quick\" or \"full\"")
    end
end

# ---- basis cache ------------------------------------------------------------
#
# Bases are shared across dt and problem, as in `run_sweep`, and their build time is one of
# the three things measured. The timed build is preceded by an untimed one at S = 2 for the
# same (T, activation, symbolic, cse, inplace) so that Julia's compilation of
# `build_network_derivatives` and of Symbolics' code generation lands there instead of in
# the number we report; what is left is the codegen work itself.

const BASIS_CACHE = Dict{Any,Any}()
const BASIS_SECS  = Dict{Any,Float64}()
const BASIS_WARM  = Set{Any}()

basis_key(::Type{T}, actlabel, S, cg::NamedTuple, symbolic::Bool) where {T} =
    (string(T), actlabel, S, symbolic, cg.cse, cg.inplace)

function get_basis(::Type{T}, act, actlabel, S::Int, cg::NamedTuple, symbolic::Bool) where {T}
    key = basis_key(T, actlabel, S, cg, symbolic)
    haskey(BASIS_CACHE, key) && return BASIS_CACHE[key]
    warmkey = (key[1], key[2], key[4], key[5], key[6])   # the key without S
    if !(warmkey in BASIS_WARM)
        ShallowNetBasis{T}(act, 2; symbolic = symbolic, cse = cg.cse, inplace = cg.inplace)
        push!(BASIS_WARM, warmkey)
    end
    secs = @elapsed basis = ShallowNetBasis{T}(act, S;
        symbolic = symbolic, cse = cg.cse, inplace = cg.inplace)
    BASIS_SECS[key] = secs
    return BASIS_CACHE[key] = basis
end

basis_secs(::Type{T}, actlabel, S, cg, symbolic) where {T} =
    get(BASIS_SECS, basis_key(T, actlabel, S, cg, symbolic), NaN)

# ---- CSV --------------------------------------------------------------------

const BACKEND_CSV_HEADER =
    "method,backend,codegen,problem,T,dt,steps,R,S,activation,solver,linesearch," *
    "initial_guess,seed,lambda,status,ref_err,ham_drift,iterations,warm_secs,cold_secs,basis_secs"

const KERNEL_CSV_HEADER =
    "T,S,activation,kernel,backend,codegen,calls,min_secs,median_secs,bytes"

const AGREEMENT_CSV_HEADER = "T,S,activation,kernel,max_rel_diff"

# Parse the sweep CSV back into rows whose field names are the ones `shallownet_report.jl`
# expects (`problem`, `T`, `dt`, `status`, `ref_err`, `ham_drift`, `iterations`,
# `total_secs`), so `groupby`, `group_stats`, `_stats_table`, `plot_success_bars` and
# `plot_metric_vs_dt` all work on them unchanged. `total_secs` is the *warm* solve.
function read_backend_results(path::AbstractString)
    rows = NamedTuple[]
    isfile(path) || return rows
    lines = readlines(path)
    length(lines) <= 1 && return rows
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        f = split(ln, ",")
        length(f) == 22 || continue
        push!(rows, (method = String(f[1]), backend = String(f[2]), codegen = String(f[3]),
                     problem = String(f[4]), T = String(f[5]), dt = _parsef(f[6]),
                     steps = round(Int, _parsef(f[7])), R = round(Int, _parsef(f[8])),
                     S = round(Int, _parsef(f[9])), activation = String(f[10]),
                     solver = String(f[11]), linesearch = String(f[12]),
                     initial_guess = String(f[13]), seed = String(f[14]),
                     lambda = _parsef(f[15]), status = String(f[16]),
                     ref_err = _parsef(f[17]), ham_drift = _parsef(f[18]),
                     iterations = _parsef(f[19]), total_secs = _parsef(f[20]),
                     cold_secs = _parsef(f[21]), basis_secs = _parsef(f[22])))
    end
    return rows
end

function read_agreement_results(path::AbstractString)
    rows = NamedTuple[]
    isfile(path) || return rows
    lines = readlines(path)
    length(lines) <= 1 && return rows
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        f = split(ln, ",")
        length(f) == 5 || continue
        push!(rows, (T = String(f[1]), S = round(Int, _parsef(f[2])),
                     activation = String(f[3]), kernel = String(f[4]),
                     max_rel_diff = _parsef(f[5])))
    end
    return rows
end

function read_kernel_results(path::AbstractString)
    rows = NamedTuple[]
    isfile(path) || return rows
    lines = readlines(path)
    length(lines) <= 1 && return rows
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        f = split(ln, ",")
        length(f) == 10 || continue
        push!(rows, (T = String(f[1]), S = round(Int, _parsef(f[2])),
                     activation = String(f[3]), kernel = String(f[4]),
                     backend = String(f[5]), codegen = String(f[6]),
                     calls = round(Int, _parsef(f[7])), min_secs = _parsef(f[8]),
                     median_secs = _parsef(f[9]), bytes = _parsef(f[10])))
    end
    return rows
end

# ---- the end-to-end sweep ---------------------------------------------------

# Each case is solved twice on a freshly built integrator. The first solve carries the
# specialization of the generated kernels (symbolic) or of the ForwardDiff tape (autodiff)
# for this element type and this network size; the second is the steady-state cost and is
# what every accuracy, drift and timing column reports.
function run_backend_sweep(cfg, mode::AbstractString)
    mkpath(RESULTS_DIR)
    csvpath = joinpath(RESULTS_DIR, "$(NAME)_$(mode).csv")

    total = length(cfg.problems) * length(cfg.types) * length(cfg.activations) *
            length(VARIANTS) * length(cfg.dts)

    println("="^124)
    println("Derivative backends: symbolic (cse+inplace / plain) vs ForwardDiff  " *
            "[mode=$(mode)]  —  $(total) cases, 10 steps each, solved twice (cold + warm)")
    println("="^124)
    @printf("%-38s %-20s %-8s %-6s | %-10s %-10s %-10s %-5s %-9s %-9s %-9s\n",
            "integrator", "problem", "T", "dt", "status", "ref_err", "ham_drift",
            "iter", "warm_s", "cold_s", "basis_s")
    println("-"^140)

    dt_min = minimum(cfg.dts)
    open(csvpath, "w") do io
        println(io, BACKEND_CSV_HEADER)
        flush(io)
        for spec in cfg.problems
            iseven(spec.S) || error("problem $(spec.name) has an odd S = $(spec.S); the " *
                                    "reversible variants require mirrored neuron pairs")
            refcache = Dict{Float64,Any}()
            for T in cfg.types, (actlabel, act) in cfg.activations
                λ = LAM.f(T)
                for v in VARIANTS
                    basis  = get_basis(T, act, actlabel, spec.S, v.codegen, v.symbolic)
                    bsecs  = basis_secs(T, actlabel, spec.S, v.codegen, v.symbolic)
                    method = v.ctor(basis, QuadratureRules.GaussLegendreQuadrature(T, spec.R);
                                show_status = false, bias_interval = [-T(pi), T(pi)],
                                dict_amount = DICT_AMOUNT,
                                initial_trajectory_method = IG.extrap)
                    for dt in cfg.dts
                        prob   = spec.build(T, (T(0), T(10 * dt)), T(dt))
                        params = prob.parameters
                        refq   = get!(() -> build_gauss_reference(spec.build, dt, dt_min),
                                      refcache, dt)

                        cold = run_case(prob, method, T, IG, STRAT, λ, nothing, refq, spec.ham, params)
                        warm = run_case(prob, method, T, IG, STRAT, λ, nothing, refq, spec.ham, params)

                        @printf("%-38s %-20s %-8s %-6.3g | %-10s %-10s %-10s %-5s %-9s %-9s %-9s\n",
                                v.label, spec.name, string(T), dt, warm.status,
                                isnan(warm.ref_err)    ? "—" : @sprintf("%.2e", warm.ref_err),
                                isnan(warm.ham_drift)  ? "—" : @sprintf("%.2e", warm.ham_drift),
                                isnan(warm.iters)      ? "—" : string(round(Int, warm.iters)),
                                isnan(warm.total_secs) ? "—" : @sprintf("%.3f", warm.total_secs),
                                isnan(cold.total_secs) ? "—" : @sprintf("%.3f", cold.total_secs),
                                isnan(bsecs)           ? "—" : @sprintf("%.3f", bsecs))

                        row = join((v.label, v.backend, v.codegen.label, spec.name, string(T),
                                    csvnum(dt), "10", csvnum(spec.R), csvnum(spec.S), actlabel,
                                    STRAT.solver, STRAT.linesearch, IG.label, v.seed,
                                    csvnum(Float64(λ)), warm.status, csvnum(warm.ref_err),
                                    csvnum(warm.ham_drift), csvint(warm.iters),
                                    csvnum(warm.total_secs), csvnum(cold.total_secs),
                                    csvnum(bsecs)), ",")
                        println(io, row)
                        flush(io)
                    end
                end
            end
        end
    end
    println("-"^140)
    println("Wrote $(csvpath)")
    return csvpath
end

# ---- the derivative-kernel microbenchmark -----------------------------------

# Hand-rolled rather than pulling in BenchmarkTools: one warm-up call to get the method
# compiled, then `calls` repeats timed individually so both the minimum (the cost with no
# interference) and the median (the cost one actually pays) can be reported.
function bench_call(f; calls::Int)
    f()
    ts = Vector{Float64}(undef, calls)
    for i in 1:calls
        t0 = time_ns()
        f()
        ts[i] = (time_ns() - t0) * 1e-9
    end
    bytes = @allocated f()
    return (min = minimum(ts), median = median(ts), bytes = Float64(bytes))
end

# Parameters in the two layouts the two backends want: the nested `NetworkParameters`
# the generated kernels take (the shape `cache(int).ps` holds), and the flat `[W2; W1; b1]`
# vector `apply_NN` indexes.
function kernel_params(::Type{T}, S::Int) where {T}
    rng = Random.MersenneTwister(42)
    ps = (L1 = (W = rand(rng, T, S, 1), b = rand(rng, T, S)), L2 = (W = rand(rng, T, 1, S),))
    ps_vec = vcat(vec(ps.L2.W), vec(ps.L1.W), ps.L1.b)
    return NetworkParameters(ps), ps_vec
end

# `cse` and `inplace` change the emitted code, not the mathematics, so the two settings have
# to compute the same derivative from the same parameters. This is where that is checkable:
# end to end the Newton solve amplifies a last-bit difference into a different accepted
# iterate, so the integrated results are *not* a test of the code generation.
function codegen_max_rel_diff(::Type{T}, a, b) where {T}
    va = NI.flatten_params(a)
    vb = NI.flatten_params(b)
    length(va) == length(vb) || return NaN
    scale = max(maximum(abs, va), eps(T))
    return Float64(maximum(abs.(va .- vb)) / scale)
end

function run_kernel_benchmark(cfg)
    mkpath(RESULTS_DIR)
    csvpath = joinpath(RESULTS_DIR, "$(NAME)_kernels.csv")
    agrpath = joinpath(RESULTS_DIR, "$(NAME)_codegen_agreement.csv")
    actlabel, act = cfg.activations[1]

    println()
    println("="^96)
    println("Derivative kernels in isolation — $(cfg.kernel_calls) calls each, activation $(actlabel)")
    println("="^96)
    @printf("%-8s %-4s %-8s %-10s %-12s | %-12s %-12s %-10s\n",
            "T", "S", "kernel", "backend", "codegen", "min [µs]", "median [µs]", "bytes")
    println("-"^96)

    # Both handles opened with the `do` form: an exception anywhere in the sweep below has to
    # close them, or `DERIV_BENCH_REUSE=true` reads a truncated agreement CSV as a complete one.
    open(agrpath, "w") do agr
        println(agr, AGREEMENT_CSV_HEADER)
        open(csvpath, "w") do io
            println(io, KERNEL_CSV_HEADER)
            flush(io)
            for T in cfg.types, S in cfg.kernel_Ss
                default = get_basis(T, act, actlabel, S, CG_DEFAULT, true)
                plain   = get_basis(T, act, actlabel, S, CG_PLAIN,   true)
                nnp, ps_vec = kernel_params(T, S)
                t  = T(0.3)                       # an interior quadrature-node-like input
                q̄  = T(0.5); q = T(0.7)           # the endpoint values the autodiff ansatz needs
                input = [t]

                for (kernel, f) in (("dqdθ", b -> b.dqdθ(input, nnp)),
                                    ("dvdθ", b -> b.dvdθ(input, nnp)))
                    d = codegen_max_rel_diff(T, f(default), f(plain))
                    println(agr, join((string(T), csvnum(S), actlabel, kernel, csvnum(d)), ","))
                end
                flush(agr)

                entries = [
                    ("dqdθ", "symbolic", CG_DEFAULT.label, () -> default.dqdθ(input, nnp)),
                    ("dvdθ", "symbolic", CG_DEFAULT.label, () -> default.dvdθ(input, nnp)),
                    ("dqdθ", "symbolic", CG_PLAIN.label,   () -> plain.dqdθ(input, nnp)),
                    ("dvdθ", "symbolic", CG_PLAIN.label,   () -> plain.dvdθ(input, nnp)),
                    ("dqdθ", "autodiff", CG_NA.label,
                        () -> NI.∂NN_ansatz_∂params(ps_vec, S, act, t, q̄, q)),
                    ("dvdθ", "autodiff", CG_NA.label,
                        () -> NI.∂VNN_ansatz_∂params(ps_vec, S, act, t, q̄, q)),
                ]
                for (kernel, backend, codegen, f) in entries
                    r = bench_call(f; calls = cfg.kernel_calls)
                    @printf("%-8s %-4d %-8s %-10s %-12s | %-12.3f %-12.3f %-10d\n",
                            string(T), S, kernel, backend, codegen, 1e6 * r.min, 1e6 * r.median,
                            round(Int, r.bytes))
                    println(io, join((string(T), csvnum(S), actlabel, kernel, backend, codegen,
                                      csvnum(cfg.kernel_calls), csvnum(r.min), csvnum(r.median),
                                      csvnum(r.bytes)), ","))
                    flush(io)
                end
            end
        end
    end
    println("-"^96)
    println("Wrote $(csvpath)")
    println("Wrote $(agrpath)")
    return csvpath, agrpath
end

# ---- report -----------------------------------------------------------------

_med_finite(xs) = (v = filter(isfinite, xs); isempty(v) ? NaN : median(v))

backend_of(label) = get(VARIANT_BY_LABEL, label, (backend = "?",)).backend
codegen_of(label) = get(VARIANT_BY_LABEL, label, (codegen = CG_NA,)).codegen.label

# ref_err for cases that produced a trajectory, otherwise the failure status. Same convention
# as `compare_float16_activations.jl`'s head-to-head table.
#
# `activation` is part of the key, not just `(problem, dt)`: `full` mode sweeps two of them, and
# keying without it would make `findfirst` return whichever came first and drop the other
# activation from the table with nothing to show it had been there.
function _cell(rows, problem, activation, dt, method)
    hit = findfirst(r -> r.problem == problem && r.activation == activation &&
                         r.dt == dt && r.method == method, rows)
    hit === nothing && return "—"
    r = rows[hit]
    has_metrics(r) ? (isnan(r.ref_err) ? "$(r.status)(NaN)" : fmt_sci(r.ref_err)) : r.status
end

# How far the two codegen settings drift apart once the Newton solve is wrapped around them.
# Pairs the plain-codegen run of each integrator with its default-codegen counterpart by
# (problem, T, activation, dt) — every axis of the sweep, so that the only thing left between
# the two rows of a pair is the code generation. Dropping `activation` from the key would pair
# a `gelu` row against a `tanh` one in `full` mode and report the activation difference as the
# codegen spread. This is *not* a check on the code generation — see `codegen_max_rel_diff`
# for that — it is a measurement of the amplification.
function _codegen_endtoend_spread(io, rows)
    cells = Vector{Vector{String}}()
    for v in VARIANTS
        v.codegen === CG_PLAIN || continue
        base = only(filter(w -> w.base == v.base && w.codegen === CG_DEFAULT, VARIANTS))
        pairs = Tuple{Float64,Float64}[]
        iterdiff = 0
        mismatched = 0
        for r in rows
            r.method == v.label || continue
            hit = findfirst(s -> s.method == base.label && s.problem == r.problem &&
                                 s.T == r.T && s.activation == r.activation &&
                                 s.dt == r.dt, rows)
            hit === nothing && continue
            d = rows[hit]
            d.status == r.status || (mismatched += 1)
            # `isequal`, not `==`: an iteration count the solver state did not yield is `NaN`,
            # and two cases that both failed that way have not disagreed about anything.
            isequal(d.iterations, r.iterations) || (iterdiff += 1)
            (isfinite(d.ref_err) && isfinite(r.ref_err) && d.ref_err != 0) || continue
            push!(pairs, (d.ref_err, r.ref_err))
        end
        maxrel = isempty(pairs) ? NaN : maximum(abs(p - d) / abs(d) for (d, p) in pairs)
        push!(cells, [base.base, string(length(pairs)),
                      isnan(maxrel) ? "—" : @sprintf("%.1e", maxrel),
                      string(iterdiff), mismatched == 0 ? "yes" : "no ($(mismatched))"])
    end
    _table(io, ["integrator", "paired cases", "max rel. Δ ref_err", "cases with a different iteration count",
                "statuses agree"], cells)
end

# The real check: the same derivative, from the same parameters, under both codegen settings.
function _codegen_agreement(io, arows)
    isempty(arows) && return println(io, "No kernel-level comparison was run.\n")
    cells = [[r.T, string(r.S), r.kernel,
              isnan(r.max_rel_diff) ? "—" : @sprintf("%.1e", r.max_rel_diff)]
             for r in sort(arows, by = r -> (r.T, r.S, r.kernel))]
    _table(io, ["T", "S", "kernel", "max rel. Δ (max-norm)"], cells)
end

function write_backend_report(rows, krows, arows; mode, outdir)
    mkpath(outdir)
    prefix = NAME
    methods_present = [v.label for v in VARIANTS if any(r -> r.method == v.label, rows)]
    headtohead = filter(in(DISTINCT_METHODS), methods_present)

    p_conv = "$(prefix)_convergence_method.png"
    p_heat = "$(prefix)_convergence_heatmap.png"
    p_time = "$(prefix)_runtime_vs_dt.png"
    p_acc  = "$(prefix)_accuracy_vs_dt.png"
    p_ener = "$(prefix)_energy_drift_vs_dt.png"
    p_iter = "$(prefix)_iterations_vs_dt.png"

    have_conv = plot_success_bars(rows, r -> r.method, "Integrator",
                                  "Convergence by integrator", joinpath(outdir, p_conv))
    have_heat = plot_success_heatmap(rows, joinpath(outdir, p_heat);
                                     keyfn = r -> r.method, keylabel = "integrator")
    have_time = plot_metric_vs_dt(rows, :total_secs, "Warm run time [s]",
                                  "Run time vs timestep", joinpath(outdir, p_time);
                                  colorby = :method, colortitle = "Integrator")
    have_acc  = plot_metric_vs_dt(rows, :ref_err, "Relative error vs reference",
                                  "Accuracy vs timestep", joinpath(outdir, p_acc);
                                  colorby = :method, colortitle = "Integrator")
    have_ener = plot_metric_vs_dt(rows, :ham_drift, "Relative Hamiltonian drift",
                                  "Energy drift vs timestep", joinpath(outdir, p_ener);
                                  colorby = :method, colortitle = "Integrator")
    have_iter = plot_metric_vs_dt(rows, :iterations, "Nonlinear iterations (final step)",
                                  "Nonlinear iterations vs timestep", joinpath(outdir, p_iter);
                                  ylog = false, colorby = :method, colortitle = "Integrator")

    md = joinpath(outdir, "$(prefix).md")
    open(md, "w") do io
        ntot = length(rows); nok = count(is_ok, rows); nmeas = count(has_metrics, rows)
        println(io, "# Derivative backends — symbolic vs ForwardDiff\n")
        println(io, "*Generated $(Dates.format(now(), "yyyy-mm-dd HH:MM")) — mode `$(mode)`.*\n")

        println(io, "`ShallowNet` and `ShallowNetReversible` read the derivatives")
        println(io, "`SymbolicNeuralNetworks.jl` compiles into the basis at construction time;")
        println(io, "`ShallowNetAutodiff` and `ShallowNetAutodiffReversible` call")
        println(io, "`ForwardDiff.gradient` on a hand-written ansatz at every evaluation. The")
        println(io, "symbolic pair is run under two code-generation settings:\n")
        println(io, "| codegen | meaning |")
        println(io, "|---|---|")
        println(io, "| `cse+inplace` | the defaults — common-subexpression elimination, so the forward pass shared by the gradient blocks is emitted once, and a batch evaluated by an in-place kernel writing into one preallocated array |")
        println(io, "| `plain` | `cse = false, inplace = false`: the shared forward pass re-emitted per block, and a batch evaluated out of place. Same mathematics, different emitted code. |\n")
        println(io, "Every axis other than problem, precision, activation and `dt` is held")
        println(io, "fixed: **$(STRAT_LABEL)**, midpoint initial trajectory, `λ = 16·√eps(T)`,")
        println(io, "10 steps per case, `timespan = (0, 10·dt)`, and the per-problem `R`/`S`")
        println(io, "the standard sweep uses. `quick` sweeps one activation, `full` two, so")
        println(io, "`activation` is a column of the head-to-head table and part of the key")
        println(io, "every paired comparison below joins on.\n")
        println(io, "- Total cases: **$(ntot)**  •  converged (`ok`): **$(nok)** " *
                    "($(fmt_pct(ntot == 0 ? 0.0 : nok/ntot)))  •  produced a trajectory " *
                    "(`ok` + `maxiter`): **$(nmeas)**.")
        println(io, "- Every case is solved **twice**: `cold_secs` is the first solve on a")
        println(io, "  fresh integrator (it carries the specialization of the generated kernels")
        println(io, "  or of the ForwardDiff tape), `warm_secs` is the second and is what all")
        println(io, "  other columns report. `basis_secs` is the one-off `ShallowNetBasis` build.\n")

        println(io, "## Two things this does not measure\n")
        println(io, "1. **The two backends do not discretize the same thing.** The symbolic pair")
        println(io, "   uses the raw network `q(t) = NN(t; θ)`; the autodiff pair uses the")
        println(io, "   boundary-interpolating ansatz `q_h(t) = (1-t)q̄ + t·q + t(1-t)·NN(t)`,")
        println(io, "   with a different unknown layout and a different `update!`. Differences")
        println(io, "   in `ref_err`, `ham_drift` and iteration count **between the pairs** are")
        println(io, "   properties of the method; only the timings compare backends. The two")
        println(io, "   codegen settings, by contrast, are the same method — see the agreement")
        println(io, "   check below.")
        println(io, "2. **Each integrator keeps its own default OGA seed** (see the `seed`")
        println(io, "   column): `ShallowNetAutodiff` selects on the normalized inner product,")
        println(io, "   the other three on the raw one. Those are tuned baselines — forcing one")
        println(io, "   seed on all four would measure a detuned method, not a backend.\n")

        println(io, "## Status breakdown\n")
        statuses = sort(collect(Set(r.status for r in rows)))
        _table(io, ["status", "count"],
               [[s, string(count(r -> r.status == s, rows))] for s in statuses])

        println(io, "## By integrator\n")
        println(io, "`med total_s` is the median **warm** solve; the cold solve and the basis")
        println(io, "build are broken out in *Cost of the backend* below.\n")
        _stats_table(io, rows, r -> r.method, "integrator")
        have_conv && println(io, "![convergence by integrator]($(p_conv))\n")
        have_heat && println(io, "![convergence heatmap]($(p_heat))\n")

        println(io, "### By backend\n")
        _stats_table(io, rows, r -> r.backend, "backend")

        println(io, "### By code generation\n")
        println(io, "`—` is the autodiff pair, which generates no code.\n")
        _stats_table(io, rows, r -> r.codegen, "codegen")

        println(io, "### By precision\n")
        _stats_table(io, rows, r -> r.T, "precision")

        if length(unique(r.problem for r in rows)) > 1
            println(io, "### By problem\n")
            _stats_table(io, rows, r -> r.problem, "problem")
        end

        println(io, "## Cost of the backend (goal of this file)\n")
        println(io, "Medians over every case that produced a trajectory. `basis build` is the")
        println(io, "one-off basis construction. For the autodiff pair, which is handed a basis")
        println(io, "built with `symbolic = false`, that is the network alone and lands in the")
        println(io, "microseconds; for the symbolic pair it is the code generation and lands in")
        println(io, "the tens to hundreds of milliseconds.\n")
        cost = Vector{Vector{String}}()
        for m in methods_present
            sub  = [r for r in rows if r.method == m]
            meas = filter(has_metrics, sub)
            push!(cost, [m, backend_of(m), codegen_of(m),
                         fmt_secs(_med_finite([r.basis_secs for r in sub])),
                         fmt_secs(_med_finite([r.cold_secs  for r in meas])),
                         fmt_secs(_med_finite([r.total_secs for r in meas])),
                         fmt_iter(_med_finite([r.iterations for r in meas]))])
        end
        _table(io, ["integrator", "backend", "codegen", "med basis build [s]", "med cold [s]",
                    "med warm [s]", "med iter"], cost)
        println(io, "Wall clock and iteration count have to be read together: a case that runs")
        println(io, "to the iteration cap spends its time there, not in the derivative kernel.")
        println(io, "The per-call cost of the kernels alone is the last section.\n")
        have_time && println(io, "![run time]($(p_time))\n")
        have_iter && println(io, "![nonlinear iterations]($(p_iter))\n")

        println(io, "## Code generation changes the code, not the mathematics\n")
        println(io, "`cse` and `inplace` change only the emitted code, so the two settings must")
        println(io, "compute the same derivative from the same parameters. Evaluated directly,")
        println(io, "at the parameters of the kernel benchmark, they do:\n")
        _codegen_agreement(io, arows)
        println(io, "That agreement does **not** survive the Newton solve, and the paired")
        println(io, "end-to-end runs measure how much it is amplified:\n")
        _codegen_endtoend_spread(io, rows)
        println(io, "This is the conditioning of the solve, not a code-generation bug. The")
        println(io, "residual stalls near the round-off floor, so a last-bit difference in the")
        println(io, "derivative decides which iterate gets accepted — hence the differing")
        println(io, "iteration counts — and a `ref_err` already at 1e-13 then moves by orders of")
        println(io, "magnitude. It is also a reminder for the tables above: compare the codegen")
        println(io, "settings on their **timings**, not on their accuracy.\n")

        println(io, "## Accuracy head-to-head\n")
        println(io, "One column per distinct *method* (the plain-codegen runs are the same")
        println(io, "methods again and are covered by the agreement check above). `ref_err`")
        println(io, "where the case produced a trajectory, otherwise the failure status. Read")
        println(io, "this as a comparison of methods (caveat 1), not backends.\n")
        probs = unique(r.problem for r in rows)
        dts   = sort(unique(r.dt for r in rows))
        Ts    = sort(unique(r.T for r in rows))
        acts  = sort(unique(r.activation for r in rows))
        # One row per full sweep key. `activation` is a column rather than an implicit
        # constant because `full` mode sweeps two of them.
        cells = [[p, T, a, @sprintf("%.3g", dt),
                  (_cell([r for r in rows if r.T == T], p, a, dt, m) for m in headtohead)...]
                 for p in probs for T in Ts for a in acts for dt in dts]
        _table(io, ["problem", "T", "activation", "dt", headtohead...], cells)
        have_acc  && println(io, "![accuracy]($(p_acc))\n")
        have_ener && println(io, "![energy drift]($(p_ener))\n")

        println(io, "## Derivative kernels in isolation\n")
        if isempty(krows)
            println(io, "No kernel measurements.\n")
        else
            println(io, "One call each, off the integrator: the compiled `basis.dqdθ` /")
            println(io, "`basis.dvdθ` under both codegen settings, against `ForwardDiff.gradient`")
            println(io, "of the hand-written ansatz. Same `S`, same activation — but the autodiff")
            println(io, "column is not the same expression (caveat 1), so it is the cost of *a*")
            println(io, "parameter gradient in that backend rather than of the same one. The two")
            println(io, "symbolic columns *are* the same expression.\n")
            _kernel_tables(io, krows)
        end
    end
    return md
end

# The three measured configurations, joined on (T, S, kernel) and split into a timing and an
# allocation table — eleven columns in one table is not a table anyone reads.
function _kernel_tables(io, krows)
    pick(T, S, kernel, backend, codegen) =
        findfirst(r -> r.T == T && r.S == S && r.kernel == kernel &&
                       r.backend == backend && r.codegen == codegen, krows)

    times = Vector{Vector{String}}()
    bytes = Vector{Vector{String}}()
    for T in sort(unique(r.T for r in krows)),
        S in sort(unique(r.S for r in krows)),
        kernel in sort(unique(r.kernel for r in krows))
        i_def = pick(T, S, kernel, "symbolic", CG_DEFAULT.label)
        i_pln = pick(T, S, kernel, "symbolic", CG_PLAIN.label)
        i_ad  = pick(T, S, kernel, "autodiff", CG_NA.label)
        any(isnothing, (i_def, i_pln, i_ad)) && continue
        d, p, a = krows[i_def], krows[i_pln], krows[i_ad]
        rel(x) = (isfinite(d.median_secs) && d.median_secs > 0) ?
                 @sprintf("%.1f×", x.median_secs / d.median_secs) : "—"
        push!(times, [T, string(S), kernel,
                      @sprintf("%.3f", 1e6 * d.median_secs),
                      @sprintf("%.3f", 1e6 * p.median_secs),
                      @sprintf("%.3f", 1e6 * a.median_secs), rel(p), rel(a)])
        push!(bytes, [T, string(S), kernel, string(round(Int, d.bytes)),
                      string(round(Int, p.bytes)), string(round(Int, a.bytes))])
    end
    println(io, "### Time per call (median)\n")
    _table(io, ["T", "S", "kernel", "cse+inplace [µs]", "plain [µs]", "autodiff [µs]",
                "plain ÷ cse+inplace", "autodiff ÷ cse+inplace"], times)
    println(io, "### Allocations per call\n")
    _table(io, ["T", "S", "kernel", "cse+inplace [B]", "plain [B]", "autodiff [B]"], bytes)
end

# ---- driver -----------------------------------------------------------------

let mode = pick_mode()
    cfg  = backend_preset(mode)
    csv  = joinpath(RESULTS_DIR, "$(NAME)_$(mode).csv")
    kcsv = joinpath(RESULTS_DIR, "$(NAME)_kernels.csv")
    acsv = joinpath(RESULTS_DIR, "$(NAME)_codegen_agreement.csv")

    # `DERIV_BENCH_REUSE=true` regenerates the report and plots from the CSVs a previous run
    # left behind, in the spirit of `report.jl`: reporting is decoupled from measuring, so
    # editing a table does not cost another sweep.
    if get(ENV, "DERIV_BENCH_REUSE", "false") == "true" && all(isfile, (csv, kcsv, acsv))
        println("Reusing the CSVs in $(RESULTS_DIR) (DERIV_BENCH_REUSE=true)")
    else
        csv = run_backend_sweep(cfg, mode)
        kcsv, acsv = run_kernel_benchmark(cfg)
    end

    md = write_backend_report(read_backend_results(csv), read_kernel_results(kcsv),
                              read_agreement_results(acsv); mode = mode, outdir = RESULTS_DIR)
    println("Wrote $(md)")
end
