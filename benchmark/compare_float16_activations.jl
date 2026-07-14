# Float16 activation comparison for the one-layer GML integrator: elu vs gelu vs
# tanh across all four benchmark problems, using the "quick" preset scalar axes
# (10 steps; DogLeg; midpoint initial guess; λ = 16·√eps; dt ∈ {0.1, 1, 10}).
#
#   julia --project=benchmark benchmark/compare_float16_activations.jl
#
# ReLU-family activations diverge at Float16 (polynomial growth exceeds the range),
# so they are excluded here. Writes results/float16_activations.csv (standard 17-column
# sweep format, so it round-trips through gml_report's `read_results`) and a markdown
# report results/float16_activations.md with by-activation / by-problem stats and an
# accuracy head-to-head table. It is a diagnostic, not part of the standard sweep.

include(joinpath(@__DIR__, "gml_benchmark_common.jl"))

using GeometricProblems.Pendulum
using GeometricProblems.HarmonicOscillator
using GeometricProblems.TodaLattice
using GeometricProblems.DoublePendulum
using Dates

const N_TODA = 16

# --- per-problem builders (mirroring the run_*.jl drivers) --------------------
function pendulum_prob(::Type{T}, timespan, timestep) where {T}
    d = Pendulum.iodeproblem(); q0 = T.(d.ics.q); p0 = T.(d.ics.p)
    Pendulum.iodeproblem(q0, p0; timespan, timestep,
        parameters = Pendulum.default_parameters(T))
end
pendulum_ham(t, q, p, params) = Pendulum.hamiltonian(t, q, p, params)

harmonic_prob(::Type{T}, timespan, timestep) where {T} =
    HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)]; timespan, timestep,
        parameters = HarmonicOscillator.default_parameters(T))
harmonic_ham(t, q, p, params) = HarmonicOscillator.hamiltonian(t, q, p, params)

function toda_prob(::Type{T}, timespan, timestep) where {T}
    d = TodaLattice.lodeproblem(N_TODA); q0 = T.(d.ics.q); p0 = T.(d.ics.p)
    TodaLattice.lodeproblem(q0, p0; timespan, timestep,
        parameters = TodaLattice.default_parameters(T))
end
toda_ham(t, q, p, params) = TodaLattice.hamiltonian(t, q, p, params, N_TODA)

function double_prob(::Type{T}, timespan, timestep) where {T}
    d = DoublePendulum.lodeproblem(); q0 = T.(d.ics.q); p0 = T.(d.ics.p)
    DoublePendulum.lodeproblem(q0, p0; timespan, timestep,
        parameters = DoublePendulum.default_parameters(T))
end
double_ham(t, q, p, params) = DoublePendulum.hamiltonian(t, q, p, params)

# problem specs: (name, build_prob, ham, R, S) — R/S follow the quick preset, with
# the harder problems using the same (16, 8) override their drivers apply.
const PROBLEMS = [
    ("pendulum",            pendulum_prob, pendulum_ham,  8, 4),
    ("harmonic_oscillator", harmonic_prob, harmonic_ham,  8, 4),
    ("toda_lattice",        toda_prob,     toda_ham,     16, 8),
    ("double_pendulum",     double_prob,   double_ham,   16, 8),
]

const ACTS  = [("elu", elu), ("gelu", gelu), ("tanh", tanh)]
const DTS   = [0.1, 1.0, 10.0]
const T     = Float16
const STRAT = SOLVERS_QUICK[1]         # DogLeg
const IG    = IGS_QUICK[1]             # midpoint
const LAM   = LAMBDAS_QUICK[1]         # 16·sqrt(eps)
const MAXIT = 100
const dt_min = minimum(DTS)

const NAME = "float16_activations"
mkpath(RESULTS_DIR)
const CSVPATH = joinpath(RESULTS_DIR, "$(NAME).csv")

println("="^80)
println("Float16 comparison — elu / gelu / tanh  (quick-preset axes, 10 steps)")
println("="^80)
@printf("%-20s %-5s %-5s | %-10s %-12s\n", "problem", "dt", "act", "status", "ref_err")
println("-"^60)

# --- sweep, writing the standard 17-column CSV --------------------------------
open(CSVPATH, "w") do io
    println(io, CSV_HEADER)
    flush(io)
    for (name, build_prob, ham, R, S) in PROBLEMS
        refcache = Dict{Float64,Any}()
        for (actlabel, act) in ACTS
            basis  = OneLayerNetwork_GML{T}(act, S)
            method = NonLinear_OneLayer_GML(basis, QuadratureRules.GaussLegendreQuadrature(T, R);
                        bias_interval = [-T(pi), T(pi)], dict_amount = DICT_AMOUNT,
                        initial_trajectory = IG.extrap)
            for dt in DTS
                prob   = build_prob(T, (T(0), T(10 * dt)), T(dt))
                params = prob.parameters
                refq   = get!(() -> build_gauss_reference(build_prob, dt, dt_min), refcache, dt)
                λ = LAM.f(T)
                r = run_case(prob, method, T, IG, STRAT, λ, MAXIT, refq, ham, params)
                @printf("%-20s %-5.3g %-5s | %-10s %-12s\n", name, dt, actlabel, r.status,
                        isnan(r.ref_err) ? "—" : @sprintf("%.3e", r.ref_err))
                row = join((name, string(T), csvnum(dt), "10", csvnum(R), csvnum(S),
                            actlabel, STRAT.solver, STRAT.linesearch, IG.label, csvnum(Float64(λ)),
                            r.status, csvnum(r.ref_err), csvnum(r.ham_drift), csvint(r.iters),
                            csvnum(r.solve_secs), csvnum(r.total_secs)), ",")
                println(io, row)
                flush(io)
            end
        end
    end
end
println("-"^60)
println("Wrote $(CSVPATH)")

# --- markdown report (reuses gml_report helpers) ------------------------------
rows = read_results(CSVPATH)
hcell(prob, dt, act) = begin
    hit = findfirst(r -> r.problem == prob && r.dt == dt && r.activation == act, rows)
    hit === nothing && return "—"
    r = rows[hit]
    is_ok(r) ? (isnan(r.ref_err) ? "ok(NaN)" : fmt_sci(r.ref_err)) : r.status
end

md = joinpath(RESULTS_DIR, "$(NAME).md")
open(md, "w") do io
    ntot = length(rows); nok = count(is_ok, rows)
    println(io, "# One-layer GML — Float16 activation comparison (elu / gelu / tanh)\n")
    println(io, "*Generated $(Dates.format(now(), "yyyy-mm-dd HH:MM")).*\n")
    println(io, "- Total cases: **$(ntot)**  •  converged (`ok`): **$(nok)** ($(fmt_pct(ntot == 0 ? 0.0 : nok/ntot)))")
    println(io, "- Axes: precision **Float16**; `dt ∈ {0.1, 1, 10}`; 10 steps; **DogLeg** solver;")
    println(io, "  **midpoint** initial guess; `λ = 16·√eps`. R/S follow the quick preset (8/4,")
    println(io, "  and 16/8 for toda_lattice and double_pendulum).")
    println(io, "- ReLU-family activations are **excluded** — they diverge at Float16.")
    println(io, "- `ref_err` is the relative max-norm error of the final state vs a `Gauss(8)` /")
    println(io, "  Float64 reference at the smallest timestep.\n")

    println(io, "## Status breakdown\n")
    statuses = sort(collect(Set(r.status for r in rows)))
    _table(io, ["status", "count"], [[s, string(count(r -> r.status == s, rows))] for s in statuses])

    println(io, "## By activation\n")
    _stats_table(io, rows, r -> r.activation, "activation")

    println(io, "## By problem\n")
    _stats_table(io, rows, r -> r.problem, "problem")

    println(io, "## Accuracy head-to-head (ref_err by problem × dt)\n")
    println(io, "Cell shows `ref_err` for converged cases, otherwise the failure status.\n")
    cells = [[prob, @sprintf("%.3g", dt), hcell(prob, dt, "elu"),
              hcell(prob, dt, "gelu"), hcell(prob, dt, "tanh")]
             for (prob, _, _, _, _) in PROBLEMS for dt in DTS]
    _table(io, ["problem", "dt", "elu", "gelu", "tanh"], cells)
end
println("Wrote $(md)")
