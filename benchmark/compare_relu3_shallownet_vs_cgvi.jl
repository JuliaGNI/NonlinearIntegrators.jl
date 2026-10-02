# ShallowNet(ReLU³, S = 4, R = 4)  vs  CGVI(Lagrange, S = 4, R = 4)
#
#   julia --project=benchmark benchmark/compare_relu3_shallownet_vs_cgvi.jl
#   julia --project=benchmark benchmark/compare_relu3_shallownet_vs_cgvi.jl harmonic_oscillator
#   julia --project=benchmark benchmark/compare_relu3_shallownet_vs_cgvi.jl --final-time=100
#
# Question: do the two integrators reach the *same* error level for
# h ∈ {0.1, 0.2, 0.5, 1.0, 2.0, 5.0}?
#
# Why one would expect it: the shallow net here is q(t) = Σᵢ aᵢ · max(0, wᵢ t + bᵢ)³
# (`Dense(1,S,σ) → Dense(S,1, no bias)`). As long as no kink −bᵢ/wᵢ falls inside the
# step, each neuron is a cubic (wᵢ t + bᵢ)³, and four cubics with distinct shifts span
# P₃ — exactly the space of CGVI with a Lagrange basis on 4 nodes (degree 3). Both use
# the same 4-node Gauss–Legendre quadrature, so a stationary point of the discrete
# action in the network parameters should coincide with the CGVI solution. If a kink
# lands inside the step, the network space is piecewise-cubic and the errors can differ.
#
# Measured per (problem, h, integrator), Float64, over (0, final_time):
#   q_err     max over the time grid of |q − q_ref|∞ / max|q_ref|∞
#   qend_err  same, at the final time only
#   ham_drift max |H − H₀| / |H₀|
# and between the two integrators directly:
#   q_diff    max over the time grid of |q_SN − q_CGVI|∞ / max|q_ref|∞
# The reference is Gauss(8) at DT_REF = 0.005 (every h in the sweep is an integer
# multiple of it), computed once per problem and subsampled onto each coarse grid.
#
# Writes results/relu3_shallownet_vs_cgvi.csv, a markdown report and a PNG plot.

include(joinpath(@__DIR__, "shallownet_benchmark_common.jl"))  # relu_k, csvnum, classify_error, RESULTS_DIR, …

using GeometricProblems.HarmonicOscillator
using GeometricProblems.Pendulum
using GeometricProblems.DoublePendulum
using CairoMakie
using Dates

# `Lagrange` lives in CompactBasisFunctions, which GeometricIntegrators imports but does not
# re-export and which is not a direct dependency of the benchmark project. Reach it through
# GeometricIntegrators rather than adding a dependency.
const CBF = GeometricIntegrators.Integrators.CompactBasisFunctions

# ---- configuration ----------------------------------------------------------

const T = Float64
const K_RELU = 3
const S_NET = 4
const R_QUAD = 4
const S_CGVI = 4                  # Lagrange nodes = the R_QUAD Gauss nodes when S_CGVI == R_QUAD
const DTS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
const DT_REF = 0.005
const DEFAULT_FINAL_TIME = 50.0   # a multiple of every h in DTS
const SAME_RTOL = 0.05            # "same error level" ⇔ |q_err_SN / q_err_CGVI − 1| ≤ SAME_RTOL

# ShallowNet solver options (as in scripts/experiments.jl `NVI_SOLVER_OPTIONS`).
const NVI_REG = 1e-5
const NVI_MAXIT = 1000
const NVI_SOLVER = () -> SimpleSolvers.Newton()
const NVI_LINESEARCH = () -> SimpleSolvers.Backtracking(T)

# ---- problems ---------------------------------------------------------------

function ho_prob(timespan, timestep)
    HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)]; timespan, timestep,
        parameters = HarmonicOscillator.default_parameters(T))
end
ho_ham(t, q, p, params) = HarmonicOscillator.hamiltonian(t, q, p, params)

function pendulum_prob(timespan, timestep)
    d = Pendulum.iodeproblem()
    Pendulum.iodeproblem(T.(d.ics.q), T.(d.ics.p); timespan, timestep,
        parameters = Pendulum.default_parameters(T))
end
pendulum_ham(t, q, p, params) = Pendulum.hamiltonian(t, q, p, params)

function double_prob(timespan, timestep)
    d = DoublePendulum.lodeproblem()
    DoublePendulum.lodeproblem(T.(d.ics.q), T.(d.ics.p); timespan, timestep,
        parameters = DoublePendulum.default_parameters(T))
end
double_ham(t, q, p, params) = DoublePendulum.hamiltonian(t, q, p, params)

const ALL_PROBLEMS = [
    (name = "harmonic_oscillator", build = ho_prob, ham = ho_ham),
    (name = "pendulum", build = pendulum_prob, ham = pendulum_ham),
    (name = "double_pendulum", build = double_prob, ham = double_ham)
]
const DEFAULT_PROBLEMS = ["harmonic_oscillator", "pendulum"]

# ---- methods ----------------------------------------------------------------

function shallownet_method()
    basis = ShallowNetBasis{T}(relu_k(K_RELU), S_NET)
    ShallowNet(basis, QuadratureRules.GaussLegendreQuadrature(T, R_QUAD);
        show_status = false, bias_interval = [-T(pi), T(pi)], dict_amount = DICT_AMOUNT)
end

function cgvi_method()
    quad = QuadratureRules.GaussLegendreQuadrature(T, R_QUAD)
    # Lagrange basis on S_CGVI nodes: the quadrature nodes when S == R (as in
    # scripts/experiments.jl `galerkin_method`), otherwise Gauss nodes of order S.
    nodes = S_CGVI == R_QUAD ? QuadratureRules.nodes(quad) :
            QuadratureRules.nodes(QuadratureRules.GaussLegendreQuadrature(T, S_CGVI))
    CGVI(CBF.Lagrange(nodes), quad)
end

# ---- helpers ----------------------------------------------------------------

solution_of(result) = result isa Tuple ? first(result) : result
qmatrix(sol) = reduce(hcat, [Float64.(collect(q)) for q in collect(sol.q[:])])   # D × (N+1)
pmatrix(sol) = reduce(hcat, [Float64.(collect(p)) for p in collect(sol.p[:])])

function reference_on_grid(qref_fine, h)
    stride = round(Int, h / DT_REF)
    abs(stride * DT_REF - h) < 1e-12 || error("h = $h is not a multiple of DT_REF = $DT_REF")
    qref_fine[:, 1:stride:end]
end

relerr(q, qref) = maximum(abs, q .- qref) / maximum(abs, qref)

function ham_drift(sol, ham, params)
    Q, P = qmatrix(sol), pmatrix(sol)
    H = [Float64(ham(0, Q[:, i], P[:, i], params)) for i in axes(Q, 2)]
    (isfinite(H[1]) && H[1] != 0) || return NaN
    maximum(abs.((H .- H[1]) ./ H[1]))
end

"""
    run_one(which, prob, method) -> (; status, sol, iters, secs)

Integrate `prob` with one method. For ShallowNet the integrator is built explicitly so the
final-step iteration count can be compared with the cap (`maxiter` status), like
`run_case` in shallownet_benchmark_common.jl.
"""
function run_one(which, prob, method)
    status, sol, iters, secs = "ok", nothing, NaN, NaN
    try
        if which == :shallownet
            int = GeometricIntegrator(prob, method;
                solver = NVI_SOLVER(), linesearch = NVI_LINESEARCH(),
                regularization_factor = T(NVI_REG), max_iterations = NVI_MAXIT)
            itcap = SimpleSolvers.config(solver(int)).max_iterations
            t0 = time()
            sol = solution_of(integrate(int))
            secs = time() - t0
            try
                iters = Float64(solverstate(int).iterations)
            catch
            end
            (!isnan(iters) && iters ≥ itcap) && (status = "maxiter")
        else
            t0 = time()
            sol = solution_of(integrate(prob, method))
            secs = time() - t0
        end
        any(x -> !isfinite(x), qmatrix(sol)) && (status = "nonfinite")
    catch e
        e isa InterruptException && rethrow()
        status = classify_error(e)
        @warn "$(which) failed" exception = (e, catch_backtrace())
        sol = nothing
    end
    (; status, sol, iters, secs)
end

# ---- sweep ------------------------------------------------------------------

function parse_args(args)
    final_time = DEFAULT_FINAL_TIME
    names = String[]
    for a in args
        if startswith(a, "--final-time=")
            final_time = parse(Float64, split(a, "=")[2])
        else
            push!(names, a)
        end
    end
    isempty(names) && (names = DEFAULT_PROBLEMS)
    known = [p.name for p in ALL_PROBLEMS]
    bad = filter(∉(known), names)
    isempty(bad) || error("unknown problem(s) $(bad); choose from $(known)")
    for h in DTS
        abs(round(final_time / h) * h - final_time) < 1e-9 ||
            error("final time $(final_time) is not a multiple of h = $(h)")
    end
    (filter(p -> p.name in names, ALL_PROBLEMS), final_time)
end

const NAME = "relu3_shallownet_vs_cgvi"
const CSV_COLS = "problem,h,steps,method,status,q_err,qend_err,ham_drift,q_diff_to_cgvi,iterations,secs"

function main(args)
    problems, final_time = parse_args(args)
    mkpath(RESULTS_DIR)
    csvpath = joinpath(RESULTS_DIR, "$(NAME).csv")

    println("="^100)
    println("ShallowNet(relu^$(K_RELU), S=$(S_NET), R=$(R_QUAD))  vs  CGVI(Lagrange, S=$(S_CGVI), R=$(R_QUAD))" *
            "   T=$(T), t ∈ (0, $(final_time))")
    println("="^100)

    sn_method = shallownet_method()          # symbolic build once
    cg_method = cgvi_method()
    rows = NamedTuple[]

    open(csvpath, "w") do io
        println(io, CSV_COLS)
        flush(io)
        for P in problems
            println("\n── $(P.name) ── reference: Gauss(8), dt = $(DT_REF)")
            ref = integrate(P.build((T(0), T(final_time)), T(DT_REF)), Gauss(8))
            qref_fine = qmatrix(ref)
            @printf("%-8s %-6s | %-10s %-11s %-11s %-11s | %-10s %-11s %-11s %-11s | %-9s %-11s %s\n",
                "h", "steps", "SN status", "SN q_err", "SN qend", "SN ΔH",
                "CG status", "CG q_err", "CG qend", "CG ΔH", "ratio", "|q_SN−q_CG|", "same?")
            println("-"^150)
            for h in DTS
                steps = round(Int, final_time / h)
                qref = reference_on_grid(qref_fine, h)
                prob = P.build((T(0), T(final_time)), T(h))

                res = Dict(:shallownet => run_one(:shallownet, prob, sn_method),
                           :cgvi => run_one(:cgvi, prob, cg_method))

                metric(r, f) = r.sol === nothing || r.status == "nonfinite" ? NaN : f(r.sol)
                m = Dict(k => (
                    q_err = metric(r, s -> size(qmatrix(s)) == size(qref) ? relerr(qmatrix(s), qref) : NaN),
                    qend_err = metric(r, s -> relerr(qmatrix(s)[:, end], qref[:, end])),
                    ham = metric(r, s -> ham_drift(s, P.ham, prob.parameters))) for (k, r) in res)

                sn, cg = res[:shallownet], res[:cgvi]
                qdiff = (sn.sol === nothing || cg.sol === nothing) ? NaN :
                        maximum(abs, qmatrix(sn.sol) .- qmatrix(cg.sol)) / maximum(abs, qref)
                ratio = m[:shallownet].q_err / m[:cgvi].q_err
                same = isfinite(ratio) && abs(ratio - 1) ≤ SAME_RTOL

                f(x) = isnan(x) ? "—" : @sprintf("%.3e", x)
                @printf("%-8g %-6d | %-10s %-11s %-11s %-11s | %-10s %-11s %-11s %-11s | %-9s %-11s %s\n",
                    h, steps, sn.status, f(m[:shallownet].q_err), f(m[:shallownet].qend_err),
                    f(m[:shallownet].ham), cg.status, f(m[:cgvi].q_err), f(m[:cgvi].qend_err),
                    f(m[:cgvi].ham), isnan(ratio) ? "—" : @sprintf("%.4f", ratio), f(qdiff),
                    same ? "yes" : "NO")

                for (k, label) in ((:shallownet, "ShallowNet"), (:cgvi, "CGVI"))
                    r = res[k]
                    println(io, join((P.name, csvnum(h), string(steps), label, r.status,
                            csvnum(m[k].q_err), csvnum(m[k].qend_err), csvnum(m[k].ham),
                            k == :shallownet ? csvnum(qdiff) : "0", csvint(r.iters),
                            csvnum(r.secs)), ","))
                end
                flush(io)
                push!(rows, (; problem = P.name, h, steps, sn_status = sn.status,
                    cg_status = cg.status, sn = m[:shallownet], cg = m[:cgvi],
                    ratio, qdiff, same, sn_secs = sn.secs, cg_secs = cg.secs))
            end
        end
    end
    println("\nWrote $(csvpath)")

    write_markdown(rows, problems, final_time)
    write_plot(rows, problems)
    return rows
end

# ---- report -----------------------------------------------------------------

function write_markdown(rows, problems, final_time)
    md = joinpath(RESULTS_DIR, "$(NAME).md")
    f(x) = isnan(x) ? "—" : @sprintf("%.3e", x)
    open(md, "w") do io
        println(io, "# ShallowNet (ReLU³, S=$(S_NET), R=$(R_QUAD)) vs CGVI (Lagrange, S=$(S_CGVI), R=$(R_QUAD))\n")
        println(io, "*Generated $(Dates.format(now(), "yyyy-mm-dd HH:MM")).*\n")
        println(io, "- Precision Float64, t ∈ (0, $(final_time)); ShallowNet: Newton/Backtracking, " *
                    "`regularization_factor = $(NVI_REG)`, `max_iterations = $(NVI_MAXIT)`, " *
                    "`dict_amount = $(DICT_AMOUNT)`, bias interval [−π, π].")
        println(io, "- Reference: Gauss(8) at dt = $(DT_REF), subsampled to each coarse grid.")
        println(io, "- `q_err` = max over the grid of ‖q − q_ref‖∞ / max‖q_ref‖∞; " *
                    "`|q_SN − q_CG|` uses the same normalisation.")
        println(io, "- **same** ⇔ |q_err(SN) / q_err(CGVI) − 1| ≤ $(SAME_RTOL).\n")
        for P in problems
            println(io, "## $(P.name)\n")
            println(io, "| h | steps | SN status | SN q_err | CGVI status | CGVI q_err | ratio | SN ΔH | CGVI ΔH | \\|q_SN − q_CG\\| | same? |")
            println(io, "|---|---|---|---|---|---|---|---|---|---|---|")
            for r in filter(r -> r.problem == P.name, rows)
                println(io, "| $(r.h) | $(r.steps) | $(r.sn_status) | $(f(r.sn.q_err)) | $(r.cg_status) | " *
                            "$(f(r.cg.q_err)) | $(isnan(r.ratio) ? "—" : @sprintf("%.4f", r.ratio)) | " *
                            "$(f(r.sn.ham)) | $(f(r.cg.ham)) | $(f(r.qdiff)) | $(r.same ? "yes" : "**no**") |")
            end
            n = count(r -> r.problem == P.name && r.same, rows)
            println(io, "\n$(n) / $(length(DTS)) step sizes at the same error level.\n")
        end
    end
    println("Wrote $(md)")
end

function write_plot(rows, problems)
    fig = Figure(size = (520 * length(problems), 420))
    for (j, P) in enumerate(problems)
        rs = filter(r -> r.problem == P.name, rows)
        ax = Axis(fig[1, j]; xscale = log10, yscale = log10, xlabel = "h",
            ylabel = "max relative error in q", title = P.name)
        pos(v) = [isfinite(x) && x > 0 ? x : NaN for x in v]
        hs = [r.h for r in rs]
        scatterlines!(ax, hs, pos([r.cg.q_err for r in rs]); label = "CGVI S$(S_CGVI) R$(R_QUAD)",
            marker = :circle, markersize = 14, linestyle = :dash)
        scatterlines!(ax, hs, pos([r.sn.q_err for r in rs]); label = "ShallowNet ReLU³ S$(S_NET) R$(R_QUAD)",
            marker = :xcross, markersize = 12)
        scatterlines!(ax, hs, pos([r.qdiff for r in rs]); label = "|q_SN − q_CGVI|",
            marker = :diamond, markersize = 8, linestyle = :dot, color = :gray)
        j == 1 && axislegend(ax; position = :lt)
    end
    png = joinpath(RESULTS_DIR, "$(NAME).pdf")
    save(png, fig)
    println("Wrote $(png)")
end

main(ARGS)
