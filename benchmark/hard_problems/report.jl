# Figures and summary of the hard-problems benchmark (PLAN.md §3).
#
#   julia --project=benchmark benchmark/hard_problems/report.jl
#
# Reads every results/<case>_baselines.csv, and <case>_nvi.csv where present, and writes per case
#   results/<case>_pareto.png       q error against wall time of the converged runs with q error
#                                   below MAX_ERR, with the Pareto front of each linear family and
#                                   of all linear methods together, and the network runs on top
#   results/<case>_invariants.png   invariant errors of the converged linear runs against h Ω, one
#                                   panel per family and invariant, one line per s (m = 1)
# and results/baselines.md (linear methods: Pareto-optimal runs, failure counts) and
# results/nvi.md (networks per run, failure rates, the N1 ≡ CGVI(P₃) check, and the "better"
# criterion of PLAN.md).
#
# A run counts as converged if its status is ok, i.e. the residual of every step is at most
# RES_TOL; for the problems in EXACT_INVARIANTS the summary cross-checks this criterion against an
# invariant that the methods preserve exactly.

using CairoMakie
using Printf

const RESULTS_DIR = joinpath(@__DIR__, "results")

# categorical slots 1–2 of the reference palette (dataviz skill), validated for all pairs
const FAMILY_COLORS = Dict("CGVI" => "#2a78d6", "Gauss" => "#eb6834")
const FRONT_COLOR = "#0b0b0b"
# the networks: categorical slot 3, told apart by the marker
const NVI_COLOR = "#1baf7a"
const NVI_MARKERS = Dict("N1 ReLU3 kinkfree S=4" => :utriangle, "N2 ReLU3 S=4" => :dtriangle,
    "N2 tanh S=4" => :diamond, "N2 tanh S=8" => :star5)
nvi_label(d, i) = "$(d["method"][i]) $(d["activation"][i]) S=$(Int(d["S"][i]))"

# ---- reading -----------------------------------------------------------------------------------

"Columns of a results CSV as a Dict of vectors (numbers parsed where possible)."
function read_csv(path)
    lines = readlines(path)
    header = split(lines[1], ",")
    rows = split.(lines[2:end], ",")
    parsecol(v) = all(x -> tryparse(Float64, x) !== nothing, v) ? parse.(Float64, v) : String.(v)
    Dict(String(h) => parsecol([r[i] for r in rows]) for (i, h) in enumerate(header))
end

const FIXED_COLUMNS = ["case", "c", "h", "family", "s", "m", "steps", "status", "unconverged",
    "max_res", "warnings", "q_err", "iterations", "secs", "method", "activation", "S", "R",
    "n_equiv", "n_spline", "n_degen", "diff_cgvi3", "secs_cgvi3"]
invariant_columns(d) = sort(filter(k -> k ∉ FIXED_COLUMNS, collect(keys(d))))

# A run converged iff every step's residual is at most RES_TOL (status ok, see integration.jl).
converged(d) = d["status"] .== "ok"

# Invariants that both CGVI and Gauss preserve exactly for an exact solution of their step
# equations (the angular momentum of the Kepler problem is quadratic and a Noether invariant). A
# converged run with a larger error than EXACT_TOL would contradict the residual criterion; their
# number is reported as a cross-check.
const EXACT_INVARIANTS = Dict("P3" => "L")
const EXACT_TOL = 1E-10

# Runs that converged and whose q error is below 1; a larger relative error is no solution, so
# these runs are left out of the fronts and figures.
const MAX_ERR = 1.0
accurate(d) = findall(converged(d) .& isfinite.(d["q_err"]) .& (0 .< d["q_err"] .< MAX_ERR))

"Indices of the runs on the Pareto front of (cost, err) among `idx`, by increasing cost."
function pareto(cost, err, idx)
    front = Int[]
    for i in idx[sortperm(cost[idx])]
        (isempty(front) || err[i] < err[front[end]]) && push!(front, i)
    end
    front
end

label(d, i) = d["family"][i] == "Gauss" ? "Gauss($(Int(d["s"][i])))" :
              "CGVI(P$(Int(d["s"][i])), m=$(Int(d["m"][i])))"

# ---- figures -----------------------------------------------------------------------------------

function pareto_figure(d, name, nvi)
    ok = accurate(d)
    fig = Figure(size = (640, 520))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "wall time [s]",
        ylabel = "max relative q error", title = name)
    for fam in ("CGVI", "Gauss")
        idx = filter(i -> d["family"][i] == fam, ok)
        isempty(idx) && continue
        scatter!(ax, d["secs"][idx], d["q_err"][idx]; color = (FAMILY_COLORS[fam], 0.35),
            markersize = 8, label = fam)
        f = pareto(d["secs"], d["q_err"], idx)
        stairs!(ax, d["secs"][f], d["q_err"][f]; step = :post, color = FAMILY_COLORS[fam],
            linewidth = 2, label = "$(fam) front")
    end
    f = pareto(d["secs"], d["q_err"], ok)
    isempty(f) || stairs!(ax, d["secs"][f], d["q_err"][f]; step = :post, color = FRONT_COLOR,
        linewidth = 1, linestyle = :dash, label = "linear front")
    if nvi !== nothing
        for i in accurate(nvi)
            scatter!(ax, [nvi["secs"][i]], [nvi["q_err"][i]]; color = NVI_COLOR, markersize = 12,
                marker = NVI_MARKERS[nvi_label(nvi, i)], strokecolor = :white, strokewidth = 1,
                label = nvi_label(nvi, i))
        end
    end
    isempty(ok) || Legend(fig[2, 1], ax; orientation = :horizontal, framevisible = false,
        nbanks = 3, merge = true)
    fig
end

function invariants_figure(d, name)
    invs = invariant_columns(d)
    fig = Figure(size = (max(640, 420 * length(invs)), 860))
    for (k, fam) in enumerate(("CGVI", "Gauss"))
        plotted = false
        local legax
        for (j, inv) in enumerate(invs)
            ax = Axis(fig[2k - 1, j]; xscale = log10, yscale = log10, xlabel = "h Ω",
                ylabel = "error of $(inv)", title = "$(name): $(fam)")
            ss = sort(unique(d["s"][d["family"] .== fam]))
            cmap = cgrad([:grey80, FAMILY_COLORS[fam]], max(length(ss), 2); categorical = true)
            for (l, s) in enumerate(ss)
                idx = findall((d["family"] .== fam) .& (d["s"] .== s) .& (d["m"] .== 1) .&
                              converged(d) .& isfinite.(d[inv]) .& (d[inv] .> 0))
                isempty(idx) && continue
                idx = idx[sortperm(d["c"][idx])]
                scatterlines!(ax, d["c"][idx], d[inv][idx]; color = cmap[l], linewidth = 2,
                    markersize = 8, label = "s = $(Int(s))")
                plotted = true
                legax = ax
            end
        end
        plotted && Legend(fig[2k, :], legax; orientation = :horizontal, framevisible = false,
            merge = true, nbanks = 2)
    end
    fig
end

# ---- summary -----------------------------------------------------------------------------------

"`selfcheck` of results/<case>_reference.txt: the difference of the Gauss(8) runs with dt and dt/2."
function selfcheck(name)
    line = filter(startswith("selfcheck"), readlines(joinpath(RESULTS_DIR, "$(name)_reference.txt")))
    parse(Float64, strip(split(only(line), "=")[2]))
end

function summary(io, d, name)
    println(io, "\n## $(name)\n")
    sc = selfcheck(name)
    sc > 0 && @printf(io, "Reference self-check (dt vs dt/2): %.2e. Errors below 10× this value (†) are not resolved by the reference.\n\n", sc)
    for fam in ("CGVI", "Gauss")
        idx = findall(d["family"] .== fam)
        conv = converged(d)
        bad = count(i -> !conv[i], idx)
        inaccurate = count(i -> conv[i] && !(d["q_err"][i] < MAX_ERR), idx)
        @printf(io, "- %s: %d runs, %d not converged (%.1f %%), %d converged with q error ≥ %g\n",
            fam, length(idx), bad, 100 * bad / length(idx), inaccurate, MAX_ERR)
    end
    inv = get(EXACT_INVARIANTS, first(split(name, "_")), nothing)
    if inv !== nothing
        n = count(i -> converged(d)[i] && !(d[inv][i] ≤ EXACT_TOL), eachindex(d["status"]))
        @printf(io, "- cross-check: %d converged runs with an error of %s above %g\n", n, inv,
            EXACT_TOL)
    end
    println(io, "\nPareto front of all linear methods (converged runs with q error < $(MAX_ERR)):\n")
    println(io, "| method | h Ω | q error | wall time [s] | Newton iterations |")
    println(io, "|---|---|---|---|---|")
    ok = accurate(d)
    for i in pareto(d["secs"], d["q_err"], ok)
        @printf(io, "| %s | %g | %.2e%s | %.3g | %d |\n", label(d, i), d["c"][i], d["q_err"][i],
            d["q_err"][i] < 10sc ? " †" : "", d["secs"][i], d["iterations"][i])
    end
    nothing
end

# ---- networks ----------------------------------------------------------------------------------

"""
Error of the linear Pareto front at wall time `t`: the smallest q error of an accurate linear run
that is at most as expensive (Inf if there is none).
"""
front_error(d, t) = minimum((d["q_err"][i] for i in accurate(d) if d["secs"][i] ≤ t); init = Inf)

# PLAN.md: a network is "better" iff its error is FACTOR times below the linear front at equal cost
const FACTOR = 3

fmt(x) = isfinite(x) ? @sprintf("%.2e", x) : "—"

function nvi_summary(io, nvi, d, name)
    println(io, "\n## $(name)\n")
    invs = invariant_columns(nvi)
    println(io, "| network | h Ω | status | unconv. steps | max residual | q error | ",
        join(invs, " | "), " | wall time [s] | P₃-equiv / spline / degen | diff to CGVI(P₃) | ",
        "linear front at that time |")
    println(io, "|", repeat("---|", 10 + length(invs)))
    for i in eachindex(nvi["status"])
        @printf(io, "| %s | %g | %s | %d | %s | %s | %s | %.3g | %s | %s | %s |\n",
            nvi_label(nvi, i), nvi["c"][i], nvi["status"][i], nvi["unconverged"][i],
            fmt(nvi["max_res"][i]), fmt(nvi["q_err"][i]), join(fmt.(nvi[k][i] for k in invs), " | "),
            nvi["secs"][i], nvi["activation"][i] == "tanh" ? "—" :
                            "$(Int(nvi["n_equiv"][i])) / $(Int(nvi["n_spline"][i])) / $(Int(nvi["n_degen"][i]))",
            fmt(nvi["diff_cgvi3"][i]), fmt(front_error(d, nvi["secs"][i])))
    end
    println(io)
    acc = accurate(nvi)
    for lab in unique(nvi_label(nvi, i) for i in eachindex(nvi["status"]))
        idx = findall(i -> nvi_label(nvi, i) == lab, eachindex(nvi["status"]))
        conv = converged(nvi)
        bad = count(i -> !conv[i], idx)
        better = count(i -> i in acc && isfinite(front_error(d, nvi["secs"][i])) &&
                            nvi["q_err"][i] * FACTOR < front_error(d, nvi["secs"][i]), idx)
        @printf(io, "- %s: %d runs, %d not converged (%.0f %%), %d better than the linear front by %d×\n",
            lab, length(idx), bad, 100 * bad / length(idx), better, FACTOR)
    end
    # N1 ≡ CGVI(P₃) whenever every step is P₃-equivalent and converged (Theorem 1)
    eq = findall(i -> nvi["method"][i] == "N1" && conv_all_equiv(nvi, i), eachindex(nvi["status"]))
    isempty(eq) || @printf(io, "- N1 with every step P₃-equivalent and converged: %d runs, max diff to CGVI(P₃) %s\n",
        length(eq), fmt(maximum(nvi["diff_cgvi3"][eq])))
    nothing
end

conv_all_equiv(nvi, i) = nvi["status"][i] == "ok" && nvi["n_spline"][i] == 0 &&
                         nvi["n_degen"][i] == 0 && nvi["n_equiv"][i] > 0

function main()
    files = sort(filter(f -> endswith(f, "_baselines.csv"), readdir(RESULTS_DIR)))
    nvifile(name) = joinpath(RESULTS_DIR, "$(name)_nvi.csv")
    open(joinpath(RESULTS_DIR, "nvi.md"), "w") do io
        println(io, "# Networks of the package on the hard problems (phase 2)\n")
        println(io, """
        N1 = ShallowNet(ReLU³, S = 4, R = 4) with the kink-free bias interval [1.1, π]; N2 = the same
        with [-π, π], and ShallowNet(tanh) with S = R = 4 and S = R = 8. DogLeg, regularisation 1e-5,
        ≤ 1000 iterations per step. Step classes are counted per step and degree of freedom. "Linear
        front at that time": the smallest q error of an accurate linear run that took at most as long.""")
        for f in files
            name = replace(f, "_baselines.csv" => "")
            isfile(nvifile(name)) || continue
            nvi_summary(io, read_csv(nvifile(name)), read_csv(joinpath(RESULTS_DIR, f)), name)
        end
    end
    open(joinpath(RESULTS_DIR, "baselines.md"), "w") do io
        println(io, "# Linear baselines (hard problems, phase 1)\n")
        println(io, """
        L1 = CGVI(P_s, R = s + 1), s = 2..6, with m = 1..4 substeps of h/m; L2 = Gauss(s), s = 1..6,
        without substeps. h Ω ∈ {0.1, 0.3, 1, 3, 10}, Ω = 1 (ω ≈ 1 for P1 and P4, the mean motion for
        P3). q error: max over the grid n h (t ≤ Terr) of the relative ∞-norm error. A run is
        converged if no solve threw and the residual ∞-norm of every step is at most 1e-10. Wall
        times include no compilation; runs under 0.1 s are the minimum of 3 repeats.""")
        for f in files
            name = replace(f, "_baselines.csv" => "")
            d = read_csv(joinpath(RESULTS_DIR, f))
            nvi = isfile(nvifile(name)) ? read_csv(nvifile(name)) : nothing
            save(joinpath(RESULTS_DIR, "$(name)_pareto.png"), pareto_figure(d, name, nvi))
            save(joinpath(RESULTS_DIR, "$(name)_invariants.png"), invariants_figure(d, name))
            summary(io, d, name)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
