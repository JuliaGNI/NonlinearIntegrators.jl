# Figures and summary of the hard-problems benchmark (PLAN.md §3).
#
#   julia --project=benchmark benchmark/hard_problems/report.jl
#
# Reads every results/<case>_baselines.csv and writes, per case,
#   results/<case>_pareto.png       q error against wall time of the converged runs with q error
#                                   below MAX_ERR, with the Pareto front of each family and of all
#                                   linear methods together
#   results/<case>_invariants.png   invariant errors of the converged runs against h Ω, one panel
#                                   per family and invariant, one line per s (m = 1)
#
# A run counts as converged if its status is ok and, for the problems in EXACT_INVARIANTS, the
# error of an invariant that the methods preserve exactly stays below EXACT_TOL.
# and results/baselines.md with the Pareto-optimal runs and the failure counts.

using CairoMakie
using Printf

const RESULTS_DIR = joinpath(@__DIR__, "results")

# categorical slots 1–2 of the reference palette (dataviz skill), validated for all pairs
const FAMILY_COLORS = Dict("CGVI" => "#2a78d6", "Gauss" => "#eb6834")
const FRONT_COLOR = "#0b0b0b"

# ---- reading -----------------------------------------------------------------------------------

"Columns of a results CSV as a Dict of vectors (numbers parsed where possible)."
function read_csv(path)
    lines = readlines(path)
    header = split(lines[1], ",")
    rows = split.(lines[2:end], ",")
    parsecol(v) = all(x -> tryparse(Float64, x) !== nothing, v) ? parse.(Float64, v) : String.(v)
    Dict(String(h) => parsecol([r[i] for r in rows]) for (i, h) in enumerate(header))
end

const FIXED_COLUMNS = ["case", "c", "h", "family", "s", "m", "steps", "status", "floor",
    "unconverged", "q_err", "iterations", "secs"]
invariant_columns(d) = sort(filter(k -> k ∉ FIXED_COLUMNS, collect(keys(d))))

# Invariants that both CGVI and Gauss preserve exactly for an exact solution of their step
# equations (the angular momentum of the Kepler problem is quadratic and a Noether invariant). A
# larger error than EXACT_TOL shows that the solver accepted steps that do not solve them.
const EXACT_INVARIANTS = Dict("P3" => "L")
const EXACT_TOL = 1E-10

function converged(d)
    ok = d["status"] .== "ok"
    inv = get(EXACT_INVARIANTS, first(split(d["case"][1], "_")), nothing)
    inv === nothing ? ok : ok .& (d[inv] .< EXACT_TOL)
end

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

function pareto_figure(d, name)
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
    isempty(ok) || Legend(fig[2, 1], ax; orientation = :horizontal, framevisible = false,
        nbanks = 2)
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

function main()
    files = sort(filter(f -> endswith(f, "_baselines.csv"), readdir(RESULTS_DIR)))
    open(joinpath(RESULTS_DIR, "baselines.md"), "w") do io
        println(io, "# Linear baselines (hard problems, phase 1)\n")
        println(io, """
        L1 = CGVI(P_s, R = s + 1), s = 2..6, with m = 1..4 substeps of h/m; L2 = Gauss(s), s = 1..6,
        without substeps. h Ω ∈ {0.1, 0.3, 1, 3, 10}, Ω = 1 (ω ≈ 1 for P1 and P4, the mean motion for
        P3). q error: max over the grid n h (t ≤ Terr) of the relative ∞-norm error. A run is
        converged if no solve gave up or threw and, for P3, the angular momentum (preserved exactly by
        both families) is conserved to $(EXACT_TOL). Wall times include no
        compilation; runs under 0.1 s are the minimum of 3 repeats.""")
        for f in files
            name = replace(f, "_baselines.csv" => "")
            d = read_csv(joinpath(RESULTS_DIR, f))
            save(joinpath(RESULTS_DIR, "$(name)_pareto.png"), pareto_figure(d, name))
            save(joinpath(RESULTS_DIR, "$(name)_invariants.png"), invariants_figure(d, name))
            summary(io, d, name)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
