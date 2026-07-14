# Reporting for the `NonLinear_OneLayer_GML` benchmark.
#
# Reads the CSV(s) produced by the sweep (so a report can be regenerated without
# re-running the sweep), draws CairoMakie plots — convergence, and metric-vs-timestep
# scatters for accuracy, energy drift, run time and nonlinear iterations (in the spirit
# of SolverBenchmark.jl) — and writes a markdown summary with tables and findings keyed
# to the benchmark goals:
#   (a) which configs work well per problem, (b) failure hot-spots, (c) robust solvers.
# Used both by the per-problem run files and by the aggregating `report.jl`.

using CairoMakie
using Printf
using Dates
using Statistics: median

# ---- CSV parsing (no quoting needed; no field contains commas) --------------

_parsef(s) = (s == "NaN" || isempty(s)) ? NaN : parse(Float64, s)

function read_results(paths::AbstractVector{<:AbstractString})
    rows = NamedTuple[]
    for path in paths
        isfile(path) || continue
        lines = readlines(path)
        length(lines) <= 1 && continue
        for ln in lines[2:end]
            isempty(strip(ln)) && continue
            f = split(ln, ",")
            length(f) == 17 || continue
            push!(rows, (problem = String(f[1]), T = String(f[2]), dt = _parsef(f[3]),
                         steps = round(Int, _parsef(f[4])), R = round(Int, _parsef(f[5])),
                         S = round(Int, _parsef(f[6])), activation = String(f[7]),
                         solver = String(f[8]), linesearch = String(f[9]),
                         initial_guess = String(f[10]), lambda = _parsef(f[11]),
                         status = String(f[12]), ref_err = _parsef(f[13]),
                         ham_drift = _parsef(f[14]), iterations = _parsef(f[15]),
                         solve_secs = _parsef(f[16]), total_secs = _parsef(f[17])))
        end
    end
    return rows
end
read_results(path::AbstractString) = read_results([path])

# ---- helpers ----------------------------------------------------------------

strategy_label(r) = r.solver == "DogLeg" ? "DogLeg" : "$(r.solver)/$(r.linesearch)"
is_ok(r) = r.status == "ok"

# Group rows by a key function, returning key => Vector{row} with keys sorted.
function groupby(rows, keyfn)
    d = Dict{Any,Vector{Any}}()
    for r in rows
        push!(get!(() -> Any[], d, keyfn(r)), r)
    end
    return sort(collect(d), by = p -> string(first(p)))
end

_median_finite(xs) = (v = filter(isfinite, xs); isempty(v) ? NaN : median(v))

# A one-line summary of a group of rows.
function group_stats(rows)
    n = length(rows); ok = count(is_ok, rows)
    okrows = filter(is_ok, rows)
    (n = n, ok = ok, frac = n == 0 ? 0.0 : ok / n,
     med_ref  = _median_finite([r.ref_err   for r in okrows]),
     med_ham  = _median_finite([r.ham_drift for r in okrows]),
     med_iter = _median_finite([r.iterations for r in okrows]),
     med_secs = _median_finite([r.solve_secs for r in okrows]))
end

fmt_pct(x)  = @sprintf("%.0f%%", 100x)
fmt_sci(x)  = isnan(x) ? "—" : @sprintf("%.2e", x)
fmt_iter(x) = isnan(x) ? "—" : string(round(Int, x))
fmt_secs(x) = isnan(x) ? "—" : @sprintf("%.3f", x)

# ---- plots ------------------------------------------------------------------

# Red/green endpoints for the convergence colour scale (red = not converged, green = converged).
const CONV_RED   = Makie.RGBf(163 / 255, 49 / 255, 42 / 255)
const CONV_GREEN = Makie.RGBf(74 / 255, 137 / 255, 92 / 255)

# Success-fraction bar chart over a categorical axis (solver strategy / initial guess).
function plot_success_bars(rows, keyfn, xlabel, title, path)
    isempty(rows) && return false
    groups = groupby(rows, keyfn)
    labels = [string(k) for (k, _) in groups]
    fracs  = [group_stats(v).frac for (_, v) in groups]
    fig = Figure(size = (max(500, 120 * length(labels)), 380))
    ax = Axis(fig[1, 1]; xlabel = xlabel, ylabel = "Success rate (status = ok)",
              title = title, xticks = (1:length(labels), labels),
              xticklabelrotation = π/8, limits = (nothing, (0, 1.05)))
    barplot!(ax, 1:length(labels), fracs; color = :steelblue)
    text!(ax, 1:length(labels), fracs; text = fmt_pct.(fracs), align = (:center, :bottom), offset = (0, 2))
    save(path, fig)
    return true
end

# Heatmap of success rate: solver strategy (rows) × precision (cols).
function plot_success_heatmap(rows, path)
    isempty(rows) && return false
    strategies = sort(unique(strategy_label.(rows)))
    precisions = sort(unique(r.T for r in rows))
    M = fill(NaN, length(strategies), length(precisions))
    for (i, s) in enumerate(strategies), (j, p) in enumerate(precisions)
        sub = [r for r in rows if strategy_label(r) == s && r.T == p]
        isempty(sub) || (M[i, j] = group_stats(sub).frac)
    end
    fig = Figure(size = (150 * length(precisions) + 300, 90 * length(strategies) + 150))
    ax = Axis(fig[1, 1]; title = "Convergence: success rate by solver × precision",
              xticks = (1:length(precisions), precisions),
              yticks = (1:length(strategies), strategies))
    hm = heatmap!(ax, 1:length(precisions), 1:length(strategies), permutedims(M);
                  colormap = cgrad([CONV_RED, CONV_GREEN]), colorrange = (0, 1))
    for (i, s) in enumerate(strategies), (j, p) in enumerate(precisions)
        isnan(M[i, j]) || text!(ax, j, i; text = fmt_pct(M[i, j]), align = (:center, :center),
                                color = :white)
    end
    Colorbar(fig[1, 2], hm; label = "Success rate")
    save(path, fig)
    return true
end

# Scatter of a metric vs timestep, one colour per series (with a small multiplicative
# x-jitter per series so overlapping points at the same dt stay distinguishable).
# `colorby` selects the series field: `:T` (precision — the default, used by per-problem
# reports) or `:problem` (used by the combined report so the four problems stay
# distinguishable). Each dot is one converged case; the spread at a given dt reflects the
# remaining swept axes (activation, R, S, solver, λ, initial guess).
function plot_metric_vs_dt(rows, field, ylabel, title, path;
                           ylog = true, colorby = :T, colortitle = "Precision")
    getv(r) = Float64(getproperty(r, field))
    data = [(r.dt, getv(r), string(getproperty(r, colorby))) for r in rows
            if is_ok(r) && isfinite(getv(r)) && isfinite(r.dt) && (!ylog || getv(r) > 0)]
    isempty(data) && return false
    series = sort(unique(s for (_, _, s) in data))
    palette = Makie.wong_colors()
    fig = Figure(size = (760, 440))
    ax = Axis(fig[1, 1]; xlabel = "Timestep dt", ylabel = ylabel, title = title,
              xscale = log10, yscale = ylog ? log10 : identity)
    for (i, p) in enumerate(series)
        sub = [(d, v) for (d, v, s) in data if s == p]
        isempty(sub) && continue
        j = 1 + 0.05 * (i - (length(series) + 1) / 2)
        # Marker style matching SolverBenchmark.jl (markersize 12, thin stroke).
        scatter!(ax, [d * j for (d, _) in sub], [v for (_, v) in sub];
                 label = p, color = palette[mod1(i, length(palette))],
                 markersize = 12, strokewidth = 0.5)
    end
    # Legend outside the axes (SolverBenchmark-style), to the right.
    Legend(fig[1, 2], ax, colortitle; framevisible = false)
    save(path, fig)
    return true
end

# ---- markdown report --------------------------------------------------------

function _table(io, header, rows_of_cells)
    println(io, "| " * join(header, " | ") * " |")
    println(io, "|" * join(fill("---", length(header)), "|") * "|")
    for cells in rows_of_cells
        println(io, "| " * join(cells, " | ") * " |")
    end
    println(io)
end

function _stats_table(io, rows, keyfn, colname)
    groups = groupby(rows, keyfn)
    cells = [[string(k), string(s.n), string(s.ok), fmt_pct(s.frac),
              fmt_sci(s.med_ref), fmt_sci(s.med_ham), fmt_iter(s.med_iter), fmt_secs(s.med_secs)]
             for (k, v) in groups for s in (group_stats(v),)]
    _table(io, [colname, "n", "ok", "success", "med ref_err", "med ham_drift", "med iter", "med solve_s"], cells)
end

# Best (lowest ref_err among converged) configuration per problem.
function _best_configs(io, rows)
    rowsout = Vector{Vector{String}}()
    for (prob, v) in groupby(rows, r -> r.problem)
        cand = [r for r in v if is_ok(r) && isfinite(r.ref_err)]
        if isempty(cand)
            push!(rowsout, [string(prob), "— no converged run with a finite reference error —", "", "", "", "", ""])
            continue
        end
        b = cand[argmin([r.ref_err for r in cand])]
        push!(rowsout, [string(prob), fmt_sci(b.ref_err), b.T, @sprintf("%.3g", b.dt),
                        "R$(b.R) S$(b.S) $(b.activation)", strategy_label(b),
                        "$(b.initial_guess), λ=$(@sprintf("%.1e", b.lambda))"])
    end
    _table(io, ["problem", "best ref_err", "T", "dt", "network", "solver", "iguess/λ"], rowsout)
end

"""
    write_report(rows; title, mode, outdir, prefix)

Generate the plots (PNG) and a markdown report from parsed `rows` into `outdir`.
Returns the markdown path.
"""
function write_report(rows; title, mode, outdir, prefix)
    mkpath(outdir)
    # When more than one problem is present (the combined report), colour the metric
    # scatters by problem so the problems stay distinguishable, and add a
    # convergence-by-problem bar chart. A single-problem report keeps the precision
    # colouring and omits the (trivial) problem bar.
    multi = length(unique(r.problem for r in rows)) > 1
    colorby    = multi ? :problem : :T
    colortitle = multi ? "Problem" : "Precision"

    # (name, generated?) — plots are skipped when there is no data to show.
    p_solver  = "$(prefix)_convergence_solver.png"
    p_iguess  = "$(prefix)_convergence_iguess.png"
    p_problem = "$(prefix)_convergence_problem.png"
    p_heat    = "$(prefix)_convergence_heatmap.png"
    p_acc     = "$(prefix)_accuracy_vs_dt.png"
    p_energy  = "$(prefix)_energy_drift_vs_dt.png"
    p_time    = "$(prefix)_runtime_vs_dt.png"
    p_iter    = "$(prefix)_iterations_vs_dt.png"

    have_solver = plot_success_bars(rows, strategy_label, "Solver strategy",
                                    "Convergence by solver strategy", joinpath(outdir, p_solver))
    have_iguess = length(unique(r.initial_guess for r in rows)) > 1 &&
                  plot_success_bars(rows, r -> r.initial_guess, "Initial-guess strategy",
                                    "Convergence by initial guess", joinpath(outdir, p_iguess))
    have_problem = multi &&
                  plot_success_bars(rows, r -> r.problem, "Problem",
                                    "Convergence by problem", joinpath(outdir, p_problem))
    have_heat   = plot_success_heatmap(rows, joinpath(outdir, p_heat))
    have_acc    = plot_metric_vs_dt(rows, :ref_err,   "Relative error vs reference",
                                    "Accuracy vs timestep",        joinpath(outdir, p_acc);
                                    colorby = colorby, colortitle = colortitle)
    have_energy = plot_metric_vs_dt(rows, :ham_drift, "Relative Hamiltonian drift",
                                    "Energy drift vs timestep",    joinpath(outdir, p_energy);
                                    colorby = colorby, colortitle = colortitle)
    have_time   = plot_metric_vs_dt(rows, :solve_secs, "Solve time [s]",
                                    "Run time vs timestep",        joinpath(outdir, p_time);
                                    colorby = colorby, colortitle = colortitle)
    have_iter   = plot_metric_vs_dt(rows, :iterations, "Nonlinear iterations (final step)",
                                    "Nonlinear iterations vs timestep", joinpath(outdir, p_iter);
                                    ylog = false, colorby = colorby, colortitle = colortitle)

    ntot = length(rows); nok = count(is_ok, rows)
    md = joinpath(outdir, "$(prefix).md")
    open(md, "w") do io
        println(io, "# $(title)\n")
        println(io, "*Generated $(Dates.format(now(), "yyyy-mm-dd HH:MM")) — mode `$(mode)`.*\n")
        println(io, "- Total cases: **$(ntot)**  •  converged (`ok`): **$(nok)** ($(fmt_pct(ntot == 0 ? 0.0 : nok/ntot)))")
        println(io, "- Each case integrates **10 steps**. `ref_err` is the relative max-norm error")
        println(io, "  of the final state vs a `Gauss(8)` / Float64 reference at the smallest timestep;")
        println(io, "  `ham_drift` is the max relative Hamiltonian drift; `iter` is the nonlinear-solver")
        println(io, "  iteration count of the final step; `solve_s` is the summed nonlinear-solve time.\n")

        println(io, "## Status breakdown\n")
        statuses = sort(collect(Set(r.status for r in rows)))
        _table(io, ["status", "count"], [[s, string(count(r -> r.status == s, rows))] for s in statuses])

        println(io, "## Convergence & robustness (goal c)\n")
        println(io, "### By solver strategy\n")
        _stats_table(io, rows, strategy_label, "solver")
        have_solver && println(io, "![convergence by solver]($(p_solver))\n")
        have_heat   && println(io, "![convergence heatmap]($(p_heat))\n")

        if length(unique(r.initial_guess for r in rows)) > 1
            println(io, "### By initial-guess strategy\n")
            _stats_table(io, rows, r -> r.initial_guess, "initial guess")
            have_iguess && println(io, "![convergence by initial guess]($(p_iguess))\n")
        end

        println(io, "### By precision\n")
        _stats_table(io, rows, r -> r.T, "precision")

        if length(unique(r.problem for r in rows)) > 1
            println(io, "### By problem\n")
            _stats_table(io, rows, r -> r.problem, "problem")
            have_problem && println(io, "![convergence by problem]($(p_problem))\n")
        end

        println(io, "## Performance metrics vs timestep\n")
        println(io, "Each dot is one converged case, coloured by precision (small x-jitter per")
        println(io, "precision keeps overlapping points visible).\n")
        have_acc    && println(io, "![accuracy]($(p_acc))\n")
        have_energy && println(io, "![energy drift]($(p_energy))\n")
        have_time   && println(io, "![run time]($(p_time))\n")
        have_iter   && println(io, "![nonlinear iterations]($(p_iter))\n")

        println(io, "## Best configuration per problem (goal a)\n")
        _best_configs(io, rows)

        println(io, "## Failure hot-spots (goal b)\n")
        fails = [r for r in rows if !is_ok(r)]
        if isempty(fails)
            println(io, "No failures recorded.\n")
        else
            println(io, "Non-`ok` cases grouped by precision and timestep:\n")
            _table(io, ["precision", "dt", "failures"],
                   [[k[1], @sprintf("%.3g", k[2]), string(length(v))]
                    for (k, v) in groupby(fails, r -> (r.T, r.dt))])
        end
    end
    return md
end
