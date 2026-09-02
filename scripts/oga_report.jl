# Reporting for the OGA studies (Tier A seed study, Tier B integrator sweeps).
#
# Reads the CSVs the studies write, so a report can be regenerated without re-running
# anything, and emits CairoMakie figures plus a markdown summary.
#
# ---- colour ------------------------------------------------------------------
#
# Three rules, and they are not stylistic:
#
#   * Magnitude (fit error, condition number, success rate) is encoded on a **single-hue
#     sequential ramp**, light → dark. Not a rainbow, and specifically *not* the red→green
#     ramp that success-rate grids usually reach for: red↔green is the one pair
#     red–green colourblind readers cannot separate, which is 8% of men.
#   * Every heatmap cell also carries its **numeric value as text**, so nothing is encoded
#     by colour alone and the figure doubles as a table.
#   * Series identity (precision) uses a **fixed categorical order**, never a cycled or
#     generated hue, and lines are direct-labelled as well as legended.
#
# The palette below is validated for colour-vision deficiency: worst all-pairs CVD ΔE 9.2,
# worst normal-vision ΔE 24.0 (OKLab ×100).

using CairoMakie
using Printf
using Statistics: median

const RESULTS_DIR = joinpath(@__DIR__, "results")

# Categorical slots, in fixed assignment order: blue, orange, aqua.
const OGA_SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
# Single-hue sequential ramp (blue 100 → 700) for magnitude.
const OGA_SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
const OGA_INK = "#0b0b0b"
const OGA_INK_MUTED = "#52514e"
const OGA_SURFACE = "#fcfcfb"
const OGA_GRID = "#e6e5e1"

# Precisions in a fixed order, so a given precision keeps its colour across every figure
# regardless of which ones a particular sweep happened to produce.
const OGA_PRECISIONS = ("Float16", "Float32", "Float64")
function precision_colour(T::AbstractString)
    (i = findfirst(==(T), OGA_PRECISIONS); OGA_SERIES[i === nothing ? 1 : i])
end

# ---- CSV --------------------------------------------------------------------

csvnum(x::Integer) = string(x)
csvnum(x) = isfinite(x) ? @sprintf("%.8e", x) : "NaN"

"""
    read_oga_csv(path) -> Vector{Dict{String,String}}

Read a CSV written by one of the OGA studies, keyed by its header. Kept schema-agnostic so
the two studies can carry different columns without two parsers; no field contains a comma,
so no quoting is needed.
"""
function read_oga_csv(path::AbstractString)
    rows = Dict{String, String}[]
    isfile(path) || return rows
    lines = readlines(path)
    length(lines) <= 1 && return rows
    header = split(lines[1], ",")
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        f = split(ln, ",")
        length(f) == length(header) || continue
        push!(rows, Dict(String(header[i]) => String(f[i]) for i in eachindex(header)))
    end
    return rows
end

function read_oga_csv(paths::AbstractVector)
    reduce(vcat, (read_oga_csv(p) for p in paths); init = Dict{String, String}[])
end

fnum(r, k) = (v = get(r, k, "NaN"); (v == "NaN" || isempty(v)) ? NaN : parse(Float64, v))
inum(r, k) = (v = fnum(r, k); isfinite(v) ? round(Int, v) : -1)

# ---- helpers ----------------------------------------------------------------

function oga_groupby(rows, keyfn)
    d = Dict{Any, Vector{Any}}()
    for r in rows
        push!(get!(() -> Any[], d, keyfn(r)), r)
    end
    return d
end

oga_median(xs) = (v = filter(isfinite, collect(xs)); isempty(v) ? NaN : median(v))
oga_ok(r) = get(r, "status", "") == "ok"
variant_label(r) = string(r["dictionary"], "/", r["selection"], "/", r["fit"])

fmt_e(x) = isfinite(x) ? @sprintf("%.2e", x) : "—"
fmt_pct(x) = isfinite(x) ? @sprintf("%.0f%%", 100x) : "—"

# Sorted unique values of a column, with the fixed precision order honoured.
function levels(rows, key)
    vals = unique(String[r[key] for r in rows])
    key == "T" && return [t for t in OGA_PRECISIONS if t in vals]
    return sort(vals)
end

# ---- the heatmap primitive --------------------------------------------------

"""
    labelled_heatmap!(ax, values; kwargs...)

A magnitude heatmap on the single-hue sequential ramp, with every cell's value written
into it. `fmt` renders a cell value as text; `NaN` cells are left blank on the surface
colour, so "no data" never reads as "low value".

The in-cell text is what makes this accessible: it is the table view, and it satisfies the
relief rule for the lighter ramp steps, whose contrast against the surface is below 3:1.
"""
function labelled_heatmap!(
        ax, values::Matrix{Float64}; fmt = fmt_e, lo = nothing, hi = nothing)
    finite = filter(isfinite, vec(values))
    lo = lo === nothing ? (isempty(finite) ? 0.0 : minimum(finite)) : lo
    hi = hi === nothing ? (isempty(finite) ? 1.0 : maximum(finite)) : hi
    hi ≤ lo && (hi = lo + 1)

    cmap = cgrad(OGA_SEQ)
    heatmap!(ax, 1:size(values, 1), 1:size(values, 2), values;
        colormap = cmap, colorrange = (lo, hi), nan_color = OGA_SURFACE)

    for i in axes(values, 1), j in axes(values, 2)

        v = values[i, j]
        isfinite(v) || continue
        # Ink flips to the light token on the dark end of the ramp so the label keeps its
        # contrast against the cell rather than against the surface.
        frac = (v - lo) / (hi - lo)
        text!(ax, i, j; text = fmt(v), align = (:center, :center), fontsize = 9,
            color = frac > 0.55 ? OGA_SURFACE : OGA_INK)
    end
    return ax
end

function styled_axis!(fig, pos; title = "", xlabel = "", ylabel = "",
        xticks = nothing, yticks = nothing, xticklabelrotation = 0.0,
        yscale = identity)
    ax = Axis(
        fig[pos...]; title = title, xlabel = xlabel, ylabel = ylabel, yscale = yscale,
        backgroundcolor = OGA_SURFACE, titlecolor = OGA_INK,
        xlabelcolor = OGA_INK_MUTED, ylabelcolor = OGA_INK_MUTED,
        xticklabelcolor = OGA_INK_MUTED, yticklabelcolor = OGA_INK_MUTED,
        xgridcolor = OGA_GRID, ygridcolor = OGA_GRID,
        titlesize = 12, xlabelsize = 10, ylabelsize = 10,
        xticklabelsize = 9, yticklabelsize = 9,
        xticklabelrotation = xticklabelrotation)
    xticks !== nothing && (ax.xticks = (1:length(xticks), xticks))
    yticks !== nothing && (ax.yticks = (1:length(yticks), yticks))
    return ax
end

# ---- Tier A report ----------------------------------------------------------

"""
    write_fit_study_report(csvpath) -> (mdpath, pngpaths)

Summarise the seed-quality study: which (dictionary, selection, fit) combinations keep the
fit accurate and the design matrix away from rank deficiency, per precision.
"""
function write_fit_study_report(csvpath::AbstractString)
    rows = read_oga_csv(csvpath)
    isempty(rows) && (@warn "no rows in $csvpath"; return (nothing, String[]))

    Ts = levels(rows, "T")
    dicts = levels(rows, "dictionary")
    sels = levels(rows, "selection")
    fits = levels(rows, "fit")
    pngs = String[]

    # One figure per metric: rows = fit, columns = selection, one panel per (dictionary, T).
    for (metric, key, fmt, transform) in (
        ("fit error", "fit_err", (x -> fmt_e(exp10(x))), log10),
        ("condition number", "cond", (x -> fmt_e(exp10(x))), log10))
        fig = Figure(size = (240 * length(dicts) + 160, 200 * length(Ts) + 80),
            backgroundcolor = OGA_SURFACE)
        Label(fig[0, 1:length(dicts)],
            "OGA seed $(metric) — median over activations and targets (log scale, darker = larger)";
            fontsize = 14, color = OGA_INK, font = :bold)

        # Build every panel first, then take the colour range from the *plotted medians*.
        # Ranging over the raw rows instead would stretch the scale to cover per-activation
        # spread that no cell displays, flattening the medians into one indistinguishable
        # shade — the encoding has to be scaled to what is actually on screen.
        panels = Dict{Tuple{Int, Int}, Matrix{Float64}}()
        for (ti, T) in enumerate(Ts), (di, dict) in enumerate(dicts)

            M = fill(NaN, length(fits), length(sels))
            for (fi, fit) in enumerate(fits), (si, sel) in enumerate(sels)

                sub = [r
                       for r in rows
                       if r["T"] == T && r["dictionary"] == dict &&
                              r["fit"] == fit && r["selection"] == sel]
                M[fi, si] = oga_median(transform(x)
                for x in (fnum(r, key) for r in sub)
                if isfinite(x) && x > 0)
            end
            panels[(ti, di)] = M
        end
        # A comprehension, not `vec.(values(panels))`: a `Dict` value iterator has no
        # `axes`, so it is iterable but not broadcastable.
        plotted = filter(isfinite, reduce(vcat, [vec(M) for M in values(panels)]))
        lo, hi = isempty(plotted) ? (0.0, 1.0) : (minimum(plotted), maximum(plotted))

        for (ti, T) in enumerate(Ts), (di, dict) in enumerate(dicts)

            ax = styled_axis!(fig, (ti, di);
                title = "$(T) · $(dict)",
                xticks = fits, yticks = di == 1 ? sels : fill("", length(sels)),
                xticklabelrotation = pi / 4)
            labelled_heatmap!(ax, panels[(ti, di)]; fmt = fmt, lo = lo, hi = hi)
        end
        png = joinpath(RESULTS_DIR, "oga_fit_study_" * replace(metric, " " => "_") * ".png")
        save(png, fig)
        push!(pngs, png)
    end

    mdpath = joinpath(RESULTS_DIR, "oga_fit_study.md")
    open(mdpath, "w") do io
        println(io, "# OGA seed-quality study (Tier A)\n")
        println(io, "Greedy fit only — no integrator, no Newton solve. `fit_err` is the ")
        println(io, "quadrature-weighted L² error of the seed, recomputed in `Float64` from the")
        println(io, "returned parameters so precisions share one scale; `cond`/`σ_min` describe the")
        println(io, "seed's design matrix, the proxy for whether the Newton system it feeds is")
        println(io, "solvable.\n")
        println(io, "$(length(rows)) cases. Failures (non-finite or thrown): ",
            count(!oga_ok, rows), ".\n")

        println(io, "## Median fit error by variant and precision\n")
        print(io, "| dictionary | selection | fit |")
        for T in Ts
            print(io, " $T err | $T cond | $T σ_min |")
        end
        println(io)
        print(io, "|---|---|---|")
        for _ in Ts
            print(io, "---|---|---|")
        end
        println(io)
        for dict in dicts, sel in sels, fit in fits
            print(io, "| `$dict` | `$sel` | `$fit` |")
            for T in Ts
                sub = [r
                       for r in rows
                       if r["T"] == T && r["dictionary"] == dict &&
                              r["fit"] == fit && r["selection"] == sel]
                print(io, " ", fmt_e(oga_median(fnum(r, "fit_err") for r in sub)),
                    " | ", fmt_e(oga_median(fnum(r, "cond") for r in sub)),
                    " | ", fmt_e(oga_median(fnum(r, "sigma_min") for r in sub)), " |")
            end
            println(io)
        end

        println(io, "\n## Best variant per precision and activation\n")
        println(io, "Ranked by median fit error over the four targets.\n")
        println(io, "| precision | activation | best variant | fit err | cond | neurons placed |")
        println(io, "|---|---|---|---|---|---|")
        for T in Ts, act in levels(rows, "activation")

            groups = oga_groupby(
                [r for r in rows if r["T"] == T && r["activation"] == act],
                variant_label)
            best, bestv, bestc, bestn = "—", Inf, NaN, "—"
            for (label, sub) in groups
                v = oga_median(fnum(r, "fit_err") for r in sub)
                if isfinite(v) && v < bestv
                    best, bestv = label, v
                    bestc = oga_median(fnum(r, "cond") for r in sub)
                    bestn = string(oga_median(Float64(inum(r, "neurons")) for r in sub))
                end
            end
            println(io, "| $T | `$act` | `$best` | ", fmt_e(bestv), " | ", fmt_e(bestc),
                " | $bestn |")
        end

        println(io, "\n## Rank behaviour\n")
        println(io, "`rejected` counts candidate atoms the greedy step refused for adding no new")
        println(io, "direction; `neurons` is how many of the requested ones it could place. A case")
        println(io, "placing fewer traded fit quality for a guaranteed full-rank seed.\n")
        println(io, "Reported as counts and extremes rather than medians: rank trouble is a *tail*")
        println(io, "phenomenon — it shows up in a minority of (activation, target) combinations, and")
        println(io, "a median over the whole group reports 4 neurons and 0 rejections either way.\n")
        println(io,
            "| precision | selection | cases | short of full width | fewest neurons | cases with rejections | most rejected | non-finite |")
        println(io, "|---|---|---|---|---|---|---|---|")
        for T in Ts, sel in sels

            sub = [r for r in rows if r["T"] == T && r["selection"] == sel]
            isempty(sub) && continue
            neurons = [inum(r, "neurons") for r in sub]
            rejects = [inum(r, "rejected") for r in sub]
            wanted = maximum(neurons)
            println(io, "| $T | `$sel` | $(length(sub)) | ",
                count(<(wanted), neurons), " | ", minimum(neurons), " | ",
                count(>(0), rejects), " | ", maximum(rejects), " | ",
                count(!oga_ok, sub), " |")
        end
        println(io, "\n### Figures\n")
        for p in pngs
            println(io, "* `$(basename(p))`")
        end
    end
    println("Wrote $(mdpath)")
    return (mdpath, pngs)
end

# ---- Tier B report ----------------------------------------------------------

"""
    write_sweep_report(csvpaths, name) -> (mdpath, pngpaths)

Summarise an end-to-end integrator sweep: convergence and accuracy by seed variant,
precision, regularization factor and activation.
"""
function write_sweep_report(csvpaths, name::AbstractString)
    rows = read_oga_csv(csvpaths isa AbstractString ? [csvpaths] : csvpaths)
    isempty(rows) && (@warn "no rows for $name"; return (nothing, String[]))

    Ts = levels(rows, "T")
    seeds = levels(rows, "seed")
    acts = levels(rows, "activation")
    # λ values are labelled by their multiple of √eps(T) — `λ = 16√eps(T)` says what the
    # shift is, where an index into a list does not. 0 is the λ = 0 control.
    multiples = sort(unique(Int[inum(r, "lambda_multiple") for r in rows]))
    pngs = String[]

    # (1) Success rate: seed × precision, per activation. Magnitude ⇒ sequential ramp with
    #     the rate written into each cell.
    fig = Figure(size = (200 * length(acts) + 220, 26 * length(seeds) + 170),
        backgroundcolor = OGA_SURFACE)
    Label(fig[0, 1:length(acts)], "$name — convergence rate over the λ ladder";
        fontsize = 14, color = OGA_INK, font = :bold)
    for (ai, act) in enumerate(acts)
        M = fill(NaN, length(Ts), length(seeds))
        for (ti, T) in enumerate(Ts), (si, seed) in enumerate(seeds)

            sub = [r
                   for r in rows
                   if r["T"] == T && r["seed"] == seed && r["activation"] == act]
            isempty(sub) && continue
            M[ti, si] = count(oga_ok, sub) / length(sub)
        end
        ax = styled_axis!(fig, (1, ai); title = act, xticks = Ts,
            yticks = ai == 1 ? seeds : fill("", length(seeds)),
            xticklabelrotation = pi / 6)
        labelled_heatmap!(ax, M; fmt = fmt_pct, lo = 0.0, hi = 1.0)
    end
    png = joinpath(RESULTS_DIR, "$(name)_success.png")
    save(png, fig)
    push!(pngs, png)

    # (2) Accuracy vs regularization factor, one line per precision, one panel per seed.
    #     Identity by a
    #     fixed categorical slot, with a legend *and* end-of-line direct labels.
    ncol = min(3, length(seeds))
    nrow = cld(length(seeds), ncol)
    fig2 = Figure(size = (330 * ncol, 250 * nrow + 90), backgroundcolor = OGA_SURFACE)
    Label(fig2[0, 1:ncol],
        "$name — accuracy vs regularization factor (0 = no regularization)";
        fontsize = 14, color = OGA_INK, font = :bold)
    for (si, seed) in enumerate(seeds)
        r0, c0 = fldmod1(si, ncol)
        ax = styled_axis!(fig2, (r0, c0); title = seed, xlabel = "λ / √eps(T)",
            ylabel = c0 == 1 ? "median error" : "", yscale = log10)
        anyplotted = false
        for T in Ts
            xs, ys = Float64[], Float64[]
            for m in multiples
                sub = [r
                       for r in rows
                       if r["T"] == T && r["seed"] == seed &&
                              inum(r, "lambda_multiple") == m && oga_ok(r)]
                v = oga_median(fnum(r, "ref_err") for r in sub)
                if isfinite(v) && v > 0
                    push!(xs, m)
                    push!(ys, v)
                end
            end
            isempty(xs) && continue
            col = precision_colour(T)
            lines!(ax, xs, ys; color = col, linewidth = 2)
            scatter!(ax, xs, ys; color = col, markersize = 8,
                strokecolor = OGA_SURFACE, strokewidth = 2)
            # Direct label at the line end: the aqua slot's contrast against the surface is
            # below 3:1, so identity may not rest on a legend swatch alone.
            text!(ax, xs[end], ys[end]; text = " " * T, align = (:left, :center),
                fontsize = 9, color = OGA_INK_MUTED)
            anyplotted = true
        end
        # A seed that converged nowhere leaves an empty panel, and a log-scaled axis cannot
        # derive limits from no data. Pin them and say so, rather than let the figure fail.
        if !anyplotted
            ylims!(ax, 1e-16, 1.0)
            text!(ax, 0.5, 0.5; text = "no converged runs", align = (:center, :center),
                fontsize = 10, color = OGA_INK_MUTED, space = :relative)
        end
    end
    # Built from explicit elements rather than harvested from an axis: not every panel
    # contains every precision, so a legend scraped from one of them would be incomplete.
    Legend(fig2[nrow + 1, 1:ncol],
        [LineElement(color = precision_colour(T), linewidth = 2) for T in Ts],
        collect(Ts); orientation = :horizontal, framevisible = false,
        labelcolor = OGA_INK_MUTED, labelsize = 10)
    png2 = joinpath(RESULTS_DIR, "$(name)_accuracy.png")
    save(png2, fig2)
    push!(pngs, png2)

    mdpath = joinpath(RESULTS_DIR, "$(name).md")
    open(mdpath, "w") do io
        println(io, "# $name\n")
        println(io, "$(length(rows)) end-to-end runs. Converged: ", count(oga_ok, rows),
            " (", fmt_pct(count(oga_ok, rows) / length(rows)), ").\n")

        println(io, "## Convergence by seed variant and precision\n")
        println(io, "Converged runs per precision, out of the seven regularization factors swept:")
        println(io, "the `λ = 0` control plus six values of `λ = multiple · √eps(T)`, times the")
        println(io, "activations. The error column is the median over *all* converged runs of that")
        println(io, "seed, across precisions and activations.\n")
        print(io, "| seed |")
        for T in Ts
            print(io, " $T |")
        end
        println(io, " median err (converged) |")
        print(io, "|---|")
        for _ in Ts
            print(io, "---|")
        end
        println(io, "---|")
        for seed in seeds
            print(io, "| `$seed` |")
            for T in Ts
                sub = [r for r in rows if r["seed"] == seed && r["T"] == T]
                print(
                    io, " ", isempty(sub) ? "—" :
                             "$(count(oga_ok, sub))/$(length(sub))", " |")
            end
            ok = [r for r in rows if r["seed"] == seed && oga_ok(r)]
            println(io, " ", fmt_e(oga_median(fnum(r, "ref_err") for r in ok)), " |")
        end

        println(io, "\n## The regularization ladder\n")
        println(io, "Each `λ` is quoted as its multiple of `√eps(T)`; multiple 0 is the")
        println(io, "no-regularization control. Where every nonzero factor behaves alike, λ is acting")
        println(io, "as a *threshold* rather than a tuned value.\n")
        println(io, "| precision | λ / √eps(T) | λ | converged | median err | median iters |")
        println(io, "|---|---|---|---|---|---|")
        for T in Ts, m in multiples

            sub = [r for r in rows if r["T"] == T && inum(r, "lambda_multiple") == m]
            isempty(sub) && continue
            ok = filter(oga_ok, sub)
            println(io, "| $T | $m | ", fmt_e(fnum(first(sub), "lambda")), " | ",
                "$(length(ok))/$(length(sub)) | ",
                fmt_e(oga_median(fnum(r, "ref_err") for r in ok)), " | ",
                fmt_e(oga_median(fnum(r, "iterations") for r in ok)), " |")
        end

        println(io, "\n## Activations\n")
        println(io, "| activation | ", join(Ts, " | "), " | best seed |")
        println(io, "|---|", repeat("---|", length(Ts)), "---|")
        for act in acts
            print(io, "| `$act` |")
            for T in Ts
                sub = [r for r in rows if r["activation"] == act && r["T"] == T]
                print(io, " ", isempty(sub) ? "—" : "$(count(oga_ok, sub))/$(length(sub))", " |")
            end
            groups = oga_groupby([r for r in rows if r["activation"] == act], r -> r["seed"])
            best, bestrate = "—", -1.0
            for (seed, sub) in groups
                rate = count(oga_ok, sub) / length(sub)
                rate > bestrate && ((best, bestrate) = (seed, rate))
            end
            println(io, " `$best` (", fmt_pct(bestrate), ") |")
        end

        println(io, "\n## Failure modes\n")
        println(io, "| status | count |")
        println(io, "|---|---|")
        for (st, sub) in sort(collect(oga_groupby(rows, r -> r["status"])), by = p -> -length(p[2]))
            println(io, "| `$st` | $(length(sub)) |")
        end
        println(io, "\n### Figures\n")
        for p in pngs
            println(io, "* `$(basename(p))`")
        end
    end
    println("Wrote $(mdpath)")
    return (mdpath, pngs)
end
