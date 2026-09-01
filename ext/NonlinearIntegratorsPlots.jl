module NonlinearIntegratorsPlots

using Makie
# Import GeometricSolutions symbols explicitly: `using Makie` also exports names such as
# `TimeSeries` (its recipe), which would otherwise clash with `GeometricSolutions.TimeSeries`.
using GeometricSolutions: GeometricSolution

import NonlinearIntegrators: Trajectory, dimension, window_stem
# The stubs live in the `Diagnostics` submodule rather than at the package's top level, because
# `plot_solution` and `plot_convergence` are names `GeometricProblems` already exports — see the
# `Diagnostics` docstring in `src/plots.jl`. The methods below are therefore defined as
# `Diagnostics.plot_…`, which extends the stub rather than creating a shadowing function here.
import NonlinearIntegrators.Diagnostics

# ---- what is deliberately *not* here ----------------------------------------
#
# Only two functions, because only two things are missing from `GeometricProblems`. Everything
# else a run of these integrators wants plotted, that package already plots, and a second
# implementation of it here would be a second thing to keep right:
#
#   * the relative error of an invariant over time, and its drift →
#     `GeometricProblems.Diagnostics.plot_energy_error` / `plot_energy_drift` /
#     `plot_invariant_error` / `plot_invariant_drift`. These take `energy = <function>`, which is
#     what a `lodeproblem` built by `EulerLagrange` needs, since it carries `NullInvariants` and so
#     has no `:h` key to look up.
#   * a phase portrait, a trajectory, traces → the per-problem recipes,
#     `GeometricProblems.HarmonicOscillator.plot_phase_portrait` and friends.
#
# What is left is the pair those cannot express:
#
#   * `plot_solution` — *several* integrators in one figure, with the **continuous** solution
#     between the discrete steps drawn through them. No recipe in `GeometricProblems` takes more
#     than one solution, and none knows about `integrate`'s second return value, which is the
#     whole subject of these integrators.
#   * `plot_convergence` — *several* series with a reference slope per family.
#     `GeometricProblems.Diagnostics.plot_convergence` does one series and one slope, which is the
#     right thing for one method's order and cannot show a neural family against a polynomial one.

# The shared Makie theme of this ecosystem: larger fonts and thicker lines than the Makie defaults,
# tuned for the fixed figure sizes of the `GeometricProblems` recipes.
#
# Kept **identical** to the copy in `GeometricExamples/src/common.jl` and the publication companion
# packages, so a figure from this package sits beside one of theirs without a visible change of
# typeface size. Reached as `Diagnostics.plot_theme()` — see the stub's docstring in
# `src/plots.jl` for why it is a function and not a `const`.
const PLOT_THEME = Theme(
    fontsize = 18,
    Lines = (linewidth = 2,),
    Scatter = (markersize = 10,),
    Axis = (
        xlabelsize = 22,
        ylabelsize = 22,
        xticklabelsize = 16,
        yticklabelsize = 16,
        titlesize = 20
    )
)

Diagnostics.plot_theme() = PLOT_THEME

# Series colours come from Makie's own colourblind-safe cycle rather than a palette of this
# package's own — one fewer thing to validate, and it is what the rest of the ecosystem's Makie
# code uses. The order is fixed, so a given role keeps its colour across every figure of a set.
series_colour(i::Integer) = Makie.wong_colors()[mod1(i, length(Makie.wong_colors()))]

# Every figure here is 2:1 — twice as wide as it is tall — so that figures with different panel
# counts sit on a slide the same way and can be included at one width. The panels divide the height
# between them; a caller who wants something else passes `figsize`.
const FIGURE_WIDTH = 1200
const FIGURE_ASPECT = 2.0

# The reference curve is deliberately outside that cycle: it is the thing the others are measured
# against, so it is black dashes underneath all of them.
const REFERENCE_COLOUR = :black

# ---- labels ------------------------------------------------------------------
#
# Written out per component rather than built with `latexstring("q_$(d)")`, which would make
# `LaTeXStrings` a second weakdep of this extension for no gain — every problem this package
# integrates has `D ≤ 4`, and past that the plain label is the better one anyway.
const Q_LATEX = (L"q_1", L"q_2", L"q_3", L"q_4")
const P_LATEX = (L"p_1", L"p_2", L"p_3", L"p_4")
const Q_PLAIN = ("q₁", "q₂", "q₃", "q₄")
const P_PLAIN = ("p₁", "p₂", "p₃", "p₄")

function q_label(d, D, latex)
    D == 1 && return latex ? L"q" : "q"
    d ≤ length(Q_LATEX) ? (latex ? Q_LATEX[d] : Q_PLAIN[d]) : "q[$(d)]"
end

function p_label(d, D, latex)
    D == 1 && return latex ? L"p" : "p"
    d ≤ length(P_LATEX) ? (latex ? P_LATEX[d] : P_PLAIN[d]) : "p[$(d)]"
end

t_label(latex) = latex ? L"t" : "t"
error_label(latex) = latex ? L"|\Delta H / H_0|" : "|ΔH / H₀|"

# ---- helpers -----------------------------------------------------------------

"""
    log_points(times, values)

`(times, values)` with every point a logarithmic axis cannot take removed from **both**.

Every invariant-error series starts with an exact zero — `(H(t₀) - H₀)/H₀` is `0` by construction —
and `log10(0)` is `-Inf`. One such point does not merely go missing: it drags the axis limits to
`-Inf`, Makie falls back to a default decade range, and the whole panel comes out **empty**, with
nothing to say why. That is not hypothetical; it is what the first render of the figures this was
written for did, with three series whose real values sat around `1e-8` on an axis running `1e0` to
`1e3`.

Dropping the pair and not masking the value with `NaN`, which is what this did first: `NaN` breaks
the polyline through it, and a series sitting at round-off hits an exact zero repeatedly rather than
only at `t₀`. The global Fourier fit of the perturbed pendulum conserves `H` to `1e-16` and is
exactly zero at 22 of its 101 samples, so the masked error panel came out as scattered fragments and
five isolated dots instead of one curve. Removing the point from `times` as well keeps every
remaining point at its own time, so nothing shifts left — which is the property masking was chosen
for. This is what `GeometricProblems.Diagnostics.plot_convergence` does, and `plot_convergence`
below.
"""
function log_points(times, values)
    keep = [i for (i, v) in enumerate(values) if v > 0 && isfinite(v)]
    (collect(times)[keep], [float(v) for v in values][keep])
end

# A step size as a tick label: the value itself, trimmed of the trailing zero an integer-valued
# `Float64` prints. `0.03125` stays `0.03125`, `4.0` becomes `4`.
_steplabel(h) = isinteger(h) ? string(Int(h)) : string(h)

function _relabel(traj::Trajectory, label::AbstractString)
    traj.label == label && return traj
    Trajectory(label, traj.t, traj.q, traj.p;
        continuous_t = traj.continuous_t, continuous_q = traj.continuous_q,
        invariant_error = traj.invariant_error)
end

# Solutions and `"label" => solution` pairs are accepted wherever a `Trajectory` is, and converted
# here — a caller who has just called `integrate` holds a solution, and writing the conversion out
# at every call site is the boilerplate this extension exists to remove.
#
# A `Trajectory` is already built, so the build keywords have nothing left to do.
_as_trajectory(traj::Trajectory, ::AbstractString; kwargs...) = traj

function _as_trajectory(sol::GeometricSolution, label::AbstractString; kwargs...)
    Trajectory(label, sol; kwargs...)
end

# The label written at this call site wins over any the value came with.
function _as_trajectory(entry::Pair, ::AbstractString; kwargs...)
    label, value = entry
    value isa Trajectory ? _relabel(value, label) : _as_trajectory(value, label; kwargs...)
end

function _as_trajectories(values, fallback::AbstractString; kwargs...)
    [_as_trajectory(v, fallback; kwargs...) for v in values]
end

# ---- the comparison figure ---------------------------------------------------

"""
    plot_solution(primary; reference, comparisons, training_region, title, latex, figsize)

Several integrators of the same problem in one figure: the **continuous** solution of `primary`
between its discrete steps, its discrete steps, any number of comparisons, and — where `primary`
carries one — the relative Hamiltonian error of all of them on a logarithmic axis.

For one degree of freedom this is three panels **stacked**, `q(t)`, `p(t)` and the error, sharing
one time axis; for `D > 1` a `D`×2 grid of `qᵈ(t)` and `pᵈ(t)` with the error panel spanning the
width beneath.

Stacked and not side by side, which is a legibility decision and not a preference. These runs go to
`t = 1000` at a period of order one, so a panel gets a hundred oscillations; at a third of the
figure width that is a solid block of ink, and at the full width it is a trajectory. It is also
what the figures this replaces did.

This is the one solution plot `GeometricProblems` cannot make: its per-problem recipes each take a
single solution, and none of them knows about `integrate`'s second return value. For anything with
one solution in it — an invariant error, a drift, a phase portrait, a set of traces — use
`GeometricProblems.Diagnostics` and the per-problem recipes instead.

# Arguments

  - `primary`: a [`Trajectory`](@ref). A `GeometricSolution` is accepted too, in which case the
    second positional argument is `integrate`'s `internal_values` and the `Trajectory` keywords
    (`label`, `hamiltonian`, `parameters`, `timestep`) are passed through.

# Keyword arguments

  - `reference = nothing`: a `Trajectory`, a solution, or a `"label" => …` pair, in black dashes
    under everything else — a high-order integrator at a fraction of the step, or the exact
    solution where there is one.
  - `comparisons = []`: the same, each in the next colour of Makie's cycle.
  - `training_region = nothing`: a time up to which the ansatz was fitted, shaded and marked.
    What happens *outside* it is the point of the VISE figures.
  - `timespan = nothing`: the time interval **every** panel spans, as `(t_begin, t_end)`. Defaults
    to the extent of `primary`, which is the problem's own timespan. Set explicitly on all panels
    rather than left to Makie, so that the trace panels and the error panel share one axis exactly —
    without it each autoscales to its own data and they disagree, most visibly where the error
    panel's first point is masked off a logarithmic axis and its axis therefore starts one step in.
    Sharing the axis is also what lets a single time label at the bottom serve the whole column.
  - `title = ""`: a heading across the whole figure. The panels are labelled but not titled, so
    without it a figure taken out of its directory does not say which run it is.
  - `latex = true`: LaTeX axis labels.
  - `figsize = nothing`: defaults to a shape suiting the panel count.

Font sizes, line widths and marker sizes all come from the ambient Makie theme — see
[`PLOT_THEME`](@ref).
"""
function Diagnostics.plot_solution(primary::Trajectory;
        reference = nothing,
        comparisons = Trajectory[],
        training_region = nothing,
        timespan = nothing,
        title = "",
        latex = true,
        figsize = nothing)
    D = dimension(primary)
    show_error = primary.invariant_error !== nothing

    refs = reference === nothing ? nothing : _as_trajectory(reference, "Reference")
    comps = _as_trajectories(comparisons, "Comparison")

    for c in comps
        dimension(c) == D || throw(ArgumentError(
            "comparison \"$(c.label)\" has $(dimension(c)) degrees of freedom, the primary " *
            "trajectory has $(D)."))
    end

    npanels = D == 1 ? (show_error ? 3 : 2) : D + (show_error ? 1 : 0)

    # 2:1 overall, whatever the panel count — a fixed figure shape rather than a fixed panel height,
    # so a three-panel and a five-panel figure sit on a slide the same way and can be included at the
    # same width. The panels divide the height between them.
    if figsize === nothing
        figsize = (FIGURE_WIDTH, round(Int, FIGURE_WIDTH / FIGURE_ASPECT))
    end
    fig = Figure(size = figsize)

    # Row 0, so the panel grid below keeps the row indices the layout computes. An empty title adds
    # no row.
    # `tellwidth = false` for the same reason as on the `Legend` below: a `Label` reports its width,
    # and in a single-column figure the column is then sized to the *title*. Measured on the
    # harmonic-oscillator figure, a 50-character title held the axes to 386 pt of a 900 pt page.
    isempty(title) || Label(fig[0, :], title;
        font = :bold, padding = (0, 0, 6, 0), tellwidth = false)

    # `D == 1`: one column, `q` over `p` over the error. Otherwise a `D`×2 grid.
    ncols = D == 1 ? 1 : 2

    axes_q = Axis[]
    axes_p = Axis[]
    for d in 1:D
        push!(axes_q, Axis(fig[D == 1 ? 1 : d, 1]; ylabel = q_label(d, D, latex)))
        push!(axes_p, Axis(fig[D == 1 ? 2 : d, ncols]; ylabel = p_label(d, D, latex)))
    end

    ax_err = nothing
    if show_error
        ax_err = Axis(fig[D == 1 ? 3 : D + 1, D == 1 ? 1 : (1:2)];
            ylabel = error_label(latex), yscale = log10)
    end

    # Every panel spans the same interval, so only the **bottom** of each column carries the time
    # label and its tick labels: stacked panels share one axis, and repeating it three times wastes
    # the height the trajectories need. This is what `GeometricProblems`' `_plot_components` does,
    # for the same reason.
    trace_axes = vcat(axes_q, axes_p)
    all_axes = vcat(trace_axes, ax_err === nothing ? Axis[] : [ax_err])
    bottom = if ax_err !== nothing
        [ax_err]
    elseif D == 1
        [axes_p[end]]
    else
        [axes_q[end], axes_p[end]]
    end
    for ax in all_axes
        if ax in bottom
            ax.xlabel = t_label(latex)
        else
            ax.xticklabelsvisible = false
        end
    end

    # Reference first, so it lies underneath everything else.
    if refs !== nothing
        for d in 1:D
            lines!(axes_q[d], refs.t, refs.q[d];
                color = REFERENCE_COLOUR, linestyle = :dash, label = refs.label)
            lines!(axes_p[d], refs.t, refs.p[d];
                color = REFERENCE_COLOUR, linestyle = :dash, label = refs.label)
        end
    end

    own = series_colour(1)
    for d in 1:D
        if primary.continuous_q !== nothing
            lines!(axes_q[d], primary.continuous_t, primary.continuous_q[d];
                color = own, label = primary.label)
        end
        scatter!(axes_q[d], primary.t, primary.q[d];
            color = own, label = "$(primary.label), discrete")
        scatter!(axes_p[d], primary.t, primary.p[d];
            color = own, label = "$(primary.label), discrete")
    end

    # A comparison at the same step as the primary is a handful of points and reads as markers; one
    # computed on a much finer grid is a curve, and drawing it as markers buries the figure — a
    # midpoint solve at `h/20` over 40 time units is 1600 of them, which is what the double-pendulum
    # figure looked like before this. The threshold is "more than twice as many points as the
    # primary has steps", which separates the two cases without a keyword nobody would remember to
    # set.
    dense(c) = length(c.t) > 2 * length(primary.t)

    for (i, c) in enumerate(comps)
        colour = series_colour(i + 1)
        draw! = dense(c) ? lines! : scatter!
        for d in 1:D
            draw!(axes_q[d], c.t, c.q[d]; color = colour, label = c.label)
            draw!(axes_p[d], c.t, c.p[d]; color = colour, label = c.label)
        end
    end

    if show_error
        # A series with nothing plottable in it at all — every value zero or not finite — is left
        # out rather than drawn empty, as `plot_convergence` leaves out a configuration that failed
        # at every step. Makie has no limits to take from an empty log axis and falls back to a
        # default decade range, which is the same empty panel `log_points` exists to prevent.
        te, ee = log_points(primary.t, primary.invariant_error)
        isempty(ee) || scatterlines!(ax_err, te, ee; color = own, label = primary.label)
        for (i, c) in enumerate(comps)
            c.invariant_error === nothing && continue
            tc, ec = log_points(c.t, c.invariant_error)
            isempty(ec) && continue
            # Same distinction as the traces: a dense series gets a line, not 1600 markers.
            draw_err! = dense(c) ? lines! : scatterlines!
            draw_err!(ax_err, tc, ec; color = series_colour(i + 1), label = c.label)
        end
    end

    if training_region !== nothing
        for ax in vcat(axes_q, axes_p)
            vspan!(ax, first(primary.t), training_region; color = (own, 0.08))
            vlines!(ax, [training_region]; color = :grey40, linestyle = :dashdot)
        end
    end

    # The one place the time axis is set. `primary.t` runs from the problem's initial time to its
    # final one, so its extent *is* the timespan unless a caller says otherwise.
    span = timespan === nothing ? (first(primary.t), last(primary.t)) : timespan
    for ax in all_axes
        xlims!(ax, span...)
    end

    # One legend for the whole figure, taken from the first `q` panel — every curve appears there,
    # and a legend per panel would repeat the same entries three times. `merge = true` collapses
    # the duplicates the `q`/`p` pairs produce.
    # `tellwidth = false` is load-bearing, not cosmetic. A `Legend` reports its own width to the
    # layout by default, and since it sits in the same column as the panels, the column is then
    # sized to the *legend* — measured on the harmonic-oscillator figure, that left the axes
    # spanning 386 pt of a 900 pt page, 43% of the width, centred, with the rest white. With it
    # false the legend takes what height it needs and the panels get the full width.
    Legend(fig[npanels + 1, 1:ncols], axes_q[1];
        orientation = :horizontal, framevisible = false, merge = true, nbanks = 2,
        tellwidth = false, tellheight = true)

    # Equal column widths, so the `q` and `p` panels of a multi-degree-of-freedom figure come out the
    # same size. Without this Makie sizes each column to its content and the two differ by the width
    # of their y tick labels — `p` running to ±40 against `q` to ±2 makes the `q` panel visibly wider.
    #
    # Only for `ncols > 1`. A single column is already the full width, and pinning it to
    # `Relative(1.0)` would leave the axis decorations nowhere to go.
    if ncols > 1
        for c in 1:ncols
            colsize!(fig.layout, c, Auto(1.0))
        end
    end

    # The panels share the height equally, the title and legend taking what they need. Stated rather
    # than left to `Auto`, because a `log10` error panel carries wider tick labels than a linear one
    # and would otherwise be given a different height from the traces it is compared against.
    for r in 1:npanels
        rowsize!(fig.layout, r, Auto(1.0))
    end

    return fig
end

# The convenience form, for a caller holding what `integrate` just returned.
function Diagnostics.plot_solution(sol::GeometricSolution, internal_values = nothing;
        label = "Continuous solution",
        timestep = nothing,
        hamiltonian = nothing,
        parameters = nothing,
        nplot = 1,
        nt = :auto,
        reference = nothing,
        comparisons = [],
        kwargs...)
    build = (; timestep = timestep, hamiltonian = hamiltonian, parameters = parameters,
        nplot = nplot, nt = nt)
    Diagnostics.plot_solution(
        Trajectory(label, sol; internal_values = internal_values, build...);
        reference = reference === nothing ? nothing :
                    _as_trajectory(reference, "Reference"; build...),
        comparisons = _as_trajectories(comparisons, "Comparison"; build...),
        kwargs...)
end

# The tuple `integrate` returns, in either of its two shapes — `(sol, internal_values)` for the
# network integrators, `(sol, internal_values, x_list)` for `VISE`.
function Diagnostics.plot_solution(result::Tuple; kwargs...)
    length(result) ≥ 2 ||
        throw(ArgumentError("expected the tuple `integrate` returns, of at least " *
                            "(solution, internal_values)."))
    Diagnostics.plot_solution(result[1], result[2]; kwargs...)
end

# ---- convergence over several series ----------------------------------------

"""
    plot_convergence(timesteps, errors; labels, linestyles, reference_orders, …)

Several error series against the time step on logarithmic axes, with a dotted reference slope per
order in `reference_orders`.

For **one** series against **one** expected order, use
`GeometricProblems.Diagnostics.plot_convergence`, which is what that is for. This exists for the
case it cannot express: two *families* of method — neural and polynomial, say — where the claim of
the figure is which of them follows the reference slope, so several series and more than one slope
have to be on the same axes at once.

# Arguments

  - `timesteps`: one vector shared by every series, or a vector of vectors, one per series, for a
    study where not every configuration ran at every step.
  - `errors`: one error vector per series.

# Keyword arguments

  - `labels`: one per series. Required — a convergence plot with an unlabelled series says nothing.
  - `linestyles = nothing`: one per series. Separating the families by style as well as by colour
    is what makes the comparison readable at a glance.
  - `colors = nothing`: one per series. Worth passing whenever there are more series than the seven
    of `Makie.wong_colors()`: the cycle then wraps, and in a figure whose whole point is one family
    against another, a solid blue and a dashed blue read as related when they are not.
  - `reference_orders = (2, 4, 6)`: dotted `h^p` guides, anchored to the largest error plotted so
    they bound the data from above rather than run through it.

    These three and not `(2, 3)`: a continuous Galerkin variational integrator on `R` Gauss nodes
    has order **`2R - 2`**, so `CGVI(2)`, `CGVI(3)` and `CGVI(4)` are second, fourth and sixth
    order. Measured on both problems here, between every pair of successive steps, the observed
    orders are `2.00`, `4.00` and `6.00` to three digits. An `h³` guide matches nothing in such a
    figure.
  - `xlabel`, `ylabel`, `title`: axis labels. The figures this replaces carried none at all.
  - `latex = true`, `figsize = nothing` — 2:1, as every figure here.

The `h` axis is ticked at the step sizes actually run, not at the decades Makie would choose: a
convergence study over `h ∈ {0.03125, …, 4}` otherwise gets an axis labelled `10^-1.5`, which is
not a step size anyone chose.
"""
function Diagnostics.plot_convergence(timesteps, errors;
        labels,
        linestyles = nothing,
        colors = nothing,
        reference_orders = (2, 4, 6),
        xlabel = nothing,
        ylabel = nothing,
        title = "",
        latex = true,
        figsize = nothing)
    length(errors) == length(labels) ||
        throw(ArgumentError("$(length(errors)) error series against $(length(labels)) labels."))
    linestyles === nothing || length(linestyles) == length(errors) ||
        throw(ArgumentError("$(length(linestyles)) line styles against $(length(errors)) " *
                            "series."))
    colors === nothing || length(colors) == length(errors) ||
        throw(ArgumentError("$(length(colors)) colours against $(length(errors)) series."))

    steps_of(i) = eltype(timesteps) <: AbstractVector ? timesteps[i] : timesteps
    colour_of(i) = colors === nothing ? series_colour(i) : colors[i]

    if figsize === nothing
        figsize = (FIGURE_WIDTH, round(Int, FIGURE_WIDTH / FIGURE_ASPECT))
    end
    fig = Figure(size = figsize)
    ax = Axis(fig[1, 1];
        title = title,
        xlabel = xlabel === nothing ? (latex ? L"h" : "h") : xlabel,
        ylabel = ylabel === nothing ?
                 (latex ? L"\max_n |\Delta H / H_0|" : "max |ΔH / H₀|") : ylabel,
        xscale = log10, yscale = log10)

    ticks = sort(unique(reduce(vcat, [collect(steps_of(i)) for i in eachindex(errors)])))
    ax.xticks = (ticks, [_steplabel(h) for h in ticks])
    ax.xticklabelrotation = π / 4

    for i in eachindex(errors)
        h = collect(steps_of(i))
        e = collect(errors[i])
        length(h) == length(e) || throw(ArgumentError(
            "series \"$(labels[i])\" has $(length(e)) errors against $(length(h)) time steps."))
        # Dropped, not masked, as in `GeometricProblems.Diagnostics.plot_convergence`: here `h` and
        # `ε` are the two ends of one point, so a point with no plottable error has no `h` either.
        keep = findall(k -> isfinite(e[k]) && e[k] > 0, eachindex(e))
        isempty(keep) && continue
        scatterlines!(ax, h[keep], e[keep];
            color = colour_of(i),
            linestyle = linestyles === nothing ? :solid : linestyles[i],
            label = labels[i])
    end

    all_h = reduce(vcat, [collect(steps_of(i)) for i in eachindex(errors)])
    all_e = filter(x -> isfinite(x) && x > 0, reduce(vcat, [collect(e) for e in errors]))
    if !isempty(all_e) && !isempty(reference_orders)
        hmin, hmax = minimum(all_h), maximum(all_h)
        hs = [hmin, hmax]
        emax = maximum(all_e)
        for p in reference_orders
            guide = emax .* (hs ./ hmax) .^ p
            lines!(ax, hs, guide; color = :grey40, linestyle = :dot,
                label = latex ? L"O(h^{%$p})" : "O(h^$(p))")
        end
    end

    Legend(fig[1, 2], ax; framevisible = false)
    return fig
end

# ---- figures from an archive --------------------------------------------------
#
# The layer above the two plot functions: given one run's archive, build every figure that run
# earns, and name each one. This is what makes a renderer a dispatcher — glob the run directory,
# hand each archive over, save what comes back — rather than a second registry of which experiment
# produces which picture, kept in step with the first by hand.
#
# It takes a plain `AbstractDict` of the flat keys `scripts/archives.jl` documents. Deliberately not
# JLD2, not a problem registry, not a filesystem path: this extension composes figures out of
# vectors and knows nothing about how they were stored or which study produced them.

# `upto` truncates every series to `t ≤ upto`, the error panel included, so each figure of a
# windowed set is internally consistent. Applied to the whole figure and not just the traces: an
# earlier version narrowed only `q` and `p`, which made the trace panels and the error panel
# disagree and then needed a second time axis to say so.
#
# `components` is a vector of one series per degree of freedom, so it is indexed per component and
# never by the time indices — which is the one way to get this wrong.
_keep(t, upto) = upto === nothing ? eachindex(t) : findall(≤(upto), t)
_cut(components, idx) = [series[idx] for series in components]

function _primary(data; upto = nothing)
    idx = _keep(data["t"], upto)
    cidx = _keep(data["continuous_t"], upto)
    Trajectory(data["label"], data["t"][idx], _cut(data["q"], idx), _cut(data["p"], idx);
        continuous_t = data["continuous_t"][cidx],
        continuous_q = _cut(data["continuous_q"], cidx),
        invariant_error = data["hamiltonian_error"][idx])
end

function _reference(data; upto = nothing)
    # `exact` where the problem has a closed-form solution, the high-order solve otherwise. Both are
    # drawn as the black dashed reference because that is the role they play; the distinction belongs
    # in the legend, not in the styling.
    #
    # The substep factor comes from the archive rather than from a constant, because it is not the
    # same everywhere — one run uses `h/20` where the rest use `h/40` — and a label naming the wrong
    # grid is worse than no label.
    exact = haskey(data, "exact_t")
    prefix = exact ? "exact" : "reference"
    label = exact ? "Exact solution" :
            "Reference (Gauss(8), h/$(get(data, "reference_substeps", "?")))"
    idx = _keep(data["$(prefix)_t"], upto)
    Trajectory(label, data["$(prefix)_t"][idx],
        _cut(data["$(prefix)_q"], idx), _cut(data["$(prefix)_p"], idx))
end

function _comparisons(data; upto = nothing)
    # Sorted, so the colour a given comparison gets does not depend on dictionary iteration order —
    # the same integrator has to keep its colour across every figure of a set.
    map(sort(collect(get(data, "comparisons", Dict{String, Any}())); by = first)) do (
        label, c)
        idx = _keep(c["t"], upto)
        err = get(c, "hamiltonian_error", nothing)
        Trajectory(label, c["t"][idx], _cut(c["q"], idx), _cut(c["p"], idx);
            invariant_error = err === nothing ? nothing : err[idx])
    end
end

# `Δt` appears only when the run has one. A global fit over a whole window steps nothing, so its
# archive carries no `"timestep"` and a `Δt` in its title would name a quantity the method does not
# have. That absence is the whole reason these runs can share this function instead of needing a
# near-copy of it with one clause removed.
function _solution_title(data, shown)
    step = haskey(data, "timestep") ? "Δt = $(data["timestep"]), " : ""
    "$(data["problem_label"]) — $(data["label"]), $(step)t ∈ [0, $(Int(shown))]"
end

function _solution_figure(data; upto = nothing)
    shown = upto === nothing ? data["final_time"] : upto
    Diagnostics.plot_solution(_primary(data; upto = upto);
        reference = _reference(data; upto = upto),
        comparisons = _comparisons(data; upto = upto),
        timespan = (0.0, shown),
        title = _solution_title(data, shown))
end

# Colours for a convergence figure, and the reason they are not left to the default cycle: the
# largest of these families has ten series and `Makie.wong_colors()` has seven, so the cycle wraps
# and a solid blue lands beside a dashed blue — which reads as "related" in the one figure whose
# entire point is that the two families are not.
#
# The dashed series are the *reference* family, so they get a greyscale ramp, which says that
# without a legend and leaves the whole colour cycle for what the figure is actually about.
const REFERENCE_GREYS = (:black, :grey35, :grey55, :grey70)

function _convergence_colours(linestyles)
    solid = 0
    dashed = 0
    map(linestyles) do style
        if style === :solid
            solid += 1
            Makie.wong_colors()[mod1(solid, length(Makie.wong_colors()))]
        else
            dashed += 1
            REFERENCE_GREYS[mod1(dashed, length(REFERENCE_GREYS))]
        end
    end
end

function _convergence_figure(data)
    styles = haskey(data, "linestyles") ? Symbol.(data["linestyles"]) : nothing

    # A series that failed at every step contributes no line, and so no legend entry — it would
    # simply be absent from the figure with nothing to say it had been tried. Named in the title
    # instead, which is the difference between an omission and a result.
    absent = [l
              for (l, e) in zip(data["labels"], data["errors"])
              if !any(x -> isfinite(x) && x > 0, e)]
    title = data["title"] *
            (isempty(absent) ? "" : "\nno solve completed for: " * join(absent, ", "))

    Diagnostics.plot_convergence(data["timesteps"], data["errors"];
        labels = data["labels"],
        linestyles = styles,
        colors = styles === nothing ? nothing : _convergence_colours(styles),
        reference_orders = Tuple(get(data, "reference_orders", (2, 4, 6))),
        title = title)
end

"""
    figures(data) -> Vector{Pair{String, Figure}}

Every figure one archived run earns, each paired with the stem it should be saved under.

`data` is a run's archive as a plain dictionary — see the schema in `scripts/archives.jl`. Two of
its keys drive this: `"kind"`, which selects the shape of figure, and `"stem"`, which names it.

# Kinds

  - `"solution"` — the traces of one integrator against a reference and any comparisons, through
    [`plot_solution`](@ref). A run whose `"windows"` is non-empty additionally yields one figure per
    window, over `t ∈ [0, window]`, named `<stem>-t<window>`.
  - `"convergence"` — several error series against the time step, through
    [`plot_convergence`](@ref), with the reference slopes the study recorded in
    `"reference_orders"`.

# Why the caller does the saving

This returns figures; it does not write them. Keeping every `save` in one place in the calling
script is what lets a figure be restyled without re-running a solve, and it keeps this extension
free of any opinion about where output goes.
"""
function Diagnostics.figures(data::AbstractDict)
    haskey(data, "kind") ||
        throw(ArgumentError("the archive has no \"kind\"; cannot tell what to draw."))
    haskey(data, "stem") ||
        throw(ArgumentError("the archive has no \"stem\"; cannot tell what to call the figure."))
    stem = data["stem"]
    kind = data["kind"]

    if kind == "convergence"
        return [stem => _convergence_figure(data)]
    elseif kind == "solution"
        out = [stem => _solution_figure(data)]
        for upto in get(data, "windows", Float64[])
            push!(out, window_stem(stem, upto) => _solution_figure(data; upto = upto))
        end
        return out
    else
        throw(ArgumentError("unknown archive kind \"$(kind)\"; " *
                            "this extension draws \"solution\" and \"convergence\"."))
    end
end

end
