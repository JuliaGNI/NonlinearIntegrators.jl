# The plotting API: the data it plots and the function stubs here, the Makie methods in
# `ext/NonlinearIntegratorsPlots.jl`.
#
# The stubs live in `src/` so that `plot_solution` is a documented name whether or not a Makie
# backend is loaded — the convention across this ecosystem (`GeometricProblems`,
# `ChargedParticleDynamics`, `ElectromagneticFields`, `PoincareInvariants`, `PoissonBrackets`).
# The docstrings of the implementations live in the extension, beside them.

"""
    continuous_solution(internal_values, timestep; dof = 1, t₀ = 0)

The continuous solution *between* the discrete time steps, as a `(t, q)` pair of vectors.

This is the one piece of post-processing every network and symbolic integrator here needs and
none of them provided: `integrate` returns the per-step record of the ansatz as its second value,
and turning that into something plottable takes knowing two things the array does not say.

# Arguments

  - `internal_values`: the second element of what `integrate` returns — a vector of
    `record_grid_points × D` matrices, one per time step. The whole returned tuple is also
    accepted, in either of its shapes.
  - `timestep`: the macro time step the problem was integrated with.

# Keyword arguments

  - `dof = 1`: which degree of freedom to extract, i.e. which column.
  - `t₀ = 0`: the initial time.

# Implementation

Two things are easy to get wrong here, and were, in each of the six scripts that used to do it by
hand.

**Row 1 of every step is the *left* endpoint.** `record_finer_solution!` evaluates the ansatz on
`range(0, 1, record_grid_points)` mapped onto `[tₙ₋₁, tₙ]`, so the first row of step `n` is the
same time as the last row of step `n-1`. Concatenating the matrices whole therefore duplicates
every interior step boundary. Row 1 is dropped, which is why the returned grid starts at
`t₀ + timestep/(G-1)` rather than at `t₀` — the initial condition is in the solution, not here.

**The grid size is a property of the method, not a constant.** `record_grid_points` is a `VISE`
and network-integrator keyword whose default happens to be 41, and every hand-written version of
this assumed the 41. It is read off `size(internal_values[1], 1)` instead, so a method built with
`record_grid_points = 21` does not silently return a `t` and a `q` of different lengths.
"""
function continuous_solution(internal_values::AbstractVector{<:AbstractMatrix},
        timestep::Real; dof::Int = 1, t₀::Real = 0)
    isempty(internal_values) && throw(ArgumentError("`internal_values` is empty."))

    nsteps = length(internal_values)
    npoints = size(internal_values[1], 1)
    npoints > 1 ||
        throw(ArgumentError("a recording grid of $(npoints) point(s) carries no interior " *
                            "values; construct the method with `record_grid_points ≥ 2`."))
    1 ≤ dof ≤ size(internal_values[1], 2) ||
        throw(ArgumentError("`dof = $(dof)` is out of range for a problem of dimension " *
                            "$(size(internal_values[1], 2))."))

    T = promote_type(eltype(internal_values[1]), typeof(timestep), typeof(t₀))
    Δ = timestep / (npoints - 1)

    t = Vector{T}(undef, nsteps * (npoints - 1))
    q = Vector{T}(undef, nsteps * (npoints - 1))

    k = 0
    for n in 1:nsteps
        values = internal_values[n]
        size(values, 1) == npoints ||
            throw(ArgumentError("step $(n) records $(size(values, 1)) grid points, " *
                                "step 1 records $(npoints)."))
        # `2:npoints`, dropping the left endpoint — see the note above.
        for i in 2:npoints
            k += 1
            t[k] = t₀ + (n - 1) * timestep + (i - 1) * Δ
            q[k] = values[i, dof]
        end
    end

    return t, q
end

# `integrate` returns `(sol, internal_values)` for the network integrators and
# `(sol, internal_values, x_list)` for `VISE`. Both are accepted, so a caller does not have to
# remember which of the two it is holding.
function continuous_solution(result::Tuple, timestep::Real; kwargs...)
    length(result) ≥ 2 ||
        throw(ArgumentError("expected the tuple `integrate` returns, of at least " *
                            "(solution, internal_values)."))
    continuous_solution(result[2], timestep; kwargs...)
end

"""
    relative_invariant_error(values) -> Vector

`|(I - I₀) / I₀|` for a series of invariant values. The absolute value and the division by the
*initial* value are what every script in this package's history wrote out by hand. A zero initial
value falls back to the absolute error, which is the only sensible reading of "relative" there and
is otherwise a silent `Inf` in the middle of a logarithmic axis.
"""
function relative_invariant_error(values::AbstractVector)
    I₀ = first(values)
    iszero(I₀) ? abs.(values .- I₀) : abs.((values .- I₀) ./ I₀)
end

"""
    Trajectory(label, t, q, p; continuous_t, continuous_q, invariant_error)

Everything a figure needs about one integrator's run, and nothing else.

# Why this exists rather than passing solutions around

A `GeometricSolution` is the wrong currency for a figure in two directions. It carries less than
a figure needs — the continuous solution between the steps lives in `integrate`'s *second* return
value, and the invariant error is not in either — and it carries more, since it is tied to the
version of `GeometricSolutions` that built it, so a result archived for later plotting cannot be
a solution object without pinning that version into the archive.

So a `Trajectory` is plain vectors: it survives a round trip through JLD2 or CSV, it is what the
plot functions actually consume, and a comparison curve that never was a solution — a closed-form
expression, a digitised reference — is expressible without pretending to be one.

`Trajectory(label, sol; …)` builds one from a solution, which is the common case.

# Fields

  - `label`: the legend entry.
  - `t`: the discrete times.
  - `q`, `p`: one series per degree of freedom, so `q[d][n]` is component `d` at step `n`.
  - `continuous_t`, `continuous_q`: the between-steps solution and its grid, or `nothing`.
    `continuous_q[d]` matches `continuous_t`, which is shared — the record is on one grid.
  - `invariant_error`: `|ΔH/H₀|` over the discrete steps, or `nothing` if it was not computed.
"""
struct Trajectory{T}
    label::String
    t::Vector{T}
    q::Vector{Vector{T}}
    p::Vector{Vector{T}}
    continuous_t::Union{Nothing, Vector{T}}
    continuous_q::Union{Nothing, Vector{Vector{T}}}
    invariant_error::Union{Nothing, Vector{T}}

    function Trajectory(label::AbstractString, t::AbstractVector,
            q::AbstractVector{<:AbstractVector}, p::AbstractVector{<:AbstractVector};
            continuous_t = nothing, continuous_q = nothing, invariant_error = nothing)
        T = promote_type(eltype(t), eltype(eltype(q)), eltype(eltype(p)))

        length(q) == length(p) ||
            throw(ArgumentError("$(length(q)) coordinate series against $(length(p)) " *
                                "momentum series."))
        all(s -> length(s) == length(t), q) && all(s -> length(s) == length(t), p) ||
            throw(ArgumentError("every component series must have the length of `t` " *
                                "($(length(t)))."))
        if continuous_q !== nothing
            continuous_t === nothing &&
                throw(ArgumentError("`continuous_q` was given without `continuous_t`."))
            length(continuous_q) == length(q) ||
                throw(ArgumentError("$(length(continuous_q)) continuous series against " *
                                    "$(length(q)) degrees of freedom."))
            all(s -> length(s) == length(continuous_t), continuous_q) ||
                throw(ArgumentError("every continuous series must have the length of " *
                                    "`continuous_t` ($(length(continuous_t)))."))
        end
        if invariant_error !== nothing
            length(invariant_error) == length(t) ||
                throw(ArgumentError("`invariant_error` has length " *
                                    "$(length(invariant_error)), `t` has $(length(t))."))
        end

        new{T}(String(label),
            collect(T, t),
            [collect(T, s) for s in q],
            [collect(T, s) for s in p],
            continuous_t === nothing ? nothing : collect(T, continuous_t),
            continuous_q === nothing ? nothing : [collect(T, s) for s in continuous_q],
            invariant_error === nothing ? nothing : collect(T, invariant_error))
    end
end

"""
    Trajectory(label, sol; internal_values, timestep, hamiltonian, parameters, nplot, nt)

Build a [`Trajectory`](@ref) from a `GeometricSolution`.

# Keyword arguments

  - `internal_values = nothing`: `integrate`'s second return value; supplies the continuous
    solution. The whole returned tuple is accepted too.
  - `timestep = nothing`: the macro step. Defaults to `sol.t[1] - sol.t[0]`, right for the
    uniform grid `internal_values` is recorded on.
  - `hamiltonian = nothing`, `parameters = nothing`: both required to fill `invariant_error`;
    with either missing it stays `nothing` rather than being silently wrong.
  - `nplot = 1`, `nt = :auto`: downsample and truncate the discrete series.
"""
function Trajectory(label::AbstractString, sol;
        internal_values = nothing,
        timestep = nothing,
        hamiltonian = nothing,
        parameters = nothing,
        nplot::Int = 1,
        nt = :auto)
    last_step = nt === :auto ? ntime(sol) : min(nt, ntime(sol))
    idx = 0:nplot:last_step

    D = length(sol.q[0])
    t = [sol.t[n] for n in idx]
    q = [[sol.q[n][d] for n in idx] for d in 1:D]
    p = [[sol.p[n][d] for n in idx] for d in 1:D]

    h = timestep === nothing ? (sol.t[1] - sol.t[0]) : timestep
    values = internal_values isa Tuple ? internal_values[2] : internal_values

    continuous_t = nothing
    continuous_q = nothing
    if values !== nothing
        pairs = [continuous_solution(values, h; dof = d, t₀ = sol.t[0]) for d in 1:D]
        continuous_t = first(first(pairs))
        continuous_q = [last(pair) for pair in pairs]
    end

    invariant_error = nothing
    if hamiltonian !== nothing && parameters !== nothing
        invariant_error = relative_invariant_error(
            [hamiltonian(sol.t[n], sol.q[n], sol.p[n], parameters) for n in idx])
    end

    Trajectory(label, t, q, p;
        continuous_t = continuous_t, continuous_q = continuous_q,
        invariant_error = invariant_error)
end

"""
    dimension(traj::Trajectory)

The number of degrees of freedom the trajectory carries.
"""
dimension(traj::Trajectory) = length(traj.q)

export continuous_solution, relative_invariant_error, Trajectory

"""
    NonlinearIntegrators.Diagnostics

Figures for the integrators of this package. The methods are implemented in the
`NonlinearIntegratorsPlots` extension, which loads together with `Makie`/`CairoMakie`.

# Why a submodule rather than four more exports

`plot_solution`, `plot_phase_portrait` and `plot_convergence` are all names `GeometricProblems`
already exports — the first two from every problem submodule (`HarmonicOscillator`, `Pendulum`,
`PointVortices`, …), the third from its own `Diagnostics`. Exporting them from this package's top
level would make all three ambiguous in any scope that also did
`using GeometricProblems.HarmonicOscillator`, which is what a script integrating a problem
naturally writes — and what this package's own test setup does.

So the same shape `GeometricProblems.Diagnostics` uses: a submodule, deliberately *not* exported,
reached as `NonlinearIntegrators.Diagnostics`, or brought in wholesale with
`using NonlinearIntegrators.Diagnostics` by a caller who knows it has no clash.
`continuous_solution`, `relative_invariant_error` and `Trajectory` stay at the top level — those
names are nobody else's.
"""
module Diagnostics

export plot_theme, plot_solution, plot_convergence

"""
    plot_theme()

The shared Makie theme of this ecosystem — larger fonts and thicker lines than the Makie defaults,
kept identical to the copy in `GeometricExamples/src/common.jl` and the publication companion
packages, so a figure from this package sits beside one of theirs without a visible change of
typeface size.

A function rather than a `const`, because a `Theme` is a Makie type and this file is loaded whether
or not Makie is: the value can only exist in the extension. Applied by the caller, not on load —
the choice of theme belongs to whoever is making the figure:

```julia
using CairoMakie
import NonlinearIntegrators.Diagnostics as NIP
set_theme!(NIP.plot_theme())
```

Nothing in the extension sets a font size, colour or line width of its own; every one comes from
the ambient theme, so a caller who wants something else sets that and gets it everywhere.
"""
function plot_theme end

# Two functions, because two things are missing from `GeometricProblems` and no more. A single
# solution's invariant error and drift are `GeometricProblems.Diagnostics.plot_energy_error` and
# `plot_energy_drift` (both take `energy = <function>`, which is what a `lodeproblem` built by
# `EulerLagrange` needs, since it carries `NullInvariants` and has no `:h` key to look up); a phase
# portrait, a trajectory or a set of traces are the per-problem recipes there. Reimplementing any of
# those here would be a second thing to keep right.

"""
    plot_solution(primary; reference, comparisons, kwargs...)

*Several* integrators of one problem in one figure, with the **continuous** solution between the
discrete steps. Implemented in the `NonlinearIntegratorsPlots` extension.

No `GeometricProblems` recipe takes more than one solution, and none knows about `integrate`'s
second return value — which is the subject of these integrators.
"""
function plot_solution end

"""
    plot_convergence(timesteps, errors; labels, kwargs...)

*Several* error series against the time step on logarithmic axes, with a reference slope per order.
Implemented in the `NonlinearIntegratorsPlots` extension.

For one series against one expected order, use `GeometricProblems.Diagnostics.plot_convergence`.
"""
function plot_convergence end

end
