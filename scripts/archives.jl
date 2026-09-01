# The shared layer under every driver in this directory: where output goes, how a run is archived,
# how arguments are parsed, and how progress is printed.
#
# Nothing here executes at top level, so it is safe to `include` from several files in one session.
#
# It is separate from `experiments.jl` because of what it *depends on*, not because of its size.
# `figures.jl` needs the archive layer and nothing else; `experiments.jl` pulls in Symbolics,
# GeometricIntegrators, GeometricProblems and SimpleSolvers to build the ansätze. Folded together,
# a script that draws PDFs out of plain vectors paid one to two minutes of load time for solver
# machinery it never calls.

using JLD2
using Printf

using NonlinearIntegrators

# ---- where output goes -------------------------------------------------------
#
# Code in `scripts/`, data in `runs/`, figures in `results/` — the tree-wide rule, in
# `Packages/CLAUDE.md`, with the reasoning in `Knowledge/AI/Folder-Structure.md`.
#
# `Ref`s rather than `const` paths, because `--runs-dir` and `--results-dir` have to be able to
# redirect them. Deriving an output directory from `@__DIR__` as a constant is what forces a caller
# to *copy* a script in order to write somewhere else, and it is why the talk that these scripts
# came from held its own copy of all of them.

const RUNS_DIR = Ref(joinpath(@__DIR__, "..", "runs"))
const RESULTS_DIR = Ref(joinpath(@__DIR__, "..", "results"))

# ---- arguments ---------------------------------------------------------------

const COMMON_OPTIONS = ("--runs-dir", "--results-dir")

"""
    parse_arguments(args, extra = ()) -> (positional, options)

Split `args` into positional names and `--flag value` pairs, accepting `--runs-dir` and
`--results-dir` everywhere plus whatever `extra` a driver adds, and **rejecting anything else**.

One parser rather than one per driver, and the rejection is the reason. The six scripts this
replaces each parsed their own arguments and disagreed about what an unrecognised one meant:
three crashed with a bare `MethodError` from indexing a `findfirst` that had returned `nothing`,
one pushed the flag onto its list of problem names and then, finding no match, silently fell back
to running the entire sweep, and one rendered nothing at all, printed `done`, and exited 0. That
last shape is the dangerous one — a regeneration recipe with a mistyped flag looks like it worked.

Sets `RUNS_DIR` and `RESULTS_DIR` as a side effect, so a driver calls this before it touches
either.
"""
function parse_arguments(args, extra = ())
    known = (COMMON_OPTIONS..., extra...)
    options = Dict{String, String}()
    positional = String[]

    i = firstindex(args)
    while i ≤ lastindex(args)
        argument = args[i]
        if startswith(argument, "--")
            argument in known || throw(ArgumentError(
                "unknown option `$(argument)`; this script takes " * join(known, ", ")))
            i < lastindex(args) || throw(ArgumentError("`$(argument)` needs a value"))
            options[argument] = args[i + 1]
            i += 2
        else
            push!(positional, argument)
            i += 1
        end
    end

    haskey(options, "--runs-dir") && (RUNS_DIR[] = options["--runs-dir"])
    haskey(options, "--results-dir") && (RESULTS_DIR[] = options["--results-dir"])

    return positional, options
end

"""
    option_steps(options, default) -> Tuple

The `--steps h1,h2,…` value, or `default`.
"""
function option_steps(options, default)
    haskey(options, "--steps") || return default
    Tuple(parse.(Float64, split(options["--steps"], ",")))
end

"""
    option_final_time(options, default) -> Float64

The `--final-time` value, or `default`. A driver that accepts this **prints what it used**: the
plotted quantity of a convergence study is a maximum over a run, so it can only grow with a longer
window, and two curves computed over different windows are not comparable.
"""
function option_final_time(options, default)
    haskey(options, "--final-time") || return default
    parse(Float64, options["--final-time"])
end

# ---- archives ----------------------------------------------------------------
#
# One JLD2 file per run, under `runs/`. The keys are flat and named for what they hold rather than
# for the variable that happened to hold it — the archives this replaces used keys like
# `HO_PR_sol_q` and a stored typo `hamltonian`, and no two problems agreed on a scheme.
#
# Two keys carry the schema itself:
#
#   "kind"    "solution" or "convergence" — what shape of figure this run draws.
#   "stem"    the figure's name, so that a renderer can glob `runs/` and needs no registry to
#             know what it is looking at or what to call the output.
#
# Everything else is per kind, and `Diagnostics.figures` in the plotting extension is what reads
# it back. Stored for a `"solution"` run:
#
#   "label", "problem", "problem_label"      the run's identity
#   "timestep"                               absent for a run that does not step (a global fit)
#   "final_time", "dimension"
#   "t", "q", "p"                            discrete times; one component series each
#   "continuous_t", "continuous_q"           the between-steps solution, on one shared grid
#   "hamiltonian_error"                      |ΔH/H₀| over the discrete steps
#   "max_hamiltonian_error"                  its maximum
#   "reference_t/q/p" or "exact_t/q/p"       the dashed reference
#   "comparisons"                            label => (t, q, p, hamiltonian_error)
#   "windows"                                extra final times to draw the same run over
#   "weights"                                converged ansatz weights per step (VISE only)
#
# and for a `"convergence"` run: "timesteps", "errors", "labels", "linestyles", "title",
# "reference_orders".

archive_path(stem) = joinpath(RUNS_DIR[], stem * ".jld2")

"""
    normalise_schema!(data) -> data

Bring an archive written by an older revision up to the keys the renderer reads.

Kept here, at the point of reading, rather than in the plotting extension: which spellings an
archive has had is a property of this directory's history, and the extension should see one
schema. The alternative — teaching the figure code every key an archive has ever used — spreads
that history across two repositories.

  - `figure_window` → `windows`. One run drawn over several intervals was one mechanism with two
    names, a scalar for the global fits and a vector for the network runs.
"""
function normalise_schema!(data)
    if haskey(data, "figure_window") && !haskey(data, "windows")
        data["windows"] = [data["figure_window"]]
    end
    return data
end

"""
    archive_kind(data) -> String or nothing

What shape of figure an archive draws: its `"kind"` if it has one, otherwise inferred from which
series it carries, and `nothing` if it carries neither shape.

The inference is not a fallback for sloppiness — it is what keeps the archive directory readable
across revisions of the writer. `"kind"` was added after these runs already existed on disk, and
requiring it strictly would mean re-running forty-five minutes of solves to redraw a figure from
an archive that already holds every number the figure needs. That is exactly the cost the split
between `runs/` and `results/` exists to avoid.

A convergence run carries error series against a step ladder; a solution run carries a
trajectory. Nothing carries both, so the shape is unambiguous.
"""
function archive_kind(data)
    haskey(data, "kind") && return data["kind"]
    haskey(data, "timesteps") && haskey(data, "errors") && return "convergence"
    haskey(data, "t") && haskey(data, "q") && haskey(data, "p") && return "solution"
    return nothing
end

"""
    store_run!(stem, data) -> String

Write one run's archive and return its path.

`jldopen` with explicit writes rather than `jldsave(path; data...)`: splatting a
`Dict{String, Any}` into keyword arguments requires `Symbol` keys, and these keys are deliberately
strings — they are data, read back by name, not identifiers.
"""
function store_run!(stem::String, data::Dict{String, Any})
    haskey(data, "kind") ||
        throw(ArgumentError("the archive for `$(stem)` has no \"kind\"; a renderer globbing " *
                            "`runs/` cannot tell what shape of figure it draws."))
    mkpath(RUNS_DIR[])
    data["stem"] = stem
    path = archive_path(stem)
    jldopen(path, "w") do file
        for (key, value) in data
            file[key] = value
        end
    end
    return path
end

load_run(stem::String) = load(archive_path(stem))

"""
    load_runs() -> Vector{Dict}

Every archive under `runs/`, sorted by name.

Sorted, and read from the directory rather than enumerated from a registry, because those are two
different failure modes and only one of them is visible. A registry that has drifted from the stem
a driver actually wrote produces a figure that silently is not drawn; a missing file produces a
figure that is silently not drawn *and* is indistinguishable from a run that was never made. The
directory is the record of what exists.
"""
function load_runs()
    isdir(RUNS_DIR[]) || return Dict{String, Any}[]
    runs = Dict{String, Any}[]
    for file in sort(filter(endswith(".jld2"), readdir(RUNS_DIR[])))
        data = load(joinpath(RUNS_DIR[], file))
        # The filename *is* the stem, by construction in `store_run!`. Filling it in when it is
        # absent is what lets a directory of archives written before `"stem"` existed still be
        # read: a run directory accumulates across revisions of the registry, and the older files
        # are not corrupt, only older.
        get!(data, "stem", chop(file; tail = length(".jld2")))
        normalise_schema!(data)
        push!(runs, data)
    end
    runs
end

"""
    solution_data(sol, D) -> (t, q, p)

A `GeometricSolution` reduced to plain vectors, which is what an archive should hold: storing the
solution object itself would tie the file to the version of GeometricSolutions that wrote it.
"""
function solution_data(sol, D)
    idx = 0:NonlinearIntegrators.ntime(sol)
    t = [sol.t[n] for n in idx]
    q = [[sol.q[n][d] for n in idx] for d in 1:D]
    p = [[sol.p[n][d] for n in idx] for d in 1:D]
    return t, q, p
end

# ---- reporting ---------------------------------------------------------------

report(label, value) = @printf("  %-46s %s\n", label, value)

report_error(label, value) = @printf("  %-46s %.3e\n", label, value)

banner(text) = println("\n", text, "\n", repeat("-", length(text)))

"""
    report_path(label, path)

Print a path relative to the repository root rather than to the script, so that a redirected
`--runs-dir` does not print `../../../../…`.
"""
function report_path(label, path)
    root = normpath(joinpath(@__DIR__, ".."))
    shown = startswith(normpath(path), root) ? relpath(path, root) : path
    report(label, shown)
end
