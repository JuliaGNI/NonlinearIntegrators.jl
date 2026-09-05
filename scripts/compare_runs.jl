# Compare two directories of run archives, numerically.
#
#   julia --project=scripts scripts/compare_runs.jl <reference-dir> [--runs-dir <dir>] [stem]...
#
# For every archive present in both, every numeric series is compared elementwise and the worst
# relative difference reported. Exits non-zero if any exceeds `TOLERANCE`.
#
# Written to answer one question — *did moving this harness change any result?* — but kept because
# the question recurs: it is the check for any change that is supposed to be a refactoring, and it
# is the only way to tell a restyled figure from a different experiment drawn the same way.
#
# The comparison is on the archives and not on the figures deliberately. Two PDFs of the same data
# differ in ways that mean nothing (font hinting, a timestamp) and agree in ways that mean nothing
# either (a plot whose axis limits hide a changed curve). The numbers are the claim.

include(joinpath(@__DIR__, "archives.jl"))

# Solver-level agreement, not bitwise. Two runs of the same experiment can differ in the last few
# digits — a different BLAS path, a reassociated sum — without differing as experiments. Anything
# above this is a change in what was computed.
const TOLERANCE = 1e-10

# `relative` for scale-carrying quantities, absolute where the reference is ~0: a Hamiltonian error
# archived as exactly 0.0 at t₀ is not a 100 % disagreement with 1e-17.
function worst_difference(a::AbstractArray, b::AbstractArray)
    size(a) == size(b) || return Inf
    worst = 0.0
    for (x, y) in zip(a, b)
        (x isa Number && y isa Number) || continue
        # A non-finite pair has no relative difference to take. Two identical ones — both `NaN`,
        # both the same infinity — are still agreement; anything else is a change.
        if !(isfinite(x) && isfinite(y))
            x === y || return Inf
            continue
        end
        scale = max(abs(x), abs(y))
        d = scale < 1e-12 ? abs(x - y) : abs(x - y) / scale
        worst = max(worst, d)
    end
    worst
end

worst_difference(a::Number, b::Number) = worst_difference([a], [b])

function worst_difference(a::AbstractVector{<:AbstractVector},
        b::AbstractVector{<:AbstractVector})
    length(a) == length(b) || return Inf
    isempty(a) ? 0.0 : maximum(worst_difference(x, y) for (x, y) in zip(a, b))
end

worst_difference(a, b) = a == b ? 0.0 : Inf

# Wall-clock timings, and `stem`, which `store_run!` writes as a property of the file. These
# measure the machine and the run's name, not the experiment: a rerun on a warm cache differs from
# a cold one by tens of percent while computing exactly the same trajectory. Excluded so that the
# tolerance can stay at solver level rather than being loosened until timings fit under it — which
# would also hide a real change.
const NOT_A_RESULT = ("stem", "vise_seconds", "seconds", "timings")

# Everything else is compared, numeric or not. Metadata — labels, the problem name — falls through
# to an equality check, which is what catches a run that kept its numbers and changed its identity.
function compare_archive(reference, current)
    worst = 0.0
    culprit = ""
    for key in sort(collect(intersect(keys(reference), keys(current))))
        key in NOT_A_RESULT && continue
        d = worst_difference(reference[key], current[key])
        if d > worst
            worst = d
            culprit = key
        end
    end

    # Keys on one side only, reported separately rather than folded into the worst difference.
    # Comparing the intersection alone would pass a run that had silently *stopped writing* a
    # series, which is the one failure an archive comparison exists to catch and the one it cannot
    # express as a number. A renamed key shows up here as one dropped and one added.
    dropped = sort(collect(setdiff(keys(reference), keys(current), NOT_A_RESULT)))
    added = sort(collect(setdiff(keys(current), keys(reference), NOT_A_RESULT)))

    return worst, culprit, dropped, added
end

function main(args)
    stems, _ = parse_arguments(args)
    isempty(stems) && throw(ArgumentError(
        "give the reference run directory as the first argument"))
    reference_dir = popfirst!(stems)
    isdir(reference_dir) || throw(ArgumentError("no such directory: $(reference_dir)"))

    isdir(RUNS_DIR[]) || throw(ArgumentError(
        "no run directory at $(RUNS_DIR[]); run a driver first, or point `--runs-dir` at one"))

    banner("Comparing $(RUNS_DIR[])\n       against $(reference_dir)")

    available = filter(endswith(".jld2"), readdir(RUNS_DIR[]))
    selected = isempty(stems) ? available :
               filter(f -> any(s -> occursin(s, f), stems), available)

    worst_overall = 0.0
    compared = 0
    missing_reference = String[]

    for file in sort(selected)
        path = joinpath(reference_dir, file)
        if !isfile(path)
            push!(missing_reference, file)
            continue
        end
        worst, culprit, dropped, added = compare_archive(
            load(path), load(joinpath(RUNS_DIR[], file)))
        compared += 1
        worst_overall = max(worst_overall, worst)
        status = worst ≤ TOLERANCE ? "ok" : "DIFFERS"
        @printf("  %-8s %-52s %.3e  %s\n", status, chop(file; tail = 5), worst, culprit)
        isempty(dropped) || println("           no longer written: ", join(dropped, ", "))
        isempty(added) || println("           newly written:    ", join(added, ", "))
    end

    banner("$(compared) archive(s) compared")
    isempty(missing_reference) || begin
        report("not in the reference", length(missing_reference))
        foreach(f -> println("    ", f), missing_reference)
    end
    report_error("worst relative difference", worst_overall)

    if worst_overall > TOLERANCE
        error("$(worst_overall) exceeds the $(TOLERANCE) tolerance: this is a change in what " *
              "was computed, not a refactoring.")
    end
    compared == 0 && error("nothing was compared; check the reference directory.")
    banner("identical to solver tolerance")
end

main(ARGS)
