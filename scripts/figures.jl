# Render every figure of the experiment suite, out of the archives in `runs/`.
#
#   julia --project=scripts scripts/figures.jl [stem]...
#
# With no argument, every archive. With arguments, only the runs whose stem contains one of them,
# which is how a single figure is regenerated without re-rendering forty.
#
#   julia --project=scripts scripts/figures.jl harmonic-oscillator
#   julia --project=scripts scripts/figures.jl --results-dir ~/talk/figures
#
# Nothing here integrates. The separation is deliberate: a figure can be restyled — or a caption
# argued about — without paying for the solves again, and a missing figure stays distinguishable
# from a failed run.
#
# Nor is any figure *built* here. That is `NonlinearIntegrators.Diagnostics.figures`, in the
# `NonlinearIntegratorsPlots` extension, which turns one archive into the `stem => Figure` pairs it
# earns; this file only decides which archives to read and where to write the results. Keeping the
# Makie in one place is what stops a per-driver copy of it from drifting.
#
# It includes `archives.jl` and **not** `experiments.jl`. Drawing PDFs out of plain vectors needs
# neither the ansätze nor the solvers, and including the registry made this script load Symbolics,
# GeometricIntegrators, GeometricProblems and SimpleSolvers to do it.

include(joinpath(@__DIR__, "archives.jl"))

using CairoMakie
# `import … as`, not `using`: `plot_solution` is also exported by every `GeometricProblems` problem
# submodule, and `plot_convergence` by its `Diagnostics`.
import NonlinearIntegrators.Diagnostics as NIP

function emit(fig, stem; extension = "pdf")
    mkpath(RESULTS_DIR[])
    path = joinpath(RESULTS_DIR[], stem * "." * extension)
    save(path, fig)
    report_path("wrote", path)
    return path
end

function main(args)
    stems, _ = parse_arguments(args)

    # The shared theme of this ecosystem — larger fonts and thicker lines than the Makie defaults,
    # identical to the copy in `GeometricExamples/src/common.jl` so that a figure from here sits
    # beside one from there without a visible change of typeface size. The extension sets no size
    # of its own, so this is the single place that decides how every figure looks.
    #
    # Applied inside `main` and not at load time. At load time it is a global `set_theme!` that
    # outlives this script: a session that includes both this file and `oga_report.jl` would find
    # the OGA reports silently restyled, since those set their sizes per axis on top of the Makie
    # defaults.
    with_theme(NIP.plot_theme()) do
        runs = load_runs()
        if isempty(runs)
            # An empty run directory is a state, not a failure: the drivers have not been run yet.
            report("no archives in", RUNS_DIR[])
            return true
        end

        selected = isempty(stems) ? runs :
                   filter(d -> any(s -> occursin(s, d["stem"]), stems), runs)
        if isempty(selected)
            # A stem that matches nothing is a mistyped argument, and it gets the same treatment
            # as an unrecognised flag rather than an exit 0 that reads as "there was nothing to do".
            report("no archive matched", join(stems, ", "))
            return false
        end

        # An archive this renderer cannot draw is reported and skipped, not fatal. A run directory
        # accumulates across revisions of the registry, and it demonstrably holds files written
        # before `"kind"` existed; letting one of those abort the run costs every other figure and
        # says nothing useful about the one at fault. Same reasoning as the per-run guard in
        # `run_nvi.jl`: one failure must not take the other forty-eight with it.
        #
        # Skipping is not the same as succeeding, though, so the exception's own message is printed
        # and `main` exits non-zero. A renderer that failed on *every* archive and still printed
        # `done` and exited 0 is the shape a regeneration recipe cannot detect — the same failure
        # the shared argument parser exists to remove.
        skipped = String[]
        for data in selected
            stem = data["stem"]
            kind = archive_kind(data)
            if kind === nothing
                push!(skipped, stem)
                continue
            end
            data["kind"] = kind
            banner(stem)
            try
                for (name, fig) in NIP.figures(data)
                    emit(fig, name)
                end
            catch exception
                exception isa InterruptException && rethrow()
                push!(skipped, stem)
                report("FAILED", "$(nameof(typeof(exception))): " *
                                 failure_message(exception))
            end
        end

        isempty(skipped) || begin
            banner("$(length(skipped)) archive(s) not drawn")
            foreach(s -> println("    ", s), skipped)
        end
        return isempty(skipped)
    end
end

ok = main(ARGS)
banner("done")
ok || exit(1)
