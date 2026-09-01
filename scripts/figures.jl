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
# earns; this file only decides which archives to read and where to write the results. The six
# scripts this replaces carried some 450 lines of copy-pasted Makie between them, three near
# identical times over.
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
        isempty(runs) && return report("no archives in", RUNS_DIR[])

        selected = isempty(stems) ? runs :
                   filter(d -> any(s -> occursin(s, d["stem"]), stems), runs)
        isempty(selected) && return report("no archive matched", join(stems, ", "))

        for data in selected
            banner(data["stem"])
            for (stem, fig) in NIP.figures(data)
                emit(fig, stem)
            end
        end
    end

    banner("done")
end

main(ARGS)
