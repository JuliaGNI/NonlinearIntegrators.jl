using Documenter
using DocumenterCitations
using NonlinearIntegrators

# Regenerate the shallow-net benchmark figures embedded in the Benchmarks page.
#
# Rather than committing the PNGs, we run the `quick` benchmark suite (under
# `benchmark/`) in its own environment and copy the figures into
# `docs/src/benchmarks/figures/`, where `benchmarks/benchmarks.md` references them. Set
# `SKIP_SHALLOWNET_BENCH=true` to skip this (useful while iterating on unrelated docs locally).
#
# Two families of figures are copied: the per-problem plots (prefix `<problem>_quick`,
# each coloured by precision) for the per-problem sections of the page, and the combined
# summary plots (prefix `shallownet_benchmark`, scatters coloured by problem) for the
# summary section.
function generate_benchmark_figures()
    benchdir = normpath(joinpath(@__DIR__, "..", "benchmark"))
    figdir   = joinpath(@__DIR__, "src", "benchmarks", "figures")
    resdir   = joinpath(benchdir, "results")
    mkpath(figdir)
    julia = Base.julia_cmd()

    # The Toda lattice is deliberately absent. Its quick sweep costs about five hours against
    # seven minutes for the other three combined, because every `Float64` case runs its full
    # iteration budget: the network width has not been chosen for it the way it now has for
    # the others (see the `Ss` override in each `run_*.jl`), so its residual floors above the
    # convergence target and the solve iterates to the cap. `benchmark/run_toda_lattice.jl`
    # still works and is included in `full`; it returns here once the width is measured.
    problems = ["harmonic_oscillator", "pendulum", "double_pendulum"]
    for p in problems
        run(`$(julia) --project=$(benchdir) $(joinpath(benchdir, "run_$(p).jl")) quick`)
    end
    run(`$(julia) --project=$(benchdir) $(joinpath(benchdir, "report.jl"))`)

    # Per-problem figures (coloured by precision).
    per_problem_metrics = ["accuracy_vs_dt", "energy_drift_vs_dt", "runtime_vs_dt",
                           "iterations_vs_dt", "convergence_heatmap"]
    figs = ["$(p)_quick_$(m).png" for p in problems for m in per_problem_metrics]

    # Combined summary figures (scatters coloured by problem).
    append!(figs, ["shallownet_benchmark_convergence_problem.png",
                   "shallownet_benchmark_convergence_solver.png",
                   "shallownet_benchmark_convergence_heatmap.png",
                   "shallownet_benchmark_accuracy_vs_dt.png",
                   "shallownet_benchmark_energy_drift_vs_dt.png",
                   "shallownet_benchmark_runtime_vs_dt.png",
                   "shallownet_benchmark_iterations_vs_dt.png"])

    # A plot the reporting step skipped (no data) leaves no file, so copy what is there.
    for fig in figs
        src = joinpath(resdir, fig)
        isfile(src) && cp(src, joinpath(figdir, fig); force=true)
    end

    # The reporting step skips any plot with no measured cases, which used to surface
    # half an hour later as a batch of Documenter `invalid local link/image` errors.
    # Check the figures the page actually references and fail here instead, naming them.
    page = read(joinpath(@__DIR__, "src", "benchmarks", "benchmarks.md"), String)
    referenced = unique(m[1] for m in eachmatch(r"\]\(figures/([\w.\-]+\.png)\)", page))
    absent = filter(f -> !isfile(joinpath(figdir, f)), referenced)
    isempty(absent) || error("""
        Benchmarks page references $(length(absent)) figure(s) the sweep did not produce:
          $(join(absent, "\n  "))
        The reporting step only plots cases that produced a trajectory (`ok` or `maxiter`),
        so this usually means the benchmark harness itself failed — check the per-case
        `status` column above.""")
    return nothing
end

if get(ENV, "SKIP_SHALLOWNET_BENCH", "false") != "true"
    generate_benchmark_figures()
end

DocMeta.setdocmeta!(NonlinearIntegrators, :DocTestSetup, :(using NonlinearIntegrators); recursive=true)

# Create bibliography
bib = CitationBibliography(joinpath(@__DIR__, "NonlinearIntegrators.bib"))
println(joinpath(@__DIR__, "NonlinearIntegrators.bib"))
makedocs(
    sitename="NonlinearIntegrators.jl",
    plugins=[bib,],
    modules=[NonlinearIntegrators],
    authors="Michael Kraus <michael.kraus@ipp.mpg.de>, Zeyuan Li <zeyuan.li@ipp.mpg.de> and contributors",
    format=Documenter.HTML(;
        canonical="https://JuliaGNI.github.io/NonlinearIntegrators.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Orthogonal Greedy Algorithm" => [
            "Overview" => "oga/oga.md",
            "Theory" => "oga/theory.md",
            "Algorithms" => "oga/algorithms.md",
            "Usage" => "oga/usage.md",
            "Precision" => "oga/precision.md",
            "Studies" => "oga/studies.md",
        ],
        "Variational Integrator with Symbolic Expression" =>
            "vise/vise.md",
        "Neural Variational Integrators" => [
            "ShallowNet"                 => "nvi/shallownet.md",
            # "ShallowNet (Reversible)"    => "nvi/shallownet_reversible.md",
            # "ShallowNet (Autodiff)"      => "nvi/shallownet_autodiff.md",
            # "ShallowNet (Autodiff+Rev.)" => "nvi/shallownet_autodiff_reversible.md",
            # "DenseNet"                   => "nvi/densenet.md",
            # "VISE Results"               => "nvi/vise_results.md",
        ],
        "Benchmarks" => "benchmarks/benchmarks.md",
    ],
    warnonly = [:cross_references],
)

# `devurl` is deliberately absent, i.e. left at Documenter's default of `"dev"`. It used to be
# `"stable"`, which published `main` at `/stable/` — and that is unusable once the package has a
# tagged release. `deploydocs` defaults to `versions = ["stable" => "v^", "v#.#", devurl =>
# devurl]`, so `devurl = "stable"` puts two entries under the one name: the symlink to the newest
# release, and the devurl directory. On a tag build Documenter creates `v0.2.0/`, then tries the
# link and finds the name taken:
#
#     ArgumentError: link `"stable" => "v0.2.0"` cannot overwrite `devurl = stable` with the same
#     name.
#
# It never surfaced before v0.2.0 because this workflow only deploys a release build on a tag push
# and the repository had no version tags at all — `v^` matched nothing, so no link was attempted.
# The first release is exactly what exposes it. With the default, `main` goes to `/dev/`, each tag
# to `/vX.Y.Z/`, and `/stable/` is the symlink to the newest release, which is what "stable" should
# mean for a registered package.
deploydocs(;
    repo="github.com/JuliaGNI/NonlinearIntegrators.jl",
    devbranch="main",
)
