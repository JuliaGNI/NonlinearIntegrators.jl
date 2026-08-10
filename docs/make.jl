using Documenter
using DocumenterCitations
using NonlinearIntegrators

# Regenerate the one-layer GML benchmark figures embedded in the Benchmarks page.
#
# Rather than committing the PNGs, we run the `quick` benchmark suite (under
# `benchmark/`) in its own environment and copy the figures into
# `docs/src/Benchmarks/figures/`, where `Benchmarks/benchmarks.md` references them. Set
# `SKIP_GML_BENCH=true` to skip this (useful while iterating on unrelated docs locally).
#
# Two families of figures are copied: the per-problem plots (prefix `<problem>_quick`,
# each coloured by precision) for the per-problem sections of the page, and the combined
# summary plots (prefix `onelayer_gml_benchmark`, scatters coloured by problem) for the
# summary section.
function generate_benchmark_figures()
    benchdir = normpath(joinpath(@__DIR__, "..", "benchmark"))
    figdir   = joinpath(@__DIR__, "src", "Benchmarks", "figures")
    resdir   = joinpath(benchdir, "results")
    mkpath(figdir)
    julia = Base.julia_cmd()

    problems = ["harmonic_oscillator", "pendulum", "double_pendulum", "toda_lattice"]
    for p in problems
        run(`$(julia) --project=$(benchdir) $(joinpath(benchdir, "run_$(p).jl")) quick`)
    end
    run(`$(julia) --project=$(benchdir) $(joinpath(benchdir, "report.jl"))`)

    # Per-problem figures (coloured by precision).
    per_problem_metrics = ["accuracy_vs_dt", "energy_drift_vs_dt", "runtime_vs_dt",
                           "iterations_vs_dt", "convergence_heatmap"]
    figs = ["$(p)_quick_$(m).png" for p in problems for m in per_problem_metrics]

    # Combined summary figures (scatters coloured by problem).
    append!(figs, ["onelayer_gml_benchmark_convergence_problem.png",
                   "onelayer_gml_benchmark_convergence_solver.png",
                   "onelayer_gml_benchmark_convergence_heatmap.png",
                   "onelayer_gml_benchmark_accuracy_vs_dt.png",
                   "onelayer_gml_benchmark_energy_drift_vs_dt.png",
                   "onelayer_gml_benchmark_runtime_vs_dt.png",
                   "onelayer_gml_benchmark_iterations_vs_dt.png"])

    # A plot the reporting step skipped (no data) leaves no file, so copy what is there.
    for fig in figs
        src = joinpath(resdir, fig)
        isfile(src) && cp(src, joinpath(figdir, fig); force=true)
    end

    # The reporting step skips any plot with no converged cases, which used to surface
    # half an hour later as a batch of Documenter `invalid local link/image` errors.
    # Check the figures the page actually references and fail here instead, naming them.
    page = read(joinpath(@__DIR__, "src", "Benchmarks", "benchmarks.md"), String)
    referenced = unique(m[1] for m in eachmatch(r"\]\(figures/([\w.\-]+\.png)\)", page))
    absent = filter(f -> !isfile(joinpath(figdir, f)), referenced)
    isempty(absent) || error("""
        Benchmarks page references $(length(absent)) figure(s) the sweep did not produce:
          $(join(absent, "\n  "))
        The reporting step only plots converged (`ok`) cases, so this usually means the
        benchmark harness itself failed — check the per-case `status` column above.""")
    return nothing
end

if get(ENV, "SKIP_GML_BENCH", "false") != "true"
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
        "Orthogonal Greedy Algorithm" => "Orthogonal Greedy Algorithm/OGA.md",
        "Variational Integrator with Symbolic Expression" => "Variational Integrator with Symbolic Expression/VISE.md",
        "Benchmarks" => "Benchmarks/benchmarks.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGNI/NonlinearIntegrators.jl",
    devbranch="main",
    devurl="stable",
)
