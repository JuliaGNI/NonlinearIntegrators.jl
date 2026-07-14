using Documenter
using DocumenterCitations
using NonlinearIntegrators

# Regenerate the one-layer GML benchmark figures embedded in the Benchmarks page.
#
# Rather than committing the PNGs, we run the `quick` benchmark suite (under
# `benchmark/`) in its own environment and copy the six combined-report figures into
# `docs/src/Benchmarks/figures/`, where `Benchmarks/benchmarks.md` references them. Set
# `SKIP_GML_BENCH=true` to skip this (useful while iterating on unrelated docs locally).
function generate_benchmark_figures()
    benchdir = normpath(joinpath(@__DIR__, "..", "benchmark"))
    figdir   = joinpath(@__DIR__, "src", "Benchmarks", "figures")
    resdir   = joinpath(benchdir, "results")
    mkpath(figdir)
    julia = Base.julia_cmd()

    runs = ["run_harmonic_oscillator.jl", "run_pendulum.jl",
            "run_double_pendulum.jl", "run_toda_lattice.jl"]
    for f in runs
        run(`$(julia) --project=$(benchdir) $(joinpath(benchdir, f)) quick`)
    end
    run(`$(julia) --project=$(benchdir) $(joinpath(benchdir, "report.jl"))`)

    figs = ["onelayer_gml_benchmark_convergence_solver.png",
            "onelayer_gml_benchmark_convergence_heatmap.png",
            "onelayer_gml_benchmark_accuracy_vs_dt.png",
            "onelayer_gml_benchmark_energy_drift_vs_dt.png",
            "onelayer_gml_benchmark_runtime_vs_dt.png",
            "onelayer_gml_benchmark_iterations_vs_dt.png"]
    for fig in figs
        cp(joinpath(resdir, fig), joinpath(figdir, fig); force=true)
    end
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
        "Benchmarks" => "Benchmarks/benchmarks.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGNI/NonlinearIntegrators.jl",
    devbranch="main",
    devurl="stable",
)
