using Documenter
using DocumenterCitations
using NonlinearIntegrators

# Set the plotting backend and no window display
ENV["GKSwstype"] = "100"

DocMeta.setdocmeta!(NonlinearIntegrators, :DocTestSetup, :(using NonlinearIntegrators); recursive=true)

# Create bibliography
bib = CitationBibliography(joinpath(@__DIR__, "NonlinearIntegrators.bib"))
println(joinpath(@__DIR__, "NonlinearIntegrators.bib"))
makedocs(
    sitename="NonlinearIntegrators.jl",
    plugins=[bib, ],
    modules=[NonlinearIntegrators],
    authors="Michael Kraus <michael.kraus@ipp.mpg.de> and contributors",
    format=Documenter.HTML(;
        canonical="https://JuliaGNI.github.io/NonlinearIntegrators.jl",
        edit_link="zeyuan",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGNI/NonlinearIntegrators.jl",
    devbranch="zeyuan",
    devurl="stable",
)
