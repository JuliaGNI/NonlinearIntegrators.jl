using NonlinearIntegrators
using Documenter

DocMeta.setdocmeta!(NonlinearIntegrators, :DocTestSetup, :(using NonlinearIntegrators); recursive=true)

makedocs(;
    modules=[NonlinearIntegrators],
    authors="Michael Kraus <michael.kraus@ipp.mpg.de> and contributors",
    sitename="NonlinearIntegrators.jl",
    format=Documenter.HTML(;
        canonical="https://JuliaGNI.github.io/NonlinearIntegrators.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGNI/NonlinearIntegrators.jl",
    devbranch="main",
)
