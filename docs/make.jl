using StateVectors
using Documenter

DocMeta.setdocmeta!(StateVectors, :DocTestSetup, :(using StateVectors); recursive=true)

makedocs(;
    modules=[StateVectors],
    authors="F. Capolupo",
    sitename="StateVectors.jl",
    format=Documenter.HTML(;
        canonical="https://FraCpl.github.io/StateVectors.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/FraCpl/StateVectors.jl",
    devbranch="master",
)
