using Documenter
using QuantumToolbox
using Clustering
using QuantumCumulants
using DiSLOUTrajectories
using TOML

repository = if haskey(ENV, "GITHUB_REPOSITORY")
    ENV["GITHUB_REPOSITORY"]
else
    project = TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))
    replace(project["repo"], r"^https://github\.com/" => "", r"\.git$" => "")
end
repository_parts = split(repository, '/'; limit = 2)
length(repository_parts) == 2 ||
    error("repository must have owner/name form, got $repository")
repository_owner, repository_name = repository_parts

DocMeta.setdocmeta!(
    DiSLOUTrajectories,
    :DocTestSetup,
    quote
        using LinearAlgebra
        using QuantumToolbox
        using DiSLOUTrajectories
    end;
    recursive = true,
)

mktempdir() do root
    source = mkdir(joinpath(root, "src"))
    cp(joinpath(@__DIR__, "..", "README.md"), joinpath(source, "README.md"))
    makedocs(;
        root,
        sitename = "README examples",
        doctest = :only,
        remotes = nothing,
        format = Documenter.HTML(edit_link = nothing),
    )
end

makedocs(;
    modules = [DiSLOUTrajectories],
    authors = "Pietro Pacchioni and Fabrizio Minganti",
    repo = Remotes.GitHub(repository_owner, repository_name),
    sitename = "DiSLOUTrajectories.jl",
    checkdocs = :exports,
    doctest = true,
    linkcheck = true,
    # Anonymous GitHub links cannot be checked until the source repository is public.
    linkcheck_ignore = [r"^https://github\.com/Alice-Bob-SW/DiSLOUTrajectories\.jl(?:/|$)"],
    format = Documenter.HTML(
        canonical = "https://$(lowercase(repository_owner)).github.io/$(repository_name)/",
        edit_link = "main",
        sidebar_sitename = false,
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
    pages = [
        "Home" => [
            "Introduction" => "index.md",
            "Installation" => "getting_started/installation.md",
            "Quick start" => "getting_started/quickstart.md",
            "Cite" => "getting_started/cite.md",
        ],
        "Examples" => [
            "Overview" => "examples.md",
            "Driven Kerr resonator" => "examples/kerr_resonator.md",
            "Two-mode ideal cat" => "examples/two_mode_ideal_cat.md",
            "Two-mode detuned cat" => "examples/two_mode_detuned_cat.md",
        ],
        "API reference" => "api.md",
    ],
)

if get(ENV, "DOCUMENTER_DEPLOY", "false") == "true"
    Documenter.deploydocs(;
        repo = "github.com/$(repository).git",
        devbranch = "main",
        push_preview = false,
    )
end
