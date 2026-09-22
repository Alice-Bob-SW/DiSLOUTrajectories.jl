using Aqua
using DiSLOUTrajectories
using JET
using Random
using Test

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

include("../reporting/check.jl")

@testset "Code quality" begin
    @testset "Aqua" begin
        # Aqua's subprocess probe flakes on hosted Julia 1.12
        Aqua.test_all(DiSLOUTrajectories; persistent_tasks = false)
    end
    @testset "JET" begin
        # Broad definition analysis cannot narrow mutable optional recording fields or
        # the unloaded CUDA extension. Keep it alongside strict concrete checks below.
        JET.test_package(DiSLOUTrajectories; target_modules = (DiSLOUTrajectories,), ignore_missing_comparison = true, mode = :typo)

        H = ComplexF64[0 0; 0 1]
        C = [ComplexF64[0 0.3; 0 0]]
        psi = ComplexF64[0, 1]
        cache = DiSLOUTrajectories._diagonal_cache_from_matrices(H, C)
        c = cache.Vfac \ psi
        sol = dislou_solve(
            H, psi, [0.0, 1.0], C;
            gauge_set = zeros(ComplexF64, 1, 1), e_ops = [H],
            ntraj = 2, ensemblealg = :serial
        )
        # Basic error analysis of public accessors. Optimization checks below focus
        # on hot kernels: Julia 1.10 cannot narrow expect_sem's optional matrix field.
        for (f, args) in ((backend_info, ()), (expect_mean, (sol, 1)), (expect_sem, (sol, 1)))
            @testset "$(nameof(f))" begin
                JET.test_call(f, typeof.(args); target_modules = (DiSLOUTrajectories,), mode = :basic)
            end
        end
        # Error and optimization analysis; no report types or call sites suppressed.
        for (f, args) in (
                (DiSLOUTrajectories._survival_probability!, (similar(c), similar(c), c, cache.Λ, cache.G, 0.3, cache.Gnorm)),
                (DiSLOUTrajectories._survival_and_rate!, (similar(c), c, cache.Λ, cache.G, cache.Gnorm)),
                (DiSLOUTrajectories._sample_channel, ([0.1, 0.3], Xoshiro(1))),
                (DiSLOUTrajectories._find_first_passage!, (DiSLOUTrajectories._FirstPassageBuffers(cache), cache, c, 0.5, 1.0)),
            )
            @testset "$(nameof(f))" begin
                JET.test_call(f, typeof.(args); target_modules = (DiSLOUTrajectories,), mode = :basic)
                JET.test_opt(f, typeof.(args); target_modules = (DiSLOUTrajectories,))
            end
        end
    end

    @testset "every export has a docstring" begin
        # Inspect registered docstrings directly; Docs.hasdoc requires Julia 1.11.
        undocumented = sort!(
            String[
                string(name) for name in names(DiSLOUTrajectories; all = false, imported = false)
                    if !haskey(Docs.meta(DiSLOUTrajectories), Docs.Binding(DiSLOUTrajectories, name))
            ]
        )
        @test isempty(undocumented)
    end

    # Documenter is configured to fail on undocumented exports, failing
    # doctests, and broken links; assert that configuration is still in place
    # rather than re-implementing those checks here.
    @testset "documentation build stays strict" begin
        makefile = read(joinpath(ROOT, "docs", "make.jl"), String)
        @test occursin(r"checkdocs\s*=\s*:exports", makefile)
        @test occursin(r"doctest\s*=\s*true", makefile)
        @test !occursin("warnonly", makefile)
    end

    # A new test_*.jl file that nobody wired into runtests.jl runs nowhere.
    # The three optional-dependency suites are owned by dedicated CI jobs.
    @testset "every package test file runs somewhere" begin
        extended = Set(
            [
                "test_cuda_extension.jl",
                "test_distributed.jl",
                "test_semiclassical.jl",
            ]
        )
        runtests = read(joinpath(ROOT, "test", "runtests.jl"), String)
        included = Set(
            match.captures[1] for match in
                eachmatch(r"""include\("([^"]+)"\)""", runtests)
        )
        @test all(path -> isfile(joinpath(ROOT, "test", path)), included)
        present = Set(
            filter(
                path -> startswith(path, "test_") && endswith(path, ".jl"),
                readdir(joinpath(ROOT, "test")),
            )
        )
        @test setdiff(present, included) == extended
        @test all(path -> isfile(joinpath(ROOT, "test", path)), extended)
    end
end
