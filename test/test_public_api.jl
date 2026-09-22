@testset "DiSLOUTrajectories exports exactly the production API" begin
    exported = Set(names(DiSLOUTrajectories; all = false, imported = false))
    delete!(exported, :DiSLOUTrajectories)
    @test exported == Set(
        (
            :dislou_solve, :discover_gauges, :DiSLOUSolution,
            :expect_mean, :expect_sem,
            :FirstPassageConvergenceError, :backend_info,
            :about, :cite, :versioninfo,
        )
    )
end

@testset "backend information uses the stable public shape" begin
    info = backend_info()
    @test keys(info) == (:cpu, :cuda_extension_loaded, :cuda_enabled)
    @test info.cpu === :lapack
    @test info.cuda_extension_loaded isa Bool
    @test info.cuda_enabled isa Bool
end

@testset "dislou_solve requires an explicit gauge set" begin
    @test_throws UndefKeywordError dislou_solve(
        zeros(ComplexF64, 1, 1), ComplexF64[1], [0.0],
        [zeros(ComplexF64, 1, 1)];
        ntraj = 1,
        ensemblealg = :serial,
    )
end

@testset "a discovered gauge set passes through the solver boundary" begin
    d = 5
    a = destroy(d)
    H, psi0, ts, c_ops = 0.1 * num(d), fock(d, 2), [0.0, 0.05, 0.1], [sqrt(0.2) * a]
    found = (;
        shifts = zeros(ComplexF64, 1, 1),
        method = :trajectories,
        centers = zeros(ComplexF64, 1, 1),
        weights = [1.0],
        diagnostics = (;),
    )
    reused = dislou_solve(
        H, psi0, ts, c_ops;
        gauge_set = found, ntraj = 8, seed = 17, ensemblealg = :serial
    )

    @test reused isa DiSLOUSolution
    @test reused.gauge_diagnostics.method === :trajectories
end

@testset "the diagnostic reports write to any IO" begin
    report = sprint(DiSLOUTrajectories.versioninfo)
    @test occursin("DiSLOUTrajectories.jl", report)
    @test occursin("Julia:", report)
    @test occursin("CUDA extension:", report)
    @test sprint(DiSLOUTrajectories.about) == report

    bibtex = sprint(DiSLOUTrajectories.cite)
    @test occursin("Pacchioni2026DiSLOU", bibtex)
    @test occursin("Pacchioni2026Diagonal", bibtex)
end
