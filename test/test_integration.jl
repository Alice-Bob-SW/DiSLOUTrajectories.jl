isdefined(Main, :DrivenKerrParams) ||
    include(joinpath(@__DIR__, "fixtures", "driven_kerr_model.jl"))

@testset "driven non-normal evolution agrees with the master equation" begin
    d = 8
    a = destroy(d)
    H = 0.4 * (a + a')
    psi0 = fock(d, 0)
    tlist = collect(0.0:0.25:2.0)
    c_ops = [sqrt(0.8) * a]
    e_ops = [num(d), a + a']
    sol = dislou_solve(
        H, psi0, tlist, c_ops;
        e_ops, gauge_set = zeros(ComplexF64, 1, 1), ntraj = 1_500,
        seed = 71, ensemblealg = :serial
    )
    exact = mesolve(
        H, psi0, tlist, c_ops;
        e_ops, progress_bar = Val(false)
    ).expect

    @test all(abs.(sol.expect .- exact) .<= 7 .* sol.expect_sem .+ 2.0e-3)
    @test sum(length, sol.col_times) == sol.njumps_total
    @test sum(sol.jumps_by_channel) == sol.njumps_total
end

@testset "all first-passage methods retain the physical trajectory stream" begin
    d = 4
    a = destroy(d)
    tlist = collect(0.0:0.2:1.0)
    common = (;
        e_ops = [num(d)],
        gauge_set = zeros(ComplexF64, 1, 1),
        ntraj = 32,
        seed = 77,
        ensemblealg = :serial,
    )
    direct = dislou_solve(
        0.1 * (a + a'), fock(d, 2), tlist, [0.9 * a];
        common..., first_passage_method = :survival
    )
    logarithmic = dislou_solve(
        0.1 * (a + a'), fock(d, 2), tlist, [0.9 * a];
        common..., first_passage_method = :log_survival
    )
    predictor = dislou_solve(
        0.1 * (a + a'), fock(d, 2), tlist, [0.9 * a];
        common..., first_passage_method = :log_survival_predictor
    )

    @test direct.col_which == logarithmic.col_which == predictor.col_which
    @test direct.col_gauge == logarithmic.col_gauge == predictor.col_gauge
    @test all(
        isapprox(direct.col_times[i], logarithmic.col_times[i]; atol = 1.0e-10)
            for i in 1:direct.ntraj
    )
    @test all(
        isapprox(logarithmic.col_times[i], predictor.col_times[i]; atol = 1.0e-10)
            for i in 1:direct.ntraj
    )
end

@testset "bistable Kerr agrees with the master equation across two gauges" begin
    model = driven_kerr_model()
    gauge_set = reshape(
        -sqrt(model.κ) .* ComplexF64[model.αlow, model.αhigh], 1, :
    )
    e_ops = [model.nop, model.xop]
    tlist = collect(0.0:0.25:2.0)
    sol = dislou_solve(
        model.H, model.ψ0, tlist, model.c_ops;
        e_ops, gauge_set, ntraj = 2_000, seed = 71, ensemblealg = :serial
    )
    exact = mesolve(
        model.H, model.ψ0, tlist, model.c_ops;
        e_ops, progress_bar = Val(false)
    ).expect

    @test all(abs.(sol.expect .- exact) .<= 7 .* sol.expect_sem .+ 2.0e-3)
    # The displaced frames are only exercised if trajectories actually switch.
    @test sol.gauge_diagnostics.switches > 0
    @test all(>(0), sol.gauge_diagnostics.jumps_by_gauge)
end
