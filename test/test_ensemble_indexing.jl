@testset "trajectory RNG stream version 2 uses the ordered seed-index pair" begin
    words(seed, index) = rand(SM._traj_rng(UInt64(seed), index), UInt64, 4)
    @test words(23, 3) == words(23, 3)
    @test all(words(23, index) != words(24, index - 1) for index in 2:50)
end

@testset "trajectory streams derive only from seed and trajectory number" begin
    @test SM._splitmix64(UInt64(0)) == 0xe220a8397b1dcdaf
    d = 4
    a = destroy(d)
    H = 0.15 * (a + a')
    psi0 = fock(d, 3)
    tlist = collect(0.0:0.2:1.0)
    common = (;
        e_ops = [num(d)],
        gauge_set = ComplexF64[0 0.6],
        ntraj = 12,
        seed = 0x1234,
        ensemblealg = :serial,
        saveat = tlist,
        save_trajectories = true,
        save_final_states = true,
    )
    first_run = dislou_solve(H, psi0, tlist, [1.2 * a]; common...)
    repeated = dislou_solve(H, psi0, tlist, [1.2 * a]; common...)
    another_seed = dislou_solve(
        H, psi0, tlist, [1.2 * a];
        common..., seed = 0x1235
    )

    @test first_run.col_times == repeated.col_times
    @test first_run.col_which == repeated.col_which
    @test first_run.col_gauge == repeated.col_gauge
    @test first_run.trajectory_expect == repeated.trajectory_expect
    @test first_run.final_states == repeated.final_states
    @test first_run.seed == UInt64(0x1234)
    @test (first_run.col_times, first_run.col_which, first_run.col_gauge) !=
        (another_seed.col_times, another_seed.col_which, another_seed.col_gauge)
end
