@testset "DiSLOUSolution has the exact public data contract" begin
    @test fieldnames(DiSLOUSolution) == (
        :ntraj, :times, :times_states, :states, :expect,
        :col_times, :col_which, :col_gauge, :njumps_total, :jumps_by_channel,
        :converged, :seed, :survival_rtol, :time_rtol, :time_atol,
        :expect_std, :expect_sem, :gauge_diagnostics, :layer3_diagnostics,
        :eigensystem_backend, :trajectory_states, :trajectory_expect,
        :final_states, :final_density,
    )
end

@testset "single trajectory statistics remain explicit" begin
    d = 2
    a = destroy(d)
    sol = dislou_solve(
        0 * a, fock(d, 1), [0.0, 0.1], [a];
        e_ops = [num(d)], gauge_set = zeros(ComplexF64, 1, 1),
        ntraj = 1, seed = 12, ensemblealg = :serial
    )

    @test all(iszero, sol.expect_std)
    @test all(isnan, sol.expect_sem)
    @test sol.converged
    @test sol.seed == UInt64(12)
    @test sol.eigensystem_backend in (:lapack, :cuda, :mixed)
end
