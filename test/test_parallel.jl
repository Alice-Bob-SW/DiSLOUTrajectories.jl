function same_saved_trajectories(left, right)
    @test left.col_times == right.col_times
    @test left.col_which == right.col_which
    @test left.col_gauge == right.col_gauge
    @test left.trajectory_expect == right.trajectory_expect
    @test size(left.trajectory_states) == size(right.trajectory_states)
    return @test all(
        isapprox(
            left.trajectory_states[i, k].data * left.trajectory_states[i, k].data',
            right.trajectory_states[i, k].data * right.trajectory_states[i, k].data';
            atol = 1.0e-13,
        ) for i in axes(left.trajectory_states, 1), k in axes(left.trajectory_states, 2)
    )
end

@testset "serial and threaded streams agree trajectory by trajectory" begin
    was_enabled = backend_info().cuda_enabled
    DiSLOUTrajectories._disable_cuda_diagonalization!()
    try
        d = 5
        a = destroy(d)
        H = 0.2 * (a + a')
        psi0 = fock(d, 3)
        tlist = collect(0.0:0.2:1.2)
        common = (;
            e_ops = [num(d)],
            gauge_set = ComplexF64[0 0.8],
            ntraj = 64,
            seed = 0x1234,
            first_passage_method = :log_survival_predictor,
            saveat = tlist,
            save_trajectories = true,
            save_final_states = true,
        )
        for observable_storage in (:dense, :sparse)
            serial = dislou_solve(
                H, psi0, tlist, [1.2 * a];
                common..., ensemblealg = :serial, observable_storage
            )
            threaded = dislou_solve(
                H, psi0, tlist, [1.2 * a];
                common..., ensemblealg = :threads, observable_storage
            )

            same_saved_trajectories(serial, threaded)
            @test serial.njumps_total == threaded.njumps_total
            @test serial.jumps_by_channel == threaded.jumps_by_channel
            @test serial.gauge_diagnostics.switches == threaded.gauge_diagnostics.switches
            @test serial.gauge_diagnostics.jumps_by_gauge ==
                threaded.gauge_diagnostics.jumps_by_gauge
            @test serial.expect ≈ threaded.expect atol = 1.0e-14 rtol = 0
            @test serial.final_density ≈ threaded.final_density atol = 1.0e-14 rtol = 0
            @test serial.eigensystem_backend === threaded.eigensystem_backend === :lapack
        end
    finally
        was_enabled ? DiSLOUTrajectories._enable_cuda_diagonalization!() :
            DiSLOUTrajectories._disable_cuda_diagonalization!()
    end
end

@testset "threaded solves compose with outer threading" begin
    dimension = 3
    lowering = destroy(dimension)
    common = (;
        e_ops = [num(dimension)],
        gauge_set = zeros(ComplexF64, 1, 1),
        ntraj = 8,
        seed = 0x5eed,
        save_final_states = true,
        ensemblealg = :threads,
    )
    solve() = dislou_solve(
        0.2 * num(dimension), fock(dimension, 1), [0.0, 0.1, 0.2],
        [sqrt(0.4) * lowering]; common...
    )

    reference = solve()
    nested = Vector{Any}(undef, 2)
    Threads.@threads :static for index in eachindex(nested)
        nested[index] = solve()
    end
    overlapping = Vector{Any}(undef, 2)
    @sync for index in eachindex(overlapping)
        Threads.@spawn overlapping[index] = solve()
    end

    for solution in (nested..., overlapping...)
        @test solution.njumps_total == reference.njumps_total
        @test solution.expect == reference.expect
        @test solution.final_density == reference.final_density
        @test solution.gauge_diagnostics == reference.gauge_diagnostics
    end
end

@testset "threaded discovery indices compose and stay ordered" begin
    expected = Any[index^2 for index in 1:8]
    nested = Vector{Any}(undef, 2)
    Threads.@threads :static for run in eachindex(nested)
        nested[run] = DiSLOUTrajectories._run_discovery_indices(index -> index^2, 8, :threads)
    end
    overlapping = Vector{Any}(undef, 2)
    @sync for run in eachindex(overlapping)
        Threads.@spawn overlapping[run] =
            DiSLOUTrajectories._run_discovery_indices(index -> index^2, 8, :threads)
    end

    @test all(==(expected), nested)
    @test all(==(expected), overlapping)
end
