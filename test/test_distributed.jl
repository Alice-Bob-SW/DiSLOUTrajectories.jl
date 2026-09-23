using Distributed

# Mutations caught: omitting the public solver check or testing `nworkers() == 0`,
# even though process 1 is reported as a worker in a worker-free Julia session.
# This must stay above the `addprocs` below: it asserts the worker-free state,
# which the parent process already provides without spawning a probe.
function rejects_worker_free_boundary(f)
    caught = try
        f()
        nothing
    catch error
        error
    end
    return caught isa ArgumentError &&
        occursin("worker processes", sprint(showerror, caught))
end

@testset "worker-free boundaries reject distributed execution" begin
    @test nprocs() == 1
    a = destroy(2)
    @test rejects_worker_free_boundary() do
        dislou_solve(
            num(2), fock(2, 0), [0.0], QuantumObject[];
            gauge_set = zeros(ComplexF64, 0, 1), ntraj = 1,
            ensemblealg = :distributed
        )
    end
    @test rejects_worker_free_boundary() do
        discover_gauges(
            num(2), [a];
            method = :trajectories, mode_ops = [a], mode_dims = [2],
            discovery_time = 0.1, seed_radii = [0.1], cluster_scales = [1.0],
            step = 0.1, nseeds = 1, terminal_window = 0.0,
            dbscan_radius = 1.0, min_neighbors = 1, min_weight = 0.0,
            ensemblealg = :distributed
        )
    end
end

@testset "distributed execution preserves indexed trajectory streams" begin
    project = Base.active_project()
    workers_added = addprocs(2; exeflags = `--project=$project --startup-file=no`)
    was_enabled = backend_info().cuda_enabled
    try
        DiSLOUTrajectories._disable_cuda_diagonalization!()
        @everywhere using DiSLOUTrajectories, QuantumToolbox
        d = 5
        a = destroy(d)
        H = 0.2 * (a + a')
        psi0 = fock(d, 3)
        tlist = collect(3.0:0.2:4.2)
        c_ops = [1.2 * a]
        common = (;
            e_ops = [num(d)],
            gauge_set = ComplexF64[0 0.8],
            ntraj = 60,
            seed = 0x1234,
            first_passage_method = :log_survival_predictor,
            saveat = tlist,
            save_trajectories = true,
            save_final_states = true,
        )
        for observable_storage in (:dense, :sparse)
            serial = dislou_solve(
                H, psi0, tlist, c_ops;
                common..., ensemblealg = :serial, observable_storage
            )
            distributed = dislou_solve(
                H, psi0, tlist, c_ops;
                common..., ensemblealg = :distributed, observable_storage
            )

            @test length.(serial.col_times) == length.(distributed.col_times)
            @test all(
                isapprox(
                    serial.col_times[i], distributed.col_times[i];
                    atol = 1.0e-14, rtol = 0
                )
                    for i in eachindex(serial.col_times)
            )
            @test serial.col_which == distributed.col_which
            @test serial.col_gauge == distributed.col_gauge
            @test serial.trajectory_expect ≈ distributed.trajectory_expect atol = 1.0e-13 rtol = 0
            @test serial.njumps_total == distributed.njumps_total
            @test serial.jumps_by_channel == distributed.jumps_by_channel
            @test serial.gauge_diagnostics.switches ==
                distributed.gauge_diagnostics.switches
            @test serial.gauge_diagnostics.jumps_by_gauge ==
                distributed.gauge_diagnostics.jumps_by_gauge
            @test serial.expect ≈ distributed.expect atol = 1.0e-14 rtol = 0
            @test serial.final_density ≈ distributed.final_density atol = 1.0e-14 rtol = 0
            @test all(3.0 <= time <= 4.2 for time in Iterators.flatten(distributed.col_times))
            @test all(
                isapprox(
                    serial.trajectory_states[i, k].data * serial.trajectory_states[i, k].data',
                    distributed.trajectory_states[i, k].data * distributed.trajectory_states[i, k].data';
                    atol = 1.0e-13,
                ) for i in axes(serial.trajectory_states, 1), k in axes(serial.trajectory_states, 2)
            )
            @test serial.eigensystem_backend === distributed.eigensystem_backend === :lapack
        end

        # Mutations caught: removing only the public guard, dropping Layer III
        # worker preparation, or silently running exact trajectories on workers.
        @testset "distributed Layer III runs worker-local spaces" begin
            dimension = 4
            lowering = destroy(dimension)
            initial_state = fock(dimension, 1)
            layer3_times = [0.0, 0.5, 1.0]
            layer3_ops = [sqrt(2.0) * lowering]
            layer3_common = (;
                e_ops = [num(dimension)],
                gauge_set = zeros(ComplexF64, 1, 1),
                ntraj = 4,
                seed = 37,
                layer3 = true,
                layer3_sizes = 2,
                residual_tolerance = 1.0e-12,
            )
            layer3_serial = dislou_solve(
                0.2 * num(dimension), initial_state, layer3_times,
                layer3_ops; layer3_common..., ensemblealg = :serial
            )
            layer3_distributed = dislou_solve(
                0.2 * num(dimension), initial_state, layer3_times,
                layer3_ops; layer3_common..., ensemblealg = :distributed
            )

            @test layer3_distributed.layer3_diagnostics.sizes == [2]
            @test layer3_distributed.layer3_diagnostics.accepted_projections > 0
            @test layer3_distributed.layer3_diagnostics.fallback_segments == 0
            @test layer3_distributed.layer3_diagnostics.maximum_residual <= 1.0e-12
            @test layer3_distributed.col_times == layer3_serial.col_times
            @test layer3_distributed.col_which == layer3_serial.col_which
            @test layer3_distributed.col_gauge == layer3_serial.col_gauge
            @test layer3_distributed.jumps_by_channel ==
                layer3_serial.jumps_by_channel
            @test layer3_distributed.layer3_diagnostics ==
                layer3_serial.layer3_diagnostics
            @test layer3_distributed.expect ≈ layer3_serial.expect atol = 1.0e-14 rtol = 0
            @test layer3_distributed.expect_sem ≈
                layer3_serial.expect_sem atol = 1.0e-14 rtol = 0
        end

        active_workers = workers()
        worker_states = Dict(
            worker => remotecall_fetch(
                () -> DiSLOUTrajectories.backend_info().cuda_enabled, worker
            )
                for worker in active_workers
        )
        try
            for worker in active_workers
                fetch(
                    Distributed.remotecall_eval(
                        Main, worker, quote
                            const _DISLOU_WORKER_CUDA_CALLS = Ref(0)
                            const _DISLOU_WORKER_CUDA_FAIL_AT = Ref(typemax(Int))
                            const _DISLOU_WORKER_PERTURB_PAIR = Ref(false)
                            function DiSLOUTrajectories._cuda_prepare_diagonal_data(
                                    H::Matrix{ComplexF64},
                                    C::Vector{Matrix{ComplexF64}},
                                    Z::Vector{Matrix{ComplexF64}}
                                )
                                _DISLOU_WORKER_CUDA_CALLS[] += 1
                                _DISLOU_WORKER_CUDA_CALLS[] ==
                                    _DISLOU_WORKER_CUDA_FAIL_AT[] &&
                                    error("mock worker CUDA preparation failure")
                                prepared = DiSLOUTrajectories._cpu_prepare_diagonal_data(H, C, Z)
                                values = copy(prepared.Λ)
                                if _DISLOU_WORKER_PERTURB_PAIR[] && length(values) == 2
                                    values[2] = values[1] + 1.0e-12
                                end
                                return merge(
                                    prepared, (;
                                        Λ = values,
                                        Γ = -2 .* imag.(values),
                                        backend = :cuda,
                                    )
                                )
                            end
                        end
                    )
                )
            end

            # Mutation caught: omitting the parent's Layer III preparation from
            # distributed preparation backend report.
            function DiSLOUTrajectories._cuda_prepare_diagonal_data(
                    H::Matrix{ComplexF64},
                    C::Vector{Matrix{ComplexF64}},
                    Z::Vector{Matrix{ComplexF64}}
                )
                return merge(
                    DiSLOUTrajectories._cpu_prepare_diagonal_data(H, C, Z),
                    (; backend = :cuda)
                )
            end
            try
                for worker in active_workers
                    remotecall_fetch(DiSLOUTrajectories._disable_cuda_diagonalization!, worker)
                end
                DiSLOUTrajectories._enable_cuda_diagonalization!()
                parent_cuda = dislou_solve(
                    Matrix(Diagonal(ComplexF64[0, 1])), ComplexF64[1, 0],
                    [0.0], Matrix{ComplexF64}[];
                    gauge_set = zeros(ComplexF64, 0, 1),
                    ntraj = length(active_workers), seed = 7,
                    ensemblealg = :distributed, layer3 = true,
                    layer3_sizes = 1, residual_tolerance = 1.0e-12
                )
                @test parent_cuda.eigensystem_backend === :mixed
            finally
                Base.delete_method(
                    which(
                        DiSLOUTrajectories._cuda_prepare_diagonal_data,
                        Tuple{
                            Matrix{ComplexF64}, Vector{Matrix{ComplexF64}},
                            Vector{Matrix{ComplexF64}},
                        }
                    )
                )
                DiSLOUTrajectories._disable_cuda_diagonalization!()
            end

            # Mutation caught: reporting parent-realized Layer III sizes when a
            # heterogeneous worker completes a different degenerate cluster.
            H_near = Matrix(Diagonal(ComplexF64[1, 1 + 2.0e-8]))
            for (index, worker) in enumerate(active_workers)
                remotecall_fetch(worker, index == 1) do perturb
                    Main._DISLOU_WORKER_CUDA_CALLS[] = 0
                    Main._DISLOU_WORKER_CUDA_FAIL_AT[] = typemax(Int)
                    Main._DISLOU_WORKER_PERTURB_PAIR[] = perturb
                    perturb ? DiSLOUTrajectories._enable_cuda_diagonalization!() :
                        DiSLOUTrajectories._disable_cuda_diagonalization!()
                end
            end
            size_error = try
                dislou_solve(
                    H_near, ComplexF64[1, 0], [0.0],
                    Matrix{ComplexF64}[];
                    gauge_set = zeros(ComplexF64, 0, 1),
                    ntraj = length(active_workers), seed = 7,
                    ensemblealg = :distributed, layer3 = true,
                    layer3_sizes = 1, residual_tolerance = 1.0e-12
                )
                nothing
            catch error
                error
            end
            @test size_error isa ArgumentError
            @test size_error isa Exception && occursin(
                "distributed Layer III realized sizes differ",
                sprint(showerror, size_error)
            )

            for (index, worker) in enumerate(active_workers)
                remotecall_fetch(worker, index == 1) do fail_first
                    Main._DISLOU_WORKER_CUDA_CALLS[] = 0
                    Main._DISLOU_WORKER_CUDA_FAIL_AT[] =
                        fail_first ? 1 : typemax(Int)
                    Main._DISLOU_WORKER_PERTURB_PAIR[] = false
                    DiSLOUTrajectories._enable_cuda_diagonalization!()
                end
            end
            mixed = dislou_solve(
                zeros(ComplexF64, 2, 2), ComplexF64[1, 0],
                [0.0], Matrix{ComplexF64}[];
                gauge_set = zeros(ComplexF64, 0, 1),
                ntraj = length(active_workers), seed = 7,
                ensemblealg = :distributed
            )
            @test mixed.eigensystem_backend === :mixed
            @test !remotecall_fetch(
                () -> DiSLOUTrajectories.backend_info().cuda_enabled,
                first(active_workers),
            )
        finally
            for worker in active_workers
                fetch(
                    Distributed.remotecall_eval(
                        Main, worker, quote
                            Base.delete_method(
                                which(
                                    DiSLOUTrajectories._cuda_prepare_diagonal_data,
                                    Tuple{
                                        Matrix{ComplexF64}, Vector{Matrix{ComplexF64}},
                                        Vector{Matrix{ComplexF64}},
                                    }
                                )
                            )
                        end
                    )
                )
                remotecall_fetch(worker, worker_states[worker]) do enabled
                    enabled ? DiSLOUTrajectories._enable_cuda_diagonalization!() :
                        DiSLOUTrajectories._disable_cuda_diagonalization!()
                end
            end
        end

        @testset "distributed trajectory discovery needs no extension on workers" begin
            # testsetup.jl loads Clustering before addprocs, so the workers never load it.
            @test !any(
                remotecall_fetch(
                    () -> Base.get_extension(DiSLOUTrajectories, :DiSLOUTrajectoriesClusteringExt) !== nothing,
                    worker
                ) for worker in workers_added
            )
            a = destroy(4)
            kw = (;
                method = :trajectories, mode_ops = [a], mode_dims = [4],
                discovery_time = 0.2, seed_radii = [1.0], cluster_scales = [1.0],
                step = 0.05, nseeds = 6, min_neighbors = 1, min_weight = 0.0, seed = 3,
            )
            distributed = discover_gauges(0.1 * num(4), [a]; kw..., ensemblealg = :distributed)
            serial = discover_gauges(0.1 * num(4), [a]; kw..., ensemblealg = :serial)
            @test distributed.shifts == serial.shifts
            @test distributed.diagnostics.terminal_means == serial.diagnostics.terminal_means
        end
    finally
        rmprocs(workers_added)
        was_enabled ? DiSLOUTrajectories._enable_cuda_diagonalization!() :
            DiSLOUTrajectories._disable_cuda_diagonalization!()
    end
end
