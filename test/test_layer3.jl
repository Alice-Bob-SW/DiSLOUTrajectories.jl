isdefined(Main, :DrivenKerrParams) ||
    include(joinpath(@__DIR__, "fixtures", "driven_kerr_model.jl"))

function layer3_jump_fixture()
    H = Matrix(Diagonal(ComplexF64[1, 0, 2]))
    first_jump = zeros(ComplexF64, 3, 3)
    first_jump[3, 1] = sqrt(10.0)
    second_jump = zeros(ComplexF64, 3, 3)
    second_jump[2, 3] = 10.0
    observable = Matrix(Diagonal(ComplexF64[0, 1, 2]))
    return H, ComplexF64[1, 0, 0], [0.0, 0.5, 1.0],
        [first_jump, second_jump], [observable], zeros(ComplexF64, 2, 1)
end

@testset "Layer III fallback uses selected sparse post-jump algebra" begin
    H = zeros(ComplexF64, 2, 2)
    C = [ComplexF64[0 5; 0 0]]
    ψ0 = ComplexF64[0, 1]
    data = DiSLOUTrajectories._validated_gauge_data(zeros(ComplexF64, 1, 1), 1)
    prepared = DiSLOUTrajectories._prepare_layer1(
        H, C, nothing, data, 0.5; observable_storage = :sparse
    )
    layer3 = DiSLOUTrajectories._prepare_layer3(prepared, ψ0, [1], 1.0e-12)
    fill!(only(prepared.system.gauges).cache.M[1], NaN)
    accumulator = DiSLOUTrajectories._TrajectoryAccumulator(0, 2, 1)
    diagnostics = DiSLOUTrajectories._GaugeDiagnostics(1)
    DiSLOUTrajectories._run_layer3_trajectory!(
        accumulator,
        diagnostics,
        prepared,
        layer3,
        ψ0,
        [0.0, 1.0],
        1.0,
        Xoshiro(715),
        DiSLOUTrajectories._WorkBuffers(prepared.system.cache);
        max_jumps = 10,
    )
    @test diagnostics.fallback_jumps == 1
end


function layer3_cross_gauge_fallback_fixture()
    H = zeros(ComplexF64, 2, 2)
    routing = zeros(ComplexF64, 2, 2)
    routing[1, 1] = 2
    routing[2, 1] = 1
    down = zeros(ComplexF64, 2, 2)
    down[1, 2] = 1
    up = zeros(ComplexF64, 2, 2)
    up[2, 1] = 3
    shifts = zeros(ComplexF64, 3, 2)
    shifts[1, 2] = -2
    observable = Matrix(Diagonal(ComplexF64[0, 1]))
    return H, ComplexF64[0, 1], collect(0.0:0.2:2.0),
        [routing, down, up], [observable], shifts
end

function layer3_rng_tail(prepared, layer3, ψ0, tlist, seed, trajectory)
    physical = prepared.system.cache
    ngauges = length(prepared.system.gauges)
    accumulator = DiSLOUTrajectories._TrajectoryAccumulator(
        physical.Ne, length(tlist), physical.Nc
    )
    diagnostics = DiSLOUTrajectories._GaugeDiagnostics(ngauges)
    rng = DiSLOUTrajectories._traj_rng(UInt64(seed), trajectory)
    buffers = DiSLOUTrajectories._WorkBuffers(physical, max(physical.N, ngauges))
    if layer3 === nothing
        DiSLOUTrajectories._run_exact_trajectory!(
            accumulator, diagnostics, prepared,
            ψ0, tlist, last(tlist), rng, buffers;
            first_passage_method = :log_survival
        )
    else
        DiSLOUTrajectories._run_layer3_trajectory!(
            accumulator, diagnostics, prepared,
            layer3, ψ0, tlist, last(tlist), rng, buffers;
            first_passage_method = :log_survival
        )
    end
    return (; tail = ntuple(_ -> rand(rng), 4), accumulator, diagnostics)
end

function layer3_storage_fixture(
        H, ψ0, tlist, c_ops, e_ops, shifts;
        observable_storage, layer3_size, ntraj, seed,
        residual_tolerance = 1.0e-12, hysteresis = 0.5,
        first_passage_method = :log_survival
    )
    data = DiSLOUTrajectories._validated_gauge_data(shifts, length(c_ops))
    prepared = DiSLOUTrajectories._prepare_layer1(
        H, c_ops, e_ops, data, hysteresis; observable_storage
    )
    layer3 = DiSLOUTrajectories._prepare_layer3(
        prepared, ψ0, fill(layer3_size, size(shifts, 2)), residual_tolerance
    )
    accumulator, diagnostics, final_states, records, col_gauge =
        DiSLOUTrajectories._solve_gauges_serial(
        prepared, layer3, ψ0, tlist, last(tlist), ntraj, UInt64(seed);
        first_passage_method, save_final_states = true
    )
    layer3_diagnostics = (;
        accepted_projections = diagnostics.accepted_projections,
        fallback_segments = diagnostics.fallback_segments,
    )
    return (;
        expect = accumulator.mean,
        col_times = [record.col_times for record in records],
        col_which = [record.col_which for record in records],
        col_gauge,
        final_states,
        layer3_diagnostics,
    )
end

@testset "Layer III is residual-gated and exactly falls back" begin
    H, ψ0, tlist, c_ops, e_ops, shifts = layer3_jump_fixture()
    common = (;
        e_ops, gauge_set = shifts, ntraj = 16, seed = 91,
        ensemblealg = :serial, save_final_states = true,
        observable_storage = :dense,
    )
    exact = dislou_solve(H, ψ0, tlist, c_ops; common...)
    gated = dislou_solve(
        H, ψ0, tlist, c_ops; common...,
        layer3 = true, layer3_sizes = 1,
        residual_tolerance = eps(Float64)
    )

    @test keys(gated.layer3_diagnostics) ==
        (:sizes, :accepted_projections, :fallback_segments, :maximum_residual)
    @test gated.layer3_diagnostics.sizes == [1]
    @test gated.layer3_diagnostics.fallback_segments > 0
    @test gated.layer3_diagnostics.maximum_residual > eps(Float64)
    @test gated.expect == exact.expect
    @test gated.expect_sem == exact.expect_sem
    @test gated.col_times == exact.col_times
    @test gated.col_which == exact.col_which
    @test gated.col_gauge == exact.col_gauge
    @test gated.jumps_by_channel == exact.jumps_by_channel
    @test gated.final_states == exact.final_states
    @test gated.gauge_diagnostics == exact.gauge_diagnostics
end

@testset "Layer III accepts invariant local dynamics" begin
    dimension = 4
    lowering = destroy(dimension)
    H = 0.2 * num(dimension)
    ψ0 = fock(dimension, 1)
    tlist = [0.0, 0.5, 1.0]
    c_ops = [sqrt(2.0) * lowering]
    common = (;
        e_ops = [num(dimension)], gauge_set = zeros(ComplexF64, 1, 1),
        ntraj = 16, seed = 37, ensemblealg = :serial,
    )
    exact = dislou_solve(H, ψ0, tlist, c_ops; common...)
    reduced = dislou_solve(
        H, ψ0, tlist, c_ops; common...,
        layer3 = true, layer3_sizes = 2, residual_tolerance = 1.0e-12
    )

    @test reduced.layer3_diagnostics.sizes == [2]
    @test reduced.layer3_diagnostics.accepted_projections > 0
    @test reduced.layer3_diagnostics.fallback_segments == 0
    @test reduced.layer3_diagnostics.maximum_residual <= 1.0e-12
    @test reduced.col_which == exact.col_which
    @test all(
        all(isapprox.(left, right; atol = 1.0e-12, rtol = 0))
            for (left, right) in zip(reduced.col_times, exact.col_times)
    )
    @test reduced.expect ≈ exact.expect atol = 1.0e-12 rtol = 0
end

@testset "Layer III gates nonzero physical jump images" begin
    H, ψ0, tlist, c_ops, e_ops, shifts = layer3_jump_fixture()
    common = (;
        e_ops, gauge_set = shifts, ntraj = 16, seed = 91,
        ensemblealg = :serial, save_final_states = true,
    )
    exact = dislou_solve(H, ψ0, tlist, c_ops; common...)
    gated = dislou_solve(
        H, ψ0, tlist, c_ops; common...,
        layer3 = true, layer3_sizes = 2, residual_tolerance = 1.0e-12
    )
    dense_layer3 = layer3_storage_fixture(
        H, ψ0, tlist, c_ops, e_ops, shifts;
        observable_storage = :dense, layer3_size = 2, ntraj = 16, seed = 91
    )
    sparse_layer3 = layer3_storage_fixture(
        H, ψ0, tlist, c_ops, e_ops, shifts;
        observable_storage = :sparse, layer3_size = 2, ntraj = 16, seed = 91
    )

    @test gated.layer3_diagnostics.accepted_projections >= gated.ntraj
    @test gated.layer3_diagnostics.fallback_segments > 0
    @test gated.layer3_diagnostics.maximum_residual > 0.99
    @test gated.jumps_by_channel[1] > 0
    @test gated.col_which == exact.col_which
    @test gated.col_gauge == exact.col_gauge
    @test all(
        all(isapprox.(left, right; atol = 1.0e-11, rtol = 0))
            for (left, right) in zip(gated.col_times, exact.col_times)
    )
    @test gated.expect ≈ exact.expect atol = 1.0e-12 rtol = 0
    @test gated.final_states ≈ exact.final_states atol = 1.0e-12 rtol = 0
    @test sparse_layer3.expect ≈ dense_layer3.expect atol = 2.0e-12 rtol = 0
    @test all(
        all(isapprox.(left, right; atol = 1.0e-11, rtol = 0))
            for (left, right) in zip(
                sparse_layer3.col_times,
                dense_layer3.col_times
            )
    )
    @test sparse_layer3.col_which == dense_layer3.col_which
    @test sparse_layer3.col_gauge == dense_layer3.col_gauge
    @test sparse_layer3.final_states ≈ dense_layer3.final_states atol = 1.0e-12 rtol = 0
    @test sparse_layer3.layer3_diagnostics.accepted_projections ==
        dense_layer3.layer3_diagnostics.accepted_projections
    @test sparse_layer3.layer3_diagnostics.accepted_projections > 0
    @test sparse_layer3.layer3_diagnostics.fallback_segments ==
        dense_layer3.layer3_diagnostics.fallback_segments
end


@testset "Layer III completes degeneracies and uses thin QR projection" begin
    cache = DiSLOUTrajectories._DiagonalCache(
        QuantumObject(Matrix(Diagonal(ComplexF64[0, 0]))), QuantumObject[]
    )
    space = DiSLOUTrajectories._build_local_eigenspace(cache, 1)
    @test space isa DiSLOUTrajectories._LocalEigenspace
    @test length(space.reduced.I) == 2
    @test size(space.Q) == (2, 2)
    @test size(space.R) == (2, 2)

    ψ = ComplexF64[0.6 + 0.2im, -0.3 + 0.7im]
    @test DiSLOUTrajectories._relative_residual!(
        zeros(ComplexF64, 2), zeros(ComplexF64, 2), space, ψ
    ) < 1.0e-14

    solution = dislou_solve(
        zeros(ComplexF64, 2, 2), ComplexF64[1, 0], [0.0],
        Matrix{ComplexF64}[]; gauge_set = zeros(ComplexF64, 0, 1),
        layer3 = true, layer3_sizes = 1, residual_tolerance = 1.0e-12,
        ntraj = 1, seed = 2, ensemblealg = :serial
    )
    @test solution.layer3_diagnostics.sizes == [2]
end

@testset "Layer III routes accepted jumps across gauges" begin
    model = driven_kerr_model(DrivenKerrParams(N = 12))
    shifts = reshape(
        -sqrt(model.κ) .* ComplexF64[
            model.αlow, model.αhigh, model.αmid,
        ], 1, :
    )
    tlist = collect(0.0:0.25:4.0)
    common = (;
        e_ops = [model.nop, model.xop], gauge_set = shifts,
        ntraj = 5, seed = 71, hysteresis = 0.7,
        ensemblealg = :serial, save_final_states = true,
    )
    exact = dislou_solve(model.H, model.ψ0, tlist, model.c_ops; common...)
    reduced = dislou_solve(
        model.H, model.ψ0, tlist, model.c_ops; common...,
        layer3 = true, layer3_sizes = 12, residual_tolerance = 1.0e-10
    )

    @test exact.gauge_diagnostics.switches > 0
    @test reduced.gauge_diagnostics.switches == exact.gauge_diagnostics.switches
    @test reduced.layer3_diagnostics.accepted_projections > reduced.ntraj
    @test reduced.layer3_diagnostics.fallback_segments == 0
    @test reduced.col_which == exact.col_which
    @test reduced.col_gauge == exact.col_gauge
    @test all(
        all(isapprox.(left, right; atol = 1.0e-11, rtol = 0))
            for (left, right) in zip(reduced.col_times, exact.col_times)
    )
    @test reduced.expect ≈ exact.expect atol = 1.0e-12 rtol = 0
    @test reduced.final_states ≈ exact.final_states atol = 1.0e-11 rtol = 0
end

# Mutations caught:
# - solving a rejected jump image in the source rather than destination gauge;
# - drawing again when switching to exact fallback.
@testset "Layer III destination-gauge fallback preserves exact streams" begin
    H, ψ0, tlist, c_ops, e_ops, shifts = layer3_cross_gauge_fallback_fixture()
    common = (;
        e_ops, gauge_set = shifts, ntraj = 1, seed = 41,
        hysteresis = 0.9, ensemblealg = :serial,
        first_passage_method = :log_survival_predictor, saveat = tlist,
        save_trajectories = true, save_final_states = true,
    )
    exact = dislou_solve(H, ψ0, tlist, c_ops; common...)
    gated = dislou_solve(
        H, ψ0, tlist, c_ops; common...,
        layer3 = true, layer3_sizes = 1, residual_tolerance = 1.0e-12
    )
    dense_layer3 = layer3_storage_fixture(
        H, ψ0, tlist, c_ops, e_ops, shifts;
        observable_storage = :dense, layer3_size = 1, ntraj = 1, seed = 41,
        hysteresis = 0.9, first_passage_method = :log_survival_predictor
    )
    sparse_layer3 = layer3_storage_fixture(
        H, ψ0, tlist, c_ops, e_ops, shifts;
        observable_storage = :sparse, layer3_size = 1, ntraj = 1, seed = 41,
        hysteresis = 0.9, first_passage_method = :log_survival_predictor
    )

    @test gated.layer3_diagnostics.accepted_projections > 0
    @test gated.layer3_diagnostics.fallback_segments > 0
    @test gated.layer3_diagnostics.maximum_residual == 1.0
    @test exact.gauge_diagnostics.switches > 0
    @test gated.gauge_diagnostics == exact.gauge_diagnostics
    @test gated.col_times == exact.col_times
    @test gated.col_which == exact.col_which
    @test gated.col_gauge == exact.col_gauge
    @test gated.njumps_total == exact.njumps_total
    @test gated.jumps_by_channel == exact.jumps_by_channel
    @test gated.expect == exact.expect
    @test gated.expect_std == exact.expect_std
    @test isequal(gated.expect_sem, exact.expect_sem)
    @test gated.states == exact.states
    @test gated.trajectory_states == exact.trajectory_states
    @test gated.trajectory_expect == exact.trajectory_expect
    @test gated.final_states == exact.final_states
    @test sparse_layer3.expect ≈ dense_layer3.expect atol = 2.0e-12 rtol = 0
    @test sparse_layer3.col_times == dense_layer3.col_times
    @test sparse_layer3.col_which == dense_layer3.col_which
    @test sparse_layer3.col_gauge == dense_layer3.col_gauge
    @test sparse_layer3.final_states == dense_layer3.final_states
    @test sparse_layer3.layer3_diagnostics.accepted_projections ==
        dense_layer3.layer3_diagnostics.accepted_projections
    @test sparse_layer3.layer3_diagnostics.fallback_segments ==
        dense_layer3.layer3_diagnostics.fallback_segments
    @test sparse_layer3.layer3_diagnostics.fallback_segments > 0

    data = DiSLOUTrajectories._validated_gauge_data(shifts, length(c_ops))
    prepared = DiSLOUTrajectories._prepare_layer1(H, c_ops, e_ops, data, 0.9)
    layer3 = DiSLOUTrajectories._prepare_layer3(
        prepared, ψ0, [1, 1], 1.0e-12
    )
    exact_tail = layer3_rng_tail(prepared, nothing, ψ0, tlist, 41, 1)
    gated_tail = layer3_rng_tail(prepared, layer3, ψ0, tlist, 41, 1)
    @test gated_tail.diagnostics.switches > 0
    @test gated_tail.diagnostics.fallback_segments > 0
    @test gated_tail.diagnostics.fallback_jumps ==
        gated_tail.diagnostics.fallback_segments == 1
    @test 0 < gated_tail.diagnostics.fallback_residence_time <= last(tlist)
    @test gated_tail.accumulator.jumps_by_channel ==
        exact_tail.accumulator.jumps_by_channel
    @test gated_tail.tail == exact_tail.tail
end

# The public keyword battery in test_edge_cases.jl owns the invalid
# residual_tolerance and layer3_sizes values; what is specific here is a size
# list whose length disagrees with the gauge count, and the accepted case.
@testset "Layer III sizes must match the gauge count" begin
    H, ψ0, _, c_ops, _, _ = layer3_jump_fixture()
    base = (;
        gauge_set = zeros(ComplexF64, 2, 2), ntraj = 1, seed = 9,
        ensemblealg = :serial, layer3 = true,
    )

    @test_throws ArgumentError dislou_solve(
        H, ψ0, [0.0], c_ops; base..., layer3_sizes = [1]
    )
    two_gauges = dislou_solve(
        H, ψ0, [0.0], c_ops;
        base..., layer3_sizes = 2, residual_tolerance = 1.0e-12
    )
    @test two_gauges.layer3_diagnostics.sizes == [2, 2]
end

@testset "Layer III slow-mode ordering is deterministic and cluster complete" begin
    cache = DiSLOUTrajectories._DiagonalCache(
        QuantumObject(Matrix(Diagonal(ComplexF64[0, 1, 2, 3]))), QuantumObject[]
    )
    cache.Γ .= Float64[1, 1, 2, 3]
    cache.Λ .= ComplexF64[2 - 0.5im, 1 - 0.5im, 3 - 1im, 4 - 1.5im]
    cache.deg_clusters = [[1], [2], [3], [4]]
    @test DiSLOUTrajectories._slow_mode_indices(cache, 1) == [2]

    cache.Λ .= ComplexF64[1 - 0.5im, 1 - 0.5im, 3 - 1im, 4 - 1.5im]
    @test DiSLOUTrajectories._slow_mode_indices(cache, 1) == [1]

    cache.deg_clusters = [[1, 2], [3], [4]]
    @test DiSLOUTrajectories._slow_mode_indices(cache, 1) == [1, 2]
end

@testset "Layer III serial and threaded trajectory streams agree" begin
    dimension = 4
    lowering = destroy(dimension)
    common = (;
        e_ops = [num(dimension)], gauge_set = zeros(ComplexF64, 1, 1),
        ntraj = 32, seed = 37, layer3 = true, layer3_sizes = 2,
        residual_tolerance = 1.0e-12, save_final_states = true,
    )
    serial = dislou_solve(
        0.2 * num(dimension), fock(dimension, 1),
        [0.0, 0.5, 1.0], [sqrt(2.0) * lowering];
        common..., ensemblealg = :serial
    )
    threaded = dislou_solve(
        0.2 * num(dimension), fock(dimension, 1),
        [0.0, 0.5, 1.0], [sqrt(2.0) * lowering];
        common..., ensemblealg = :threads
    )

    @test threaded.col_times == serial.col_times
    @test threaded.col_which == serial.col_which
    @test threaded.col_gauge == serial.col_gauge
    @test threaded.jumps_by_channel == serial.jumps_by_channel
    @test threaded.gauge_diagnostics == serial.gauge_diagnostics
    @test threaded.layer3_diagnostics == serial.layer3_diagnostics
    @test threaded.expect ≈ serial.expect atol = 1.0e-14 rtol = 0
    @test threaded.expect_sem ≈ serial.expect_sem atol = 1.0e-14 rtol = 0
    @test threaded.final_states == serial.final_states
end
