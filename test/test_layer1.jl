using Test
using DiSLOUTrajectories
using LinearAlgebra

@testset "Layer I routes on total shifted activity with strict hysteresis" begin
    H = zeros(ComplexF64, 2, 2)
    C = [ComplexF64[0 0; 0 2], Matrix{ComplexF64}(I, 2, 2)]
    data = DiSLOUTrajectories._validated_gauge_data(ComplexF64[0 -2; 0 3], 2)
    prepared = DiSLOUTrajectories._prepare_layer1(H, C, nothing, data, 1.0)
    ψ_left = ComplexF64[1, 0]
    ψ_right = ComplexF64[0, 1]
    wb = DiSLOUTrajectories._WorkBuffers(prepared.system.cache, 2)

    @test DiSLOUTrajectories._initial_gauge!(wb, prepared, ψ_left) == 1
    @test DiSLOUTrajectories._initial_gauge!(wb, prepared, ψ_right) == 1
    nshift = length(prepared.system.gauges[1].meanV)
    means = view(wb.moments, 1:nshift)
    second = view(wb.w, 1:prepared.system.cache.Nc)
    @test DiSLOUTrajectories._activity_from_moments(
        prepared.system.gauges[1], means, second, nshift
    ) ≈ 5.0
    @test DiSLOUTrajectories._activity_from_moments(
        prepared.system.gauges[2], means, second, nshift
    ) ≈ 16.0
    DiSLOUTrajectories._solve_coordinates!(
        wb.c, prepared.system.gauges[1].cache, ψ_right
    )
    @test DiSLOUTrajectories._postjump_gauge!(wb, prepared, 1) == 1

    # The buffered post-jump scan scores every gauge from the current-gauge
    # coordinates already sitting in wb.c and must select the expected gauge.
    expected_selected = similar(wb.cplus)
    expected_current = similar(wb.cplus)
    DiSLOUTrajectories._solve_coordinates!(
        expected_selected, prepared.system.gauges[1].cache, ψ_right
    )
    DiSLOUTrajectories._normalize_coordinates!(
        expected_selected, similar(expected_selected),
        prepared.system.gauges[1].cache.G, prepared.system.gauges[1].cache.Gnorm
    )
    DiSLOUTrajectories._solve_coordinates!(
        expected_current, prepared.system.gauges[2].cache, ψ_right
    )
    copyto!(wb.c, expected_current)
    @test DiSLOUTrajectories._postjump_gauge!(wb, prepared, 2) == 1
    # On a switch it leaves wb.cplus holding normalised coordinates in the new gauge.
    @test DiSLOUTrajectories._postjump_gauge_coordinates!(wb, prepared, 2, ψ_right) == 1
    @test wb.cplus ≈ expected_selected
    switched_state = prepared.system.gauges[1].cache.V * wb.cplus
    @test switched_state / norm(switched_state) ≈ ψ_right / norm(ψ_right)
    @test !isapprox(wb.cplus, expected_current; atol = 1.0e-12, rtol = 1.0e-12)
end

@testset "operator storage selects sparse post-jump algebra" begin
    H = zeros(ComplexF64, 4, 4)
    C = [
        ComplexF64[0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0],
        Matrix{ComplexF64}(Diagonal(0:3)),
    ]
    shifts = ComplexF64[0.2 -0.3; 0.1im -0.2im]
    data = DiSLOUTrajectories._validated_gauge_data(shifts, length(C))
    dense = DiSLOUTrajectories._prepare_layer1(
        H, C, nothing, data, 0.5; observable_storage = :dense
    )
    sparse_prepared = DiSLOUTrajectories._prepare_layer1(
        H, C, nothing, data, 0.5; observable_storage = :sparse
    )

    @test isempty(dense.system.C_sparse)
    @test all(isempty(gauge.C_sparse) for gauge in dense.system.gauges)
    @test sparse_prepared.system.C_sparse == sparse.(C)
    for (index, gauge) in enumerate(sparse_prepared.system.gauges)
        @test gauge.C_sparse == [
            sparse(C[channel] + shifts[channel, index] * I)
                for channel in eachindex(C)
        ]
    end
end

@testset "sparse post-jump helper matches dense routing" begin
    H, c_ops = random_system(N = 6, seed = 711)
    C = Matrix{ComplexF64}[Matrix(op.data) for op in c_ops]
    shifts = ComplexF64[0.2 -0.3; 0.1im -0.2im]
    data = DiSLOUTrajectories._validated_gauge_data(shifts, length(C))
    dense = DiSLOUTrajectories._prepare_layer1(
        H, C, nothing, data, 0.5; observable_storage = :dense
    )
    sparse_prepared = DiSLOUTrajectories._prepare_layer1(
        H, C, nothing, data, 0.5; observable_storage = :sparse
    )
    gauge = 1
    cache = dense.system.gauges[gauge].cache
    ψ = normalize!(randn(Xoshiro(712), ComplexF64, cache.N))
    Dc = cache.Vfac \ ψ
    dense_wb = DiSLOUTrajectories._WorkBuffers(dense.system.cache, cache.N)
    sparse_wb = DiSLOUTrajectories._WorkBuffers(sparse_prepared.system.cache, cache.N)
    dense_rng = Xoshiro(713)
    sparse_rng = Xoshiro(713)

    dense_channel = DiSLOUTrajectories._apply_exact_jump_coordinates!(
        dense_wb.c, dense_wb, cache, Dc, dense_rng
    )
    dense_next = DiSLOUTrajectories._postjump_gauge_coordinates!(
        dense_wb, dense, gauge, dense_wb.ψ
    )
    dense_means = copy(view(dense_wb.moments, 1:cache.Nc))
    dense_second = copy(view(dense_wb.w, 1:cache.Nc))

    for local_gauge in sparse_prepared.system.gauges
        foreach(matrix -> fill!(matrix, NaN), local_gauge.cache.M)
        foreach(matrix -> fill!(matrix, NaN), local_gauge.meanV)
        foreach(matrix -> fill!(matrix, NaN), local_gauge.secondV)
    end
    sparse_channel, sparse_next = DiSLOUTrajectories._apply_exact_jump_and_route!(
        sparse_wb, sparse_prepared, gauge, Dc, sparse_rng
    )

    @test sparse_channel == dense_channel
    @test sparse_next == dense_next
    @test sparse_wb.ψ ≈ dense_wb.ψ atol = 1.0e-12 rtol = 1.0e-12
    @test sparse_wb.c ≈ dense_wb.c atol = 1.0e-11 rtol = 1.0e-11
    @test view(sparse_wb.moments, 1:cache.Nc) ≈ dense_means atol = 1.0e-11 rtol = 1.0e-11
    @test view(sparse_wb.w, 1:cache.Nc) ≈ dense_second atol = 1.0e-11 rtol = 1.0e-11
    @test rand(sparse_rng) == rand(dense_rng)
    dense_next == gauge ||
        @test sparse_wb.cplus ≈ dense_wb.cplus atol = 1.0e-11 rtol = 1.0e-11

    DiSLOUTrajectories._apply_exact_jump_and_route!(
        sparse_wb, sparse_prepared, gauge, Dc, sparse_rng
    )
    route_without_result! = () -> begin
        DiSLOUTrajectories._apply_exact_jump_and_route!(
            sparse_wb, sparse_prepared, gauge, Dc, sparse_rng
        )
        return nothing
    end
    route_without_result!()
    @test (@allocated route_without_result!()) == 0
end

@testset "exact trajectories use selected sparse post-jump algebra" begin
    H = zeros(ComplexF64, 2, 2)
    C = [ComplexF64[0 5; 0 0]]
    data = DiSLOUTrajectories._validated_gauge_data(zeros(ComplexF64, 1, 1), 1)
    prepared = DiSLOUTrajectories._prepare_layer1(
        H, C, nothing, data, 0.5; observable_storage = :sparse
    )
    fill!(only(prepared.system.gauges).cache.M[1], NaN)
    accumulator = DiSLOUTrajectories._TrajectoryAccumulator(0, 2, 1)
    diagnostics = DiSLOUTrajectories._GaugeDiagnostics(1)
    DiSLOUTrajectories._run_exact_trajectory!(
        accumulator,
        diagnostics,
        prepared,
        ComplexF64[0, 1],
        [0.0, 1.0],
        1.0,
        Xoshiro(714),
        DiSLOUTrajectories._WorkBuffers(prepared.system.cache);
        max_jumps = 10,
    )
    @test accumulator.njumps_total == 1
end

@testset "single-gauge post-jump routing is a no-op" begin
    H = zeros(ComplexF64, 2, 2)
    C = [ComplexF64[0 1; 0 0]]
    data = DiSLOUTrajectories._validated_gauge_data(zeros(ComplexF64, 1, 1), 1)
    prepared = DiSLOUTrajectories._prepare_layer1(H, C, nothing, data, 0.5)
    wb = DiSLOUTrajectories._WorkBuffers(prepared.system.cache)
    fill!(wb.cplus, 2 + 3im)
    fill!(wb.Gc, 4 + 5im)
    fill!(wb.moments, 6 + 7im)
    fill!(wb.w, 8.0)
    ψ = ComplexF64[1, 0]

    @test DiSLOUTrajectories._postjump_gauge!(wb, prepared, 1) == 1
    @test wb.cplus == fill(2 + 3im, 2)
    @test wb.Gc == fill(4 + 5im, 2)
    @test wb.moments == fill(6 + 7im, 1)
    @test wb.w == fill(8.0, length(wb.w))
    @test_throws BoundsError DiSLOUTrajectories._postjump_gauge!(wb, prepared, 0)
end

@testset "buffered post-jump scan ignores Layer III routed-activity tail slots" begin
    # Layer III writes per-gauge routed activities into view(wb.w, 1:ngauges).
    # When ngauges > Nc the slots beyond Nc are never overwritten by the scan's
    # second moments, so they must not be summed in as phantom ⟨C_μ†C_μ⟩_ψ terms:
    # they shift every gauge's activity by the same constant, which leaves the
    # argmin intact but silently corrupts the hysteresis ratio test.
    H = zeros(ComplexF64, 2, 2)
    C = [ComplexF64[0 0; 0 2]]
    data = DiSLOUTrajectories._validated_gauge_data(ComplexF64[0 -2], 1)   # 2 gauges, Nc = 1
    prepared = DiSLOUTrajectories._prepare_layer1(H, C, nothing, data, 0.5)
    ngauges = length(prepared.system.gauges)
    cache = prepared.system.cache
    @test ngauges > cache.Nc
    ψ = ComplexF64[0.5, sqrt(0.75)]                            # activities 3.0 and 1.0

    for poison in (0.0, 1.0, 100.0, 1.0e12)
        wb = DiSLOUTrajectories._WorkBuffers(cache, ngauges)
        DiSLOUTrajectories._solve_coordinates!(wb.c, prepared.system.gauges[1].cache, ψ)
        fill!(wb.w, 0.0)
        wb.w[cache.Nc + 1] = poison
        # A poisoned tail slot adds `poison` to both activities; the switch test
        # 1 + p < 0.5 * (3 + p) fails for any p >= 1 if the tail leaks in.
        @test DiSLOUTrajectories._postjump_gauge!(wb, prepared, 1) == 2
        @test wb.w[cache.Nc + 1] == poison
    end
end

@testset "Layer I uses strict hysteresis below one" begin
    H = zeros(ComplexF64, 2, 2)
    C = [ComplexF64[0 0; 0 2]]
    data = DiSLOUTrajectories._validated_gauge_data(ComplexF64[0 -2], 1)
    prepared = DiSLOUTrajectories._prepare_layer1(H, C, nothing, data, 0.5)
    wb = DiSLOUTrajectories._WorkBuffers(prepared.system.cache)
    equality = ComplexF64[sqrt(1 / 3), sqrt(2 / 3)]
    better = ComplexF64[0.5, sqrt(0.75)]
    DiSLOUTrajectories._solve_coordinates!(wb.c, prepared.system.gauges[1].cache, equality)
    @test DiSLOUTrajectories._postjump_gauge!(wb, prepared, 1) == 1
    DiSLOUTrajectories._solve_coordinates!(wb.c, prepared.system.gauges[1].cache, better)
    @test DiSLOUTrajectories._postjump_gauge!(wb, prepared, 1) == 2
end

@testset "Layer I gauge shifts preserve the Lindblad generator" begin
    # Paper: ρ̇ = -i[H,ρ] + Σ_μ D[C_μ]ρ (Eq. 1).
    function lindblad_rhs(H, collapse, density)
        derivative = -im .* (H * density .- density * H)
        for operator in collapse
            derivative .+= operator * density * operator' .-
                0.5 .* (
                operator' * operator * density .+
                    density * operator' * operator
            )
        end
        return derivative
    end

    H, c_ops = random_system(N = 6, seed = 2)
    Hmat = Matrix{ComplexF64}(H.data)
    collapse = Matrix{ComplexF64}[Matrix{ComplexF64}(op.data) for op in c_ops]
    for shifts in (
            ComplexF64[0, 0], ComplexF64[0.7 - 0.3im, -0.2im],
            ComplexF64[-1.2, 0.9im],
        )
        shifted_H, shifted_collapse = DiSLOUTrajectories._shifted_problem(Hmat, collapse, shifts)
        for seed in 1:10
            sample = randn(Xoshiro(seed), ComplexF64, 6, 6)
            density = sample * sample'
            density ./= tr(density)
            baseline = lindblad_rhs(Hmat, collapse, density)
            shifted = lindblad_rhs(shifted_H, shifted_collapse, density)
            @test norm(baseline - shifted) / max(norm(baseline), 1.0e-300) < 1.0e-10
        end
    end
end

# Mutation caught: using the wrong conjugate on either shifted-cache cross term
# changes physical first or second collapse moments for complex shifts.
@testset "Layer I derives physical moments from shifted caches" begin
    H, c_ops = random_system(N = 6, seed = 71)
    C = Matrix{ComplexF64}[Matrix(op.data) for op in c_ops]
    shifts = ComplexF64[
        0.35 - 0.2im -0.1 + 0.15im
        -0.15 + 0.4im 0.25 + 0.05im
    ]
    data = DiSLOUTrajectories._validated_gauge_data(shifts, length(C))
    prepared = DiSLOUTrajectories._prepare_layer1(H, C, nothing, data, 0.8)

    for (index, gauge) in enumerate(prepared.system.gauges)
        meanV, secondV = DiSLOUTrajectories._physical_moment_matrices(
            gauge.cache, @view shifts[:, index]
        )
        for channel in eachindex(C)
            action = C[channel] * gauge.cache.V
            @test meanV[channel] ≈ gauge.cache.V' * action atol = 1.0e-11 rtol = 1.0e-10
            @test secondV[channel] ≈ action' * action atol = 1.0e-11 rtol = 1.0e-10
            @test gauge.meanV[channel] ≈ meanV[channel] atol = 1.0e-12 rtol = 0
            @test gauge.secondV[channel] ≈ secondV[channel] atol = 1.0e-12 rtol = 0
        end
    end
end

@testset "Layer I prepared gauges retain cache identities and dimensions" begin
    a = tensor(destroy(2), qeye(2))
    b = tensor(qeye(2), destroy(2))
    H = 0.15 * (a + a') + 0.07 * (b + b')
    c_ops = [0.4 * a, 0.3 * b]
    e_ops = [a' * a]
    shifts = ComplexF64[0.1 0.2; -0.05im 0.08im]
    centers = ComplexF64[1 2; 3 4]
    data = DiSLOUTrajectories._GaugeData(
        shifts, :named, centers, ones(2), (;),
        :provided
    )
    prepared = DiSLOUTrajectories._prepare_layer1(H, c_ops, e_ops, data, 0.5)
    system = prepared.system

    @test system.cache === first(system.gauges).cache
    @test system.cache.dimensions == H.dimensions
    @test system.centers == centers
    @test system.shift_channels == [1, 2]
    @test length(system.gauges) == 2
    @test all(gauge.cache.dimensions == H.dimensions for gauge in system.gauges)

    Hmat = Matrix{ComplexF64}(H.data)
    collapse = Matrix{ComplexF64}[Matrix{ComplexF64}(op.data) for op in c_ops]
    for gauge_index in eachindex(system.gauges)
        gauge = system.gauges[gauge_index]
        shifted_H, shifted_collapse = DiSLOUTrajectories._shifted_problem(
            Hmat, collapse, shifts[:, gauge_index]
        )
        effective = shifted_H -
            (im / 2) * sum(C' * C for C in shifted_collapse)
        @test gauge.shifts == shifts[:, gauge_index]
        @test DiSLOUTrajectories._effective_hamiltonian(gauge.cache.H, gauge.cache.C) ≈
            effective rtol = 1.0e-10
        @test gauge.cache.G ≈ gauge.cache.V' * gauge.cache.V rtol = 1.0e-10
        @test all(
            gauge.cache.A[μ] ≈ gauge.cache.C[μ] * gauge.cache.V
                for μ in eachindex(shifted_collapse)
        )
        @test all(
            gauge.cache.M[μ] ≈ gauge.cache.A[μ]' * gauge.cache.A[μ]
                for μ in eachindex(shifted_collapse)
        )
    end
end
