@testset "private recording kernel matches normalized dense evolution" begin
    H, c_ops = random_system(; N = 6, seed = 15)
    observable = destroy(6)
    cache = SM._DiagonalCache(H, c_ops; e_ops = [observable])
    H_eff = SM._effective_hamiltonian(cache.H, cache.C)
    tlist = collect(0.0:0.25:3.0)
    psi = normalize(randn(Xoshiro(31), ComplexF64, cache.N))
    coordinates = similar(psi)
    SM._solve_coordinates!(coordinates, cache, psi)
    coordinates ./= sqrt(real(dot(coordinates, cache.G, coordinates)))
    accumulator = SM._TrajectoryAccumulator(1, length(tlist), cache.Nc)

    SM._record_exact!(accumulator, cache, coordinates, tlist, 0.0, 1, length(tlist))
    @test all(
        isapprox(
            accumulator.mean[1, index],
            direct_expect(H_eff, Matrix(observable.data), psi, tlist[index]);
            atol = 1.0e-9,
        ) for index in eachindex(tlist)
    )
end

@testset "sparse exact recording matches dense without allocations" begin
    H, c_ops = random_system(; N = 32, seed = 212)
    observable = QuantumObject(sparse(num(32).data))
    dense = SM._DiagonalCache(
        H, c_ops; e_ops = [observable], observable_storage = :dense
    )
    sparse_cache = SM._DiagonalCache(
        H, c_ops; e_ops = [observable], observable_storage = :sparse
    )
    ψ0 = normalize(randn(Xoshiro(213), ComplexF64, 32))
    times = collect(0.0:0.1:1.0)

    function recorded(cache)
        c = cache.Vfac \ ψ0
        c ./= sqrt(real(dot(c, cache.G, c)))
        acc = SM._TrajectoryAccumulator(1, length(times), cache.Nc)
        Dc, tmp, ψ = similar(c), similar(c), similar(c)
        SM._record_exact!(acc, cache, c, times, 0.0, 1, length(times), Dc, tmp, ψ)
        bytes = @allocated SM._record_exact!(
            acc, cache, c, times, 0.0, 1, length(times), Dc, tmp, ψ
        )
        return acc.mean, bytes
    end

    dense_values, _ = recorded(dense)
    sparse_values, sparse_bytes = recorded(sparse_cache)
    @test sparse_values ≈ dense_values atol = 2.0e-13 rtol = 0
    @test sparse_bytes == 0
end

@testset "sparse local recording matches dense without allocations" begin
    H, c_ops = random_system(N = 24, seed = 214)
    observable = QuantumObject(sparse(num(24).data))
    dense = SM._DiagonalCache(
        H, c_ops; e_ops = [observable], observable_storage = :dense
    )
    sparse_cache = SM._DiagonalCache(
        H, c_ops; e_ops = [observable], observable_storage = :sparse
    )
    indices = collect(1:12)
    dense_reduced = SM._get_reduced!(dense, indices)
    sparse_reduced = SM._get_reduced!(sparse_cache, indices)
    c = normalize(randn(Xoshiro(215), ComplexF64, length(indices)))
    times = collect(0.0:0.1:0.5)

    function recorded(cache, reduced)
        acc = SM._TrajectoryAccumulator(1, length(times), cache.Nc)
        evolved = similar(c)
        metric_tmp = similar(c)
        ψ = zeros(ComplexF64, cache.N)
        observable_tmp = zeros(ComplexF64, cache.N)
        SM._record_local!(
            acc, cache, reduced, c, times, 0.0, 1,
            length(times), evolved, metric_tmp, ψ, observable_tmp
        )
        bytes = @allocated SM._record_local!(
            acc, cache, reduced, c, times, 0.0, 1, length(times),
            evolved, metric_tmp, ψ, observable_tmp
        )
        return acc.mean, bytes
    end

    dense_values, _ = recorded(dense, dense_reduced)
    sparse_values, sparse_bytes = recorded(sparse_cache, sparse_reduced)
    @test sparse_values ≈ dense_values atol = 2.0e-13 rtol = 0
    @test sparse_bytes == 0
end

@testset "state density accumulation reuses storage" begin
    ψ1 = ComplexF64[1 + 2im, -3 + im, 2 - im, -im]
    ψ2 = ComplexF64[2 - im, 1 + 3im, -2im, 4 + im]
    accumulator = SM._TrajectoryAccumulator(0, 0, 0)
    record = SM._TrajectoryRecord(length(ψ1), 0, 0, 1, true)

    SM._record_state!(accumulator, 1, ψ1, record)
    saved_state = record.states[1]
    ψ2_before = copy(ψ2)
    bytes = @allocated SM._record_state!(accumulator, 1, ψ2, record)
    expected = ψ1 * ψ1' / sum(abs2, ψ1) + ψ2 * ψ2' / sum(abs2, ψ2)

    @test bytes == 0
    @test accumulator.state_sums[1] ≈ expected
    @test record.states[1] ≈ ψ2 / norm(ψ2)
    @test record.states[1] === saved_state
    @test ψ2 == ψ2_before

    large_state = normalize!(randn(Xoshiro(13), ComplexF64, 256))
    large_accumulator = SM._TrajectoryAccumulator(0, 0, 0)
    SM._record_state!(large_accumulator, 1, large_state)
    SM._record_state!(large_accumulator, 1, large_state)
    @test @allocated(SM._record_state!(large_accumulator, 1, large_state)) == 0
end

@testset "failed trajectories do not enter aggregate statistics" begin
    d = 4
    a = destroy(d)
    data = SM._validated_gauge_data(zeros(ComplexF64, 1, 1), 1)
    prepared = SM._prepare_layer1(0 * a, [5 * a], [num(d)], data, 0.5)
    cache = prepared.system.cache
    psi0 = Vector{ComplexF64}(fock(d, 1).data)
    tlist = collect(0.0:0.1:1.0)
    accumulator = SM._TrajectoryAccumulator(1, length(tlist), cache.Nc)
    diagnostics = SM._GaugeDiagnostics(1)
    error = try
        SM._run_exact_trajectory!(
            accumulator, diagnostics, prepared, psi0,
            tlist, last(tlist), Xoshiro(901), SM._WorkBuffers(cache); max_jumps = 0
        )
        nothing
    catch caught
        caught
    end

    @test error isa ErrorException
    @test occursin("maximum number of jumps reached", sprint(showerror, error))
    @test accumulator.ntraj == 0
    @test accumulator.njumps_total == 0
    @test all(iszero, accumulator.mean)
    @test all(iszero, accumulator.jumps_by_channel)
end

@testset "public max_jumps limit reaches both trajectory kernels" begin
    H = zeros(ComplexF64, 2, 2)
    psi0 = ComplexF64[1, 0]
    tlist = [0.0]
    c_ops = Matrix{ComplexF64}[]
    common = (;
        gauge_set = zeros(ComplexF64, 0, 1), ntraj = 1,
        ensemblealg = :serial, max_jumps = 0,
    )

    @test_throws ErrorException dislou_solve(H, psi0, tlist, c_ops; common...)
    @test_throws ErrorException dislou_solve(
        H, psi0, tlist, c_ops;
        common..., layer3 = true, layer3_sizes = 1
    )
end

@testset "public first_passage_maxiter reaches both trajectory kernels" begin
    H = zeros(ComplexF64, 2, 2)
    psi0 = normalize(ComplexF64[1, 1])
    tlist = [0.0, 10.0]
    c_ops = [ComplexF64[0 1; 0 0]]
    common = (;
        gauge_set = zeros(ComplexF64, 1, 1), ntraj = 1,
        seed = 5, ensemblealg = :serial, survival_rtol = 1.0e-15,
        time_rtol = 1.0e-15, first_passage_maxiter = 1,
    )

    @test_throws DiSLOUTrajectories.FirstPassageConvergenceError dislou_solve(
        H, psi0, tlist, c_ops; common...
    )
    @test_throws DiSLOUTrajectories.FirstPassageConvergenceError dislou_solve(
        H, psi0, tlist, c_ops; common..., layer3 = true, layer3_sizes = 2
    )
end

@testset "public event columns remain aligned" begin
    d = 5
    a = destroy(d)
    sol = dislou_solve(
        0.1 * (a + a'), fock(d, 3), collect(2.0:0.2:3.0),
        [1.4 * a]; e_ops = [num(d)], gauge_set = ComplexF64[0 0.8],
        ntraj = 40, seed = 81, ensemblealg = :serial
    )

    @test length(sol.col_times) == sol.ntraj
    @test length(sol.col_which) == sol.ntraj
    @test length(sol.col_gauge) == sol.ntraj
    @test all(
        length(sol.col_times[i]) == length(sol.col_which[i]) ==
            length(sol.col_gauge[i]) for i in 1:sol.ntraj
    )
    @test sum(length, sol.col_times) == sol.njumps_total
    @test all(2.0 <= time <= 3.0 for time in Iterators.flatten(sol.col_times))
    @test all(
        1 <= channel <= length(sol.jumps_by_channel)
            for channel in Iterators.flatten(sol.col_which)
    )
    @test all(
        1 <= gauge <= sol.gauge_diagnostics.count
            for gauge in Iterators.flatten(sol.col_gauge)
    )
end
