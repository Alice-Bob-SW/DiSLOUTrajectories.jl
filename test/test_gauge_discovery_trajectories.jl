using Random

function _replay_splitmix64(x::UInt64)
    z = x + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

function _replay_trajectory_rng(seed::Integer, index::Integer)
    z = _replay_splitmix64(UInt64(seed))
    return Xoshiro(_replay_splitmix64(z ⊻ UInt64(index)))
end

@testset "trajectory discovery is deterministic and ordered" begin
    d = 8
    a = destroy(d)
    H = 0.1 * num(d)
    c_ops = [sqrt(0.3) * a]
    kw = (;
        method = :trajectories, mode_ops = [a], mode_dims = [d],
        discovery_time = 0.2, seed_radii = [sqrt(d - 1)], cluster_scales = [1.0],
        step = 0.02, nseeds = 20, terminal_window = 0.04,
        dbscan_radius = 1.5, min_neighbors = 2, min_weight = 0.05,
        seed = 31, ensemblealg = :serial,
    )
    left = discover_gauges(H, c_ops; kw...)
    right = discover_gauges(H, c_ops; kw...)
    @test propertynames(left) == (:shifts, :method, :centers, :weights, :diagnostics)
    @test left.shifts == right.shifts
    @test left.centers == right.centers
    @test left.weights == right.weights
    @test left.diagnostics.counts == right.diagnostics.counts
    @test left.diagnostics.labels == right.diagnostics.labels
    @test left.method === :trajectories
    @test size(left.shifts, 1) == length(c_ops)
    @test size(left.shifts, 2) == size(left.centers, 2) == length(left.weights)
    @test issorted(left.weights; rev = true)
    @test all(isfinite, left.shifts)
end

@testset "trajectory clustering breaks equal-weight ties by center" begin
    points = ComplexF64[1 + 2im 1 + 2im 1 + 2im -1 - 2im -1 - 2im -1 - 2im]
    clusters = ClusteringExt._cluster_terminal_means(
        points;
        cluster_scales = [1.0], dbscan_radius = 0.5, min_neighbors = 2,
        min_weight = 0.0
    )
    @test clusters.centers == ComplexF64[-1 - 2im 1 + 2im]
    @test clusters.counts == [3, 3]
    @test clusters.labels == [2, 2, 2, 1, 1, 1]
end

@testset "preliminary shifts preserve the pilot generator and not returned targets" begin
    d = 6
    a = destroy(d)
    H = 0.2 * num(d)
    c_ops = [sqrt(0.4) * a]
    preliminary = ComplexF64[0.15 + 0.05im]
    Hrun, Crun = SM._shifted_problem(
        Matrix{ComplexF64}(H.data),
        Matrix{ComplexF64}[Matrix{ComplexF64}(c.data) for c in c_ops], preliminary
    )
    expected_Hrun = Matrix{ComplexF64}(H.data) +
        (im / 2) .* (
        preliminary[1] .* Matrix{ComplexF64}(c_ops[1].data)' .-
            conj(preliminary[1]) .* Matrix{ComplexF64}(c_ops[1].data)
    )
    @test Hrun ≈ expected_Hrun atol = 1.0e-12
    @test Crun[1] ≈ Matrix{ComplexF64}(c_ops[1].data) + preliminary[1] * I atol = 1.0e-12

    result = discover_gauges(
        H, c_ops;
        method = :trajectories, mode_ops = [a], mode_dims = [d],
        discovery_time = 0.2, seed_radii = [sqrt(d - 1)], cluster_scales = [1.0],
        step = 0.02, nseeds = 20, terminal_window = 0.04,
        preliminary_shifts = preliminary, dbscan_radius = 1.5,
        min_neighbors = 2, min_weight = 0.05, seed = 19, ensemblealg = :serial
    )
    for g in axes(result.shifts, 2)
        members = findall(==(g), result.diagnostics.labels)
        @test result.shifts[:, g] ≈
            -vec(mean(result.diagnostics.terminal_collapse_means[:, members]; dims = 2))
    end
    @test result.diagnostics.preliminary_shifts == preliminary
end

@testset "semiclassical discovery has no method without its extension" begin
    @test Base.get_extension(DiSLOUTrajectories, :DiSLOUTrajectoriesQuantumCumulantsExt) === nothing
    @test_throws MethodError discover_gauges(identity, sin; method = :semiclassical)
end

function _sentinel_hamiltonian_constructor() end
function _sentinel_collapse_constructor() end
struct _SemiclassicalSentinel <: Exception end
function DiSLOUTrajectories._discover_gauges(
        ::Val{:semiclassical}, ::typeof(_sentinel_hamiltonian_constructor),
        ::typeof(_sentinel_collapse_constructor); kwargs...
    )
    throw(_SemiclassicalSentinel())
end
@testset "semiclassical constructor hook dispatch propagates implementation errors" begin
    @test_throws _SemiclassicalSentinel discover_gauges(
        _sentinel_hamiltonian_constructor, _sentinel_collapse_constructor;
        method = :semiclassical
    )
end

@testset "preliminary pilot replay uses physical collapse expectations" begin
    d = 5
    a = destroy(d)
    H = 0.17 * num(d)
    c_ops = [sqrt(0.4) * a]
    preliminary = ComplexF64[0.2 - 0.1im]
    seed = 7
    nseeds = 4
    step, discovery_time, terminal_window = 0.02, 0.1, 0.04
    result = discover_gauges(
        H, c_ops;
        method = :trajectories, mode_ops = [a], mode_dims = [d],
        discovery_time, seed_radii = [1.2], cluster_scales = [1.0], step,
        nseeds, terminal_window, preliminary_shifts = preliminary,
        dbscan_radius = 10.0, min_neighbors = 1, min_weight = 0.0,
        seed, ensemblealg = :serial
    )

    tlist = collect(0.0:step:discovery_time)
    tail = findall(t -> t >= discovery_time - terminal_window, tlist)
    Hrun, Crun = DiSLOUTrajectories._shifted_problem(
        H, c_ops, [(1, preliminary[1])]
    )
    amplitude_rng = Xoshiro(seed)
    expected = Matrix{ComplexF64}(undef, 1, nseeds)
    for index in 1:nseeds
        alpha = 1.2 * sqrt(rand(amplitude_rng)) * cis(2π * rand(amplitude_rng))
        sol = mcsolve(
            Hrun, coherent(d, alpha), tlist, Crun;
            e_ops = [a, a' * a, c_ops[1]], ntraj = 1,
            rng = _replay_trajectory_rng(seed, index), progress_bar = Val(false)
        )
        expected[1, index] = mean(@view sol.expect[3, tail])
    end
    @test result.diagnostics.terminal_collapse_means == expected
    @test result.shifts == reshape(ComplexF64[-mean(expected)], 1, 1)
end

@testset "saved preliminary trajectories are bounded, indexed, and copied" begin
    d = 5
    a = destroy(d)
    kw = (;
        method = :trajectories, mode_ops = [a], mode_dims = [d],
        discovery_time = 0.1, seed_radii = [1.0], cluster_scales = [1.0],
        step = 0.02, nseeds = 4, terminal_window = 0.02,
        dbscan_radius = 10.0, min_neighbors = 1, min_weight = 0.0,
        seed = 13, save_preliminary_trajectories = 2, ensemblealg = :serial,
    )
    left = discover_gauges(0.1 * num(d), [sqrt(0.3) * a]; kw...)
    right = discover_gauges(0.1 * num(d), [sqrt(0.3) * a]; kw...)
    traces = left.diagnostics.preliminary_traces
    @test traces.indices == [1, 2]
    @test length(traces.states) == length(traces.means) == length(traces.occupations) == 2
    @test all(size(state) == (d, length(left.diagnostics.times)) for state in traces.states)
    @test all(size(means) == (1, length(left.diagnostics.times)) for means in traces.means)
    @test all(size(occupations) == (1, length(left.diagnostics.times)) for occupations in traces.occupations)
    @test all(
        length(left.diagnostics.preliminary_jump_times[index]) ==
            length(left.diagnostics.preliminary_jump_channels[index]) for index in traces.indices
    )
    left.diagnostics.preliminary_traces.states[1][1, 1] = 99
    left.diagnostics.preliminary_traces.means[1][1, 1] = 99
    @test right.diagnostics.preliminary_traces.states[1][1, 1] != 99
    @test right.diagnostics.preliminary_traces.means[1][1, 1] != 99
end
