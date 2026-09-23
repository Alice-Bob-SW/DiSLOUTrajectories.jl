module DiSLOUTrajectoriesClusteringExt

import Clustering
import DiSLOUTrajectories
import DiSLOUTrajectories: _chunk_ranges, _shifted_problem, _traj_rng, _validated_seed
using Distances
using Distributed: nprocs, pmap
using QuantumToolbox: coherent, mcsolve, tensor
using Random: Xoshiro
using Statistics

const CF = ComplexF64

# Paper: rescaled (Re ᾱ_j^(r), Im ᾱ_j^(r)) (Appendix A.2).
function _scaled_features(points::AbstractMatrix{<:Complex}, scales::AbstractVector)
    nmodes, npoints = size(points)
    features = Matrix{Float64}(undef, 2nmodes, npoints)
    @inbounds for point in 1:npoints, mode in 1:nmodes
        z = points[mode, point] / scales[mode]
        features[2mode - 1, point] = real(z)
        features[2mode, point] = imag(z)
    end
    return features
end

_center_sort_key(center) = Tuple(Iterators.flatten((real(z), imag(z)) for z in center))

# Paper: clusters 𝓘_g and centers of ᾱ_j^(r) (Appendix A.2).
function _cluster_terminal_means(
        points::AbstractMatrix{<:Complex};
        cluster_scales::AbstractVector, dbscan_radius::Real,
        min_neighbors::Int, min_weight::Real
    )
    nmodes, npoints = size(points)
    length(cluster_scales) == nmodes || throw(DimensionMismatch("need one clustering scale per mode"))
    all(scale -> isfinite(scale) && scale > 0, cluster_scales) || throw(ArgumentError("cluster_scales must be finite and positive"))
    isfinite(dbscan_radius) && dbscan_radius > 0 || throw(ArgumentError("dbscan_radius must be finite and positive"))
    min_neighbors >= 1 || throw(ArgumentError("min_neighbors must be positive"))
    0 <= min_weight < 1 || throw(ArgumentError("min_weight must lie in [0, 1)"))
    all(isfinite, points) || throw(ArgumentError("terminal means must be finite"))
    npoints == 0 && return (; centers = zeros(CF, nmodes, 0), weights = Float64[], counts = Int[], labels = Int[])
    features = _scaled_features(points, cluster_scales)
    result = Clustering.dbscan(features, dbscan_radius; min_neighbors)
    labels = copy(Clustering.assignments(result))
    for (label, cluster) in enumerate(result.clusters), point in cluster.boundary_indices
        any(
            core -> evaluate(Euclidean(), @view(features[:, point]), @view(features[:, core])) <= dbscan_radius,
            cluster.core_indices
        ) || (labels[point] = 0)
    end
    records = NamedTuple[]
    for old in sort!(unique(filter(>(0), labels)))
        members = findall(==(old), labels)
        weight = length(members) / npoints
        weight < min_weight && continue
        center = CF.(vec(mean(@view(points[:, members]); dims = 2)))
        push!(records, (; old, count = length(members), weight, center))
    end
    sort!(records; by = record -> (-record.weight, _center_sort_key(record.center)))
    remap = Dict(record.old => index for (index, record) in enumerate(records))
    final_labels = [get(remap, label, 0) for label in labels]
    centers = isempty(records) ? zeros(CF, nmodes, 0) : hcat((copy(record.center) for record in records)...)
    return (; centers, weights = Float64[record.weight for record in records], counts = Int[record.count for record in records], labels = final_labels)
end

function _run_discovery_indices(f, nseeds::Int, ensemblealg::Symbol)
    results = Vector{Any}(undef, nseeds)
    if ensemblealg === :distributed
        for (index, result) in pmap(index -> (index, f(index)), 1:nseeds)
            results[index] = result
        end
    elseif ensemblealg === :threads && Threads.nthreads() > 1
        nchunks = max(1, min(2 * Threads.nthreads(), nseeds))
        @sync for indices in _chunk_ranges(nseeds, nchunks)
            Threads.@spawn for index in indices
                results[index] = f(index)
            end
        end
    else
        for index in 1:nseeds
            results[index] = f(index)
        end
    end
    return results
end

function _trajectory_discovery_inputs(H, c_ops, mode_ops, mode_dims, discovery_time, seed_radii, cluster_scales, step, nseeds, terminal_window, preliminary_shifts, dbscan_radius, min_neighbors, min_weight, save_preliminary_trajectories, ensemblealg)
    nmodes = length(mode_ops)
    nmodes > 0 || throw(ArgumentError("need at least one mode operator"))
    length(mode_dims) == nmodes || throw(DimensionMismatch("need one subsystem dimension per mode"))
    all(>=(2), mode_dims) || throw(ArgumentError("mode dimensions must be at least 2"))
    length(seed_radii) == nmodes || throw(DimensionMismatch("need one seed radius per mode"))
    all(radius -> isfinite(radius) && radius >= 0, seed_radii) || throw(ArgumentError("seed_radii must be finite and nonnegative"))
    isfinite(discovery_time) && discovery_time > 0 || throw(ArgumentError("discovery_time must be finite and positive"))
    isfinite(step) && step > 0 || throw(ArgumentError("step must be finite and positive"))
    isfinite(terminal_window) && terminal_window >= 0 || throw(ArgumentError("terminal_window must be finite and nonnegative"))
    nseeds >= 1 || throw(ArgumentError("nseeds must be positive"))
    save_preliminary_trajectories >= 0 || throw(ArgumentError("save_preliminary_trajectories must be nonnegative"))
    ensemblealg in (:serial, :threads, :distributed) || throw(ArgumentError("ensemblealg must be :serial, :threads, or :distributed"))
    ensemblealg === :distributed && nprocs() <= 1 && throw(ArgumentError("ensemblealg=:distributed requires worker processes"))
    tensor_dims = Tuple(Int.(mode_dims)); N = prod(tensor_dims)
    Tuple(first(H.dims)) == tensor_dims && Tuple(last(H.dims)) == tensor_dims && size(H.data) == (N, N) || throw(DimensionMismatch("mode_dims=$tensor_dims do not match H tensor dimensions $(H.dims)"))
    all(op -> size(op.data) == (N, N) && Tuple(first(op.dims)) == tensor_dims && Tuple(last(op.dims)) == tensor_dims, mode_ops) || throw(DimensionMismatch("every mode operator must have tensor dimensions $tensor_dims"))
    all(op -> size(op.data) == (N, N) && Tuple(first(op.dims)) == tensor_dims && Tuple(last(op.dims)) == tensor_dims, c_ops) || throw(DimensionMismatch("every collapse operator must have tensor dimensions $tensor_dims"))
    z = preliminary_shifts === nothing ? zeros(CF, length(c_ops)) : Vector{CF}(preliminary_shifts)
    length(z) == length(c_ops) || throw(DimensionMismatch("preliminary_shifts must contain one shift per collapse channel"))
    all(isfinite, z) || throw(ArgumentError("preliminary_shifts must be finite"))
    _cluster_terminal_means(zeros(CF, nmodes, 0); cluster_scales, dbscan_radius, min_neighbors, min_weight)
    return (; nmodes, z)
end

# Paper: ᾱ_j^(r) and terminal ⟨C_μ⟩ averages (Eqs. A.7–A.8).
function _run_preliminary_trajectories(
        H, c_ops;
        mode_ops, mode_dims, discovery_time, seed_radii, cluster_scales,
        step = discovery_time / 40, nseeds::Int = 600, terminal_window = 0.0,
        preliminary_shifts = nothing, dbscan_radius = 1.5,
        min_neighbors::Int = 10, min_weight = 0.02, seed::Integer = 1,
        save_preliminary_trajectories::Int = 0, ensemblealg::Symbol = :threads
    )
    inputs = _trajectory_discovery_inputs(H, c_ops, mode_ops, mode_dims, discovery_time, seed_radii, cluster_scales, step, nseeds, terminal_window, preliminary_shifts, dbscan_radius, min_neighbors, min_weight, save_preliminary_trajectories, ensemblealg)
    seed = _validated_seed(seed)
    tlist = collect(0.0:float(step):float(discovery_time)); last(tlist) < discovery_time && push!(tlist, float(discovery_time))
    tail = findall(t -> t >= max(first(tlist), float(discovery_time - terminal_window)), tlist); isempty(tail) && (tail = [lastindex(tlist)])
    Hrun, Crun = all(iszero, inputs.z) ? (H, c_ops) :
        _shifted_problem(H, c_ops, collect(enumerate(inputs.z)))
    number_ops = [op' * op for op in mode_ops]; e_ops = vcat(collect(mode_ops), number_ops, collect(c_ops))
    seed_rng = Xoshiro(seed); amplitudes = Matrix{CF}(undef, inputs.nmodes, nseeds)
    for point in 1:nseeds, mode in 1:inputs.nmodes
        amplitudes[mode, point] = seed_radii[mode] * sqrt(rand(seed_rng)) * cis(2π * rand(seed_rng))
    end
    nsave = min(save_preliminary_trajectories, nseeds)
    # QuantumToolbox's retained-run keyword predates DiSLOUTrajectories.
    pilot_storage = (; Symbol("keep_" * "runs_results") => Val(true))
    relax = function (point)
        psi0 = tensor((coherent(Int(mode_dims[mode]), amplitudes[mode, point]) for mode in 1:inputs.nmodes)...)
        sol = mcsolve(
            Hrun, psi0, tlist, Crun; e_ops, ntraj = 1,
            rng = _traj_rng(seed, point),
            pilot_storage..., saveat = tlist,
            progress_bar = Val(false)
        )
        return (; means = CF[mean(@view sol.expect[mode, 1, tail]) for mode in 1:inputs.nmodes], occupations = Float64[mean(real.(@view sol.expect[inputs.nmodes + mode, 1, tail])) for mode in 1:inputs.nmodes], collapses = CF[mean(@view sol.expect[2inputs.nmodes + channel, 1, tail]) for channel in eachindex(c_ops)], trace_states = point <= nsave ? hcat((CF.(vec(state.data)) for state in sol.states)...) : nothing, trace_means = point <= nsave ? Matrix{CF}(@view sol.expect[1:inputs.nmodes, 1, :]) : nothing, trace_occupations = point <= nsave ? Float64.(real.(@view sol.expect[(inputs.nmodes + 1):2inputs.nmodes, 1, :])) : nothing, jump_times = point <= nsave ? copy(sol.col_times[1]) : nothing, jump_channels = point <= nsave ? copy(sol.col_which[1]) : nothing)
    end
    results = _run_discovery_indices(relax, nseeds, ensemblealg)
    terminal_means = Matrix{CF}(undef, inputs.nmodes, nseeds); terminal_occupations = Matrix{Float64}(undef, inputs.nmodes, nseeds); terminal_collapse_means = Matrix{CF}(undef, length(c_ops), nseeds)
    for point in 1:nseeds
        result = results[point]; terminal_means[:, point] = result.means; terminal_occupations[:, point] = result.occupations; terminal_collapse_means[:, point] = result.collapses
    end
    states = [copy(results[point].trace_states) for point in 1:nsave]; traces = [copy(results[point].trace_means) for point in 1:nsave]; occupations = [copy(results[point].trace_occupations) for point in 1:nsave]; jump_times = [copy(results[point].jump_times) for point in 1:nsave]; jump_channels = [copy(results[point].jump_channels) for point in 1:nsave]
    return (;
        inputs, tlist, results, terminal_means, terminal_occupations,
        terminal_collapse_means, nsave, states, traces, occupations,
        jump_times, jump_channels,
    )
end

# Paper: ζ_μ^(g) = -⟨C_μ⟩_g, averaged within cluster g (Eq. A.8).
function _trajectory_gauge_result(data, clusters, diagnostics)
    isempty(clusters.counts) &&
        throw(ArgumentError("trajectory discovery found no retained DBSCAN clusters"))
    ngauges = length(clusters.counts)
    shifts = Matrix{CF}(undef, size(data.terminal_collapse_means, 1), ngauges)
    for gauge in 1:ngauges
        members = findall(==(gauge), clusters.labels)
        shifts[:, gauge] = -vec(
            mean(
                @view(data.terminal_collapse_means[:, members]); dims = 2
            )
        )
    end
    return (; shifts = copy(shifts), method = :trajectories, centers = copy(clusters.centers), weights = copy(clusters.weights), diagnostics)
end

# Paper: ζ_μ^(g) from preliminary trajectory clusters (Eq. A.8).
function DiSLOUTrajectories._discover_gauges(
        ::Val{:trajectories},
        H, c_ops;
        mode_ops, mode_dims, discovery_time, seed_radii, cluster_scales,
        step = discovery_time / 40, nseeds::Int = 600, terminal_window = 0.0,
        preliminary_shifts = nothing, dbscan_radius = 1.5,
        min_neighbors::Int = 10, min_weight = 0.02, seed::Integer = 1,
        save_preliminary_trajectories::Int = 0, ensemblealg::Symbol = :threads
    )
    data = _run_preliminary_trajectories(
        H, c_ops;
        mode_ops, mode_dims, discovery_time, seed_radii, cluster_scales,
        step, nseeds, terminal_window, preliminary_shifts, dbscan_radius,
        min_neighbors, min_weight, seed, save_preliminary_trajectories,
        ensemblealg
    )
    clusters = _cluster_terminal_means(
        data.terminal_means;
        cluster_scales, dbscan_radius, min_neighbors, min_weight
    )
    diagnostics = (;
        counts = copy(clusters.counts), labels = copy(clusters.labels),
        terminal_means = copy(data.terminal_means),
        terminal_occupations = copy(data.terminal_occupations),
        terminal_collapse_means = copy(data.terminal_collapse_means),
        times = copy(data.tlist),
        preliminary_traces = (;
            indices = collect(1:data.nsave),
            states = data.states, means = data.traces, occupations = data.occupations,
        ),
        preliminary_jump_times = data.jump_times,
        preliminary_jump_channels = data.jump_channels,
        mode_dims = collect(Int, mode_dims), discovery_time = float(discovery_time),
        seed_radii = Float64.(seed_radii), cluster_scales = Float64.(cluster_scales),
        step = float(step), nseeds, terminal_window = float(terminal_window),
        preliminary_shifts = copy(data.inputs.z), dbscan_radius = float(dbscan_radius),
        min_neighbors, min_weight = float(min_weight), seed = _validated_seed(seed),
        save_preliminary_trajectories = data.nsave, ensemblealg,
    )
    return _trajectory_gauge_result(data, clusters, diagnostics)
end

end
