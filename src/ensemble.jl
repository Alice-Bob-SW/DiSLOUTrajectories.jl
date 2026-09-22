# Private deterministic trajectory and ensemble execution.

@inline function _splitmix64(x::UInt64)
    z = x + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

_traj_rng(base_seed::UInt64, index::Integer) =
    Xoshiro(_splitmix64(_splitmix64(base_seed) ⊻ UInt64(index)))

_chunk_ranges(ntraj::Int, nchunks::Int) =
    [
    (((chunk - 1) * ntraj) ÷ nchunks + 1):((chunk * ntraj) ÷ nchunks)
        for chunk in 1:nchunks
]

# Paper: c/√(c†Gc), representing normalized |ψ⟩ (Eq. 17).
@inline function _normalize_coordinates!(
        coordinates::AbstractVector{CF},
        scratch::AbstractVector{CF}, gram::AbstractMatrix{CF}, gram_norm::Real
    )
    norm2 = _checked_quadratic!(
        scratch, coordinates, gram,
        gram_norm * sum(abs2, coordinates), "state norm"
    )
    coordinates ./= sqrt(norm2)
    return coordinates
end

function _trajectory_predictor(system::_GaugeSystem, method::Symbol)
    method === :log_survival_predictor || return nothing
    return _FirstPassagePredictor(system.cache; m = min(system.cache.N, 32), tol = 1.0e-8)
end

# Paper: |ψ(t)⟩ with exact Layers I–II (Section 3.4).
function _run_exact_trajectory!(
        acc::_TrajectoryAccumulator,
        diagnostics::_GaugeDiagnostics, prepared::_Layer1Prepared,
        ψ0::AbstractVector{CF}, tlist::AbstractVector{Float64}, T::Float64,
        rng, wb::_WorkBuffers; survival_rtol::Real = 1.0e-10,
        time_rtol::Real = 1.0e-12, time_atol::Real = 0.0,
        max_jumps::Int = 1_000_000,
        first_passage_maxiter::Int = 100,
        first_passage_method::Symbol = :log_survival,
        trajectory_predictor::Union{Nothing, _FirstPassagePredictor} = nothing,
        final_states::Union{Nothing, AbstractMatrix{CF}} = nothing,
        final_state_column::Int = 0,
        times_states::AbstractVector{Float64} = Float64[],
        record::Union{Nothing, _TrajectoryRecord} = nothing,
        col_gauge::Union{Nothing, Vector{Int}} = nothing
    )
    system = prepared.system
    next_time = 1
    next_state = 1
    time = 0.0
    jumps = 0
    gauge = _initial_gauge_coordinates!(wb, prepared, ψ0)

    @inbounds while next_time <= length(tlist)
        jumps >= max_jumps && error("maximum number of jumps reached")
        local_cache = system.gauges[gauge].cache
        threshold = rand(rng)
        remaining = T - time
        _normalize_coordinates!(wb.c, wb.Gc, local_cache.G, local_cache.Gnorm)
        result = _find_first_passage!(
            _FirstPassageBuffers(wb.Dc, wb.Gc, trajectory_predictor), local_cache,
            wb.c, threshold, remaining; method = first_passage_method,
            survival_rtol, time_rtol, time_atol,
            maxiter = first_passage_maxiter
        )
        if !result.jumped
            hi = searchsortedlast(tlist, T)
            _record_exact!(
                acc, local_cache, wb.c, tlist, time,
                next_time, hi, wb.Dc, wb.Gc, wb.ψ; record
            )
            state_hi = searchsortedlast(times_states, T)
            _record_state_window!(
                acc, record, local_cache.Λ, local_cache.V,
                wb.c, times_states, time, next_state, state_hi, wb.Dc, wb.ψ
            )
            diagnostics.residence_time[gauge] += T - time
            next_time = hi + 1
            next_state = state_hi + 1
            break
        end

        event_time = time + result.tau
        hi = searchsortedfirst(tlist, event_time) - 1
        _record_exact!(
            acc, local_cache, wb.c, tlist, time,
            next_time, hi, wb.Dc, wb.Gc, wb.ψ; record
        )
        state_hi = searchsortedfirst(times_states, event_time) - 1
        _record_state_window!(
            acc, record, local_cache.Λ, local_cache.V,
            wb.c, times_states, time, next_state, state_hi, wb.Dc, wb.ψ
        )
        diagnostics.residence_time[gauge] += result.tau
        next_time = hi + 1
        next_state = state_hi + 1

        _phase_evolve!(wb.Dc, local_cache.Λ, wb.c, result.tau)
        channel, next_gauge = _apply_exact_jump_and_route!(
            wb, prepared, gauge, wb.Dc, rng
        )
        jumps += 1
        acc.jumps_by_channel[channel] += 1
        diagnostics.jumps_by_gauge[gauge] += 1
        if record !== nothing
            push!(record.col_times, event_time)
            push!(record.col_which, channel)
        end
        col_gauge === nothing || push!(col_gauge, gauge)

        if next_gauge != gauge
            copyto!(wb.c, wb.cplus)
            diagnostics.switches += 1
        end
        gauge = next_gauge
        time = event_time
    end

    if final_states !== nothing
        local_cache = system.gauges[gauge].cache
        _phase_evolve!(wb.Dc, local_cache.Λ, wb.c, T - time)
        mul!(wb.ψ, local_cache.V, wb.Dc)
        wb.ψ ./= norm(wb.ψ)
        final_states[:, final_state_column] .= wb.ψ
    end
    acc.ntraj += 1
    acc.njumps_total += jumps
    return jumps
end

function _run_gauge_range!(
        accumulator, diagnostics, prepared, layer3, state, times, T,
        trajectories, slots, base_seed, final_states, records, col_gauge;
        first_passage_method, times_states, save_trajectories, kwargs...
    )
    system = prepared.system
    physical = system.cache
    wb = _WorkBuffers(physical, max(physical.N, length(system.gauges)))
    predictor = _trajectory_predictor(system, first_passage_method)
    run!, preparations = layer3 === nothing ?
        (_run_exact_trajectory!, (prepared,)) :
        (_run_layer3_trajectory!, (prepared, layer3))
    for (trajectory, slot) in zip(trajectories, slots)
        record = _TrajectoryRecord(
            physical.N, physical.Ne, length(times),
            length(times_states), save_trajectories
        )
        run!(
            accumulator, diagnostics, preparations..., state, times, T,
            _traj_rng(base_seed, trajectory), wb; kwargs..., first_passage_method,
            trajectory_predictor = predictor, final_states,
            final_state_column = slot, times_states, record,
            col_gauge = col_gauge[slot]
        )
        records[slot] = record
    end
    return nothing
end

function _solve_gauges_serial(
        prepared::_Layer1Prepared,
        layer3::Union{Nothing, _Layer3Prepared}, ψ0::AbstractVector{CF},
        tlist::AbstractVector{Float64}, T::Float64, ntraj::Int,
        base_seed::UInt64; survival_rtol::Real = 1.0e-10,
        time_rtol::Real = 1.0e-12, time_atol::Real = 0.0,
        max_jumps::Int = 1_000_000,
        first_passage_maxiter::Int = 100,
        first_passage_method::Symbol = :log_survival,
        save_final_states::Bool = false,
        times_states::Vector{Float64} = Float64[],
        save_trajectories::Bool = false
    )
    system = prepared.system
    physical = system.cache
    ngauges = length(system.gauges)
    acc = _TrajectoryAccumulator(physical.Ne, length(tlist), physical.Nc)
    diagnostics = _GaugeDiagnostics(ngauges)
    final_states = save_final_states ?
        Matrix{CF}(undef, physical.N, ntraj) : nothing
    records = Vector{_TrajectoryRecord}(undef, ntraj)
    col_gauge = [Int[] for _ in 1:ntraj]
    state = Vector{CF}(ψ0)
    times = collect(Float64, tlist)

    _run_gauge_range!(
        acc, diagnostics, prepared, layer3, state, times, T,
        1:ntraj, 1:ntraj, base_seed, final_states, records, col_gauge;
        survival_rtol, time_rtol, time_atol, max_jumps, first_passage_maxiter,
        first_passage_method, times_states, save_trajectories
    )
    return acc, diagnostics, final_states, records, col_gauge
end

function _solve_gauges_threaded(
        prepared::_Layer1Prepared,
        layer3::Union{Nothing, _Layer3Prepared}, ψ0::AbstractVector{CF},
        tlist::AbstractVector{Float64}, T::Float64, ntraj::Int,
        base_seed::UInt64; survival_rtol::Real = 1.0e-10,
        time_rtol::Real = 1.0e-12, time_atol::Real = 0.0,
        max_jumps::Int = 1_000_000,
        first_passage_maxiter::Int = 100,
        first_passage_method::Symbol = :log_survival,
        save_final_states::Bool = false,
        times_states::Vector{Float64} = Float64[],
        save_trajectories::Bool = false
    )
    system = prepared.system
    physical = system.cache
    ngauges = length(system.gauges)
    nchunks = max(1, min(2 * Threads.nthreads(), ntraj))
    ranges = _chunk_ranges(ntraj, nchunks)
    accumulators = [
        _TrajectoryAccumulator(physical.Ne, length(tlist), physical.Nc)
            for _ in 1:nchunks
    ]
    diagnostics = [_GaugeDiagnostics(ngauges) for _ in 1:nchunks]
    final_states = save_final_states ?
        Matrix{CF}(undef, physical.N, ntraj) : nothing
    records = Vector{_TrajectoryRecord}(undef, ntraj)
    col_gauge = [Int[] for _ in 1:ntraj]
    state = Vector{CF}(ψ0)
    times = collect(Float64, tlist)

    @sync for chunk in 1:nchunks
        Threads.@spawn _run_gauge_range!(
            accumulators[chunk], diagnostics[chunk], prepared, layer3,
            state, times, T, ranges[chunk], ranges[chunk], base_seed,
            final_states, records, col_gauge;
            survival_rtol, time_rtol, time_atol, max_jumps, first_passage_maxiter,
            first_passage_method, times_states, save_trajectories
        )
    end

    merged_accumulator = accumulators[1]
    merged_diagnostics = diagnostics[1]
    for chunk in 2:nchunks
        _merge_accumulator!(merged_accumulator, accumulators[chunk])
        _merge_gauge_diagnostics!(merged_diagnostics, diagnostics[chunk])
    end
    return merged_accumulator, merged_diagnostics, final_states, records, col_gauge
end

function _solve_gauges_distributed(
        H::Matrix{CF}, C::Vector{Matrix{CF}},
        Z::_ObservableMatrices, gauge_data::_GaugeData,
        ψ0::AbstractVector{CF}, tlist::AbstractVector{Float64}, T::Float64,
        ntraj::Int, base_seed::UInt64; survival_rtol::Real = 1.0e-10,
        time_rtol::Real = 1.0e-12, time_atol::Real = 0.0,
        max_jumps::Int = 1_000_000,
        first_passage_maxiter::Int = 100,
        hysteresis::Real = 0.5,
        condition_limit::Real = _DEFAULT_CONDITION_LIMIT,
        observable_storage::Symbol,
        first_passage_method::Symbol = :log_survival,
        layer3_sizes::Union{Nothing, Vector{Int}} = nothing,
        residual_tolerance::Real = 1.0e-3,
        save_final_states::Bool = false,
        times_states::Vector{Float64} = Float64[],
        save_trajectories::Bool = false
    )
    nchunks = min(max(1, nworkers()), ntraj)
    ranges = _chunk_ranges(ntraj, nchunks)
    dimension, nobservables, nchannels = size(H, 1), length(Z), length(C)
    ngauges = size(gauge_data.shifts, 2)
    state = Vector{CF}(ψ0)
    times = collect(Float64, tlist)
    worker_data = _GaugeData(
        copy(gauge_data.shifts), gauge_data.method,
        copy(gauge_data.centers), copy(gauge_data.weights), (;), gauge_data.status
    )

    results = pmap(WorkerPool(workers()), 1:nchunks) do chunk
        prepared = _prepare_layer1(
            H, C, Z, worker_data, hysteresis;
            observable_storage, condition_limit
        )
        layer3 = layer3_sizes === nothing ? nothing :
            _prepare_layer3(prepared, state, layer3_sizes, residual_tolerance)
        accumulator = _TrajectoryAccumulator(nobservables, length(times), nchannels)
        diagnostics = _GaugeDiagnostics(ngauges)
        final_states = save_final_states ?
            Matrix{CF}(undef, dimension, length(ranges[chunk])) : nothing
        records = Vector{_TrajectoryRecord}(undef, length(ranges[chunk]))
        col_gauge = [Int[] for _ in ranges[chunk]]
        _run_gauge_range!(
            accumulator, diagnostics, prepared, layer3, state, times, T,
            ranges[chunk], eachindex(records), base_seed, final_states, records,
            col_gauge; survival_rtol, time_rtol, time_atol, max_jumps,
            first_passage_maxiter, first_passage_method, times_states,
            save_trajectories
        )
        backend = _eigensystem_backend(prepared)
        realized_sizes = layer3 === nothing ? nothing :
            [length(space.reduced.I) for space in layer3.spaces]
        (
            accumulator, diagnostics, final_states, records, col_gauge, backend,
            realized_sizes,
        )
    end

    layer3_sizes === nothing ||
        all(result -> result[7] == layer3_sizes, results) ||
        throw(
        ArgumentError(
            "distributed Layer III realized sizes differ across processes"
        )
    )

    merged_accumulator, merged_diagnostics = results[1]
    merged_states = save_final_states ?
        Matrix{CF}(undef, dimension, ntraj) : nothing
    merged_records = Vector{_TrajectoryRecord}(undef, ntraj)
    merged_col_gauge = Vector{Vector{Int}}(undef, ntraj)
    for chunk in eachindex(results)
        if chunk > 1
            _merge_accumulator!(merged_accumulator, results[chunk][1])
            _merge_gauge_diagnostics!(merged_diagnostics, results[chunk][2])
        end
        merged_states === nothing ||
            (merged_states[:, ranges[chunk]] = results[chunk][3])
        merged_records[ranges[chunk]] = results[chunk][4]
        merged_col_gauge[ranges[chunk]] = results[chunk][5]
    end
    backends = unique(result[6] for result in results)
    backend = length(backends) == 1 ? only(backends) : :mixed
    return merged_accumulator, merged_diagnostics, merged_states,
        merged_records, merged_col_gauge, backend
end
