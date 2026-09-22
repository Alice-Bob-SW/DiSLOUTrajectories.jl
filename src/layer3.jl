# Private Layer III: residual-gated local eigenspaces with exact-gauge fallback.

struct _LocalEigenspace
    reduced::_ReducedCache
    Q::Matrix{ComplexF64}  # Q_m^(g) (Eq. C.1)
    R::UpperTriangular{ComplexF64, Matrix{ComplexF64}}  # R_m^(g) (Eq. C.1)
end

struct _LocalJumpRoute
    rate::Matrix{CF}
    rate_norm::Float64
    residual::Matrix{CF}
    residual_norm::Float64
    coordinates::Matrix{CF}  # J_{μ,red}^{g′←g} (Eq. 22)
end

struct _Layer3Prepared
    system::_GaugeSystem
    spaces::Vector{_LocalEigenspace}
    routes::Vector{Vector{Vector{_LocalJumpRoute}}}
    tolerance::Float64
end

# Paper: 𝒦_m^(g) indices, selected by γ_j^(g) (Eq. 20).
function _slow_mode_indices(cache::_DiagonalCache, requested::Int)
    1 <= requested <= cache.N || throw(
        ArgumentError(
            "Layer III size must be in 1:$(cache.N), got $requested"
        )
    )
    order = sortperm(
        1:cache.N;
        by = index -> (
            cache.Γ[index], real(cache.Λ[index]),
            imag(cache.Λ[index]), index,
        )
    )
    chosen = Set(order[1:requested])
    for cluster in cache.deg_clusters
        isempty(intersect(chosen, cluster)) || union!(chosen, cluster)
    end
    return sort!(collect(chosen))
end

# Paper: 𝒦_m^(g), Q_m^(g), R_m^(g) (Eqs. 20, C.1).
function _build_local_eigenspace(
        cache::_DiagonalCache,
        requested::Int
    )::_LocalEigenspace
    reduced = _get_reduced!(cache, _slow_mode_indices(cache, requested))
    factorization = qr(reduced.V_I)
    size = length(reduced.I)
    Q = Matrix{CF}(factorization.Q)[:, 1:size]
    R = UpperTriangular(Matrix{CF}(factorization.R))
    return _LocalEigenspace(reduced, Q, R)
end

# Paper: r_m^(g)(ψ/‖ψ‖) (Eq. 21).
function _relative_residual!(coeff, scratch, space::_LocalEigenspace, ψ)::Float64
    length(ψ) == size(space.Q, 1) || throw(
        DimensionMismatch(
            "Layer III state has length $(length(ψ)); expected $(size(space.Q, 1))"
        )
    )
    length(coeff) == size(space.Q, 2) || throw(
        DimensionMismatch(
            "Layer III coefficient scratch has length $(length(coeff)); expected $(size(space.Q, 2))"
        )
    )
    length(scratch) == length(ψ) || throw(
        DimensionMismatch(
            "Layer III residual scratch has length $(length(scratch)); expected $(length(ψ))"
        )
    )
    state_norm = norm(ψ)
    isfinite(state_norm) && state_norm > 0 || throw(
        ArgumentError(
            "Layer III state norm must be finite and nonzero"
        )
    )
    mul!(coeff, space.Q', ψ)
    copyto!(scratch, ψ)
    mul!(scratch, space.Q, coeff, -one(CF), one(CF))
    return Float64(norm(scratch) / state_norm)
end

function _validated_layer3_state(state, label::AbstractString, dimension::Int)
    raw = state isa AbstractVector ? state :
        hasproperty(state, :data) ? getproperty(state, :data) :
        throw(ArgumentError("$label must be a state vector or QuantumObject ket"))
    ψ = try
        Vector{CF}(raw)
    catch
        throw(ArgumentError("$label must be convertible to Vector{ComplexF64}"))
    end
    length(ψ) == dimension || throw(
        DimensionMismatch(
            "$label has length $(length(ψ)); expected $dimension"
        )
    )
    all(isfinite, ψ) || throw(ArgumentError("$label must contain only finite values"))
    state_norm = norm(ψ)
    isfinite(state_norm) || throw(ArgumentError("$label norm must be finite"))
    state_norm > 0 || throw(ArgumentError("$label must be nonzero"))
    ψ ./= state_norm
    return ψ
end

function _normalized_layer3_sizes(sizes, ngauges::Int, dimension::Int)
    if sizes isa Integer && !(sizes isa Bool)
        1 <= sizes <= dimension || throw(
            ArgumentError(
                "layer3_sizes must lie in 1:$dimension"
            )
        )
        return fill(Int(sizes), ngauges)
    end
    sizes isa AbstractVector || throw(
        ArgumentError(
            "layer3_sizes must be one integer or one integer per gauge"
        )
    )
    length(sizes) == ngauges || throw(
        ArgumentError(
            "layer3_sizes must contain one size or one size per gauge"
        )
    )
    all(size -> size isa Integer && !(size isa Bool), sizes) ||
        throw(ArgumentError("layer3_sizes must contain integers"))
    all(size -> 1 <= size <= dimension, sizes) ||
        throw(ArgumentError("layer3_sizes must lie in 1:$dimension"))
    return Int.(sizes)
end

# Paper: J_{μ,red}^{g′←g} and quadratic forms for A_g′ and r² (Eqs. 22, C.4).
function _build_local_jump_routes(
        system::_GaugeSystem,
        spaces::Vector{_LocalEigenspace}
    )
    ngauges = length(system.gauges)
    return map(1:ngauges) do source
        source_space = spaces[source]
        map(1:system.gauges[source].cache.Nc) do channel
            action = source_space.reduced.A_I[channel]
            map(1:ngauges) do destination
                destination_space = spaces[destination]
                rate = zeros(CF, size(action, 2), size(action, 2))
                for collapse in system.gauges[destination].cache.C
                    shifted_image = collapse * action
                    rate .+= shifted_image' * shifted_image
                end
                projected = destination_space.Q' * action
                missing = action - destination_space.Q * projected
                residual = missing' * missing
                _LocalJumpRoute(
                    rate, _matrix_inf_norm(rate), residual,
                    _matrix_inf_norm(residual), destination_space.R \ projected
                )
            end
        end
    end
end

# Paper: {𝒦_m^(g), J_{μ,red}^{g′←g}} (Eqs. 20, 22).
function _prepare_layer3(problem::_Layer1Prepared, ψ0, sizes, tolerance)
    isfinite(tolerance) && tolerance > 0 || throw(
        ArgumentError(
            "residual_tolerance must be finite and positive"
        )
    )
    system = problem.system
    dimension = system.cache.N
    _validated_layer3_state(ψ0, "initial state", dimension)
    requested = _normalized_layer3_sizes(sizes, length(system.gauges), dimension)
    spaces = [
        _build_local_eigenspace(
            system.gauges[gauge].cache,
            requested[gauge]
        ) for gauge in eachindex(system.gauges)
    ]
    routes = _build_local_jump_routes(system, spaces)
    return _Layer3Prepared(system, spaces, routes, Float64(tolerance))
end

# Paper: |ψ_m^(g)⟩ coordinates, accepted if r_m^(g) ≤ r_tol (Eq. 21).
function _accept_layer3_state!(
        diagnostics::_GaugeDiagnostics, wb::_WorkBuffers,
        layer3::_Layer3Prepared, gauge::Int, state::AbstractVector{CF}
    )
    space = layer3.spaces[gauge]
    size = length(space.reduced.I)
    coefficients = view(wb.cplus, 1:size)
    residual = _relative_residual!(coefficients, wb.Gc, space, state)
    diagnostics.maximum_residual = max(diagnostics.maximum_residual, residual)
    if residual <= layer3.tolerance
        ldiv!(space.R, coefficients)
        _normalize_coordinates!(
            coefficients, view(wb.Gc, 1:size),
            space.reduced.G_I, space.reduced.Gnorm
        )
        diagnostics.accepted_projections += 1
        return true
    end
    diagnostics.fallback_segments += 1
    return false
end

# Paper: A^(g′)(ϕ/‖ϕ‖), ϕ = C_μ^(g)V_m^(g)c⁻ (Eqs. 11, C.2).
function _local_jump_activities!(
        rates::AbstractVector{Float64},
        scratch::AbstractVector{CF}, layer3::_Layer3Prepared, source::Int,
        channel::Int, evolved::AbstractVector{CF}, jump_weight::Float64
    )
    norm2 = sum(abs2, evolved)
    for destination in eachindex(layer3.spaces)
        route = layer3.routes[source][channel][destination]
        rates[destination] = _checked_quadratic!(
            scratch, evolved, route.rate,
            route.rate_norm * norm2, "Layer III routed activity"
        ) / jump_weight
    end
    return rates
end

# Paper: |ψ(t)⟩ using 𝒦_m^(g), with residual-gated fallback (Section 3.3).
function _run_layer3_trajectory!(
        acc::_TrajectoryAccumulator,
        diagnostics::_GaugeDiagnostics, prepared::_Layer1Prepared,
        layer3::_Layer3Prepared, ψ0::AbstractVector{CF},
        tlist::AbstractVector{Float64}, T::Float64, rng, wb::_WorkBuffers;
        survival_rtol::Real = 1.0e-10, time_rtol::Real = 1.0e-12,
        time_atol::Real = 0.0, max_jumps::Int = 1_000_000,
        first_passage_maxiter::Int = 100,
        first_passage_method::Symbol = :log_survival,
        trajectory_predictor::Union{Nothing, _FirstPassagePredictor} = nothing,
        final_states::Union{Nothing, AbstractMatrix{CF}} = nothing,
        final_state_column::Int = 0, times_states::AbstractVector{Float64} = Float64[],
        record::Union{Nothing, _TrajectoryRecord} = nothing,
        col_gauge::Union{Nothing, Vector{Int}} = nothing,
        time_origin::Float64 = 0.0
    )
    system = prepared.system
    system === layer3.system || throw(
        ArgumentError(
            "Layer III preparation does not belong to these gauges"
        )
    )
    ngauges = length(system.gauges)
    length(wb.w) >= ngauges || throw(
        ArgumentError(
            "work buffer needs at least $ngauges routed-activity slots"
        )
    )
    copyto!(wb.ψ, ψ0)
    gauge = _initial_gauge_coordinates!(wb, prepared, wb.ψ)
    using_local = _accept_layer3_state!(diagnostics, wb, layer3, gauge, wb.ψ)
    next_time = 1
    next_state = 1
    time = 0.0
    jumps = 0

    @inbounds while next_time <= length(tlist)
        jumps >= max_jumps && error("maximum number of jumps reached")
        local_cache = system.gauges[gauge].cache
        threshold = rand(rng)
        remaining = T - time

        if !using_local
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
                duration = T - time
                diagnostics.residence_time[gauge] += duration
                diagnostics.fallback_residence_time += duration
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
            diagnostics.fallback_residence_time += result.tau
            next_time = hi + 1
            next_state = state_hi + 1
            _phase_evolve!(wb.Dc, local_cache.Λ, wb.c, result.tau)
            channel, next_gauge = _apply_exact_jump_and_route!(
                wb, prepared, gauge, wb.Dc, rng
            )
            acc.jumps_by_channel[channel] += 1
            diagnostics.jumps_by_gauge[gauge] += 1
            diagnostics.fallback_jumps += 1
            record === nothing || begin
                push!(record.col_times, event_time + time_origin)
                push!(record.col_which, channel)
            end
            col_gauge === nothing || push!(col_gauge, gauge)
            if next_gauge != gauge
                copyto!(wb.c, wb.cplus)
                diagnostics.switches += 1
            end
            gauge = next_gauge
            using_local = _accept_layer3_state!(diagnostics, wb, layer3, gauge, wb.ψ)
            time = event_time
            jumps += 1
            continue
        end

        space = layer3.spaces[gauge]
        reduced = space.reduced
        size = length(reduced.I)
        coordinates = view(wb.cplus, 1:size)
        evolved = view(wb.Dc, 1:size)
        scratch = view(wb.Gc, 1:size)
        _normalize_coordinates!(coordinates, scratch, reduced.G_I, reduced.Gnorm)
        survival = duration -> _survival_probability!(
            evolved, scratch,
            coordinates, reduced.λ_I, reduced.G_I, duration, reduced.Gnorm
        )
        rate = duration -> _jump_rate!(
            scratch,
            _phase_evolve!(evolved, reduced.λ_I, coordinates, duration),
            reduced.K_I, reduced.Knorms
        )
        result = _find_scalar_first_passage(
            survival, rate, threshold, remaining;
            survival_rtol, time_rtol, time_atol,
            maxiter = first_passage_maxiter,
            root_variant = first_passage_method
        )
        if !result.jumped
            hi = searchsortedlast(tlist, T)
            _record_local!(
                acc, local_cache, reduced, coordinates, tlist, time,
                next_time, hi, evolved, scratch, wb.ψ, wb.Gc; record
            )
            state_hi = searchsortedlast(times_states, T)
            _record_state_window!(
                acc, record, reduced.λ_I, reduced.V_I,
                coordinates, times_states, time, next_state, state_hi,
                evolved, wb.ψ
            )
            diagnostics.residence_time[gauge] += T - time
            next_time = hi + 1
            next_state = state_hi + 1
            break
        end

        event_time = time + result.tau
        hi = searchsortedfirst(tlist, event_time) - 1
        _record_local!(
            acc, local_cache, reduced, coordinates, tlist, time,
            next_time, hi, evolved, scratch, wb.ψ, wb.Gc; record
        )
        state_hi = searchsortedfirst(times_states, event_time) - 1
        _record_state_window!(
            acc, record, reduced.λ_I, reduced.V_I,
            coordinates, times_states, time, next_state, state_hi, evolved, wb.ψ
        )
        diagnostics.residence_time[gauge] += result.tau
        next_time = hi + 1
        next_state = state_hi + 1
        _phase_evolve!(evolved, reduced.λ_I, coordinates, result.tau)
        channel, jump_weight = _sample_local_channel!(
            wb.w, scratch,
            reduced, evolved, local_cache.Nc, rng
        )
        acc.jumps_by_channel[channel] += 1
        diagnostics.jumps_by_gauge[gauge] += 1
        record === nothing || begin
            push!(record.col_times, event_time + time_origin)
            push!(record.col_which, channel)
        end
        col_gauge === nothing || push!(col_gauge, gauge)

        source = gauge
        rates = view(wb.w, 1:ngauges)
        _local_jump_activities!(
            rates, scratch, layer3, source, channel,
            evolved, jump_weight
        )
        gauge = _hysteretic_activity(rates, source, prepared.hysteresis)
        gauge == source || (diagnostics.switches += 1)
        route = layer3.routes[source][channel][gauge]
        residual = sqrt(
            _checked_quadratic!(
                scratch, evolved, route.residual,
                route.residual_norm * sum(abs2, evolved),
                "Layer III jump-image residual"
            ) / jump_weight
        )
        diagnostics.maximum_residual = max(diagnostics.maximum_residual, residual)
        if residual <= layer3.tolerance
            destination = layer3.spaces[gauge]
            destination_size = length(destination.reduced.I)
            destination_coordinates = view(wb.cplus, 1:destination_size)
            mul!(destination_coordinates, route.coordinates, evolved)
            destination_coordinates ./= sqrt(jump_weight)
            _normalize_coordinates!(
                destination_coordinates,
                view(wb.Gc, 1:destination_size), destination.reduced.G_I,
                destination.reduced.Gnorm
            )
            diagnostics.accepted_projections += 1
            using_local = true
        else
            diagnostics.fallback_segments += 1
            mul!(wb.ψ, reduced.A_I[channel], evolved)
            wb.ψ ./= sqrt(jump_weight)
            exact_cache = system.gauges[gauge].cache
            _solve_coordinates!(wb.c, exact_cache, wb.ψ)
            _normalize_coordinates!(
                wb.c, wb.Gc, exact_cache.G,
                exact_cache.Gnorm
            )
            using_local = false
        end
        time = event_time
        jumps += 1
    end

    if final_states !== nothing
        if using_local
            space = layer3.spaces[gauge]
            coordinates = view(wb.cplus, 1:length(space.reduced.I))
            _phase_evolve!(
                view(wb.Dc, 1:length(coordinates)),
                space.reduced.λ_I, coordinates, T - time
            )
            mul!(
                wb.ψ, space.reduced.V_I,
                view(wb.Dc, 1:length(coordinates))
            )
        else
            _phase_evolve!(wb.Dc, system.gauges[gauge].cache.Λ, wb.c, T - time)
            mul!(wb.ψ, system.gauges[gauge].cache.V, wb.Dc)
        end
        state_norm = norm(wb.ψ)
        final_states[:, final_state_column] .= wb.ψ ./ state_norm
    end
    acc.ntraj += 1
    acc.njumps_total += jumps
    return jumps
end
