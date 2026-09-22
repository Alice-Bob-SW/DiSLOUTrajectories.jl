# Private Layer I: exact gauge preparation and total-activity routing.

struct _Gauge
    shifts::Vector{CF}
    cache::_DiagonalCache
    C_sparse::Vector{SparseMatrixCSC{CF, Int}}
    meanV::Vector{Matrix{CF}}
    secondV::Vector{Matrix{CF}}
    secondV_norms::Vector{Float64}
end

struct _GaugeSystem
    cache::_DiagonalCache
    gauges::Vector{_Gauge}
    C_sparse::Vector{SparseMatrixCSC{CF, Int}}
    shift_channels::Vector{Int}
    centers::Matrix{CF}
end

struct _Layer1Prepared
    system::_GaugeSystem
    hysteresis::Float64
end

# Paper: {H^(g), C_μ^(g), V^(g), Λ^(g), G^(g)} (Eqs. 9, 15–17).
function _prepare_layer1(
        H, c_ops, e_ops, data::_GaugeData, hysteresis::Real;
        observable_storage::Symbol = :dense, cache_policy...
    )
    isfinite(hysteresis) && 0 < hysteresis <= 1 || throw(
        ArgumentError(
            "hysteresis must satisfy 0 < hysteresis <= 1, got $hysteresis"
        )
    )
    H_input = H isa AbstractMatrix ? QuantumObject(Matrix{CF}(H)) : H
    C_input = all(Cμ -> Cμ isa AbstractMatrix, c_ops) ?
        [QuantumObject(Matrix{CF}(Cμ)) for Cμ in c_ops] : c_ops
    Hmat, dimensions = _operator_matrix(H_input, "H"; hermitian = true)
    C = Matrix{CF}[
        _operator_matrix(
            op, "c_ops[$mu]"; expected_dimensions = dimensions
        )[1]
            for (mu, op) in enumerate(C_input)
    ]
    Z = _validated_observables(
        e_ops, size(Hmat, 1), dimensions, observable_storage
    )
    return _prepare_layer1_matrices(
        Hmat, C, Z, dimensions, data, hysteresis;
        observable_storage, cache_policy...
    )
end

# Paper: V^(g)†C_μV^(g) and V^(g)†C_μ†C_μV^(g) (unshifted C_μ).
function _physical_moment_matrices(
        cache::_DiagonalCache,
        shifts::AbstractVector{CF}
    )
    length(shifts) == cache.Nc || throw(
        DimensionMismatch(
            "gauge shifts must have length $(cache.Nc), got $(length(shifts))"
        )
    )
    meanV = Vector{Matrix{CF}}(undef, cache.Nc)
    secondV = Vector{Matrix{CF}}(undef, cache.Nc)
    for channel in 1:cache.Nc
        shift = shifts[channel]
        mean = cache.V' * cache.A[channel]
        second = similar(mean)
        second .= cache.M[channel] .- shift .* mean' .-
            conj(shift) .* mean .+ abs2(shift) .* cache.G
        mean .-= shift .* cache.G
        meanV[channel] = mean
        secondV[channel] = second
    end
    return meanV, secondV
end

# Paper: {H^(g), C_μ^(g), V^(g), Λ^(g), G^(g)} (Eqs. 9, 15–17).
function _prepare_layer1_matrices(
        Hmat::Matrix{CF}, C::Vector{Matrix{CF}},
        Z::_ObservableMatrices, dimensions, data::_GaugeData,
        hysteresis::Real; observable_storage::Symbol, cache_policy...
    )
    isfinite(hysteresis) && 0 < hysteresis <= 1 || throw(
        ArgumentError(
            "hysteresis must satisfy 0 < hysteresis <= 1, got $hysteresis"
        )
    )
    size(data.shifts, 1) == length(C) || throw(
        DimensionMismatch(
            "gauge shifts must have $(length(C)) rows, got $(size(data.shifts, 1))"
        )
    )
    C_sparse = observable_storage === :sparse ?
        SparseMatrixCSC{CF, Int}[sparse(Cμ) for Cμ in C] :
        SparseMatrixCSC{CF, Int}[]
    gauges = _Gauge[]
    for b in axes(data.shifts, 2)
        shifts = Vector{CF}(@view data.shifts[:, b])
        Hξ, Cξ = _shifted_problem(Hmat, C, shifts)
        cache = _diagonal_cache_from_matrices(
            Hξ, Cξ;
            Z, observable_storage, cache_policy...
        )
        cache.dimensions = deepcopy(dimensions)
        Cξ_sparse = isempty(C_sparse) ? SparseMatrixCSC{CF, Int}[] :
            SparseMatrixCSC{CF, Int}[
                sparse(C_sparse[channel] + shifts[channel] * I)
                for channel in eachindex(C_sparse)
            ]
        meanV, secondV = _physical_moment_matrices(cache, shifts)
        secondV_norms = Float64[_matrix_inf_norm(A) for A in secondV]
        push!(
            gauges, _Gauge(
                shifts, cache, Cξ_sparse, meanV, secondV, secondV_norms
            )
        )
    end
    bu = _GaugeSystem(
        first(gauges).cache, gauges,
        C_sparse, collect(1:length(C)), copy(data.centers)
    )
    return _Layer1Prepared(bu, Float64(hysteresis))
end

# Paper: A^(g)(ψ) (Eq. 11).
@inline function _activity_from_moments(
        g::_Gauge, means::AbstractVector,
        second::AbstractVector, nshift::Int
    )
    r = 0.0
    scale = 0.0
    @inbounds for mu in eachindex(second)
        second_term = real(second[mu])
        r += second_term
        scale += abs(second_term)
    end
    @inbounds for j in 1:nshift
        ζ = g.shifts[j]
        cross_term = 2real(conj(ζ) * means[j])
        shift_term = abs2(ζ)
        r += cross_term + shift_term
        scale += abs(cross_term) + shift_term
    end
    return _checked_quadratic(r, scale, g.cache.N, "gauge event activity")
end

# The physical moments ⟨C_μ⟩_ψ and ⟨C_μ†C_μ⟩_ψ are gauge independent: they are the
# same unshifted matrices seen in each gauge's eigenbasis, and a gauge enters
# _activity_from_moments only through its scalar shifts. One coordinate solve
# therefore scores every gauge of a routing scan.
# Robustness relaxation: only the *current* gauge's secondV quadratic forms now
# run through _checked_quadratic!, so a catastrophically ill-conditioned distant
# gauge no longer raises during a scan the way the old solve-per-gauge loop did.
# Each gauge's assembled activity is still guarded by the scalar
# _checked_quadratic at the end of _activity_from_moments.
@inline function _checked_moments!(
        means::AbstractVector{CF},
        second::AbstractVector{Float64}, tmp::AbstractVector{CF},
        c::AbstractVector{CF}, g::_Gauge
    )
    den = _branch_moments!(means, second, tmp, c, g)
    isfinite(den) && den > 0 || throw(DomainError(den, "state must be nonzero"))
    return means, second
end

# Paper: g₀ = argmin_g A_g(ψ) (Eq. 12).
@inline function _lowest_gauge(
        gauges::Vector{_Gauge}, means::AbstractVector,
        second::AbstractVector, nshift::Int
    )::Int
    best = 1
    best_activity = _activity_from_moments(gauges[1], means, second, nshift)
    @inbounds for gauge in 2:length(gauges)
        activity = _activity_from_moments(gauges[gauge], means, second, nshift)
        activity < best_activity && ((best, best_activity) = (gauge, activity))
    end
    return best
end

# Paper: g′ with A_g′(ψ) < η A_g(ψ), else g (Eq. 13).
@inline function _hysteretic_gauge(
        gauges::Vector{_Gauge}, current::Int,
        means::AbstractVector, second::AbstractVector, nshift::Int,
        hysteresis::Float64
    )::Int
    current_activity = _activity_from_moments(gauges[current], means, second, nshift)
    iszero(current_activity) && return current
    best = current
    best_activity = current_activity
    @inbounds for gauge in eachindex(gauges)
        gauge == current && continue
        activity = _activity_from_moments(gauges[gauge], means, second, nshift)
        activity < best_activity && ((best, best_activity) = (gauge, activity))
    end
    return best != current && best_activity < hysteresis * current_activity ? best : current
end

# Paper: g₀ = argmin_g A_g(ψ) (Eq. 12).
@inline function _lowest_activity(rates)
    best = 1
    best_rate = rates[1]
    for gauge in 2:length(rates)
        rates[gauge] < best_rate && ((best, best_rate) = (gauge, rates[gauge]))
    end
    return best
end

# Paper: g′ with A_g′(ψ) < η A_g(ψ), else g (Eq. 13).
@inline function _hysteretic_activity(rates, current::Int, hysteresis::Real)
    current_rate = rates[current]
    iszero(current_rate) && return current
    best = _lowest_activity(rates)
    return best != current && rates[best] < hysteresis * current_rate ? best : current
end

# Paper: g₀ (Eq. 12).
function _initial_gauge!(wb::_WorkBuffers, prepared::_Layer1Prepared, ψ)::Int
    gauges = prepared.system.gauges
    g = first(gauges)
    length(ψ) == g.cache.N || throw(DimensionMismatch("state has the wrong dimension"))
    _solve_coordinates!(wb.cplus, g.cache, ψ)
    nshift = length(g.meanV)
    means = view(wb.moments, 1:nshift)
    second = view(wb.w, 1:g.cache.Nc)
    _checked_moments!(means, second, wb.Gc, wb.cplus, g)
    return _lowest_gauge(gauges, means, second, nshift)
end

# wb variant: _apply_exact_jump_coordinates! has just written the post-jump
# coordinates in gauge `current` into wb.c, so the scan needs no solve at all.
# Precondition: wb.c holds the coordinates of the state being routed, expressed
# in gauge `current`. The physical state is deliberately not a parameter — it was
# never used beyond a length check, and taking it would suggest a consistency
# guarantee this function cannot make.
# Paper: g′ after the jump (Eq. 13).
function _postjump_gauge!(
        wb::_WorkBuffers, prepared::_Layer1Prepared,
        current::Int
    )::Int
    gauges = prepared.system.gauges
    1 <= current <= length(gauges) ||
        throw(BoundsError(prepared.system.gauges, current))
    length(gauges) == 1 && return current
    g = gauges[current]
    nshift = length(g.meanV)
    # Only wb.w[1:Nc] belongs to this scan. Layer III parks routed per-gauge
    # activities in wb.w[1:ngauges], so with ngauges > Nc the tail slots hold
    # stale values that must never reach _activity_from_moments as extra ⟨C_μ†C_μ⟩_ψ.
    means = view(wb.moments, 1:nshift)
    second = view(wb.w, 1:g.cache.Nc)
    _checked_moments!(means, second, wb.Gc, wb.c, g)
    return _hysteretic_gauge(
        gauges, current, means, second, nshift,
        prepared.hysteresis
    )
end

# Paper: ⟨C_μ⟩_ψ, ⟨C_μ†C_μ⟩_ψ, and c†Gc (Eqs. 11, 17).
@inline function _branch_moments!(
        means::AbstractVector{CF},
        second::AbstractVector{Float64}, tmp::AbstractVector{CF},
        c::AbstractVector{CF}, g::_Gauge
    )
    den = _real_quadratic!(tmp, c, g.cache.G)
    @inbounds for j in eachindex(g.meanV)
        mul!(tmp, g.meanV[j], c)
        means[j] = dot(c, tmp) / den
    end
    norm2 = sum(abs2, c)
    @inbounds for mu in eachindex(g.secondV)
        second[mu] = _checked_quadratic!(
            tmp, c, g.secondV[mu],
            g.secondV_norms[mu] * norm2, "physical collapse activity"
        ) / den
    end
    return den
end


function _eigensystem_backend(prepared::_Layer1Prepared)
    gauges = prepared.system.gauges
    backend = first(gauges).cache.backend
    for gauge in @view gauges[2:end]
        gauge.cache.backend === backend || return :mixed
    end
    return backend
end

# Paper: g₀ and c = V^(g₀)⁻¹|ψ(0)⟩ (Eqs. 12, 16).
function _initial_gauge_coordinates!(
        wb::_WorkBuffers, prepared::_Layer1Prepared,
        ψ::AbstractVector{CF}
    )::Int
    gauge = _initial_gauge!(wb, prepared, ψ)
    local_cache = prepared.system.gauges[gauge].cache
    _solve_coordinates!(wb.c, local_cache, ψ)
    _normalize_coordinates!(wb.c, wb.Gc, local_cache.G, local_cache.Gnorm)
    return gauge
end

# Paper: g′ and normalized c⁺ in V^(g′) (Eqs. 13, 16).
function _postjump_gauge_coordinates!(
        wb::_WorkBuffers,
        prepared::_Layer1Prepared, current::Int,
        postjump_state::AbstractVector{CF}
    )::Int
    next = _postjump_gauge!(wb, prepared, current)
    if next != current
        new = prepared.system.gauges[next]
        _solve_coordinates!(wb.cplus, new.cache, postjump_state)
        _normalize_coordinates!(wb.cplus, wb.Gc, new.cache.G, new.cache.Gnorm)
    end
    return next
end
