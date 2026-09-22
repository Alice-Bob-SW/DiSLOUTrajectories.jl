# In-segment observable recording into a streaming accumulator.
#
# Paper: ⟨O⟩_ψ(t) = c(t)†O_m c(t) / [c(t)†G_m c(t)] (Section 3.3.2).
# Code Z_e denotes O, Z_I denotes O_m, and G_I denotes G_m. The segment
# duration is t = t_k - t0; t0 and tlist use the same trajectory time origin.
# Each tlist index is recorded exactly once across the whole trajectory (the trajectory
# loop hands each segment a contiguous index window `lo:hi`). Expectations use a
# cancellation-resistant Welford accumulator.

"""
    _TrajectoryAccumulator(Ne, Nt)

Streaming Welford state for `Ne` observables over `Nt` grid points.
"""
mutable struct _TrajectoryAccumulator
    mean::Matrix{CF}               # Ne × Nt
    M2::Matrix{Float64}            # Ne × Nt
    ntraj::Int
    njumps_total::Int
    jumps_by_channel::Vector{Int}  # length Nc
    state_sums::Union{Nothing, Vector{Matrix{CF}}}
end
_TrajectoryAccumulator(Ne::Int, Nt::Int, Nc::Int) =
    _TrajectoryAccumulator(
    zeros(CF, Ne, Nt), zeros(Float64, Ne, Nt),
    0, 0, zeros(Int, Nc), nothing
)

mutable struct _TrajectoryRecord
    states::Union{Nothing, Vector{Vector{CF}}}
    expect::Union{Nothing, Matrix{CF}}
    col_times::Vector{Float64}
    col_which::Vector{Int}
end

_TrajectoryRecord(N::Int, Ne::Int, Nt::Int, Ns::Int, save_trajectories::Bool) =
    _TrajectoryRecord(
    save_trajectories ? [zeros(CF, N) for _ in 1:Ns] : nothing,
    save_trajectories ? zeros(CF, Ne, Nt) : nothing,
    Float64[],
    Int[],
)

# Paper: ensemble ⟨O⟩ from trajectory expectations (Section 2).
function _merge_stats!(
        a::_TrajectoryAccumulator, bn::Int,
        bmean::AbstractMatrix{<:Number}, bM2::AbstractMatrix{<:Real}
    )
    bn == 0 && return a
    an = a.ntraj
    if an == 0
        copyto!(a.mean, bmean)
        copyto!(a.M2, bM2)
        a.ntraj = bn
        return a
    end
    n = an + bn
    mean_weight = bn / n
    cross_weight = an * bn / n
    @inbounds for i in eachindex(a.mean, a.M2, bmean, bM2)
        δ = bmean[i] - a.mean[i]
        a.mean[i] += δ * mean_weight
        a.M2[i] += bM2[i] + abs2(δ) * cross_weight
    end
    a.ntraj = n
    return a
end

# Paper: unnormalized ensemble sum for ρ_MC(t) (Section 4.1).
function _merge_state_sums!(
        a::_TrajectoryAccumulator,
        sums::Union{Nothing, Vector{Matrix{CF}}}
    )
    sums === nothing && return a
    if a.state_sums === nothing
        a.state_sums = copy.(sums)
    else
        length(a.state_sums) == length(sums) ||
            throw(DimensionMismatch("state-save grids do not match"))
        for k in eachindex(sums)
            destination = a.state_sums[k]
            source = sums[k]
            @inbounds for i in eachindex(destination, source)
                destination[i] += source[i]
            end
        end
    end
    return a
end

# Paper: ensemble ⟨O⟩ and sums for ρ_MC(t) (Sections 2, 4).
function _merge_accumulator!(a::_TrajectoryAccumulator, b::_TrajectoryAccumulator)
    _merge_stats!(a, b.ntraj, b.mean, b.M2)
    a.njumps_total += b.njumps_total
    a.jumps_by_channel .+= b.jumps_by_channel
    _merge_state_sums!(a, b.state_sums)
    return a
end

# Paper: ensemble ⟨O⟩ from each ⟨O⟩_ψ(t) sample (Section 2).
@inline function _record_point!(
        acc::_TrajectoryAccumulator, e::Int, idx::Int, v::Number,
        record::Union{Nothing, _TrajectoryRecord} = nothing
    )
    record === nothing || record.expect === nothing || (record.expect[e, idx] = v)
    n = acc.ntraj + 1
    δ = v - acc.mean[e, idx]
    acc.mean[e, idx] += δ / n
    acc.M2[e, idx] += real(conj(δ) * (v - acc.mean[e, idx]))
    return nothing
end

# Paper: |ψ(t)⟩⟨ψ(t)| contributions to ρ_MC(t) (Section 4.1).
function _record_state!(
        acc::_TrajectoryAccumulator, idx::Int, ψ::StridedVector{CF},
        record::Union{Nothing, _TrajectoryRecord} = nothing
    )
    invnorm = inv(norm(ψ))
    if acc.state_sums === nothing
        acc.state_sums = [zeros(CF, length(ψ), length(ψ)) for _ in 1:idx]
    elseif length(acc.state_sums) < idx
        append!(
            acc.state_sums,
            [
                zeros(CF, length(ψ), length(ψ))
                    for _ in (length(acc.state_sums) + 1):idx
            ]
        )
    end
    BLAS.ger!(CF(invnorm^2), ψ, ψ, acc.state_sums[idx])
    if record !== nothing && record.states !== nothing
        copyto!(record.states[idx], ψ)
        record.states[idx] .*= invnorm
    end
    return nothing
end

# Paper: |ψ(t_k)⟩ and projectors for ρ_MC(t_k) (Eq. 16, Section 4.1).
function _record_state_window!(
        acc::_TrajectoryAccumulator,
        record::Union{Nothing, _TrajectoryRecord}, eigenvalues, basis,
        coordinates::AbstractVector{CF}, times, origin::Float64, lo::Int, hi::Int,
        evolved::AbstractVector{CF}, state::AbstractVector{CF}
    )
    lo > hi && return nothing
    @inbounds for index in lo:hi
        _phase_evolve!(evolved, eigenvalues, coordinates, times[index] - origin)
        mul!(state, basis, evolved)
        _record_state!(acc, index, state, record)
    end
    return nothing
end


# Reduced subspace recording over grid window lo:hi (times tlist[lo:hi]). `Dc` is a
# caller-provided scratch buffer (length m).
# Paper: ⟨O⟩_ψ(t) = c(t)†O_m c(t) / [c(t)†G_m c(t)].
function _record_local_dense!(
        acc::_TrajectoryAccumulator, rc::_ReducedCache,
        c::AbstractVector{CF}, tlist,
        t0::Float64, lo::Int, hi::Int,
        Dc::AbstractVector{CF}, tmp::AbstractVector{CF};
        record::Union{Nothing, _TrajectoryRecord} = nothing
    )
    lo > hi && return nothing
    Ne = length(rc.Z_I)
    @inbounds for idx in lo:hi
        _phase_evolve!(Dc, rc.λ_I, c, tlist[idx] - t0)
        den = _real_quadratic!(tmp, Dc, rc.G_I)
        for e in 1:Ne
            value = _quadratic!(tmp, Dc, rc.Z_I[e]) / den
            _record_point!(acc, e, idx, value, record)
        end
    end
    return nothing
end

# Paper: ⟨O⟩_ψ(t) from |ψ̃(t)⟩ = V_m c(t), divided by s(t).
function _record_local_sparse!(
        acc::_TrajectoryAccumulator,
        cache::_DiagonalCache, rc::_ReducedCache,
        c::AbstractVector{CF}, tlist,
        t0::Float64, lo::Int, hi::Int,
        Dc::AbstractVector{CF},
        metric_tmp::AbstractVector{CF},
        ψ::AbstractVector{CF},
        observable_tmp::AbstractVector{CF};
        record::Union{Nothing, _TrajectoryRecord} = nothing
    )
    lo > hi && return nothing
    observables = cache.Z::Vector{SparseMatrixCSC{CF, Int}}
    @inbounds for idx in lo:hi
        _phase_evolve!(Dc, rc.λ_I, c, tlist[idx] - t0)
        den = _real_quadratic!(metric_tmp, Dc, rc.G_I)
        mul!(ψ, rc.V_I, Dc)
        for e in 1:cache.Ne
            mul!(observable_tmp, observables[e], ψ)
            _record_point!(acc, e, idx, dot(ψ, observable_tmp) / den, record)
        end
    end
    return nothing
end

# Paper: ⟨O⟩_ψ(t) in 𝒦_m^(g) (Section 3.3.2).
function _record_local!(
        acc::_TrajectoryAccumulator, cache::_DiagonalCache,
        rc::_ReducedCache, c::AbstractVector{CF}, tlist,
        t0::Float64, lo::Int, hi::Int,
        Dc::AbstractVector{CF}, metric_tmp::AbstractVector{CF},
        ψ::AbstractVector{CF}, observable_tmp::AbstractVector{CF};
        record::Union{Nothing, _TrajectoryRecord} = nothing
    )
    if cache.observable_storage === :dense
        return _record_local_dense!(
            acc, rc, c, tlist, t0, lo, hi,
            Dc, metric_tmp; record
        )
    end
    return _record_local_sparse!(
        acc, cache, rc, c, tlist, t0, lo, hi,
        Dc, metric_tmp, ψ, observable_tmp; record
    )
end

# Full diagonal-basis recording over grid window lo:hi. `Dc` is a caller-provided scratch
# buffer (length N); the allocating method below is kept for tests/standalone use.
# Paper: ⟨O⟩_ψ(t) = c(t)†V†OV c(t) / s(t).
function _record_exact_dense!(
        acc::_TrajectoryAccumulator, cache::_DiagonalCache, c::AbstractVector{CF},
        tlist, t0::Float64, lo::Int, hi::Int,
        Dc::AbstractVector{CF}, tmp::AbstractVector{CF};
        record::Union{Nothing, _TrajectoryRecord} = nothing
    )
    lo > hi && return nothing
    @inbounds for idx in lo:hi
        _phase_evolve!(Dc, cache.Λ, c, tlist[idx] - t0)
        den = _real_quadratic!(tmp, Dc, cache.G)
        for e in 1:cache.Ne
            value = _quadratic!(tmp, Dc, cache.ZV[e]) / den
            _record_point!(acc, e, idx, value, record)
        end
    end
    return nothing
end
# Paper: ⟨O⟩_ψ(t) from |ψ̃(t)⟩ = Vc(t), divided by s(t).
function _record_exact_sparse!(
        acc::_TrajectoryAccumulator, cache::_DiagonalCache,
        c::AbstractVector{CF}, tlist, t0::Float64, lo::Int, hi::Int,
        Dc::AbstractVector{CF}, tmp::AbstractVector{CF}, ψ::AbstractVector{CF};
        record::Union{Nothing, _TrajectoryRecord} = nothing
    )
    lo > hi && return nothing
    observables = cache.Z::Vector{SparseMatrixCSC{CF, Int}}
    @inbounds for idx in lo:hi
        _phase_evolve!(Dc, cache.Λ, c, tlist[idx] - t0)
        den = _real_quadratic!(tmp, Dc, cache.G)
        mul!(ψ, cache.V, Dc)
        for e in 1:cache.Ne
            mul!(tmp, observables[e], ψ)
            _record_point!(acc, e, idx, dot(ψ, tmp) / den, record)
        end
    end
    return nothing
end

# Paper: ⟨O⟩_ψ(t) in the full eigenbasis (Section 3.2.1).
function _record_exact!(
        acc::_TrajectoryAccumulator, cache::_DiagonalCache,
        c::AbstractVector{CF}, tlist, t0::Float64, lo::Int, hi::Int,
        Dc::AbstractVector{CF}, tmp::AbstractVector{CF}, ψ::AbstractVector{CF};
        record::Union{Nothing, _TrajectoryRecord} = nothing
    )
    if cache.observable_storage === :dense
        return _record_exact_dense!(
            acc, cache, c, tlist, t0, lo, hi,
            Dc, tmp; record
        )
    end
    return _record_exact_sparse!(
        acc, cache, c, tlist, t0, lo, hi,
        Dc, tmp, ψ; record
    )
end

# Paper: ⟨O⟩_ψ(t) in the full eigenbasis (Section 3.2.1).
_record_exact!(
    acc::_TrajectoryAccumulator, cache::_DiagonalCache, c::AbstractVector{CF},
    tlist, t0::Float64, lo::Int, hi::Int;
    record::Union{Nothing, _TrajectoryRecord} = nothing
) =
    _record_exact!(acc, cache, c, tlist, t0, lo, hi, similar(c), similar(c), similar(c); record)

mutable struct _GaugeDiagnostics
    ngauges::Int
    switches::Int
    residence_time::Vector{Float64}
    jumps_by_gauge::Vector{Int}
    accepted_projections::Int
    fallback_segments::Int
    fallback_residence_time::Float64
    fallback_jumps::Int
    maximum_residual::Float64
end

_GaugeDiagnostics(ngauges::Int) = _GaugeDiagnostics(
    ngauges, 0,
    zeros(Float64, ngauges), zeros(Int, ngauges), 0, 0, 0.0, 0, 0.0
)

function _merge_gauge_diagnostics!(
        left::_GaugeDiagnostics,
        right::_GaugeDiagnostics
    )
    left.ngauges == right.ngauges || throw(
        DimensionMismatch(
            "cannot merge gauge diagnostics with different gauge counts"
        )
    )
    left.switches += right.switches
    left.residence_time .+= right.residence_time
    left.jumps_by_gauge .+= right.jumps_by_gauge
    left.accepted_projections += right.accepted_projections
    left.fallback_segments += right.fallback_segments
    left.fallback_residence_time += right.fallback_residence_time
    left.fallback_jumps += right.fallback_jumps
    left.maximum_residual = max(left.maximum_residual, right.maximum_residual)
    return left
end
