# Jump-channel selection and collapse (paper Eqs. 3–4).
#
# Code weights w_μ = ‖C_μ|ψ̃⟩‖² give probability w_μ / Σ_ν w_ν; the state is
# C_μ|ψ̃⟩ / ‖C_μ|ψ̃⟩‖. Tiny negative weights from roundoff are clamped; large negatives
# signal a broken representation. Gauge superscripts are suppressed here.

# Cumulative-sum weighted choice — mirrors QuantumToolbox's mcsolve_callback_helpers.jl.
# Paper: μ sampled with Pr(μ) ∝ ⟨C_μ†C_μ⟩_ψ (Eq. 4).
@inline function _sample_channel(w::AbstractVector{<:Real}, rng, n::Int)
    s = 0.0
    last_positive = 0
    @inbounds for μ in 1:n
        weight = Float64(w[μ])
        isfinite(weight) && weight >= 0 ||
            throw(DomainError(weight, "jump-channel weights must be finite and nonnegative"))
        if weight > 0
            s += weight
            isfinite(s) || throw(DomainError(s, "total jump-channel weight must be finite"))
            last_positive = μ
        end
    end
    last_positive != 0 ||
        throw(DomainError(s, "at least one jump-channel weight must be positive"))

    r = rand(rng) * s
    acc = 0.0
    @inbounds for μ in 1:n
        w[μ] > 0 || continue
        acc += w[μ]
        acc >= r && return μ
    end
    return last_positive
end

# Paper: μ sampled with Pr(μ) ∝ ⟨C_μ†C_μ⟩_ψ (Eq. 4).
_sample_channel(w::AbstractVector{<:Real}, rng) = _sample_channel(w, rng, length(w))

# Paper: μ and ‖C_μ V_m c(τ)‖² (Eqs. 4, C.2).
function _sample_local_channel!(
        w::AbstractVector{Float64},
        tmp::AbstractVector{CF}, rc::_ReducedCache,
        Dcτ::AbstractVector{CF}, Nc::Int, rng
    )
    norm2 = sum(abs2, Dcτ)
    @inbounds for μ in 1:Nc
        K = rc.K_I[μ]
        w[μ] = _checked_quadratic!(
            tmp, Dcτ, K,
            rc.Knorms[μ] * norm2, "jump-channel weight"
        )
    end
    μ = _sample_channel(w, rng, Nc)
    return μ, w[μ]
end

# Paper: |ψ⁺⟩ = C_μ|ψ̃⟩/‖C_μ|ψ̃⟩‖ and c⁺ = V⁻¹|ψ⁺⟩ (Eq. 3).
function _apply_exact_jump_coordinates!(
        c::AbstractVector{CF}, wb::_WorkBuffers,
        cache::_DiagonalCache, Dcτ::AbstractVector{CF}, rng
    )
    norm2 = sum(abs2, Dcτ)
    @inbounds for μ in 1:cache.Nc
        M = cache.M[μ]
        wb.w[μ] = _checked_quadratic!(
            wb.Gc, Dcτ, M,
            cache.Mnorms[μ] * norm2, "jump-channel weight"
        )
    end
    μ = _sample_channel(wb.w, rng, cache.Nc)
    mul!(wb.ψ, cache.A[μ], Dcτ)
    wb.ψ ./= sqrt(wb.w[μ])
    _solve_coordinates!(c, cache, wb.ψ)
    return μ
end

# Paper: |ψ⁺⟩ under C_μ^(g), then g′ (Eqs. 3, 13).
function _apply_exact_jump_and_route!(
        wb::_WorkBuffers,
        prepared::_Layer1Prepared, current::Int,
        Dcτ::AbstractVector{CF}, rng
    )
    gauges = prepared.system.gauges
    1 <= current <= length(gauges) ||
        throw(BoundsError(gauges, current))
    gauge = gauges[current]
    cache = gauge.cache
    if isempty(gauge.C_sparse)
        channel = _apply_exact_jump_coordinates!(wb.c, wb, cache, Dcτ, rng)
        next = _postjump_gauge_coordinates!(wb, prepared, current, wb.ψ)
        return channel, next
    end

    mul!(wb.ψ, cache.V, Dcτ)
    @inbounds for channel in eachindex(gauge.C_sparse)
        mul!(wb.Gc, gauge.C_sparse[channel], wb.ψ)
        wb.w[channel] = sum(abs2, wb.Gc)
    end
    channel = _sample_channel(wb.w, rng, cache.Nc)
    mul!(wb.Gc, gauge.C_sparse[channel], wb.ψ)
    wb.Gc ./= sqrt(wb.w[channel])
    copyto!(wb.ψ, wb.Gc)
    _solve_coordinates!(wb.c, cache, wb.ψ)
    length(gauges) == 1 && return channel, current

    means = view(wb.moments, 1:cache.Nc)
    second = view(wb.w, 1:cache.Nc)
    @inbounds for index in eachindex(prepared.system.C_sparse)
        mul!(wb.Gc, prepared.system.C_sparse[index], wb.ψ)
        means[index] = dot(wb.ψ, wb.Gc)
        second[index] = sum(abs2, wb.Gc)
    end
    next = _hysteretic_gauge(
        gauges, current, means, second, cache.Nc, prepared.hysteresis
    )
    if next != current
        new = gauges[next]
        _solve_coordinates!(wb.cplus, new.cache, wb.ψ)
        _normalize_coordinates!(wb.cplus, wb.Gc, new.cache.G, new.cache.Gnorm)
    end
    return channel, next
end
