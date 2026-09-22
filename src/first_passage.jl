# Private first-passage naming boundary over the retained exact diagonal kernels.

"""
    FirstPassageConvergenceError(time, iterations, residual, bracket_width)

Exception raised when a bracketed jump-time root solve exhausts its iteration
budget before satisfying its convergence criteria.

# Arguments

The constructor arguments are also available as fields of the exception:

- `time::Float64`: Final midpoint estimate of the waiting time measured from
  the start of the current no-jump segment, not an absolute `tlist` time.
- `iterations::Int`: Number of root-solver iterations performed.
- `residual::Float64`: Signed survival-probability residual `s(time) - u`,
  where `s` is the no-jump survival probability and `u` is the sampled
  threshold. This is a probability residual even for log-survival methods.
- `bracket_width::Float64`: Width of the remaining bracket containing the
  jump time.

# Notes

- `time` and `bracket_width` have the same units as the input time grid;
  `residual` is dimensionless.
- [`dislou_solve`](@ref) controls the iteration budget with
  `first_passage_maxiter` (default `100`) and the convergence criteria with
  `survival_rtol`, `time_rtol`, and `time_atol`. Increasing the iteration budget
  allows more refinement without relaxing the requested tolerances.

See also [`dislou_solve`](@ref).

# Examples

```jldoctest
julia> using DiSLOUTrajectories

julia> err = FirstPassageConvergenceError(0.5, 100, 1e-6, 1e-4);

julia> (err.time, err.iterations, err.residual, err.bracket_width)
(0.5, 100, 1.0e-6, 0.0001)
```
"""
struct FirstPassageConvergenceError <: Exception
    time::Float64
    iterations::Int
    residual::Float64
    bracket_width::Float64
end

function Base.showerror(io::IO, err::FirstPassageConvergenceError)
    return print(
        io, "first-passage root solve did not converge after ", err.iterations,
        " iterations (time=", err.time, ", residual=", err.residual,
        ", bracket_width=", err.bracket_width, ")"
    )
end

@inline function _validated_first_passage_method(method::Symbol)::Symbol
    method === :automatic && return :log_survival
    (
        method === :survival || method === :log_survival ||
            method === :log_survival_predictor
    ) && return method
    throw(ArgumentError("first-passage method must be :automatic, :survival, :log_survival, or :log_survival_predictor, got $method"))
end

# Paper: survival s(t) and unnormalized norm-loss rate -ṡ(t) (Eqs. 18, B.1).
#
# In any (reduced or full) eigenbasis the unnormalized no-jump coefficients are
# c(t) = D(t) c, with D(t) = diag(exp(-i λ_j t)). Then
#
#     s(t) = c(t)† G c(t)          (G the Gram matrix of the eigenvectors)
#     -ṡ(t) = Σ_μ c(t)† V†C_μ†C_μV c(t) ≥ 0.
#
# Code fields S and R denote s and -ṡ; normalized activity is A = R/S.
# Quadratic forms are made real for roundoff; survival is clamped to [0,1].

@inline _clamp01(x::Real) = x < zero(x) ? zero(x) : (x > one(x) ? one(x) : x)

# Tolerate some small negativities in quadratic form
@inline function _checked_quadratic(q::Real, scale::Real, N::Integer, what::AbstractString)
    qf = Float64(q)
    sf = Float64(scale)
    isfinite(qf) || throw(DomainError(qf, "$what must be finite"))
    isfinite(sf) && sf >= 0 ||
        throw(DomainError(sf, "$what roundoff scale must be finite and nonnegative"))
    N >= 0 || throw(ArgumentError("quadratic dimension must be nonnegative, got $N"))
    kϵ = (16N + 8) * eps(Float64)
    kϵ < 1 || throw(ArgumentError("quadratic roundoff bound is undefined for dimension $N"))
    τ = kϵ / (1 - kϵ) * sf
    qf >= -τ ||
        throw(DomainError(qf, "$what is materially negative (roundoff tolerance $τ)"))
    return max(qf, 0.0)
end

# This gets overridden for CuArrays
@inline _matrix_inf_norm(A::AbstractMatrix) = opnorm(A, Inf)

# Allocation free helpers

# Paper: c†Gc for norms; also Re(c†O_m c) for Hermitian observables.
@inline function _real_quadratic!(
        tmp::AbstractVector{CF},
        x::AbstractVector{CF}, A::AbstractMatrix{CF}
    )
    mul!(tmp, A, x)
    q = 0.0
    @inbounds @simd for i in eachindex(x, tmp)
        q += real(conj(x[i]) * tmp[i])
    end
    return q
end

# Paper: c†O_m c, the unnormalized observable numerator.
@inline function _quadratic!(
        tmp::AbstractVector{CF},
        x::AbstractVector{CF}, A::AbstractMatrix{CF}
    )
    mul!(tmp, A, x)
    return dot(x, tmp)
end

# Non-negative quantities
@inline function _checked_quadratic!(
        tmp::AbstractVector{CF},
        x::AbstractVector{CF}, A::AbstractMatrix{CF}, scale::Real,
        what::AbstractString
    )
    q = _real_quadratic!(tmp, x, A)
    return _checked_quadratic(q, scale, length(x), what)
end

# Paper: c(t) = D(t)c(0) (Eq. 16), written into `out`.
@inline function _phase_evolve!(
        out::AbstractVector{CF}, λ::AbstractVector{CF},
        c::AbstractVector{CF}, s::Real
    )
    @inbounds @simd for j in eachindex(c)
        out[j] = exp((-im * s) * λ[j]) * c[j]
    end
    return out
end

# Paper: s(t) (Eq. 18), or its reduced-space counterpart.
function _survival_probability!(
        Dc::AbstractVector{CF}, tmp::AbstractVector{CF},
        c::AbstractVector{CF}, λ::AbstractVector{CF}, G::AbstractMatrix{CF},
        s::Real, Gnorm::Real
    )
    _phase_evolve!(Dc, λ, c, s)
    survival = _checked_quadratic!(
        tmp, Dc, G,
        Gnorm * sum(abs2, Dc), "survival probability"
    )
    return _clamp01(survival)
end

# Paper: -ṡ(t) = Σ_μ ‖C_μ|ψ̃(t)⟩‖² (Eq. B.1).
function _jump_rate!(
        tmp::AbstractVector{CF}, Dc::AbstractVector{CF},
        Kmats::AbstractVector{<:AbstractMatrix{CF}},
        Knorms::AbstractVector{<:Real}
    )
    r = 0.0
    norm2 = sum(abs2, Dc)
    @inbounds for μ in eachindex(Kmats)
        r += _checked_quadratic!(
            tmp, Dc, Kmats[μ],
            Knorms[μ] * norm2, "jump-channel rate"
        )
    end
    return r
end

"""
    _survival_and_rate!(Gc, Dc, λ, G, Gnorm)::NamedTuple

Evaluate survival ``s(t)`` and unnormalized norm-loss rate ``-ṡ(t)``.

Supply already evolved coefficients `Dc = D(t)c`. Apply the metric once as
`Gc = G * Dc` and reuse it to compute ``s(t) = Re(Dc† G Dc)`` and
``-ṡ(t) = -2 Im((G Dc)† Λ Dc)`` (Eq. B.2).
Return fields `S = s(t)` and `R = -ṡ(t)`; `R/S` is activity ``A(ψ(t))``.
"""
function _survival_and_rate!(
        Gc::AbstractVector{CF}, Dc::AbstractVector{CF},
        λ::AbstractVector{CF}, G::AbstractMatrix{CF},
        Gnorm::Real
    )
    mul!(Gc, G, Dc)
    norm2 = sum(abs2, Dc)
    s = 0.0
    r = 0.0
    @inbounds @simd for i in eachindex(Dc, Gc, λ)
        s += real(conj(Dc[i]) * Gc[i])
        rate_term = conj(Gc[i]) * λ[i] * Dc[i]
        r -= 2 * imag(rate_term)
    end
    s = _clamp01(
        _checked_quadratic(
            s, Gnorm * norm2, length(Dc), "survival probability"
        )
    )
    return (S = s, R = r)
end

# Paper: s(t), -ṡ(t), with G ≈ I; `error` bounds the survival error.
function _survival_and_rate_identity!(
        Dc::AbstractVector{CF},
        λ::AbstractVector{CF}, metric_error::Real
    )
    norm2 = sum(abs2, Dc)
    r = 0.0
    rate_scale = 0.0
    @inbounds @simd for i in eachindex(Dc, λ)
        rate_term = conj(Dc[i]) * λ[i] * Dc[i]
        r -= 2 * imag(rate_term)
        rate_scale += 2 * abs(rate_term)
    end
    s = _clamp01(
        _checked_quadratic(
            norm2, norm2, length(Dc), "survival probability"
        )
    )
    r = _checked_quadratic(r, rate_scale, length(Dc), "total jump rate")
    return (S = s, R = r, error = float(metric_error) * norm2)
end

@inline function _use_identity_metric(metric_error::Real, survival_rtol::Real)
    return isfinite(metric_error) && 0 <= metric_error < 1 &&
        -log1p(-float(metric_error)) <= float(survival_rtol) / 4
end

# Paper: s(t), -ṡ(t), using G ≈ I when permitted.
@inline function _full_survival_and_rate!(
        Gc::AbstractVector{CF},
        Dc::AbstractVector{CF}, λ::AbstractVector{CF}, G::AbstractMatrix{CF},
        Gnorm::Real, metric_error::Real
    )
    isfinite(metric_error) &&
        return _survival_and_rate_identity!(Dc, λ, metric_error)
    exact = _survival_and_rate!(Gc, Dc, λ, G, Gnorm)
    return (S = exact.S, R = exact.R, error = 0.0)
end

# Paper: s(t), -ṡ(t), resolving comparison with u in the full metric.
@inline function _certified_full_survival_and_rate!(
        Gc::AbstractVector{CF},
        Dc::AbstractVector{CF}, λ::AbstractVector{CF}, G::AbstractMatrix{CF},
        Gnorm::Real, metric_error::Real, threshold::Real
    )
    estimate = _full_survival_and_rate!(Gc, Dc, λ, G, Gnorm, metric_error)
    if estimate.error > 0 && abs(estimate.S - threshold) <= estimate.error
        exact = _survival_and_rate!(Gc, Dc, λ, G, Gnorm)
        return (S = exact.S, R = exact.R, error = 0.0)
    end
    return estimate
end


@inline function _verified_convergence!(
        Gc::AbstractVector{CF},
        Dc::AbstractVector{CF}, λ::AbstractVector{CF}, G::AbstractMatrix{CF},
        Gnorm::Real, estimate, threshold::Real, time::Real, T_rem::Real,
        bracket_converged::Bool, survival_rtol::Real, time_rtol::Real,
        time_atol::Real
    )
    residual_limit = float(survival_rtol) * max(float(threshold), 1.0e-300)
    if estimate.error > 0 &&
            (
            bracket_converged ||
                abs(estimate.S - threshold) <= residual_limit + estimate.error
        )
        exact = _survival_and_rate!(Gc, Dc, λ, G, Gnorm)
        estimate = (S = exact.S, R = exact.R, error = 0.0)
    end
    converged = bracket_converged ||
        (
        estimate.error == 0 && _residual_converged(
            estimate.S, estimate.R, threshold, time, T_rem,
            survival_rtol, time_rtol, time_atol
        )
    )
    return converged, estimate
end


# Reduced-basis jump-time predictor for the full diagonal-basis path.
#
# The full survival s(t) = c† D(t)† G D(t) c still defines the true jump time. This module
# only produces a cheap *predictor* t_guess by solving the analogous survival equation in a
# small active set I ⊂ {1..N} of p modes (code `m`):
#
# s_I(t) = c̄_I† D_I(t)† G_I D_I(t) c̄_I = u (Eqs. B.7–B.8),
#
# c̄_I = c[I]/√(c[I]†G_I c[I]), G_I = G[I,I], λ_I = λ[I]; hence s_I(0) = 1.
# G_I is the p×p principal submatrix of G; no per-segment ReducedCache is needed.
# -ṡ_I = -2 Im((G_I D_I c̄_I)† Λ_I D_I c̄_I) reuses _survival_and_rate!
# on the truncated vectors without channel matrices.
#
# The predictor is used ONLY to seed the full safeguarded Newton solve. Correctness is
# unchanged: the full root finder rebrackets around t_guess and converges to s(τ)=u.

"""
    _FirstPassagePredictor(N; <keyword arguments>)

Per-execution-context scratch and configuration for the reduced jump-time predictor.

Keep a separate instance for each concurrent trajectory.

# Arguments
- `N::Int`: Full Hilbert-space dimension.
- `m::Integer=8`: Number of largest-magnitude coordinates to retain, clamped to `1:N`.
- `tol::Real=1e-4`: Positive, finite relative tolerance for the reduced root solve.
- `maxiter::Integer=60`: Positive iteration limit for the reduced root solve.
"""
mutable struct _FirstPassagePredictor
    m::Int
    tol::Float64
    maxiter::Int
    sval::Vector{Float64}    # length N — ranking scores
    perm::Vector{Int}        # length N — top-m insertion scratch
    I::Vector{Int}           # length m — selected modes
    cI::Vector{CF}           # length m
    DcI::Vector{CF}          # length m
    GcI::Vector{CF}          # length m
    λI::Vector{CF}           # length m
    GI::Matrix{CF}           # m×m principal submatrix of G
end

function _FirstPassagePredictor(
        N::Int; m::Integer = 8, tol::Real = 1.0e-4, maxiter::Integer = 60
    )
    isfinite(tol) && tol > 0 ||
        throw(ArgumentError("tol must be finite and positive"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    mm = max(1, min(Int(m), N))
    return _FirstPassagePredictor(
        mm, float(tol), Int(maxiter),
        Vector{Float64}(undef, N), collect(1:N),
        Vector{Int}(undef, mm), Vector{CF}(undef, mm), Vector{CF}(undef, mm),
        Vector{CF}(undef, mm), Vector{CF}(undef, mm), Matrix{CF}(undef, mm, mm)
    )
end

_FirstPassagePredictor(cache::_DiagonalCache; kwargs...) = _FirstPassagePredictor(cache.N; kwargs...)

# Paper: auxiliary predictor indices I, with p = m (Appendix B).
function _topm_indices!(
        indices::AbstractVector{Int},
        values::AbstractVector{Float64}, m::Int
    )
    # O(Nm) works for the predictor's small fixed m, use heap if m grows.
    @inbounds for index in eachindex(values)
        if index <= m
            position = index
        else
            last_index = indices[m]
            if !(
                    isless(values[last_index], values[index]) ||
                        (
                        isequal(values[index], values[last_index]) &&
                            index < last_index
                    )
                )
                continue
            end
            position = m
        end
        while position > 1
            previous = indices[position - 1]
            if !(
                    isless(values[previous], values[index]) ||
                        (
                        isequal(values[index], values[previous]) &&
                            index < previous
                    )
                )
                break
            end
            indices[position] = previous
            position -= 1
        end
        indices[position] = index
    end
    return indices
end

# Paper: predictor τ_I from s_I(τ_I) = u (Appendix B).
# Mirrors the :log_survival branch of _find_diagonal_first_passage!, but on the m-dim truncated quadratic form.
# Returns (t, iters); iters counts reduced survival/rate evaluations inside the bracket loop.
function _reduced_root_log!(rg::_FirstPassagePredictor, r::Real, T_rem::Real)
    m = rg.m
    cI = view(rg.cI, 1:m); DcI = view(rg.DcI, 1:m)
    GcI = view(rg.GcI, 1:m); λI = view(rg.λI, 1:m); GI = view(rg.GI, 1:m, 1:m)
    Gnorm = _matrix_inf_norm(GI)

    _phase_evolve!(DcI, λI, cI, float(T_rem))
    SR_end = _survival_and_rate!(GcI, DcI, λI, GI, Gnorm)
    S_end = SR_end.S
    h = -log(float(r))

    if S_end > r
        # Reduced survival predicts no jump within [0, T_rem] (decays slower than the full one):
        # log-linear extrapolation, clamped just inside the interval so the full solve gets a
        # finite interior seed rather than the endpoint.
        H_end = -log(S_end)
        t = (isfinite(h) && isfinite(H_end) && H_end > 1.0e-300) ?
            T_rem * h / H_end : float(T_rem)
        return (min(float(T_rem), max(t, 1.0e-300 * max(T_rem, 1.0))), 0)
    end

    lo = 0.0
    hi = float(T_rem)
    atol = rg.tol * max(float(T_rem), 1.0)
    H_end = -log(S_end)
    t = (isfinite(h) && isfinite(H_end) && H_end > 0) ?
        T_rem * h / H_end : T_rem * (1 - r) / max(1 - S_end, 1.0e-300)
    (t > lo && t < hi) || (t = 0.5 * (lo + hi))
    tiny = floatmin(Float64)

    iters = 0
    for _ in 1:rg.maxiter
        _phase_evolve!(DcI, λI, cI, t)
        SR = _survival_and_rate!(GcI, DcI, λI, GI, Gnorm)
        iters += 1
        f = SR.S - r
        if abs(f) <= rg.tol || (hi - lo) <= atol
            return (t, iters)
        end
        SR.S > r ? (lo = t) : (hi = t)
        if SR.R > 1.0e-300 && SR.S > tiny
            F = -log(SR.S) - h
            tN = t - F * SR.S / SR.R
        else
            tN = SR.R > 1.0e-300 ? t + f / SR.R : 0.5 * (lo + hi)
        end
        t = (tN > lo && tN < hi) ? tN : 0.5 * (lo + hi)
    end
    return (0.5 * (lo + hi), iters)
end

"""
    _predict_first_passage!(rg, c, λ, G, r, T_rem)::NamedTuple

Compute the predictor ``τ_I`` from the reduced survival equation (Appendix B).

Solve ``s_I(t) = u`` (code threshold `r`) in the `m` largest-magnitude coordinates of `c`.
Supply full-spectrum coordinates `c`, eigenvalues `λ`, and metric `G`.
Return `(t = t_guess, jumped = reduced_root_was_evaluated)`, with
`t_guess ∈ (0, T_rem]` to seed the full Newton solve. All scratch storage is in `rg`.
"""
function _predict_first_passage!(
        rg::_FirstPassagePredictor, c::AbstractVector{CF}, λ::AbstractVector{CF},
        G::AbstractMatrix{CF}, r::Real, T_rem::Real
    )
    m = rg.m
    N = length(c)

    # ---- rank modes ---------------------------------------------------------------------
    @inbounds @simd for j in 1:N
        rg.sval[j] = abs(c[j])
    end
    _topm_indices!(rg.perm, rg.sval, m)
    @inbounds for a in 1:m
        rg.I[a] = rg.perm[a]
    end

    # ---- gather the reduced problem -----------------------------------------------------
    @inbounds for a in 1:m
        ia = rg.I[a]
        rg.cI[a] = c[ia]
        rg.λI[a] = λ[ia]
        for b in 1:m
            rg.GI[a, b] = G[ia, rg.I[b]]
        end
    end

    # Paper: c̄_I = c_I/√(c_I†G_I c_I), so s_I(0) = 1 (Eq. B.7).
    cI = view(rg.cI, 1:m); GI = view(rg.GI, 1:m, 1:m)
    nrm = real(dot(cI, GI, cI))
    if nrm > 0
        s = 1 / sqrt(nrm)
        @inbounds @simd for a in 1:m
            rg.cI[a] *= s
        end
    end

    t_guess, iters = _reduced_root_log!(rg, r, T_rem)
    return (; t = t_guess, jumped = iters >= 1)
end


# Paper: τ solves s(τ) = u on [0, T_rem], with s(0) = 1 (Eq. 19).
#
# Strategy (Appendix B): evaluate s(T_rem) first — the rare-event shortcut that
# terminates most trajectories with a single quadratic-form evaluation. If a jump is
# bracketed, use safeguarded Newton (step t - (s(t)-u)/ṡ(t), Eq. B.3) confined to
# a bisection bracket; fall back to bisection whenever Newton leaves the bracket.

"""
    _FirstPassageResult

Outcome of a jump-time solve.

`jumped` distinguishes an in-interval root from
the endpoint no-jump shortcut; `tau`, convergence state, iteration count,
survival residual, and final bracket width record the numerical result.
"""
struct _FirstPassageResult
    jumped::Bool
    tau::Float64
    converged::Bool
    iterations::Int
    survival_residual::Float64
    bracket_width::Float64
end

Base.getproperty(res::_FirstPassageResult, name::Symbol) =
    (name === :τ || name === :time) ? getfield(res, :tau) : getfield(res, name)

@inline function _jump_time_result(
        jumped::Bool, tau::Real, converged::Bool,
        iterations::Integer, survival_residual::Real, bracket_width::Real
    )
    return _FirstPassageResult(
        jumped, float(tau), converged, Int(iterations),
        float(survival_residual), float(bracket_width)
    )
end

@inline _no_jump_result(T_rem::Real, survival_residual::Real) =
    _jump_time_result(false, T_rem, true, 0, survival_residual, 0.0)

@inline _jump_converged_result(
    tau::Real, iterations::Integer,
    survival_residual::Real, bracket_width::Real
) =
    _jump_time_result(true, tau, true, iterations, survival_residual, bracket_width)

@inline function _validate_root_tolerances(
        survival_rtol::Real, time_rtol::Real,
        time_atol::Real
    )
    isfinite(survival_rtol) && survival_rtol > 0 ||
        throw(ArgumentError("survival_rtol must be finite and positive"))
    isfinite(time_rtol) && time_rtol > 0 ||
        throw(ArgumentError("time_rtol must be finite and positive"))
    isfinite(time_atol) && time_atol >= 0 ||
        throw(ArgumentError("time_atol must be finite and nonnegative"))
    return nothing
end

@inline _time_tolerance(t::Real, T_rem::Real, time_rtol::Real, time_atol::Real) =
    time_atol + time_rtol * max(abs(t), abs(T_rem))

@inline _bracket_converged(
    lo::Real, hi::Real, T_rem::Real, time_rtol::Real,
    time_atol::Real
) =
    hi - lo <= time_atol + time_rtol * max(abs(lo), abs(hi), abs(T_rem))

@inline function _residual_converged(
        S::Real, R::Real, r::Real, t::Real, T_rem::Real,
        survival_rtol::Real, time_rtol::Real, time_atol::Real
    )
    S > 0 && r > 0 && isfinite(S) && isfinite(R) && R > 0 || return false
    log_residual = abs(log(S) - log(r))
    hazard = R / S
    return isfinite(log_residual) && log_residual <= survival_rtol &&
        isfinite(hazard) && hazard > 0 &&
        log_residual / hazard <= _time_tolerance(t, T_rem, time_rtol, time_atol)
end

@inline function _throw_nonconverged_jump_time(
        tau::Real, iterations::Integer,
        survival_residual::Real, bracket_width::Real
    )
    throw(
        FirstPassageConvergenceError(
            float(tau), Int(iterations),
            float(survival_residual), float(bracket_width)
        )
    )
end

function _throw_nonconverged_full!(
        Dc, Gc, c, λ, G, Gnorm::Real,
        r::Real, lo::Real, hi::Real, iterations::Integer
    )
    tau = 0.5 * (float(lo) + float(hi))
    _phase_evolve!(Dc, λ, c, tau)
    residual = _survival_and_rate!(Gc, Dc, λ, G, Gnorm).S - r
    _throw_nonconverged_jump_time(tau, iterations, residual, float(hi) - float(lo))
end

"""
    _find_scalar_first_passage(Sval, Rval, r, T_rem; <keyword arguments>)

Find ``τ = inf{t ≥ 0 : s(t) ≤ u}`` (Eq. 19); code threshold `r` is ``u``.

Supply `Sval(t) = s(t)` with `Sval(0)=1` and nonnegative norm-loss rate
`Rval(t) = -ṡ(t)` (Eq. B.1). Return a `_FirstPassageResult`. If `Sval(T_rem) > r`,
return `jumped=false` and `tau=T_rem`. Exhausting the Newton iteration budget
raises `FirstPassageConvergenceError` with the final estimate and diagnostics.

# Arguments
- `survival_rtol::Real=1e-10`: Relative tolerance for the survival residual.
- `time_rtol::Real=1e-12`: Relative root-time and bracket-width tolerance.
- `time_atol::Real=0.0`: Absolute root-time and bracket-width tolerance.
- `maxiter::Integer=100`: Newton iteration limit.
- `root_variant::Symbol=:survival`: Use `:survival` or `:log_survival` Newton steps.
  `:log_survival_predictor` aliases `:log_survival` for these scalar closures.
"""
function _find_scalar_first_passage(
        Sval, Rval, r::Real, T_rem::Real;
        survival_rtol::Real = 1.0e-10, time_rtol::Real = 1.0e-12,
        time_atol::Real = 0.0, maxiter::Integer = 100,
        root_variant::Symbol = :survival
    )
    _validate_root_tolerances(survival_rtol, time_rtol, time_atol)
    root_variant_eff = root_variant === :log_survival_predictor ? :log_survival : root_variant
    S_end = Sval(T_rem)
    if r == 0 || S_end > r
        return _no_jump_result(T_rem, S_end - r)
    end

    lo = 0.0
    hi = float(T_rem)
    h = root_variant_eff === :log_survival ? -log(float(r)) : 0.0
    if root_variant_eff === :log_survival
        H_end = -log(S_end)
        t = (isfinite(h) && isfinite(H_end) && H_end > 0) ?
            T_rem * h / H_end : T_rem * (1 - r) / max(1 - S_end, 1.0e-300)
    else
        t = T_rem * (1 - r) / max(1 - S_end, 1.0e-300)
    end
    (t > lo && t < hi) || (t = 0.5 * (lo + hi))

    iters = 0
    tiny = floatmin(Float64)
    for _ in 1:maxiter
        St = Sval(t)
        f = St - r
        iters += 1
        St > r ? (lo = t) : (hi = t)
        _bracket_converged(lo, hi, T_rem, time_rtol, time_atol) &&
            return _jump_converged_result(t, iters, f, hi - lo)
        Rt = Rval(t)                       # = -S'(t) ≥ 0
        _residual_converged(St, Rt, r, t, T_rem, survival_rtol, time_rtol, time_atol) &&
            return _jump_converged_result(t, iters, f, hi - lo)
        if root_variant_eff === :log_survival && Rt > 1.0e-300 && St > tiny
            F = -log(St) - h
            tN = t - F * St / Rt
        else
            tN = Rt > 1.0e-300 ? t + f / Rt : 0.5 * (lo + hi)
        end
        t = (tN > lo && tN < hi) ? tN : 0.5 * (lo + hi)
    end
    tau = 0.5 * (lo + hi)
    _throw_nonconverged_jump_time(tau, iters, Sval(tau) - r, hi - lo)
end

"""
    _find_diagonal_first_passage!(Dc, Gc, c, λ, G, r, T_rem; <keyword arguments>)

Find ``τ`` from ``s(τ) = u`` (Eq. 19), with code threshold `r = u`.

Reuse `Dc` for `D(t)c` and `Gc` for `G * D(t)c`. Each Newton step evaluates survival
and rate with one metric multiplication and an `O(N)` eigenvalue contraction.
Return a `_FirstPassageResult` or raise `FirstPassageConvergenceError` on exhaustion.

# Arguments
- `survival_rtol::Real=1e-10`: Relative tolerance for the survival residual.
- `time_rtol::Real=1e-12`: Relative root-time and bracket-width tolerance.
- `time_atol::Real=0.0`: Absolute root-time and bracket-width tolerance.
- `maxiter::Integer=100`: Newton iteration limit.
- `root_variant::Symbol=:survival`: Use `:survival`, `:log_survival`, or
  `:log_survival_predictor` Newton steps.
- `rg::Union{Nothing,_FirstPassagePredictor}=nothing`: Optional reduced predictor scratch.
- `Gnorm::Real`: Infinity norm of `G`, computed from `G` by default.
- `metric_error::Real=Inf`: Bound on the departure of `G` from the identity metric.
"""
function _find_diagonal_first_passage!(
        Dc::AbstractVector{CF}, Gc::AbstractVector{CF},
        c::AbstractVector{CF}, λ::AbstractVector{CF}, G::AbstractMatrix{CF},
        r::Real, T_rem::Real; survival_rtol::Real = 1.0e-10,
        time_rtol::Real = 1.0e-12, time_atol::Real = 0.0,
        maxiter::Integer = 100,
        root_variant::Symbol = :survival,
        rg::Union{Nothing, _FirstPassagePredictor} = nothing,
        Gnorm::Real = _matrix_inf_norm(G),
        metric_error::Real = Inf
    )
    _validate_root_tolerances(survival_rtol, time_rtol, time_atol)
    fast_metric_error = _use_identity_metric(metric_error, survival_rtol) ?
        float(metric_error) : Inf
    if root_variant === :log_survival && rg !== nothing
        throw(
            ArgumentError(
                "reduced first-passage prediction requires root_variant=:log_survival_predictor"
            )
        )
    end

    root_variant_eff = root_variant === :log_survival_predictor ? :log_survival : root_variant

    if r == 0
        _phase_evolve!(Dc, λ, c, float(T_rem))
        S_end = _full_survival_and_rate!(
            Gc, Dc, λ, G, Gnorm, fast_metric_error
        ).S
        return _no_jump_result(T_rem, S_end)
    end

    if root_variant === :log_survival_predictor && rg !== nothing
        h = -log(float(r))
        gz = _predict_first_passage!(rg, c, λ, G, r, T_rem)
        t_red = gz.t

        if !(gz.jumped && t_red > 0.0 && t_red < float(T_rem))
            _phase_evolve!(Dc, λ, c, float(T_rem))
            SR_end = _certified_full_survival_and_rate!(
                Gc, Dc, λ, G, Gnorm, fast_metric_error, r
            )
            S_end = SR_end.S
            S_end > r && return _no_jump_result(T_rem, S_end - r)
            lo = 0.0
            hi = float(T_rem)
            H_end = -log(S_end)
            t = (t_red > lo && t_red < hi) ? t_red :
                (
                    (isfinite(h) && isfinite(H_end) && H_end > 0) ? T_rem * h / H_end :
                    T_rem * (1 - r) / max(1 - S_end, 1.0e-300)
                )
        else
            _phase_evolve!(Dc, λ, c, t_red)
            SR = _certified_full_survival_and_rate!(
                Gc, Dc, λ, G, Gnorm, fast_metric_error, r
            )
            f = SR.S - r
            if SR.S <= r
                lo = 0.0
                hi = t_red
                bracket_done = _bracket_converged(
                    lo, hi, T_rem, time_rtol, time_atol
                )
                converged, SR = _verified_convergence!(
                    Gc, Dc, λ, G, Gnorm, SR, r, t_red, T_rem,
                    bracket_done, survival_rtol, time_rtol, time_atol
                )
                f = SR.S - r
                if converged
                    return _jump_converged_result(t_red, 1, f, hi - lo)
                end
                F = -log(max(SR.S, floatmin(Float64))) - h
                tN = SR.R > 1.0e-300 ? t_red - F * SR.S / SR.R : 0.5 * (lo + hi)
                t = (tN > lo && tN < hi) ? tN : 0.5 * (lo + hi)
            else
                lo = t_red
                _phase_evolve!(Dc, λ, c, float(T_rem))
                SR_end = _certified_full_survival_and_rate!(
                    Gc, Dc, λ, G, Gnorm, fast_metric_error, r
                )
                if SR_end.S > r
                    return _no_jump_result(T_rem, SR_end.S - r)
                end
                hi = float(T_rem)
                _phase_evolve!(Dc, λ, c, t_red)
                bracket_done = _bracket_converged(
                    lo, hi, T_rem, time_rtol, time_atol
                )
                converged, SR = _verified_convergence!(
                    Gc, Dc, λ, G, Gnorm, SR, r, t_red, T_rem,
                    bracket_done, survival_rtol, time_rtol, time_atol
                )
                f = SR.S - r
                if converged
                    return _jump_converged_result(t_red, 1, f, hi - lo)
                end
                tiny = floatmin(Float64)
                if SR.R > 1.0e-300 && SR.S > tiny
                    F = -log(SR.S) - h
                    tN = t_red - F * SR.S / SR.R
                else
                    tN = SR.R > 1.0e-300 ? t_red + f / SR.R : 0.5 * (lo + hi)
                end
                t = (tN > lo && tN < hi) ? tN : 0.5 * (lo + hi)
            end
        end
    else
        _phase_evolve!(Dc, λ, c, float(T_rem))
        SR_end = _certified_full_survival_and_rate!(
            Gc, Dc, λ, G, Gnorm, fast_metric_error, r
        )
        S_end = SR_end.S
        S_end > r && return _no_jump_result(T_rem, S_end - r)

        lo = 0.0
        hi = float(T_rem)
        h = root_variant_eff === :log_survival ? -log(float(r)) : 0.0
        if root_variant_eff === :log_survival
            H_end = -log(S_end)
            t = (isfinite(h) && isfinite(H_end) && H_end > 0) ?
                T_rem * h / H_end : T_rem * (1 - r) / max(1 - S_end, 1.0e-300)
        else
            t = T_rem * (1 - r) / max(1 - S_end, 1.0e-300)
        end
        (t > lo && t < hi) || (t = 0.5 * (lo + hi))
    end
    tiny = floatmin(Float64)

    iters = 0
    for _ in 1:maxiter
        _phase_evolve!(Dc, λ, c, t)
        SR = _certified_full_survival_and_rate!(
            Gc, Dc, λ, G, Gnorm, fast_metric_error, r
        )
        f = SR.S - r
        iters += 1
        SR.S > r ? (lo = t) : (hi = t)
        bracket_done = _bracket_converged(lo, hi, T_rem, time_rtol, time_atol)
        converged, SR = _verified_convergence!(
            Gc, Dc, λ, G, Gnorm, SR, r, t, T_rem,
            bracket_done, survival_rtol, time_rtol, time_atol
        )
        f = SR.S - r
        if converged
            return _jump_converged_result(t, iters, f, hi - lo)
        end
        if root_variant_eff === :log_survival && SR.R > 1.0e-300 && SR.S > tiny
            F = -log(SR.S) - h
            tN = t - F * SR.S / SR.R
        else
            tN = SR.R > 1.0e-300 ? t + f / SR.R : 0.5 * (lo + hi)
        end
        t = (tN > lo && tN < hi) ? tN : 0.5 * (lo + hi)
    end
    _throw_nonconverged_full!(Dc, Gc, c, λ, G, Gnorm, r, lo, hi, iters)
end


struct _FirstPassageBuffers
    Dc::Vector{CF}
    Gc::Vector{CF}
    predictor::Union{Nothing, _FirstPassagePredictor}
end

_FirstPassageBuffers(
    cache::_DiagonalCache,
    predictor::Union{Nothing, _FirstPassagePredictor} = nothing
) =
    _FirstPassageBuffers(zeros(CF, cache.N), zeros(CF, cache.N), predictor)

# Paper: τ = inf{t ≥ 0 : s(t) ≤ u} (Eq. 19).
function _find_first_passage!(
        buffers::_FirstPassageBuffers, cache::_DiagonalCache,
        c::AbstractVector{CF}, threshold::Real, remaining::Real;
        method::Symbol = :automatic, survival_rtol::Real = 1.0e-10,
        time_rtol::Real = 1.0e-12, time_atol::Real = 0.0, maxiter::Integer = 100
    )
    return _find_diagonal_first_passage!(
        buffers.Dc, buffers.Gc, c, cache.Λ, cache.G,
        threshold, remaining; survival_rtol, time_rtol, time_atol, maxiter,
        root_variant = method,
        rg = method === :log_survival_predictor ? buffers.predictor : nothing,
        Gnorm = cache.Gnorm,
        metric_error = cache.metric_error
    )
end
