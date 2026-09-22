@doc raw"""
    DiSLOUSolution

Solution with ensemble density operators ``ρ_{MC}(t)``, observable statistics,
jump records, and diagnostics.

Obtain this subtype of `QuantumToolbox.TimeEvolutionMultiTrajSol` from
[`dislou_solve`](@ref). Observation times are in `times`, while `times_states`
contains the state-saving grid selected by `saveat`.

See also [`dislou_solve`](@ref), [`expect_mean`](@ref), [`expect_sem`](@ref).

# Examples

```jldoctest
julia> using DiSLOUTrajectories

julia> sol = dislou_solve(zeros(ComplexF64, 2, 2), ComplexF64[1, 0],
           [0.0, 1.0], Matrix{ComplexF64}[];
           gauge_set = zeros(ComplexF64, 0, 1),
           e_ops = [ComplexF64[1 0; 0 0]], ntraj = 2, ensemblealg = :serial);

julia> (sol isa DiSLOUSolution, sol.ntraj, length(sol.states))
(true, 2, 1)
```

# Extended help

Let `N`
be the Hilbert-space dimension, `Nc` the number of collapse channels, `Ng` the
number of gauges, `Ne` the number of observables, `Nt = length(times)`, and
`Ns = length(times_states)`.

# Fields

Solutions are normally constructed by [`dislou_solve`](@ref). The positional
constructor takes the following fields in order; the same names are available
as properties of the returned solution:

- `ntraj::Int`: Number of completed trajectories.
- `times::Vector{Float64}`: Observation times supplied through `tlist`.
- `times_states::Vector{Float64}`: Sorted state-saving times resolved from
  `saveat`; defaults to the final observation time.
- `states`: Vector of `Ns` `QuantumToolbox.QuantumObject` density operators,
  each the average ``ρ_{MC}(t)`` of normalized trajectory projectors at that time.
- `expect`: `Ne × Nt` `Matrix{ComplexF64}` of ensemble expectation means, or
  `nothing` if no observables were requested. Rows follow the input `e_ops` order.
- `col_times::Vector{Vector{Float64}}`: Jump times for each trajectory, in the
  absolute time coordinates of `tlist`.
- `col_which::Vector{Vector{Int}}`: One-based collapse-channel indices aligned
  with `col_times`.
- `col_gauge::Vector{Vector{Int}}`: One-based active gauge indices at each jump,
  aligned with `col_times`. Gauge indices refer to columns of the input shifts.
- `njumps_total::Int`: Total number of recorded jumps across all trajectories.
- `jumps_by_channel::Vector{Int}`: Total jump counts per collapse channel;
  has length `Nc`.
- `converged::Bool`: `true` when all trajectories completed successfully.
  This field does not assess statistical convergence of the ensemble.
- `seed::UInt64`: Base random seed supplied to the solver.
- `survival_rtol::Float64`: First-passage log-survival residual tolerance.
- `time_rtol::Float64`: Relative first-passage time tolerance.
- `time_atol::Float64`: Absolute first-passage time tolerance.
- `expect_std`: `Ne × Nt` `Matrix{Float64}` of population standard deviations
  across trajectories, or `nothing` without observables.
- `expect_sem`: `Ne × Nt` `Matrix{Float64}` of estimated standard errors of the
  ensemble means, or `nothing` without observables. All entries are `NaN` for
  one trajectory.
- `gauge_diagnostics::NamedTuple`: Gauge count, provenance, switching counts,
  residence times, and jump counts, as detailed below.
- `layer3_diagnostics`: Layer III diagnostics described below, or `nothing`
  when Layer III was disabled.
- `eigensystem_backend::Symbol`: `:lapack`, `:cuda`, or `:mixed`, identifying
  the eigensystem preparation backend actually used. `:mixed` denotes a solve
  whose gauges or participating processes used different backends.
- `trajectory_states`: `ntraj × Ns` matrix of normalized
  `QuantumToolbox.QuantumObject` kets when `save_trajectories=true`, otherwise
  `nothing`. Rows index trajectories and columns index `times_states`.
- `trajectory_expect`: `Ne × ntraj × Nt` `Array{ComplexF64,3}` when
  `save_trajectories=true`, otherwise `nothing`. Its axes index observable,
  trajectory, and observation time; the first axis is empty without observables.
- `final_states`: `N × ntraj` `Matrix{ComplexF64}` of normalized final kets
  when `save_final_states=true`, otherwise `nothing`. Each column is one ket.
- `final_density`: `N × N` `Matrix{ComplexF64}` averaging the final-state
  projectors when `save_final_states=true`, otherwise `nothing`.

# Notes

- For trajectory expectations ``x_k`` at one observable and time, let
  ``\bar{x} = n^{-1}\sum_{k=1}^n x_k`` and
  ``M_2 = \sum_{k=1}^n |x_k-\bar{x}|^2``, with ``n=\mathtt{ntraj}``.
  Then `expect_std` is ``\sqrt{M_2/n}`` and `expect_sem` is
  ``\sqrt{M_2/[n(n-1)]}`` for ``n>1``.
  A single trajectory has zero population standard deviation but an
  undefined standard error.
- `gauge_diagnostics` contains `count=Ng`; `method` (`:manual`,
  `:trajectories`, or `:semiclassical`); `status` (`:manual` for a shift-matrix
  input or `:validated` for a gauge finding result); `switches` (total gauge
  switches); `residence_time` (length-`Ng` vector of total time spent in each
  gauge across all trajectories); and `jumps_by_gauge` (length-`Ng` vector of
  total jump counts).
- `layer3_diagnostics` contains `sizes` (retained mode count per gauge),
  `accepted_projections` (total accepted projections), `fallback_segments`
  (total segments sent to full-space propagation), and `maximum_residual`
  (largest relative projection residual tested, including rejected
  projections).
- Jump channels label the shifted collapse operators of the selected
  unraveling.
- Use [`expect_mean`](@ref) and [`expect_sem`](@ref) to extract one observable.
"""
struct DiSLOUSolution{TS, TE} <: QuantumToolbox.TimeEvolutionMultiTrajSol{TS, TE}
    ntraj::Int
    times::Vector{Float64}
    times_states::Vector{Float64}
    states::TS
    expect::TE
    col_times::Vector{Vector{Float64}}
    col_which::Vector{Vector{Int}}
    col_gauge::Vector{Vector{Int}}
    njumps_total::Int
    jumps_by_channel::Vector{Int}
    converged::Bool
    seed::UInt64
    survival_rtol::Float64
    time_rtol::Float64
    time_atol::Float64
    expect_std::Union{Nothing, Matrix{Float64}}
    expect_sem::Union{Nothing, Matrix{Float64}}
    gauge_diagnostics::NamedTuple
    layer3_diagnostics::Union{Nothing, NamedTuple}
    eigensystem_backend::Symbol
    trajectory_states::Union{Nothing, Matrix{QuantumToolbox.QuantumObject}}
    trajectory_expect::Union{Nothing, Array{ComplexF64, 3}}
    final_states::Union{Nothing, Matrix{ComplexF64}}
    final_density::Union{Nothing, Matrix{ComplexF64}}
end

function _state_qobj(data::AbstractArray{CF}, dimensions)
    copied = copy(data)
    dimensions === nothing && return QuantumObject(copied)
    return QuantumObject(
        copied;
        dims = first(QuantumToolbox.dimensions_to_dims(dimensions))
    )
end

# Paper: ρ_MC(t) = mean_r |ψ_r(t)⟩⟨ψ_r(t)| (Section 4.1).
function _aggregate_states(acc::_TrajectoryAccumulator, dimensions)
    acc.state_sums === nothing && return QuantumToolbox.QuantumObject[]
    return QuantumToolbox.QuantumObject[
        _state_qobj(state_sum / acc.ntraj, dimensions)
            for state_sum in acc.state_sums
    ]
end

function _trajectory_states(records::Vector{_TrajectoryRecord}, dimensions)
    isempty(records) && return nothing
    records[1].states === nothing && return nothing
    states = Matrix{QuantumToolbox.QuantumObject}(
        undef, length(records), length(records[1].states)
    )
    for trajectory in eachindex(records), time in eachindex(records[trajectory].states)
        states[trajectory, time] =
            _state_qobj(records[trajectory].states[time], dimensions)
    end
    return states
end

function _trajectory_expect(records::Vector{_TrajectoryRecord})
    isempty(records) && return nothing
    records[1].expect === nothing && return nothing
    Ne, Nt = size(records[1].expect)
    values = Array{CF, 3}(undef, Ne, length(records), Nt)
    for trajectory in eachindex(records)
        values[:, trajectory, :] = records[trajectory].expect
    end
    return values
end

# Paper: ρ_MC(T) = mean_r |ψ_r(T)⟩⟨ψ_r(T)| (Section 4.1).
function _final_density(final_states::AbstractMatrix{CF})
    density = final_states * final_states' / size(final_states, 2)
    return (density + density') / 2
end

# Paper: ρ_MC(t) and ensemble ⟨O⟩(t) (Sections 2, 4).
function _build_solution(
        acc::_TrajectoryAccumulator, times::Vector{Float64},
        times_states::Vector{Float64},
        records::Vector{_TrajectoryRecord},
        col_gauge::Vector{Vector{Int}},
        diagnostics::_GaugeDiagnostics,
        gauge_data::_GaugeData, gauge_status::Symbol,
        dimensions;
        seed::UInt64,
        survival_rtol::Real,
        time_rtol::Real,
        time_atol::Real,
        layer3::Union{Nothing, _Layer3Prepared} = nothing,
        eigensystem_backend::Symbol = :lapack,
        final_states::Union{Nothing, Matrix{CF}} = nothing
    )
    acc.ntraj > 0 || throw(ArgumentError("cannot finalize an empty trajectory ensemble"))
    Ng = size(gauge_data.shifts, 2)
    diagnostics.ngauges == Ng || throw(
        DimensionMismatch(
            "gauge diagnostics contain $(diagnostics.ngauges) gauges; expected $Ng"
        )
    )
    expect = isempty(acc.mean) ? nothing : copy(acc.mean)
    expect_std = expect === nothing ? nothing :
        sqrt.(max.(acc.M2 ./ acc.ntraj, 0.0))
    expect_sem = expect === nothing ? nothing : acc.ntraj > 1 ?
        sqrt.(max.(acc.M2 ./ (acc.ntraj * (acc.ntraj - 1)), 0.0)) :
        fill(NaN, size(acc.mean))
    gauge_diagnostics = (;
        count = Ng,
        method = gauge_data.method,
        status = gauge_status,
        switches = diagnostics.switches,
        residence_time = copy(diagnostics.residence_time),
        jumps_by_gauge = copy(diagnostics.jumps_by_gauge),
    )
    layer3_diagnostics = layer3 === nothing ? nothing : (;
            sizes = [length(space.reduced.I) for space in layer3.spaces],
            accepted_projections = diagnostics.accepted_projections,
            fallback_segments = diagnostics.fallback_segments,
            maximum_residual = diagnostics.maximum_residual,
        )
    saved_final_states = final_states === nothing ? nothing : copy(final_states)

    return DiSLOUSolution(
        acc.ntraj,
        copy(times),
        copy(times_states),
        _aggregate_states(acc, dimensions),
        expect,
        [copy(record.col_times) for record in records],
        [copy(record.col_which) for record in records],
        copy.(col_gauge),
        acc.njumps_total,
        copy(acc.jumps_by_channel),
        true,
        seed,
        Float64(survival_rtol),
        Float64(time_rtol),
        Float64(time_atol),
        expect_std,
        expect_sem,
        gauge_diagnostics,
        layer3_diagnostics,
        eigensystem_backend,
        _trajectory_states(records, dimensions),
        _trajectory_expect(records),
        saved_final_states,
        saved_final_states === nothing ? nothing : _final_density(saved_final_states),
    )
end

"""
    expect_mean(sol::DiSLOUSolution, e::Integer = 1)::Union{Nothing,Vector{ComplexF64}}

Extract the ensemble expectation ``⟨O⟩(t)`` for one observable.

Select the observable by its one-based index `e` in the solve's `e_ops` collection.
Return a copy of `sol.expect[e, :]`, ordered by `sol.times`, or `nothing` when no
observables were requested. An out-of-range index raises `BoundsError` when
observables are present.

Values may be complex, including for non-Hermitian observables. Use
`real.(expect_mean(sol, e))` when only the real part is needed.

See also [`expect_sem`](@ref), [`dislou_solve`](@ref), [`DiSLOUSolution`](@ref).

# Examples

```jldoctest
julia> using DiSLOUTrajectories

julia> sol = dislou_solve(zeros(ComplexF64, 2, 2), ComplexF64[1, 0],
           [0.0, 1.0], Matrix{ComplexF64}[];
           gauge_set = zeros(ComplexF64, 0, 1),
           e_ops = [ComplexF64[1 0; 0 0]], ntraj = 2, ensemblealg = :serial);

julia> expect_mean(sol) == ComplexF64[1, 1]
true
```
"""
function expect_mean(sol::DiSLOUSolution, e::Integer = 1)
    return sol.expect === nothing ? nothing : sol.expect[e, :]
end

@doc raw"""
    expect_sem(sol::DiSLOUSolution, e::Integer = 1)::Union{Nothing,Vector{Float64}}

Extract the estimated standard error of the mean for one observable.

Select the observable by its one-based index `e` in the solve's `e_ops` collection.
Return a copy of `sol.expect_sem[e, :]`, ordered by `sol.times`, or `nothing` when
no observables were requested. An out-of-range index raises `BoundsError` when
observables are present. Every entry is `NaN` for a single trajectory.

This estimates Monte Carlo sampling error. It differs from `sol.expect_std`,
which measures population spread, and does not estimate numerical integration
or Layer III approximation error.

See also [`expect_mean`](@ref), [`dislou_solve`](@ref), [`DiSLOUSolution`](@ref).

# Examples

```jldoctest
julia> using DiSLOUTrajectories

julia> sol = dislou_solve(zeros(ComplexF64, 2, 2), ComplexF64[1, 0],
           [0.0, 1.0], Matrix{ComplexF64}[];
           gauge_set = zeros(ComplexF64, 0, 1),
           e_ops = [ComplexF64[1 0; 0 0]], ntraj = 2, ensemblealg = :serial);

julia> expect_sem(sol) == [0.0, 0.0]
true
```

# Extended help

For ``n`` trajectories with expectations ``x_k`` at one time, the estimate is

```math
\operatorname{SEM} =
\sqrt{\frac{\sum_{k=1}^{n}|x_k-\bar{x}|^2}{n(n-1)}},
\qquad \bar{x}=\frac{1}{n}\sum_{k=1}^{n}x_k,\quad n>1.
```

For complex expectations, squared absolute deviations define one real uncertainty,
not separate real-part and imaginary-part uncertainties.
"""
expect_sem(sol::DiSLOUSolution, e::Integer = 1) =
    sol.expect_sem === nothing ? nothing : sol.expect_sem[e, :]

function Base.show(io::IO, sol::DiSLOUSolution)
    Ne = sol.expect === nothing ? 0 : size(sol.expect, 1)
    return print(
        io, "DiSLOUSolution(ntraj=$(sol.ntraj), Ne=$Ne, ",
        "Nt=$(length(sol.times)), total_jumps=$(sol.njumps_total))"
    )
end
