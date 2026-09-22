# Top-level user-facing entry point.

function _validated_seed(seed::Integer)
    0 <= seed <= typemax(UInt64) ||
        throw(ArgumentError("seed must be in 0:typemax(UInt64)"))
    return UInt64(seed)
end

function _validated_tlist(tlist)
    tlist isa AbstractString && throw(ArgumentError("tlist must be a nonempty iterable of times"))
    tl = try
        collect(Float64, tlist)
    catch
        throw(ArgumentError("tlist must be a nonempty iterable of real times"))
    end
    isempty(tl) && throw(ArgumentError("tlist must not be empty"))
    all(isfinite, tl) || throw(ArgumentError("tlist must contain only finite times"))
    issorted(tl) || throw(ArgumentError("tlist must be sorted ascending"))
    allunique(tl) || throw(ArgumentError("tlist must contain unique times"))
    return tl
end

function _validated_saveat(saveat, tlist::Vector{Float64})
    saveat === nothing && return [last(tlist)]
    saveat isa Bool &&
        throw(ArgumentError("saveat must be a finite real scalar or nonempty iterable"))
    values = if saveat isa Real
        [Float64(saveat)]
    else
        saveat isa AbstractString &&
            throw(ArgumentError("saveat must be a finite real scalar or nonempty iterable"))
        try
            collect(Float64, saveat)
        catch
            throw(ArgumentError("saveat must be a finite real scalar or nonempty iterable"))
        end
    end
    isempty(values) && throw(ArgumentError("saveat must not be empty"))
    all(isfinite, values) || throw(ArgumentError("saveat must contain only finite times"))
    all(first(tlist) .<= values .<= last(tlist)) ||
        throw(ArgumentError("saveat must lie within the tlist interval"))
    allunique(values) || throw(ArgumentError("saveat must contain unique times"))
    sort!(values)
    return values
end

function _validate_ensemble_options(ntraj::Int, ensemblealg::Symbol, seed::Integer)
    ntraj > 0 || throw(ArgumentError("ntraj must be positive"))
    ensemblealg in (:serial, :threads, :distributed) ||
        throw(ArgumentError("ensemblealg must be :serial, :threads, or :distributed, got $ensemblealg"))
    ensemblealg === :distributed && nprocs() <= 1 &&
        throw(ArgumentError("ensemblealg=:distributed requires worker processes"))
    return _validated_seed(seed)
end

function _validated_hysteresis(hysteresis::Real)
    isfinite(hysteresis) && 0 < hysteresis <= 1 || throw(
        ArgumentError(
            "hysteresis must satisfy 0 < hysteresis <= 1, got $hysteresis"
        )
    )
    return Float64(hysteresis)
end

function _resolved_observable_storage(storage, layer3::Bool)::Symbol
    (storage === :auto || storage === :sparse || storage === :dense) || throw(
        ArgumentError(
            "observable_storage must be :auto, :sparse, or :dense, got $storage"
        )
    )
    return storage === :auto ? (layer3 ? :dense : :sparse) : storage
end

function _validated_operator_input(operator, name::AbstractString, N::Int, dimensions)
    if hasproperty(operator, :data) && hasproperty(operator, :dimensions)
        matrix, _ = _operator_matrix(
            operator, name;
            expected_dimensions = dimensions
        )
        return matrix
    end
    operator isa AbstractMatrix ||
        throw(ArgumentError("$name must be a QuantumToolbox.QuantumObject or matrix"))
    return _checked_matrix(Matrix{CF}(operator), name; N)
end

function _validated_problem(
        H, ψ0, c_ops::AbstractVector,
        e_ops::Union{Nothing, AbstractVector},
        observable_storage
    )
    Hmat, dimensions = if hasproperty(H, :data) && hasproperty(H, :dimensions)
        _operator_matrix(H, "H"; hermitian = true)
    elseif H isa AbstractMatrix
        (_checked_matrix(Matrix{CF}(H), "H"; hermitian = true), nothing)
    else
        throw(ArgumentError("H must be a QuantumToolbox.QuantumObject or matrix"))
    end
    N = size(Hmat, 1)
    C = Matrix{CF}[
        _validated_operator_input(operator, "c_ops[$index]", N, dimensions)
            for (index, operator) in enumerate(c_ops)
    ]
    Z = _validated_observables(e_ops, N, dimensions, observable_storage)

    raw_state = if hasproperty(ψ0, :data) && hasproperty(ψ0, :dimensions)
        dimensions !== nothing && ψ0.dimensions.to != dimensions.to &&
            throw(ArgumentError("ψ0 dimensions must match Hamiltonian"))
        ψ0.data
    elseif ψ0 isa AbstractVector
        ψ0
    else
        throw(ArgumentError("ψ0 must be a QuantumToolbox.QuantumObject or vector"))
    end
    raw_state isa AbstractVector || throw(ArgumentError("ψ0 must be a ket"))
    length(raw_state) == N || throw(ArgumentError("ψ0 dimension must match Hamiltonian"))
    all(isfinite, raw_state) || throw(ArgumentError("ψ0 must contain only finite values"))
    state = float.(raw_state)
    # Scale components first: even abs(z) or norm(state) can overflow for finite inputs.
    scale = maximum(z -> max(abs(real(z)), abs(imag(z))), state)
    scale > 0 || throw(ArgumentError("ψ0 must be nonzero"))
    state ./= scale
    state = normalize!(Vector{CF}(state))
    return (; H = Hmat, C, Z, ψ0 = state, dimensions)
end

function _validated_layer3_size_input(layer3_sizes, N::Int)
    if layer3_sizes isa Integer && !(layer3_sizes isa Bool)
        1 <= layer3_sizes <= N ||
            throw(ArgumentError("layer3_sizes must lie in 1:$N"))
        return Int(layer3_sizes)
    elseif layer3_sizes isa AbstractVector
        all(size -> size isa Integer && !(size isa Bool), layer3_sizes) ||
            throw(ArgumentError("layer3_sizes must contain integers"))
        all(size -> 1 <= size <= N, layer3_sizes) ||
            throw(ArgumentError("layer3_sizes must lie in 1:$N"))
        return Int.(layer3_sizes)
    else
        throw(ArgumentError("layer3_sizes must be an integer or an integer vector"))
    end
end

function _validated_layer3_sizes(layer3_sizes, Ng::Int)
    layer3_sizes isa Int && return fill(layer3_sizes, Ng)
    length(layer3_sizes) == Ng || throw(
        ArgumentError(
            "layer3_sizes must contain one size or one size per gauge"
        )
    )
    return copy(layer3_sizes)
end

@doc raw"""
    dislou_solve(H, ψ0, tlist, c_ops; <keyword arguments>)

Compute the average density matrix ``ρ_{MC}(t)`` and expectations ``⟨O⟩(t)``
using locally optimal unraveling (Sections 2–3).

Supply the required `gauge_set` as a finite shift matrix or a result from
[`discover_gauges`](@ref). Layers I and II (optimal unraveling and diagonal propagation) are run by default; set `layer3=true`
and supply `layer3_sizes` to enable reduced basis propagation.

See also [`discover_gauges`](@ref), [`DiSLOUSolution`](@ref),
[`expect_mean`](@ref), [`expect_sem`](@ref), [`backend_info`](@ref).

# Examples

A closed two-level system has constant occupation and no jump events:

```jldoctest
julia> using DiSLOUTrajectories

julia> sol = dislou_solve(zeros(ComplexF64, 2, 2), ComplexF64[1, 0],
           [0.0, 1.0], Matrix{ComplexF64}[];
           gauge_set = zeros(ComplexF64, 0, 1),
           e_ops = [ComplexF64[1 0; 0 0]], ntraj = 2, ensemblealg = :serial);

julia> (size(sol.expect), all(isone, expect_mean(sol)), sol.njumps_total)
((1, 2), true, 0)
```

# Extended help

For a Hamiltonian ``H`` and collapse operators ``C_μ``, the ensemble density
operator obeys the Lindblad equation, with ``ℏ=1``:

```math
\frac{dρ}{dt} = -i[H,ρ] + \sum_μ\left(
C_μρ C_μ^† - \frac{1}{2}\{C_μ^† C_μ,ρ\}\right).
```

Layer I selects a gauge that reduces jump activity. Layer II diagonalizes each
gauge's effective Hamiltonian and samples jump times from the no-jump survival
probability. Optional Layer III propagates in local slow-mode eigenspaces
and falls back to the full space if the projection residual is higher than the specified tolerance.

# Arguments

- `H`: Finite, Hermitian system Hamiltonian, supplied as a
  `QuantumToolbox.QuantumObject` operator or an `N × N` CPU matrix.
- `ψ0`: Nonzero, finite initial ket at `first(tlist)`, supplied as a
  `QuantumToolbox.QuantumObject` ket or a CPU vector of length `N`. The solver
  normalizes the state. Quantum-object subsystem dimensions must match `H`.
- `tlist`: Nonempty iterable of finite, strictly increasing real times.
  Evolution runs from `first(tlist)` to `last(tlist)`; expectation values are
  sampled at these times. A single time is allowed.
- `c_ops`: `AbstractVector` of time-independent collapse operators, each a
  compatible quantum operator or `N × N` CPU matrix. Include the square root
  of each decay rate in its operator. An empty vector describes a closed system.
- `e_ops`: `AbstractVector` of compatible operators whose expectation values
  are sampled at `tlist`. Operators may be non-Hermitian. Defaults to `nothing`;
  `nothing` and an empty vector both disable expectation-value recording.
- `ntraj::Int`: Positive number of independent trajectories. Defaults to `500`.
- `max_jumps::Int`: Per-trajectory jump-count limit. Defaults to `1_000_000`.
  Reaching the limit before the trajectory completes raises an error.
- `ensemblealg::Symbol`: Trajectory execution mode: `:serial`, `:threads` (default),
  or `:distributed`. Distributed execution requires worker processes with
  DiSLOUTrajectories available; add workers and load DiSLOUTrajectories on them before solving.
- `seed::Integer`: Integer in `0:typemax(UInt64)` used to derive a random stream for
  each trajectory. Defaults to `0`.
- `gauge_set`: Required finite gauge set. Supply the complete
  result of [`discover_gauges`](@ref), or an `Nc × Ng` matrix of finite complex
  shifts, where `Nc = length(c_ops)` and `Ng ≥ 1`. Column `g` contains the
  shifts for gauge `g`, in collapse-channel order. For a closed system, use
  a `0 × 1` matrix, such as `zeros(ComplexF64, 0, 1)`.
- `layer3::Bool`: Whether to enable approximate propagation in local eigenspaces.
  Defaults to `false`.
- `layer3_sizes`: Number of eigenmodes retained in each gauge's local space.
  Supply one integer for all gauges or an integer vector of length `Ng`, with
  each entry in `1:N`. Required when `layer3=true`; otherwise must be `nothing`
  (the default). The least-decaying modes are retained.
- `observable_storage::Symbol`: `:auto` (default), `:sparse`, or `:dense`. Sparse mode
  applies observables and exact post-jump collapse operators in the physical
  basis; dense mode stores their transformed matrices. `:auto` selects
  `:sparse` for Layers I–II and `:dense` when Layer III is enabled.
- `residual_tolerance::Real`: The paper's ``r_{tol}``: a positive, finite upper bound on
  ``r_m^{(g)}(ψ)`` for normalized states (Eq. 21). Defaults to `1e-3`. A rejected projection
  uses exact full-space propagation for that segment.
- `hysteresis::Real`: The paper's ``η``, with ``0 < η ≤ 1`` (Eq. 13). After a
  jump, switch only if the best gauge's activity is strictly less than
  `hysteresis` times the current gauge's activity. Defaults to `0.5`.
- `condition_limit::Real`: Finite value at least `1` bounding the LAPACK estimate of
  the one-norm condition number of every gauge's eigenvector basis. Defaults
  to `sqrt(1e-2 / eps(Float64))`, approximately `6.7e6`. An unacceptable basis
  raises an error.
- `survival_rtol::Real`: Positive, finite tolerance on the logarithmic survival
  residual in the first-passage convergence test. Defaults to `1e-10`.
- `time_rtol::Real`: Positive, finite relative tolerance on the jump time or its
  bracket width, scaled by the remaining segment duration. Defaults to `1e-12`.
- `time_atol::Real`: Nonnegative, finite absolute jump-time tolerance, in the same
  units as `tlist`. Defaults to `0.0`.
- `first_passage_maxiter::Int`: Positive iteration limit for each jump-time root
  solve. Defaults to `100`. Exhaustion raises
  [`FirstPassageConvergenceError`](@ref).
- `first_passage_method::Symbol`: `:automatic` (default, currently `:log_survival`),
  `:survival`, `:log_survival`, or `:log_survival_predictor`. The first two
  methods use ``s(t)`` or ``ℓ(t) = -\ln s(t)`` Newton steps within a bracket;
  the predictor variant supplies an initial estimate to the log-survival solve.
- `saveat`: Absolute times at which to save ensemble density operators.
  Supply a finite real scalar or a nonempty iterable of unique finite times
  within the `tlist` interval; iterable inputs are sorted. Defaults to
  `nothing`, which saves only `last(tlist)`. A scalar selects one time, not a
  sampling interval. Use `saveat=tlist` to save states at every observation time.
- `save_trajectories::Bool`: Whether to retain each trajectory's kets at `saveat`
  and expectation values at `tlist`, in addition to the ensemble results.
  Defaults to `false`.
- `save_final_states::Bool`: Whether to retain each final ket as a column of
  `sol.final_states` and their average density matrix as `sol.final_density`.
  Defaults to `false`; this option is independent of `saveat` and
  `save_trajectories`.

# Notes

- Gauge column `g` contains ``ζ_μ^{(g)}``. It defines
  ``C_μ^{(g)} = C_μ + ζ_μ^{(g)} I`` and
  ``H^{(g)} = H + \frac{i}{2}\sum_μ(ζ_μ^{(g)} C_μ^† - ζ_μ^{(g)*} C_μ)``
  (Eq. 9).
  This preserves the Lindblad generator. Jump records refer to these shifted
  operators.
- Layers I–II are exact up to floating-point and first-passage tolerances.
  Layer III accepts a projection when ``r_m^{(g)}(ψ) ≤ r_{tol}``, where
  ``P_m^{(g)} = Q_m^{(g)} Q_m^{(g)†}`` projects onto ``\mathcal K_m^{(g)}``
  (Eqs. 20–21, C.1).
- Inputs are converted to `ComplexF64` operators and states and `Float64`
  times. Every gauge requires a usable dense eigensystem, including in sparse
  observable mode. Time-dependent operators and ODE-solver keywords are not yet
  supported.
- The results may show a dependence on small changes due to the eigensystem backend. 
  See [`backend_info`](@ref); the backend actually used is recorded in 
  `sol.eigensystem_backend`.

# Returns

- `sol::DiSLOUSolution`: Ensemble density operators at `sol.times_states`,
  expectation means and uncertainties at `sol.times`, jump records,
  gauge diagnostics, and any requested trajectory data. See
  [`DiSLOUSolution`](@ref) for all fields and array dimensions, and
  [`expect_mean`](@ref) and [`expect_sem`](@ref) for observables.
"""
function dislou_solve(
        H, ψ0, tlist, c_ops;
        e_ops::Union{Nothing, AbstractVector} = nothing,
        ntraj::Int = 500,
        max_jumps::Int = 1_000_000,
        ensemblealg::Symbol = :threads,
        seed::Integer = 0,
        gauge_set,
        layer3::Bool = false,
        layer3_sizes = nothing,
        observable_storage = :auto,
        residual_tolerance::Real = 1.0e-3,
        hysteresis::Real = 0.5,
        condition_limit::Real = _DEFAULT_CONDITION_LIMIT,
        survival_rtol::Real = 1.0e-10,
        time_rtol::Real = 1.0e-12,
        time_atol::Real = 0.0,
        first_passage_maxiter::Int = 100,
        first_passage_method::Symbol = :automatic,
        saveat = nothing,
        save_trajectories::Bool = false,
        save_final_states::Bool = false
    )
    c_ops isa AbstractVector || throw(ArgumentError("c_ops must be an AbstractVector"))
    base_seed = _validate_ensemble_options(ntraj, ensemblealg, seed)
    _validate_root_tolerances(survival_rtol, time_rtol, time_atol)
    first_passage_maxiter > 0 ||
        throw(ArgumentError("first_passage_maxiter must be positive"))
    hysteresis = _validated_hysteresis(hysteresis)
    condition_limit = _validated_condition_limit(condition_limit)
    method = _validated_first_passage_method(first_passage_method)
    isfinite(residual_tolerance) && residual_tolerance > 0 ||
        throw(ArgumentError("residual_tolerance must be finite and positive"))
    layer3 || layer3_sizes === nothing ||
        throw(ArgumentError("layer3_sizes requires layer3=true"))
    layer3 && layer3_sizes === nothing &&
        throw(ArgumentError("layer3=true requires layer3_sizes"))
    resolved_observable_storage =
        _resolved_observable_storage(observable_storage, layer3)
    tl = _validated_tlist(tlist)
    times_states = _validated_saveat(saveat, tl)
    time_origin = first(tl)
    tl_local = tl .- time_origin
    states_local = times_states .- time_origin
    duration = last(tl_local)
    problem = _validated_problem(
        H, ψ0, c_ops, e_ops, resolved_observable_storage
    )
    layer3_size_input = layer3 ?
        _validated_layer3_size_input(layer3_sizes, size(problem.H, 1)) : nothing
    gauge_data = _validated_gauge_data(gauge_set, length(c_ops))
    isempty(c_ops) && size(gauge_data.shifts, 2) != 1 &&
        throw(ArgumentError("empty collapse channels require exactly one gauge"))
    validated_layer3_sizes = layer3 ? _validated_layer3_sizes(
            layer3_size_input, size(gauge_data.shifts, 2)
        ) : nothing
    prepared = ensemblealg === :distributed && !layer3 ? nothing : _prepare_layer1(
            problem.H, problem.C, problem.Z, gauge_data, hysteresis;
            observable_storage = resolved_observable_storage, condition_limit
        )

    layer3_prepared = layer3 ? _prepare_layer3(
            prepared, problem.ψ0,
            validated_layer3_sizes, residual_tolerance
        ) : nothing

    common = (;
        survival_rtol,
        time_rtol,
        time_atol,
        max_jumps,
        first_passage_maxiter,
        first_passage_method = method,
        save_final_states,
        times_states = states_local,
        save_trajectories,
    )
    if ensemblealg === :serial
        acc, diagnostics, final_states, records, col_gauge = _solve_gauges_serial(
            prepared, layer3_prepared, problem.ψ0, tl_local, duration, ntraj, base_seed;
            common...,
        )
        backend = _eigensystem_backend(prepared)
    elseif ensemblealg === :threads
        acc, diagnostics, final_states, records, col_gauge = _solve_gauges_threaded(
            prepared, layer3_prepared, problem.ψ0, tl_local, duration, ntraj, base_seed;
            common...,
        )
        backend = _eigensystem_backend(prepared)
    else
        acc, diagnostics, final_states, records, col_gauge, backend =
            _solve_gauges_distributed(
            problem.H, problem.C, problem.Z, gauge_data, problem.ψ0,
            tl_local, duration, ntraj, base_seed;
            survival_rtol, time_rtol, time_atol, max_jumps,
            first_passage_maxiter,
            hysteresis,
            condition_limit,
            observable_storage = resolved_observable_storage,
            first_passage_method = method,
            layer3_sizes = layer3_prepared === nothing ? nothing :
                [length(space.reduced.I) for space in layer3_prepared.spaces],
            residual_tolerance,
            save_final_states,
            times_states = states_local,
            save_trajectories,
        )
        parent_backend = prepared === nothing ? backend :
            _eigensystem_backend(prepared)
        backend === parent_backend || (backend = :mixed)
    end
    iszero(time_origin) ||
        foreach(record -> record.col_times .+= time_origin, records)
    return _build_solution(
        acc, tl, times_states, records, col_gauge, diagnostics,
        gauge_data, gauge_data.status, problem.dimensions;
        seed = base_seed,
        survival_rtol,
        time_rtol,
        time_atol,
        layer3 = layer3_prepared,
        eigensystem_backend = backend,
        final_states,
    )
end
