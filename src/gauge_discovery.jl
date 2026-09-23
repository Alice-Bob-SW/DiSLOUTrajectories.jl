# Finite gauge inputs and exact generator-preserving shifts.

struct _GaugeData
    shifts::Matrix{ComplexF64}
    method::Symbol
    centers::Matrix{ComplexF64}
    weights::Vector{Float64}
    diagnostics::NamedTuple
    status::Symbol
end

_copy_gauge_diagnostic(x::AbstractArray) = deepcopy(x)
_copy_gauge_diagnostic(x::NamedTuple) = NamedTuple{propertynames(x)}(
    Tuple(_copy_gauge_diagnostic(v) for v in values(x))
)
_copy_gauge_diagnostic(x) = x

function _validated_gauge_data(gauge_set, Nc::Int)::_GaugeData
    Nc >= 0 || throw(ArgumentError("number of collapse channels must be nonnegative"))
    if gauge_set isa AbstractMatrix
        shifts = Matrix{CF}(gauge_set)
        size(shifts, 1) == Nc || throw(
            DimensionMismatch(
                "gauge shifts must have $Nc rows, got $(size(shifts, 1))"
            )
        )
        size(shifts, 2) >= 1 || throw(ArgumentError("gauge shifts must contain at least one column"))
        all(isfinite, shifts) || throw(ArgumentError("gauge shifts must be finite"))
        ng = size(shifts, 2)
        return _GaugeData(
            shifts, :manual, zeros(CF, 0, ng),
            ones(Float64, ng), (;), :manual
        )
    end
    shifts = Matrix{CF}(gauge_set.shifts)
    size(shifts, 1) == Nc || throw(
        DimensionMismatch(
            "gauge shifts must have $Nc rows, got $(size(shifts, 1))"
        )
    )
    ng = size(shifts, 2)
    ng >= 1 || throw(ArgumentError("gauge shifts must contain at least one column"))
    all(isfinite, shifts) || throw(ArgumentError("gauge shifts must be finite"))
    gauge_set.centers isa AbstractMatrix || throw(ArgumentError("gauge centers must be a matrix"))
    centers = Matrix{CF}(gauge_set.centers)
    size(centers, 2) == ng || throw(
        DimensionMismatch(
            "gauge centers must have $ng columns, got $(size(centers, 2))"
        )
    )
    all(isfinite, centers) || throw(ArgumentError("gauge centers must be finite"))
    gauge_set.weights isa AbstractVector || throw(ArgumentError("gauge weights must be a vector"))
    weights = Vector{Float64}(gauge_set.weights)
    length(weights) == ng || throw(
        DimensionMismatch(
            "gauge weights must have length $ng, got $(length(weights))"
        )
    )
    all(isfinite, weights) || throw(ArgumentError("gauge weights must be finite"))
    all(>=(0), weights) || throw(ArgumentError("gauge weights must be nonnegative"))
    sum(weights) > 0 || throw(ArgumentError("gauge weights must have positive sum"))
    diagnostics = _copy_gauge_diagnostic(gauge_set.diagnostics)
    return _GaugeData(
        shifts, gauge_set.method, centers, weights,
        diagnostics, :validated
    )
end

function _validated_shifts(shifts, nchannels::Int)
    parsed = Tuple{Int, CF}[]
    for shift in shifts
        length(shift) == 2 || throw(ArgumentError("shifts must contain (channel, zeta) pairs"))
        idx, ζraw = shift
        idx isa Integer || throw(ArgumentError("shift channel indices must be integers"))
        1 <= idx <= nchannels || throw(ArgumentError("shift channel $idx is out of range"))
        isfinite(ζraw) || throw(ArgumentError("collapse shifts must be finite"))
        push!(parsed, (Int(idx), CF(ζraw)))
    end
    indices = first.(parsed)
    length(unique(indices)) == length(indices) ||
        throw(ArgumentError("shift channel indices must be unique, got $indices"))
    return parsed
end

# Paper: H^(g), C_μ^(g) (Eq. 9).
function _shifted_problem(
        H::Matrix{CF}, C::Vector{Matrix{CF}},
        z::AbstractVector{CF}
    )
    N = size(H, 1)
    size(H, 2) == N || throw(DimensionMismatch("H must be square"))
    length(z) == length(C) || throw(
        DimensionMismatch(
            "shift vector has length $(length(z)); expected $(length(C))"
        )
    )
    all(Cμ -> size(Cμ) == (N, N), C) ||
        throw(DimensionMismatch("every collapse operator must have size ($N, $N)"))
    all(isfinite, z) || throw(ArgumentError("collapse shifts must be finite"))
    Hshift = copy(H)
    Cshift = Matrix{CF}[copy(Cμ) for Cμ in C]
    for μ in eachindex(Cshift)
        ζ = z[μ]
        Hshift .+= (im / 2) .* (ζ .* C[μ]' .- conj(ζ) .* C[μ])
        @inbounds for i in 1:N
            Cshift[μ][i, i] += ζ
        end
    end
    return Hshift, Cshift
end

# Paper: H^(g), C_μ^(g) (Eq. 9).
function _shifted_problem(
        Hmat::AbstractMatrix{CF},
        C::AbstractVector{<:AbstractMatrix{CF}},
        shifts::AbstractVector{<:Tuple{Int, <:Number}}
    )
    parsed = _validated_shifts(shifts, length(C))
    z = zeros(CF, length(C))
    for (idx, ζ) in parsed
        z[idx] = ζ
    end
    return _shifted_problem(Matrix{CF}(Hmat), Matrix{CF}[Matrix{CF}(Cμ) for Cμ in C], z)
end

# Paper: H^(g), C_μ^(g) (Eq. 9).
function _shifted_problem(H, C::AbstractVector, shifts::AbstractVector{<:Tuple})
    dims = Tuple(first(H.dims))
    Tuple(last(H.dims)) == dims || throw(DimensionMismatch("H must be an operator"))
    all(op -> Tuple(first(op.dims)) == dims && Tuple(last(op.dims)) == dims, C) ||
        throw(DimensionMismatch("every collapse operator must have tensor dimensions $dims"))
    Hshift, Cshift = _shifted_problem(
        Matrix{CF}(H.data), Matrix{CF}[Matrix{CF}(op.data) for op in C], shifts
    )
    hdata = H.data isa SparseMatrixCSC ? sparse(Hshift) : Hshift
    cdata = [
        (C[i].data isa SparseMatrixCSC ? sparse(Cμ) : Cμ)
            for (i, Cμ) in enumerate(Cshift)
    ]
    return QuantumObject(hdata; dims = H.dimensions),
        [QuantumObject(cdata[i]; dims = C[i].dimensions) for i in eachindex(cdata)]
end

# Workers under ensemblealg=:distributed deserialize the pilot closure, so it and
# its runner must live in the core module: workers need not load Clustering.
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

_mean(x) = sum(x) / length(x)

# Paper: ᾱ_j^(r) and terminal ⟨C_μ⟩ averages (Eqs. A.7–A.8).
function _preliminary_pilot_results(
        Hrun, Crun, e_ops, tlist, tail, amplitudes, mode_dims, seed, nsave, ensemblealg
    )
    nmodes, nseeds = size(amplitudes)
    # QuantumToolbox's retained-run keyword predates DiSLOUTrajectories.
    pilot_storage = (; Symbol("keep_" * "runs_results") => Val(true))
    relax = function (point)
        psi0 = tensor((coherent(Int(mode_dims[mode]), amplitudes[mode, point]) for mode in 1:nmodes)...)
        sol = mcsolve(
            Hrun, psi0, tlist, Crun; e_ops, ntraj = 1,
            rng = _traj_rng(seed, point),
            pilot_storage..., saveat = tlist,
            progress_bar = Val(false)
        )
        return (; means = CF[_mean(@view sol.expect[mode, 1, tail]) for mode in 1:nmodes], occupations = Float64[_mean(real.(@view sol.expect[nmodes + mode, 1, tail])) for mode in 1:nmodes], collapses = CF[_mean(@view sol.expect[2nmodes + channel, 1, tail]) for channel in eachindex(Crun)], trace_states = point <= nsave ? hcat((CF.(vec(state.data)) for state in sol.states)...) : nothing, trace_means = point <= nsave ? Matrix{CF}(@view sol.expect[1:nmodes, 1, :]) : nothing, trace_occupations = point <= nsave ? Float64.(real.(@view sol.expect[(nmodes + 1):2nmodes, 1, :])) : nothing, jump_times = point <= nsave ? copy(sol.col_times[1]) : nothing, jump_channels = point <= nsave ? copy(sol.col_which[1]) : nothing)
    end
    return _run_discovery_indices(relax, nseeds, ensemblealg)
end

"""
    discover_gauges(H, c_ops; <keyword arguments>)
    discover_gauges(hamiltonian, collapse_operators; <keyword arguments>)

Discover the paper's shifts `ζ_μ^(g)` (Eq. 9) for [`dislou_solve`](@ref).

The default `method=:trajectories` clusters terminal amplitudes from preliminary
quantum trajectories and requires `Clustering.jl`. Otherwise, you can provide symbolic 
constructors for the Hamiltonian and collapse operators, select `method=:semiclassical` and load `QuantumCumulants.jl` to find
stable mean-field fixed points. Install these optional packages before loading them.

See also [`dislou_solve`](@ref), [`DiSLOUSolution`](@ref).

# Examples

Find the single vacuum gauge of a damped mode with trajectory discovery:

```jldoctest
julia> using DiSLOUTrajectories, QuantumToolbox, Clustering

julia> a = destroy(2);

julia> gauges = discover_gauges(0.0 * a, [a];
           mode_ops = [a], mode_dims = [2], discovery_time = 1.0,
           seed_radii = [0.0], cluster_scales = [1.0], step = 1.0,
           nseeds = 2, min_neighbors = 1, min_weight = 0.0,
           seed = 1, ensemblealg = :serial);

julia> (gauges.method, size(gauges.shifts), all(iszero, gauges.shifts))
(:trajectories, (1, 1), true)
```

Find the same vacuum fixed point from symbolic constructors:

```jldoctest
julia> using DiSLOUTrajectories, QuantumCumulants

julia> # H
       hamiltonian(a) = adjoint(a) * a;

julia> # C = a
       collapse_operators(a) = [a];

julia> gauges = discover_gauges(hamiltonian, collapse_operators;
           method = :semiclassical, limits = (1.0,));

julia> (gauges.method, size(gauges.shifts), all(z -> abs(z) < 1e-9, gauges.shifts))
(:semiclassical, (1, 1), true)
```

# Extended help

Trajectory discovery starts in random coherent product states. Each retained
cluster defines shifts from the negative cluster averages of the physical
collapse-operator expectations. Semiclassical discovery derives first-order bosonic
mean-field equations, finds stable fixed points, and evaluates collapse amplitudes
at those points.

# Arguments

The accepted arguments depend on `method`. The trajectory method is the default;
the semiclassical method must be selected explicitly.

## Trajectory discovery

- `H`: Time-independent Hermitian Hamiltonian as a
  `QuantumToolbox.QuantumObject` operator. Its tensor dimensions must match
  `mode_dims`. This method uses QuantumToolbox's `mcsolve` for the preliminary
  trajectories.
- `c_ops`: Vector of time-independent `QuantumToolbox.QuantumObject` collapse
  operators with the same tensor dimensions as `H`. Include the square root of
  each physical decay rate in the corresponding operator.
- `method::Symbol`: Set to `:trajectories` (the default).
- `mode_ops`: Nonempty collection of bosonic annihilation operators, embedded
  in the full tensor-product Hilbert space. Each operator must have the same
  dimensions as `H`; their order defines the rows of the returned centers.
- `mode_dims`: One integer Hilbert-space truncation dimension, at least `2`,
  per mode, in tensor-product order.
- `discovery_time`: Positive, finite duration of each preliminary trajectory,
  which starts at time zero.
- `seed_radii`: One finite, nonnegative coherent-amplitude radius per mode.
  Initial amplitudes are sampled uniformly in each complex disk.
- `cluster_scales`: Vector of positive, finite amplitude scales, one per mode.
  Real and imaginary parts of each terminal mode mean are divided by its
  scale before Euclidean DBSCAN clustering.
- `step`: Positive, finite spacing of the preliminary observation grid.
  Defaults to `discovery_time / 40`. The endpoint `discovery_time` is always
  included.
- `nseeds::Int`: Positive number of initial coherent product states, with one
  trajectory per state. Defaults to `600`.
- `terminal_window`: Finite, nonnegative duration at the end of the run over
  which sampled mode, occupation, and collapse expectations are averaged.
  Defaults to `0.0`, which uses the final sample. Values greater than
  `discovery_time` use the whole observation grid.
- `preliminary_shifts`: Optional vector of finite complex shifts, one per
  collapse channel, applied during the preliminary runs with a compensating
  Hamiltonian shift. Defaults to `nothing`, meaning zero shifts. Returned
  gauges use expectations of the original physical collapse operators.
- `dbscan_radius`: Positive, finite DBSCAN neighborhood radius in the scaled
  real-imaginary feature space. Defaults to `1.5`.
- `min_neighbors::Int`: Positive DBSCAN core-point neighbor-count threshold,
  passed to `Clustering.dbscan`. Defaults to `10`.
- `min_weight`: Minimum retained cluster population as a fraction of all
  `nseeds` samples. Must be finite and in `[0, 1)`; defaults to `0.02`.
- `seed::Integer`: Integer in `0:typemax(UInt64)` controlling the initial amplitudes and
  preliminary trajectory streams. Defaults to `1`.
- `save_preliminary_trajectories::Int`: Nonnegative number of preliminary
  trajectories whose state, expectation, and jump traces are retained in the
  diagnostics. Defaults to `0`; values above `nseeds` retain all trajectories.
- `ensemblealg::Symbol`: Execution mode for the preliminary trajectories: `:serial`,
  `:threads` (default), or `:distributed`. Distributed execution requires
  worker processes with DiSLOUTrajectories and QuantumToolbox loaded.

## Semiclassical discovery

- `hamiltonian::Function`: Function `hamiltonian(a₁, a₂, ...)` returning a
  QuantumCumulants Hamiltonian from symbolic bosonic annihilation operators.
- `collapse_operators::Function`: Function `collapse_operators(a₁, a₂, ...)` returning
  a tuple or vector of QuantumCumulants collapse operators. It is also called
  with complex fixed-point amplitudes and must then produce a collection of
  the same length that evaluates to finite physical collapse amplitudes.
  Include square roots of decay rates in both evaluations.
- `method::Symbol`: Set to `:semiclassical`.
- `limits`: Nonempty tuple or vector of positive, finite occupation bounds,
  one per mode. Accepted fixed points satisfy `abs2(α[m]) ≤ limits[m]` up to
  numerical tolerance. These bounds define the search region.
- `parameters`: Tuple, vector, or dictionary of unique symbolic
  `parameter => value` pairs with finite numeric values. The keys must exactly
  match the parameters of the compiled mean-field system. Defaults to `()`
  for a model without symbolic parameters.

# Notes

- Load `Clustering.jl` together with `DiSLOUTrajectories.jl` before calling
  `method=:trajectories`. It activates the extension that implements the
  DBSCAN clustering step.
- Load `QuantumCumulants.jl` together with `DiSLOUTrajectories.jl` before calling
  `method=:semiclassical`.
- The two methods have separate keyword sets. Arbitrary `mcsolve` or ODE
  solver options are not forwarded.
- Trajectory cluster weights are fractions of all preliminary samples,
  including discarded samples in the denominator. Their sum can be less than
  one. Clusters are ordered by decreasing weight, then by their centers.
  Noise points and samples from discarded clusters receive label `0`.
- Stability of the semiclassical solution is determined
  from the real mean-field Jacobian. A finite multistart search can miss roots,
  and its cost grows rapidly with the number of modes.
- Discovery raises an `ArgumentError` if no trajectory cluster or stable
  semiclassical fixed point is retained.

# Returns

- `gauges::NamedTuple`: `(; shifts, method, centers, weights, diagnostics)`.
  With `Nc` collapse channels, `Nm` modes, and `Ng` retained gauges:
  - `shifts::Matrix{ComplexF64}`: `Nc × Ng` shifts `ζ_μ^(g)`; column `g` defines
    `C_μ^(g) = C_μ + ζ_μ^(g) I` (Eq. 9) for each channel `μ`.
  - `method::Symbol`: `:trajectories` or `:semiclassical`.
  - `centers::Matrix{ComplexF64}`: `Nm × Ng` cluster-mean mode amplitudes
    or stable fixed-point amplitudes.
  - `weights::Vector{Float64}`: `Ng` cluster population fractions or uniform
    semiclassical weights, as described above.
  - `diagnostics::NamedTuple`: Method-specific discovery data described below.

For trajectory discovery, `diagnostics` contains:

- `counts` and `labels`: Cluster sizes of length `Ng` and assignments of
  length `nseeds`. Positive labels index columns of `centers`.
- `terminal_means` and `terminal_occupations`: `Nm × nseeds` terminal-window
  averages of mode amplitudes and occupations.
- `terminal_collapse_means`: `Nc × nseeds` terminal-window averages of the
  physical collapse-operator expectations.
- `times`: Preliminary observation grid, including zero and `discovery_time`.
- `preliminary_traces`: `(; indices, states, means, occupations)` for the
  retained preliminary runs. `indices` contains their one-based seed indices;
  the other fields are vectors of matrices. For `Nt = length(times)` and
  `N = prod(mode_dims)`, each state matrix is `N × Nt`, and each mode-mean
  or occupation matrix is `Nm × Nt`. These vectors are empty by default.
- `preliminary_jump_times` and `preliminary_jump_channels`: Vectors
  of event-time and one-based collapse-channel vectors for the retained runs.
- The resolved discovery settings: `mode_dims`, `discovery_time`, `seed_radii`,
  `cluster_scales`, `step`, `nseeds`, `terminal_window`, `preliminary_shifts`,
  `dbscan_radius`, `min_neighbors`, `min_weight`, `seed`, and
  `save_preliminary_trajectories`. The last field records the retained count,
  capped at `nseeds`.

For semiclassical discovery, `diagnostics` contains:

- `roots`: Matrix with one column per distinct fixed point found, including
  unstable points, and one row per mode.
- `stability`: One `(; stable, max_real_eigenvalue, residual)` record per
  column of `roots`. `max_real_eigenvalue` is the largest real part of the
  Jacobian eigenvalues, and `residual` is the norm of the mean-field drift.
- `rejected`: Matrix of the unstable fixed points excluded from `centers`.
- `stable`: Boolean vector of length `Ng`, with every entry `true`.
"""
discover_gauges(model, collapse_operators; method::Symbol = :trajectories, kwargs...) =
    _discover_gauges(Val(method), model, collapse_operators; kwargs...)

# Methods live in DiSLOUTrajectoriesClusteringExt (Val{:trajectories}) and
# DiSLOUTrajectoriesQuantumCumulantsExt (Val{:semiclassical}).
function _discover_gauges end

const _DISCOVERY_EXTENSIONS = (
    trajectories = (:DiSLOUTrajectoriesClusteringExt, "Clustering"),
    semiclassical = (:DiSLOUTrajectoriesQuantumCumulantsExt, "QuantumCumulants"),
)

# Turns the bare MethodError of an unloaded or unknown method into an actionable hint.
function _discovery_error_hint(io, exc, argtypes, kwargs)
    f = exc.f === Core.kwcall && length(exc.args) >= 2 ? exc.args[2] : exc.f
    f === _discover_gauges && !isempty(argtypes) && argtypes[1] <: Val || return
    method = argtypes[1].parameters[1]
    if !haskey(_DISCOVERY_EXTENSIONS, method)
        print(io, "\nUnsupported gauge discovery method $(repr(method)); use :trajectories or :semiclassical.")
        return
    end
    extension, package = _DISCOVERY_EXTENSIONS[method]
    Base.get_extension(@__MODULE__, extension) === nothing || return
    print(io, "\nmethod=:$method requires $package; run `using $package` before calling discover_gauges.")
    return
end
