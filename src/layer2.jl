# Private Layer II: exact diagonal representation and derived operators.

# Precomputed spectral decomposition of H_eff and all derived quantities.

const _DEFAULT_CONDITION_LIMIT = sqrt(1.0e-2 / eps(Float64))

"""
    _ReducedCache

Cached data for a reduced eigenmode subspace with sorted indices `I`.

For Layer III, `V_I`, `λ_I`, and `G_I` are the paper's `V_m`, `λ_j`
(`j ∈ I`), and `G_m` for the active gauge (Section 3.3.2).

# Fields

- `V_I`  : `N×m` matrix of the selected right eigenvectors
- `λ_I`  : the `m` eigenvalues
- `G_I`  : `V_I' V_I` (the `m×m` Gram matrix / reduced metric)
- `Gnorm`: infinity norm of `G_I`
- `A_I`  : per channel `C_μ V_I`        (used for the physical post-jump state)
- `K_I`  : per channel `V_m† C_μ† C_μ V_m` (jump-weight matrix, not the eigenspace `𝒦_m`)
- `Knorms`: infinity norms of `K_I`
- `Z_I`  : per observable `O_m = V_m† O V_m`, with code observable `Z_e = O`

Sparse-observable caches leave `Z_I` empty.
"""
struct _ReducedCache
    I::Vector{Int}
    V_I::Matrix{CF}
    λ_I::Vector{CF}
    G_I::Matrix{CF}
    Gnorm::Float64
    A_I::Vector{Matrix{CF}}
    K_I::Vector{Matrix{CF}}
    Knorms::Vector{Float64}
    Z_I::Vector{Matrix{CF}}
end

const _ObservableMatrices = Union{
    Vector{Matrix{CF}},
    Vector{SparseMatrixCSC{CF, Int}},
}

"""
    _DiagonalCache

Full Layer II eigensystem `V`, `Λ`, `G` (Eqs. 15–17), prepared once per
`(H, c_ops)`. Gauge superscripts are suppressed within each cache.

Reuse the cache across initial states and random seeds. Dense recording matrices
are empty in sparse mode.
"""
mutable struct _DiagonalCache
    N::Int
    dimensions::Any
    Nc::Int
    Ne::Int
    H::Matrix{CF}
    backend::Symbol
    C::Vector{Matrix{CF}}          # dense collapse operators C_μ
    Z::_ObservableMatrices          # paper observable O, indexed by e in code
    observable_storage::Symbol
    V::Matrix{CF}                  # right eigenvectors (columns), ⟨v_j|v_j⟩ = 1
    Λ::Vector{CF}                  # eigenvalues λ_j
    Γ::Vector{Float64}             # paper γ_j = -2 Im λ_j ≥ 0 (Section 3.3.1)
    Vfac::LU{CF, Matrix{CF}, Vector{Int}}  # cached factorization for V \ ψ
    G::Matrix{CF}                  # G = V†V; I for an orthonormal eigenbasis
    Gnorm::Float64                 # infinity norm of G
    A::Vector{Matrix{CF}}          # C_μ V (not the scalar activity A)
    M::Vector{Matrix{CF}}          # per channel V' C_μ' C_μ V = A_μ' A_μ
    Mnorms::Vector{Float64}        # infinity norms of M
    ZV::Vector{Matrix{CF}}         # V† O V, with code observable Z_e = O
    κV::Float64                    # estimated cond(V, 1)
    normal::Bool                   # eigenvectors ~orthonormal (‖G - I‖ small)
    metric_error::Float64          # √(‖G-I‖₁‖G-I‖∞), bounds metric shortcut error
    normal_tol::Float64
    degeneracy_rtol::Float64
    degeneracy_atol::Float64
    condition_limit::Float64
    deg_clusters::Vector{Vector{Int}}  # Layer III mode-selection metadata
    reduced::Dict{Vector{Int}, _ReducedCache}
    reduced_lock::ReentrantLock
end

function _identity_metric_error(G::AbstractMatrix)
    size(G, 1) == size(G, 2) || return Inf
    inf_norm = 0.0
    @inbounds for i in axes(G, 1)
        rowsum = 0.0
        for j in axes(G, 2)
            rowsum += abs(G[i, j] - (i == j ? one(eltype(G)) : zero(eltype(G))))
        end
        inf_norm = max(inf_norm, rowsum)
    end
    one_norm = 0.0
    @inbounds for j in axes(G, 2)
        colsum = 0.0
        for i in axes(G, 1)
            colsum += abs(G[i, j] - (i == j ? one(eltype(G)) : zero(eltype(G))))
        end
        one_norm = max(one_norm, colsum)
    end
    return sqrt(one_norm * inf_norm)
end

# Group eigenvalue indices into transitive relative-plus-absolute clusters.
function degeneracy_clusters(
        Λ::AbstractVector;
        rtol::Real = sqrt(eps(Float64)), atol::Real = 0.0
    )
    isfinite(rtol) && rtol >= 0 ||
        throw(ArgumentError("degeneracy_rtol must be finite and nonnegative"))
    isfinite(atol) && atol >= 0 ||
        throw(ArgumentError("degeneracy_atol must be finite and nonnegative"))
    (rtol > 0 || atol > 0) ||
        throw(ArgumentError("degeneracy_rtol and degeneracy_atol cannot both be zero"))

    n = length(Λ)
    assigned = falses(n)
    clusters = Vector{Vector{Int}}()
    for i in 1:n
        assigned[i] && continue
        cl = [i]
        assigned[i] = true
        for a in cl
            for j in 1:n
                if !assigned[j] &&
                        abs(Λ[a] - Λ[j]) <= atol + rtol * max(abs(Λ[a]), abs(Λ[j]))
                    push!(cl, j)
                    assigned[j] = true
                end
            end
        end
        sort!(cl)
        push!(clusters, cl)
    end
    return clusters
end

degeneracy_clusters(Λ::AbstractVector, atol::Real) =
    degeneracy_clusters(Λ; rtol = 0.0, atol)

function _hermitian_with_roundoff(H)
    scale = opnorm(H, Inf)
    skew = opnorm(H - H', Inf)
    return isfinite(scale) && isfinite(skew) &&
        skew <= 64 * eps(Float64) * scale
end

# Paper: H_eff (Eq. 2).
function _effective_hamiltonian(
        H::AbstractMatrix{CF},
        C::AbstractVector{<:AbstractMatrix{CF}}
    )
    H_eff = copy(H)
    for Cμ in C
        H_eff .-= (im / 2) .* (Cμ' * Cμ)
    end
    return H_eff
end

function _checked_matrix(
        data, name::AbstractString; N::Union{Nothing, Int} = nothing,
        hermitian::Bool = false
    )
    data isa Union{Matrix, SparseMatrixCSC} ||
        throw(ArgumentError("$name must use CPU Matrix or SparseMatrixCSC storage"))
    size(data, 1) == size(data, 2) || throw(ArgumentError("$name must be square"))
    size(data, 1) > 0 || throw(ArgumentError("$name must be nonempty"))
    N === nothing || size(data) == (N, N) ||
        throw(ArgumentError("$name must have dimensions ($N, $N)"))
    all(isfinite, data) || throw(ArgumentError("$name must contain only finite values"))
    dense = Matrix{CF}(data)
    if hermitian
        _hermitian_with_roundoff(dense) || throw(ArgumentError("$name must be Hermitian"))
        dense = dense / 2 + dense' / 2
    end
    return dense
end

function _operator_data(op, name::AbstractString; expected_dimensions = nothing)
    hasproperty(op, :data) && hasproperty(op, :dimensions) ||
        throw(ArgumentError("$name must be a QuantumToolbox.QuantumObject"))
    dimensions = op.dimensions
    dimensions.to == dimensions.from ||
        throw(ArgumentError("$name must act on one Hilbert space"))
    expected_dimensions === nothing || dimensions == expected_dimensions ||
        throw(ArgumentError("$name dimensions must match the Hamiltonian"))
    return op.data, dimensions
end

function _operator_matrix(
        op, name::AbstractString;
        expected_dimensions = nothing, hermitian::Bool = false
    )
    data, dimensions = _operator_data(op, name; expected_dimensions)
    return _checked_matrix(data, name; hermitian), dimensions
end

function _validated_concrete_observable_storage(storage)::Symbol
    storage in (:dense, :sparse) || throw(
        ArgumentError(
            "observable_storage must resolve to :dense or :sparse, got $storage"
        )
    )
    return storage
end

function _checked_observable(data, name::AbstractString, N::Int, storage::Symbol)
    data isa AbstractMatrix ||
        throw(ArgumentError("$name must use CPU matrix storage"))
    size(data) == (N, N) || throw(ArgumentError("$name must have dimensions ($N, $N)"))
    all(isfinite, data) || throw(ArgumentError("$name must contain only finite values"))
    storage === :dense && return Matrix{CF}(data)
    return SparseMatrixCSC{CF, Int}(data)
end

function _validated_observables(e_ops, N::Int, dimensions, storage)
    mode = _validated_concrete_observable_storage(storage)
    e_ops === nothing && return mode === :dense ? Matrix{CF}[] : SparseMatrixCSC{CF, Int}[]
    converted = map(enumerate(e_ops)) do (index, operator)
        data = if hasproperty(operator, :data) && hasproperty(operator, :dimensions)
            op_data = first(_operator_data(operator, "e_ops[$index]"; expected_dimensions = dimensions))
            op_data isa Union{Matrix, SparseMatrixCSC} ||
                throw(ArgumentError("e_ops[$index] must use CPU Matrix or SparseMatrixCSC storage"))
            op_data
        elseif operator isa AbstractMatrix
            operator
        else
            throw(ArgumentError("e_ops[$index] must be a QuantumToolbox.QuantumObject or matrix"))
        end
        _checked_observable(data, "e_ops[$index]", N, mode)
    end
    return mode === :dense ? Matrix{CF}[converted...] : SparseMatrixCSC{CF, Int}[converted...]
end

function _validated_condition_limit(condition_limit::Real)
    limit = Float64(condition_limit)
    isfinite(limit) && limit >= 1 ||
        throw(ArgumentError("condition_limit must be finite and at least 1"))
    return limit
end

function _validate_eigenbasis(
        V::AbstractMatrix{CF}, Vfac::LU, metric_error::Real;
        condition_limit::Real, normal_tol::Real
    )
    limit = _validated_condition_limit(condition_limit)
    isfinite(normal_tol) && normal_tol >= 0 ||
        throw(ArgumentError("normal_tol must be finite and nonnegative"))
    reciprocal_condition = LAPACK.gecon!('1', Vfac.factors, opnorm(V, 1))
    reciprocal_condition > 0 ||
        throw(ArgumentError("effective-Hamiltonian eigenvector basis is rank-deficient"))
    κV = inv(reciprocal_condition)
    κV <= limit ||
        throw(ArgumentError("effective-Hamiltonian eigenvector basis is too ill-conditioned"))
    return κV, metric_error < normal_tol
end

# Paper: v_j with ⟨v_j|v_j⟩ = 1 (Eq. 15).
function _normalize_eigenvectors!(V::AbstractMatrix{CF})
    V ./= sqrt.(sum(abs2, V; dims = 1))
    _, indices = findmax(abs.(V); dims = 1)
    pivots = V[indices]
    V ./= pivots ./ abs.(pivots)
    return V
end

# Paper: V, Λ, G, γ_j, and V†OV (Eqs. 15–17, Section 3.3.1).
function _prepare_diagonal_data(
        H::AbstractMatrix{CF},
        C::AbstractVector{<:AbstractMatrix{CF}},
        Z::AbstractVector{<:AbstractMatrix{CF}}; backend::Symbol
    )
    H_eff = _effective_hamiltonian(H, C)
    F = eigen(H_eff)
    Λ = F.values
    V = _normalize_eigenvectors!(F.vectors)
    G = V' * V
    Vfac = lu(V)
    Γ = -2 .* imag.(Λ)
    A = [Cμ * V for Cμ in C]
    M = [Aμ' * Aμ for Aμ in A]
    Gnorm = _matrix_inf_norm(G)
    Mnorms = Float64[_matrix_inf_norm(Mμ) for Mμ in M]
    ZV = [V' * Ze * V for Ze in Z]
    return (;
        backend, V, Λ, Γ, Vfac, G, Gnorm, A, M, Mnorms, ZV,
        metric_error = _identity_metric_error(G),
    )
end

# Paper: V, Λ, G, γ_j, and V†OV on the CPU.
_cpu_prepare_diagonal_data(H, C, Z) =
    _prepare_diagonal_data(H, C, Z; backend = :lapack)

"""
    _diagonal_cache_from_matrices(Hmat, C; <keyword arguments>)

Validate CPU operators and prepare `V`, `Λ`, `G` (Eqs. 15–17).

Prepare dense eigensystems and derived matrices on the selected CPU or CUDA
backend, then retain CPU arrays for trajectory propagation.

# Arguments
- `Hmat::AbstractMatrix{ComplexF64}`: Finite, Hermitian Hamiltonian.
- `C::AbstractVector{<:AbstractMatrix{ComplexF64}}`: Compatible collapse operators.
- `Z::AbstractVector{<:AbstractMatrix{ComplexF64}}=Matrix{ComplexF64}[]`: Observables.
- `observable_storage::Symbol=:dense`: Use `:dense` or `:sparse` observable storage.
- `normal_tol::Real=1e-9`: Tolerance for the eigenvector Gram-matrix error.
- `degeneracy_rtol::Real=sqrt(eps(Float64))`: Relative eigenvalue clustering tolerance.
- `degeneracy_atol::Real=0.0`: Absolute eigenvalue clustering tolerance.
- `condition_limit::Real=sqrt(1e-2 / eps(Float64))`: Eigenbasis condition-number limit.
"""
function _diagonal_cache_from_matrices(
        Hmat::AbstractMatrix{CF},
        C::AbstractVector{<:AbstractMatrix{CF}};
        Z::AbstractVector{<:AbstractMatrix{CF}} = Matrix{CF}[],
        observable_storage::Symbol = :dense,
        normal_tol::Real = 1.0e-9,
        degeneracy_rtol::Real = sqrt(eps(Float64)),
        degeneracy_atol::Real = 0.0,
        condition_limit::Real = _DEFAULT_CONDITION_LIMIT
    )

    H_dense = _checked_matrix(Hmat, "H"; hermitian = true)
    N = size(H_dense, 1)
    C_dense = Matrix{CF}[_checked_matrix(Cμ, "c_ops[$μ]"; N) for (μ, Cμ) in enumerate(C)]
    Nc = length(C_dense)
    mode = _validated_concrete_observable_storage(observable_storage)
    Z_checked = _validated_observables(Z, N, nothing, mode)
    Z_backend = mode === :dense ? Z_checked : Matrix{CF}[]
    Ne = length(Z_checked)

    prepared = _prepare_diagonal_cache_data(H_dense, C_dense, Z_backend)
    κV, normal = _validate_eigenbasis(
        prepared.V, prepared.Vfac, prepared.metric_error;
        condition_limit, normal_tol
    )
    deg = degeneracy_clusters(
        prepared.Λ;
        rtol = degeneracy_rtol, atol = degeneracy_atol
    )

    return _DiagonalCache(
        N, nothing, Nc, Ne, H_dense, prepared.backend,
        C_dense, Z_checked, mode,
        prepared.V, prepared.Λ, prepared.Γ, prepared.Vfac,
        prepared.G, prepared.Gnorm, prepared.A, prepared.M,
        prepared.Mnorms, prepared.ZV,
        κV, normal, prepared.metric_error,
        float(normal_tol), float(degeneracy_rtol),
        float(degeneracy_atol), float(condition_limit), deg,
        Dict{Vector{Int}, _ReducedCache}(), ReentrantLock()
    )
end

# Paper: c = V⁻¹|ψ̃⟩ (Eqs. 15–16).
_solve_coordinates!(c::AbstractVector{CF}, cache::_DiagonalCache, ψ::AbstractVector{CF}) =
    ldiv!(c, cache.Vfac, ψ)

# Paper: V, Λ, G (Eqs. 15–17), with the active gauge understood.
function _DiagonalCache(
        H, c_ops::AbstractVector;
        e_ops::Union{Nothing, AbstractVector} = nothing,
        observable_storage::Symbol = :dense,
        normal_tol::Real = 1.0e-9,
        degeneracy_rtol::Real = sqrt(eps(Float64)),
        degeneracy_atol::Real = 0.0,
        condition_limit::Real = _DEFAULT_CONDITION_LIMIT
    )

    Hmat, dimensions = _operator_matrix(H, "H"; hermitian = true)
    C = Matrix{CF}[
        _operator_matrix(op, "c_ops[$μ]"; expected_dimensions = dimensions)[1]
            for (μ, op) in enumerate(c_ops)
    ]
    mode = _validated_concrete_observable_storage(observable_storage)
    Z_ops = _validated_observables(e_ops, size(Hmat, 1), dimensions, mode)

    cache = _diagonal_cache_from_matrices(
        Hmat, C; Z = Z_ops, observable_storage = mode, normal_tol,
        degeneracy_rtol, degeneracy_atol, condition_limit
    )
    cache.dimensions = dimensions
    return cache
end

"""
    _get_reduced!(cache, I)::_ReducedCache

Fetch or build `V_m`, `G_m`, and `O_m` for mode indices `I` (Section 3.3.2).

The cache sorts the indices and synchronizes construction. Prebuild before a
parallel region to keep the hot path free of locking overhead.
"""
function _get_reduced!(cache::_DiagonalCache, I::AbstractVector{<:Integer})
    Is = sort!(collect(Int, I))
    !isempty(Is) || throw(ArgumentError("reduced-subspace indices must be nonempty"))
    allunique(Is) || throw(ArgumentError("reduced-subspace indices must be unique"))
    all(i -> 1 <= i <= cache.N, Is) ||
        throw(ArgumentError("reduced-subspace indices must be in 1:$(cache.N)"))
    return lock(cache.reduced_lock) do
        get!(cache.reduced, Is) do
            V_I = cache.V[:, Is]
            G_I = cache.G[Is, Is]
            A_I = [cache.A[μ][:, Is] for μ in 1:cache.Nc]
            K_I = [cache.M[μ][Is, Is] for μ in 1:cache.Nc]
            Z_I = cache.observable_storage === :dense ?
                [cache.ZV[e][Is, Is] for e in 1:cache.Ne] : Matrix{CF}[]
            Gnorm = _matrix_inf_norm(G_I)
            Knorms = Float64[_matrix_inf_norm(K) for K in K_I]
            _ReducedCache(
                Is, V_I, cache.Λ[Is], G_I, Gnorm,
                A_I, K_I, Knorms, Z_I
            )
        end
    end
end
