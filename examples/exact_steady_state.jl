module ExactSteadyState

using LinearAlgebra
using QuantumToolbox
using SparseArrays

import CUDA
import CUDSS
import DiSLOUTrajectories
import MUMPS
import ModelingToolkitBase
import QuantumCumulants

export exact_steadystate, semiclassical_fixed_points

const CF = ComplexF64

_sparse_complex(A) = convert(SparseMatrixCSC{CF, Int}, sparse(CF.(A)))

# ρ ↦ -i[H, ρ] (Eq. 1).
function _commutator_superoperator(H::SparseMatrixCSC{CF, Int})
    n = size(H, 1)
    Isp = spdiagm(0 => ones(CF, n))
    L = -im * (kron(Isp, H) - kron(sparse(transpose(H)), Isp))
    dropzeros!(L)
    return L
end

# D[C] (Eq. 1).
function _dissipator_superoperator(C::SparseMatrixCSC{CF, Int})
    n = size(C, 1)
    Isp = spdiagm(0 => ones(CF, n))
    CdC = sparse(C' * C)
    D = kron(sparse(conj(C)), C)
    D += -0.5 * kron(Isp, CdC)
    D += -0.5 * kron(sparse(transpose(CdC)), Isp)
    dropzeros!(D)
    return D
end

function _equal_parity_indices(mode_dims::Tuple, parity_mode::Int)
    all(>(0), mode_dims) || throw(ArgumentError("mode dimensions must be positive"))
    1 ≤ parity_mode ≤ length(mode_dims) ||
        throw(ArgumentError("parity_mode must index mode_dims"))

    n = prod(mode_dims)
    stride = prod(mode_dims[(parity_mode + 1):end]; init = 1)
    parity = Vector{Bool}(undef, n)
    @inbounds for state in 1:n
        occupation = div(state - 1, stride) % mode_dims[parity_mode]
        parity[state] = iseven(occupation)
    end

    keep = Int[]
    sizehint!(keep, div(n^2 + 1, 2))
    @inbounds for col in 1:n
        offset = (col - 1) * n
        for row in 1:n
            parity[row] == parity[col] && push!(keep, offset + row)
        end
    end
    return keep
end

# Generator of ρ̇ (Eq. 1), restricted to the selected parity sector.
function _reduced_liouvillian(H, c_ops, mode_dims::Tuple, parity_mode::Int)
    n = prod(mode_dims)
    size(H.data) == (n, n) ||
        throw(DimensionMismatch("prod(mode_dims) must equal the Hamiltonian dimension"))
    all(op -> size(op.data) == (n, n), c_ops) ||
        throw(DimensionMismatch("collapse operators must match the Hamiltonian"))

    keep = _equal_parity_indices(mode_dims, parity_mode)
    full = _commutator_superoperator(_sparse_complex(H.data))
    L = sparse(full[keep, keep])
    full = nothing
    GC.gc()

    for op in c_ops
        full = _dissipator_superoperator(_sparse_complex(op.data))
        L += sparse(full[keep, keep])
        full = nothing
        GC.gc()
    end
    dropzeros!(L)
    return L, keep
end

function _trace_constrained_system(L::SparseMatrixCSC{CF, Int}, keep, n::Int)
    trace_positions = Vector{Int}(undef, n)
    @inbounds for state in 1:n
        full_index = state + (state - 1) * n
        reduced_index = searchsortedfirst(keep, full_index)
        reduced_index ≤ length(keep) && keep[reduced_index] == full_index ||
            error("trace element missing from parity sector")
        trace_positions[state] = reduced_index
    end

    nr = size(L, 1)
    trace_row = sparse(fill(1, n), trace_positions, ones(CF, n), 1, nr)
    A = sparse(vcat(trace_row, L[2:end, :]))
    b = zeros(CF, nr)
    b[1] = 1

    rownorm = sqrt.(vec(sum(abs2, A; dims = 2)))
    scale = map(r -> r > 1.0e-15 ? inv(r) : 1.0, rownorm)
    As = sparse(spdiagm(0 => CF.(scale)) * A)
    bs = CF.(scale) .* b
    dropzeros!(As)
    return As, bs
end

function _cudss_solve(A::SparseMatrixCSC{CF, Int}, b::Vector{CF})
    CUDA.functional() || error("backend=:cudss requires a functional CUDA device")
    A_gpu = nothing
    x_gpu = nothing
    b_gpu = nothing
    solver = nothing
    try
        A_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(A)
        x_gpu = CUDA.zeros(CF, size(A, 1))
        b_gpu = CUDA.CuVector(b)
        solver = CUDSS.CudssSolver(A_gpu, "G", 'F')
        CUDSS.cudss("analysis", solver, x_gpu, b_gpu)
        CUDA.synchronize()
        CUDSS.cudss("factorization", solver, x_gpu, b_gpu; asynchronous = false)
        CUDA.synchronize()
        CUDSS.cudss("solve", solver, x_gpu, b_gpu; asynchronous = false)
        CUDA.synchronize()
        return Array(x_gpu)
    finally
        try
            CUDA.synchronize()
        catch
        end
        solver = nothing
        A_gpu = nothing
        x_gpu = nothing
        b_gpu = nothing
        GC.gc(true)
        CUDA.reclaim()
    end
end

function _mumps_solve(A::SparseMatrixCSC{CF, Int}, b::Vector{CF})
    MUMPS.MPI.Initialized() || MUMPS.MPI.Init()
    solver = MUMPS.Mumps{CF}(
        MUMPS.mumps_unsymmetric,
        MUMPS.get_icntl(; verbose = false),
        copy(MUMPS.default_cntl64),
    )
    return try
        MUMPS.factorize!(solver, A)
        solver.err < 0 && error("MUMPS factorization failed with code $(solver.err)")
        x = vec(MUMPS.solve(solver, b))
        solver.err < 0 && error("MUMPS solve failed with code $(solver.err)")
        x
    finally
        Base.finalize(solver)
    end
end

# ρ_ss: ρ̇ = 0 (Eq. 1), Tr ρ_ss = 1.
function exact_steadystate(
        H,
        c_ops;
        mode_dims,
        parity_mode,
        backend::Symbol = :auto,
        scaled_residual_tol::Real = 1.0e-9,
        liouvillian_residual_tol::Real = 2.0e-8,
    )
    dims = Tuple(Int.(mode_dims))
    selected = backend === :auto ? (CUDA.functional() ? :cudss : :mumps) : backend
    selected in (:cudss, :mumps) ||
        throw(ArgumentError("backend must be :auto, :cudss, or :mumps"))

    L, keep = _reduced_liouvillian(H, c_ops, dims, parity_mode)
    n = prod(dims)
    A, b = _trace_constrained_system(L, keep, n)
    x = selected === :cudss ? _cudss_solve(A, b) : _mumps_solve(A, b)

    scaled_residual = norm(A * x - b) /
        max(norm(A, Inf) * norm(x) + norm(b), eps(Float64))

    fullvec = zeros(CF, n^2)
    fullvec[keep] .= x
    rho = reshape(fullvec, n, n)
    rho = Matrix((rho + rho') / 2)
    raw_trace = real(tr(rho))
    abs(raw_trace) > 1.0e-14 || error("steady-state solution has vanishing trace")
    trace_error = abs(raw_trace - 1)
    rho ./= raw_trace

    reduced_rho = vec(rho)[keep]
    relative_L_residual = norm(L * reduced_rho) /
        max(norm(L, Inf) * norm(reduced_rho), eps(Float64))

    scaled_residual ≤ scaled_residual_tol || error(
        "scaled constrained-system residual $scaled_residual exceeds $scaled_residual_tol",
    )
    relative_L_residual ≤ liouvillian_residual_tol || error(
        "relative Liouvillian residual $relative_L_residual exceeds $liouvillian_residual_tol",
    )

    state = QuantumObject(rho, Operator(), H.dimensions)
    diagnostics = (;
        scaled_relative_residual = scaled_residual,
        relative_liouvillian_residual = relative_L_residual,
        trace_error,
    )
    return (; state, backend = selected, diagnostics)
end

# F_j(α, α*) = α̇_j (Eq. A.1).
function _symbolic_meanfield_drift(hamiltonian, collapse_operators)
    space = QuantumCumulants.tensor(
        QuantumCumulants.FockSpace(:memory),
        QuantumCumulants.FockSpace(:buffer),
    )
    a = QuantumCumulants.Destroy(space, :a, 1)
    b = QuantumCumulants.Destroy(space, :b, 2)
    equations = try
        QuantumCumulants.meanfield(
            [a, b], hamiltonian(a, b), collect(collapse_operators(a, b)); order = 1
        )
    catch err
        error("failed to derive the two-mode mean-field equations: ", sprint(showerror, err))
    end
    length(equations) == 2 || error(
        "first-order two-mode model produced $(length(equations)) equations; expected 2"
    )

    problem = try
        system = ModelingToolkitBase.mtkcompile(
            ModelingToolkitBase.System(equations; name = :two_mode_meanfield)
        )
        initial = QuantumCumulants.initial_values(equations, zeros(CF, 2))
        ModelingToolkitBase.ODEProblem(system, initial, (0.0, 1.0))
    catch err
        error("failed to compile the two-mode mean-field equations: ", sprint(showerror, err))
    end
    return u -> problem.f(u, problem.p, 0.0)
end

# α_j^(g) roots and max Re λℓ[J] (Eqs. A.2–A.5).
function semiclassical_fixed_points(hamiltonian, collapse_operators; limits)
    length(limits) == 2 || throw(ArgumentError("limits must contain two occupation bounds"))
    bounds = (Float64(limits[1]), Float64(limits[2]))
    all(x -> isfinite(x) && x > 0, bounds) ||
        throw(ArgumentError("occupation limits must be finite and positive, got $limits"))

    drift = _symbolic_meanfield_drift(hamiltonian, collapse_operators)
    # Gauge discovery rejects models with no stable roots; plots still need those roots.
    extension = Base.get_extension(DiSLOUTrajectories, :DiSLOUTrajectoriesQuantumCumulantsExt)
    points = [
        (;
            a = point.amplitudes[1], b = point.amplitudes[2], point.stable,
            point.max_real_eigenvalue, point.residual,
        ) for point in extension._phase_space_candidates(drift, bounds)
    ]
    return sort!(points; by = p -> (abs2(p.a), real(p.a), imag(p.a), abs2(p.b), real(p.b)))
end

end
