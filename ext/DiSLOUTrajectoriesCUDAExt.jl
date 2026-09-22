module DiSLOUTrajectoriesCUDAExt

using LinearAlgebra
import CUDA
import DiSLOUTrajectories

DiSLOUTrajectories._matrix_inf_norm(A::CUDA.CuMatrix) =
    Float64(maximum(sum(abs, A; dims = 2)))

function DiSLOUTrajectories._identity_metric_error(G::CUDA.CuMatrix)
    size(G, 1) == size(G, 2) || return Inf
    Δ = G - I
    return sqrt(
        Float64(maximum(sum(abs, Δ; dims = 1))) *
            Float64(maximum(sum(abs, Δ; dims = 2)))
    )
end

_cpu_lu(F) = LU(
    Matrix{ComplexF64}(Array(getfield(F, :factors))),
    Vector{Int}(Array(getfield(F, :ipiv))),
    getfield(F, :info),
)

# Paper: V, Λ, G, γ_j, and V†OV on CUDA.
function DiSLOUTrajectories._cuda_prepare_diagonal_data(
        H::Matrix{ComplexF64},
        C::Vector{Matrix{ComplexF64}}, Z::Vector{Matrix{ComplexF64}}
    )
    prepared = DiSLOUTrajectories._prepare_diagonal_data(
        CUDA.CuArray(H), CUDA.CuArray.(C), CUDA.CuArray.(Z); backend = :cuda
    )
    CUDA.synchronize()
    return merge(
        prepared, (;
            V = Array(prepared.V),
            Λ = Array(prepared.Λ),
            Γ = Array(prepared.Γ),
            Vfac = _cpu_lu(prepared.Vfac),
            G = Array(prepared.G),
            A = Array.(prepared.A),
            M = Array.(prepared.M),
            ZV = Array.(prepared.ZV),
        )
    )
end

function __init__()
    CUDA.functional() && DiSLOUTrajectories._enable_cuda_diagonalization!()
    return nothing
end

end
