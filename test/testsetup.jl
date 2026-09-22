using Test
using DiSLOUTrajectories
import Clustering  # activates DiSLOUTrajectoriesClusteringExt for trajectory gauge discovery
using QuantumToolbox
using LinearAlgebra
using Random
using SparseArrays
using Statistics

const SM = DiSLOUTrajectories

# Paper: s(t) = ‖exp(-i H_eff t)|ψ(0)⟩‖² (Eq. 5).
direct_survival(Heff::AbstractMatrix, psi::AbstractVector, time::Real) =
    abs2(norm(exp(-im * time * Heff) * psi))

# Paper: ⟨O⟩_ψ(t) = ⟨ψ̃(t)|O|ψ̃(t)⟩/s(t).
function direct_expect(Heff, observable, psi, time)
    evolved = exp(-im * time * Heff) * psi
    return dot(evolved, observable * evolved) / dot(evolved, evolved)
end

function random_system(; N = 6, seed = 1)
    rng = Xoshiro(seed)
    A = randn(rng, ComplexF64, N, N)
    H = QuantumObject((A + A') / 2)
    c1 = QuantumObject(triu(randn(rng, ComplexF64, N, N), 1) .* 0.5)
    c2 = QuantumObject(Matrix(0.3 .* Diagonal(randn(rng, N))) .+ 0im)
    return H, [c1, c2]
end
