"""
    DiSLOUTrajectories

Diagonal, Switching, and Locally Optimal Unraveling for finite-dimensional,
time-independent Lindblad models.

[`dislou_solve`](@ref) requires a finite
gauge set, runs exact Layers I and II (optimal unraveling and diagonal propagation), and can add optional Layer III reduced space
propagation. Inputs may be `QuantumToolbox.QuantumObject`s or compatible CPU
arrays; results use [`DiSLOUSolution`](@ref).

See also [`dislou_solve`](@ref), [`discover_gauges`](@ref).
"""
module DiSLOUTrajectories

using QuantumToolbox
using LinearAlgebra
using Random
using Distributed
using SparseArrays

export dislou_solve, discover_gauges, DiSLOUSolution, expect_mean, expect_sem
export FirstPassageConvergenceError, about, backend_info, cite, versioninfo

const CF = ComplexF64

include("backend.jl")
include("layer2.jl")
include("buffers.jl")
include("gauge_discovery.jl")
include("first_passage.jl")
include("layer1.jl")
include("jump.jl")
include("recording.jl")
include("layer3.jl")
include("ensemble.jl")
include("result.jl")
include("solve.jl")

__init__() = Base.Experimental.register_error_hint(_discovery_error_hint, MethodError)

end # module
