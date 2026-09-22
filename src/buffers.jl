# Per-trajectory scratch buffers, reused across all segments of a trajectory (and across
# trajectories within a chunk) so the hot path allocates nothing. One private buffer per
# thread/worker — never share one across concurrent trajectories.

struct _WorkBuffers
    c::Vector{CF}        # eigenbasis coordinates  V⁻¹ψ           (length N)
    Dc::Vector{CF}       # c(t) = D(t)c(0) (length N)
    cplus::Vector{CF}    # c⁺, or local eigenspace coordinates (length N)
    Gc::Vector{CF}       # metric-applied coordinates  Gc         (length N)
    moments::Vector{CF}  # unshifted moments ⟨C_μ⟩_ψ (length Nc)
    w::Vector{Float64}   # jump weights, ⟨C_μ†C_μ⟩_ψ, or routed A_g scratch
    ψ::Vector{CF}        # physical-state scratch                 (length N)
end
_WorkBuffers(N::Int, Nc::Int, nweights::Int = max(N, Nc)) =
    _WorkBuffers(
    zeros(CF, N), zeros(CF, N), zeros(CF, N), zeros(CF, N),
    zeros(CF, Nc), zeros(Float64, max(Nc, nweights)), zeros(CF, N)
)

_WorkBuffers(
    cache::_DiagonalCache,
    nweights::Int = max(cache.N, cache.Nc)
) =
    _WorkBuffers(cache.N, cache.Nc, nweights)
