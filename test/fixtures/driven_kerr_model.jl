using QuantumToolbox, LinearAlgebra
Base.@kwdef struct DrivenKerrParams
    N::Int = 30        # Fock cutoff (n_high ≈ 6.9 at defaults ⇒ 30 is ample)
    Δ::Float64 = 6.0       # detuning (H has −Δ a†a; needs Δ>0 and Δ²>3(κ/2)²)
    K::Float64 = 1.0       # Kerr (>0)
    κ::Float64 = 1.0       # single-photon loss  (THE displaced channel)
    ε::ComplexF64 = 2.6 + 0im # single-photon drive amplitude
end

# Paper: n_sc and α_sc (Eq. 8), denoted n and α below.
# Stable/unstable mean-field branches: positive roots n≥0 of
#   |ε|² = n[(κ/2)² + (Δ−Kn)²]  ⇔  K²n³ − 2ΔK n² + (Δ²+(κ/2)²)n − |ε|² = 0,
# with amplitude  α(n) = ε / (κ/2 − i(Δ−Kn)).
function driven_kerr_branches(p::DrivenKerrParams)
    a3, a2 = p.K^2, -2 * p.Δ * p.K
    a1, a0 = p.Δ^2 + (p.κ / 2)^2, -abs2(p.ε)
    b2, b1, b0 = a2 / a3, a1 / a3, a0 / a3            # monic n³ + b2 n² + b1 n + b0
    C = [0.0 0.0 -b0; 1.0 0.0 -b1; 0.0 1.0 -b2] # companion matrix (LinearAlgebra only)
    ns = sort!([real(r) for r in eigvals(C) if abs(imag(r)) < 1.0e-8 && real(r) > 1.0e-9])
    @assert length(ns) == 3 "not in the bistable regime ($(length(ns)) positive roots): \
        need Δ>0, Δ²>3(κ/2)², and |ε| between the two saddle-node drives"
    αof(n) = p.ε / (p.κ / 2 - im * (p.Δ - p.K * n))
    nlow, nmid, nhigh = ns[1], ns[2], ns[3]
    return (; nlow, nmid, nhigh, αlow = αof(nlow), αmid = αof(nmid), αhigh = αof(nhigh))
end

# Paper: H and C = √κ a (Eq. 7), initialized on the bright branch.
function driven_kerr_model(p::DrivenKerrParams = DrivenKerrParams())
    a = destroy(p.N)
    H = -p.Δ * (a' * a) + (p.K / 2) * ((a')^2 * a^2) + im * (p.ε * a' - conj(p.ε) * a)
    c_ops = [sqrt(p.κ) * a]                       # single-photon loss = the displaced channel
    b = driven_kerr_branches(p)
    ψ0 = coherent(p.N, b.αhigh)                # start localized in the HIGH branch
    nop = a' * a
    xop = (a + a') / sqrt(2)
    return (;
        H, c_ops, ψ0, a, nop, xop, κ = p.κ, loss_idx = 1,
        b.nlow, b.nmid, b.nhigh, b.αlow, b.αmid, b.αhigh, p,
    )
end
