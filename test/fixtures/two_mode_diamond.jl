const TWO_MODE_DIAMOND = let s = inv(sqrt(0.8))
    (;
        Na = 85,
        Nb = 16,
        Δa = -0.5,
        Δb = 41.0,
        Ka = -0.05 / s^2,
        Kb = -0.75 / s^2,
        χ = -0.375 / s^2,
        g2 = 1.5 / s,
        εb = s * (6.8 + 0im),
        κa = 5.0e-3,
        κb = 15.0,
    )
end

# Paper: H (Eq. 24); code Ka, Kb, χ = paper K_a/2, K_b/2, χ/2.
function two_mode_diamond_hamiltonian(a, b)
    cfg = TWO_MODE_DIAMOND
    na, nb = adjoint(a) * a, adjoint(b) * b
    return cfg.Δa * na + cfg.Δb * nb -
        cfg.Ka * (adjoint(a)^2 * a^2) -
        cfg.Kb * (adjoint(b)^2 * b^2) -
        cfg.χ * na * nb +
        cfg.g2 * (a^2 * adjoint(b) + adjoint(a)^2 * b) +
        cfg.εb * (b + adjoint(b))
end

# Paper: C_a = √κ_a a, C_b = √κ_b b (Eq. 25, κ_a = κ₁).
function two_mode_diamond_collapse_operators(a, b)
    cfg = TWO_MODE_DIAMOND
    return [sqrt(cfg.κa) * a, sqrt(cfg.κb) * b]
end

# Paper: H, C_a, C_b (Eqs. 24–25), initialized in |0,0⟩.
function two_mode_diamond_numerical_fixture(dimensions::Tuple{Int, Int})
    Na, Nb = dimensions
    a = QuantumToolbox.tensor(
        QuantumToolbox.destroy(Na), QuantumToolbox.qeye(Nb)
    )
    b = QuantumToolbox.tensor(
        QuantumToolbox.qeye(Na), QuantumToolbox.destroy(Nb)
    )
    return (;
        H = two_mode_diamond_hamiltonian(a, b),
        psi0 = QuantumToolbox.tensor(
            QuantumToolbox.fock(Na, 0), QuantumToolbox.fock(Nb, 0)
        ),
        c_ops = two_mode_diamond_collapse_operators(a, b),
    )
end
