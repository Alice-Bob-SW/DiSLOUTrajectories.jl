# API reference

```@docs
DiSLOUTrajectories
```

## Paper notation

The notation follows that of the [reference paper](./getting_started/cite.md) for the DiSLOU method.

| Code | Paper quantity |
| :----- | :--------------- |
| `gauge_set[μ, g]` or `gauges.shifts[μ, g]` | ``ζ_μ^{(g)}`` (Eq. 9) |
| `hysteresis` | ``η`` (Eq. 13) |
| `layer3_sizes[g]` | Requested ``m^{(g)}``; degenerate clusters can increase it (Eq. 20) |
| `residual_tolerance` | ``r_{tol}``, applied to ``r_m^{(g)}(ψ)`` for normalized states (Eq. 21) |
| `sol.states`, `sol.final_density` | ``ρ_{MC}(t)``, ``ρ_{MC}(T)`` (Section 4.1) |
| `sol.expect` | Ensemble ``⟨O⟩(t)`` |
| Cache `Γ[j]` | ``γ_j=-2\operatorname{Im}λ_j`` (Section 3.3.1) |
| Cache `V_I`, `G_I`, `Z_I` | Layer III ``V_m``, ``G_m``, ``O_m`` (Section 3.3.2) |
| First-passage fields `S`, `R`; threshold `r` | ``s(t)``, ``-\dot{s}(t)``; ``u`` (Appendix B) |

The normalized activity is ``A(ψ(t))=-\dot{s}(t)/s(t)``. Cache `A` holds ``C_μV``; cache `K_I` holds
``V_m^†C_μ^†C_μV_m``, while ``\mathcal K_m`` denotes the retained eigenspace.
Code observables `Z_e` denote the paper's ``O``. Expectations use
``c(t)^†V^†OVc(t)/s(t)`` to normalize the no-jump state.

## Solver

```@docs
dislou_solve
```

### Examples

The following closed two-level system illustrates the independent observation
and state grids and the optional trajectory arrays. All matrices and vectors
are CPU arrays; the empty collapse-operator vector requires a `0 × 1` gauge
matrix.

```jldoctest
using DiSLOUTrajectories

H = zeros(ComplexF64, 2, 2)
ψ0 = ComplexF64[1, 0]
tlist = [1.0, 1.5, 2.0]

sol = dislou_solve(H, ψ0, tlist, Matrix{ComplexF64}[];
                   gauge_set = zeros(ComplexF64, 0, 1),
                   e_ops = [ComplexF64[1 0; 0 0]],
                   ntraj = 2, ensemblealg = :serial,
                   saveat = [1.0, 2.0], save_trajectories = true)

(size(sol.expect), length(sol.states), size(sol.trajectory_states),
 expect_mean(sol) == ones(ComplexF64, 3),
 all(iszero, expect_sem(sol)))

# output

((1, 3), 2, (2, 2), true, true)
```

## Gauge discovery

```@docs
discover_gauges
```

## Solution and observable statistics

```@docs
DiSLOUSolution
expect_mean
expect_sem
```

## Package information

```@docs
DiSLOUTrajectories.versioninfo
DiSLOUTrajectories.about
DiSLOUTrajectories.cite
```

## Backends and exceptions

```@docs
backend_info
FirstPassageConvergenceError
```
