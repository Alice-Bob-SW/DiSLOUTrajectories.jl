![DiSLOUTrajectories.jl logo](docs/src/assets/logo-dark.svg#gh-dark-mode-only)
![DiSLOUTrajectories.jl logo](docs/src/assets/logo.svg#gh-light-mode-only)

| **Release** | [![Release][release-img]][release-url] [![License][license-img]][license-url] [![Cite][cite-img]][cite-url] |
| :-----------------: | :------------- |
| **Runtests** | [![CI][CI-img]][CI-url] [![Coverage][codecov-img]][codecov-url] |
| **Code Quality** | [![Code Quality][code-quality-img]][code-quality-url] [![Aqua QA][aqua-img]][aqua-url] [![JET][jet-img]][jet-url] [![code style: runic][runic-img]][runic-url] |

[release-img]: https://img.shields.io/github/v/release/Alice-Bob-SW/DiSLOUTrajectories.jl.svg
[release-url]: https://github.com/Alice-Bob-SW/DiSLOUTrajectories.jl/releases

[license-img]: https://img.shields.io/badge/license-BSD--3--Clause-blue.svg
[license-url]: LICENSE.md

[cite-img]: https://img.shields.io/badge/cite-ArXiV-blue
[cite-url]: https://arxiv.org/

[CI-img]: https://github.com/Alice-Bob-SW/DiSLOUTrajectories.jl/actions/workflows/CI.yml/badge.svg
[CI-url]: https://github.com/Alice-Bob-SW/DiSLOUTrajectories.jl/actions/workflows/CI.yml

[codecov-img]: https://codecov.io/gh/Alice-Bob-SW/DiSLOUTrajectories.jl/graph/badge.svg
[codecov-url]: https://codecov.io/gh/Alice-Bob-SW/DiSLOUTrajectories.jl

[code-quality-img]: https://github.com/Alice-Bob-SW/DiSLOUTrajectories.jl/actions/workflows/Code-Quality.yml/badge.svg
[code-quality-url]: https://github.com/Alice-Bob-SW/DiSLOUTrajectories.jl/actions/workflows/Code-Quality.yml

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

[jet-img]: https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a
[jet-url]: https://github.com/aviatesk/JET.jl

[runic-img]: https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black
[runic-url]: https://github.com/fredrikekre/Runic.jl

`DiSLOUTrajectories.jl` implements the *Diagonal, Switching, and Locally Optimal Unraveling* method in [`Julia`](https://julialang.org/).
It provides an efficient approach to simulate quantum trajectories of metastable open quantum systems by combining different layers of optimizations.

It is built on top of
[`QuantumToolbox.jl`](https://github.com/qutip/QuantumToolbox.jl), and the main access point maintains a similar API to `mcsolve`:

```julia
sol = dislou_solve(H, ψ0, tlist, c_ops; gauge_set, e_ops)
```

By default, only Layer I (locally optimal unravelings) and II (diagonal propagation) are enabled, while Layer III (reduced space propagation) can be used by setting `layer3 = true` and providing `layer3_sizes`.

## Installation

`DiSLOUTrajectories.jl` requires Julia 1.10 or later and QuantumToolbox 0.47. To install it, run in an interactive session (REPL):

```julia
using Pkg
Pkg.add("DiSLOUTrajectories")
```

To load the package and check version information:

```julia
using DiSLOUTrajectories
DiSLOUTrajectories.versioninfo()
DiSLOUTrajectories.about()
```

## Minimal model

A resonantly driven linear cavity with loss rate $\kappa$ and Hamiltonian
$H = i\kappa(\alpha a^\dagger - \alpha^* a)/2$ relaxes to the coherent state
$|\alpha\rangle$, with mean occupation $|\alpha|^2$. Here we start in vacuum
and set the gauge at $\alpha$, to minimize quantum jumps in the steady state. We pass $\zeta^{(g)}=-\sqrt{\kappa}\alpha$ so that the shifted
collapse operator is $\sqrt{\kappa}(a-\alpha)$.

```jldoctest
using QuantumToolbox, DiSLOUTrajectories

N = 20
κ = 1.0
α = 1.5 + 0.5im
a = destroy(N)
H = im * κ / 2 * (α * a' - conj(α) * a)
c_ops = [sqrt(κ) * a]
ψ0 = fock(N, 0)
tlist = collect(0.0:0.5:10.0) ./ κ
e_ops = [num(N)]
gauges = fill(-sqrt(κ) * α, 1, 1)

sol = dislou_solve(H, ψ0, tlist, c_ops;
                   gauge_set = gauges, e_ops, ntraj = 1_000, seed = 1)
mean_n = expect_mean(sol)
sem_n = expect_sem(sol)

(isapprox(real(mean_n[end]), abs2(α); rtol = 0.02), all(isfinite, sem_n))

# output

(true, true)
```

## Documentation

The documentation is published at <https://alice-bob-sw.github.io/DiSLOUTrajectories.jl/>.

To build the manual and run its doctests, locally:

```sh
julia --project=docs --startup-file=no -e '
    using Pkg
    Pkg.develop(path=pwd())
    Pkg.instantiate()
'
julia --project=docs --startup-file=no docs/make.jl
```

Open `docs/build/index.html` to preview the result.

## License and citation

DiSLOUTrajectories.jl is available under the [BSD-3-Clause License](LICENSE.md).

If you found `DiSLOUTrajectories.jl` useful, please cite our publication [ [ArXiv (2026)](https://arxiv.org/) ] in your work using the following bibtex entry:

```bib
something
```

Citation metadata is also provided in [`CITATION.cff`](CITATION.cff).

Since this package is built on top of [`QuantumToolbox.jl`](https://qutip.org/QuantumToolbox.jl/), please also consider citing the relevant publication  [Quantum 9, 1866 (2025)](https://doi.org/10.22331/q-2025-09-29-1866) ] using the following bibtex entry:

```bib
@article{QuantumToolbox.jl2025,
  title = {Quantum{T}oolbox.jl: {A}n efficient {J}ulia framework for simulating open quantum systems},
  author = {Mercurio, Alberto and Huang, Yi-Te and Cai, Li-Xun and Chen, Yueh-Nan and Savona, Vincenzo and Nori, Franco},
  journal = {{Quantum}},
  issn = {2521-327X},
  publisher = {{Verein zur F{\"{o}}rderung des Open Access Publizierens in den Quantenwissenschaften}},
  volume = {9},
  pages = {1866},
  month = sep,
  year = {2025},
  doi = {10.22331/q-2025-09-29-1866},
  url = {https://doi.org/10.22331/q-2025-09-29-1866}
}
```
