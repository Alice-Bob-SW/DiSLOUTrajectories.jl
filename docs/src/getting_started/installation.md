# Installation

`DiSLOUTrajectories.jl` requires Julia 1.10 or later and QuantumToolbox 0.47. To install it, run in an interactive session (REPL):

```julia
using Pkg
Pkg.add("DiSLOUTrajectories")
```

Alternatively, this can also be done in Julia's [Pkg REPL](https://julialang.github.io/Pkg.jl/v1/getting-started/) by pressing the key `]` in the REPL to use the package mode, and then type the following command:

```julia-REPL
(1.10) pkg> add DiSLOUTrajectories
```

More information about `Julia`'s package manager can be found at [`Pkg.jl`](https://julialang.github.io/Pkg.jl/v1/).

To load the package and check the version information, use either [`DiSLOUTrajectories.versioninfo()`](@ref) or [`DiSLOUTrajectories.about()`](@ref), namely

```julia
using DiSLOUTrajectories
DiSLOUTrajectories.versioninfo()
DiSLOUTrajectories.about()
```

## [QuantumToolbox.jl](https://github.com/qutip/QuantumToolbox.jl)

`DiSLOUTrajectories.jl` is built upon `QuantumToolbox.jl`, which is a cutting-edge Julia package designed for quantum physics simulations, closely emulating the popular Python [`QuTiP`](https://qutip.org/) package. It provides many useful functions to create arbitrary quantum states and operators which can be combined in all the expected ways. It uniquely combines the simplicity and power of Julia with advanced features like GPU acceleration and distributed computing, making simulation of quantum systems more accessible and efficient.

## Optional extensions

Automatic gauge discovery via initial trajectories clusters the terminal amplitudes of its preliminary runs
with DBSCAN. Install [```Clustering.jl```](https://github.com/JuliaStats/Clustering.jl) and load it to activate this method:

```julia
using Pkg
Pkg.add("Clustering")
using Clustering
```

Semiclassical gauge discovery requires instead [```QuantumCumulants.jl```](https://qojulia.github.io/QuantumCumulants.jl) to derive the semiclassical equations of motion.
Install ```QuantumCumulants.jl``` and load it to activate this method:

```julia
using Pkg
Pkg.add("QuantumCumulants")
using QuantumCumulants
```

[```CUDA.jl```](https://cuda.juliagpu.org/stable/) can be installed to perform the initial eigendecomposition and related numerical operations on NVidia GPUs.
The extension only needs the `CUDACore` and `cuSOLVER` components of CUDA.jl:

```julia
using Pkg
Pkg.add(["CUDACore", "cuSOLVER"])
using CUDACore, cuSOLVER
using DiSLOUTrajectories
```

Loading the full `CUDA` package works too.

When CUDA is functional, the extension prepares the dense diagonal-basis cache
on the GPU, including normalization, LU factorizations, and diagnostics, then
copies the completed cache to the CPU. For now, trajectory propagation and returned
arrays remain on the CPU. If GPU preparation fails, DiSLOUTrajectories.jl warns once, disables
further GPU attempts in the process and retries with CPU LAPACK.

Check extension state with `DiSLOUTrajectories.backend_info()`.

Continue with the [quick start](quickstart.md).
