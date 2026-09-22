const _CUDA_DIAGONALIZATION_ENABLED = Ref(false)

function _cuda_prepare_diagonal_data end

_enable_cuda_diagonalization!() = (_CUDA_DIAGONALIZATION_ENABLED[] = true; nothing)
_disable_cuda_diagonalization!() = (_CUDA_DIAGONALIZATION_ENABLED[] = false; nothing)

"""
    backend_info()

Inspect the eigensystem preparation backends available to the current process.

# Notes

- Loading CUDA activates DiSLOUTrajectories.jl's optional CUDA extension. GPU preparation is
  enabled when `CUDA.functional()` succeeds during extension initialization.
- CUDA accelerates eigensystem preparation, but trajectory propagation and returned
  arrays remain on the CPU. GPU preparation failures disable CUDA preparation
  in the current process and retry with LAPACK.

# Returns

- `info::NamedTuple`: `(; cpu, cuda_extension_loaded, cuda_enabled)`, where
  `cpu` is always `:lapack`, `cuda_extension_loaded::Bool` reports whether
  `DiSLOUTrajectoriesCUDAExt` is loaded, and `cuda_enabled::Bool` reports whether CUDA
  preparation is currently enabled.

See also [`dislou_solve`](@ref), [`DiSLOUSolution`](@ref).

# Examples

```jldoctest
julia> using DiSLOUTrajectories

julia> backend_info().cpu
:lapack
```
"""
function backend_info()
    return (;
        cpu = :lapack,
        cuda_extension_loaded = Base.get_extension(DiSLOUTrajectories, :DiSLOUTrajectoriesCUDAExt) !== nothing,
        cuda_enabled = _CUDA_DIAGONALIZATION_ENABLED[],
    )
end

const _REPORT_EXTENSIONS = (
    (:DiSLOUTrajectoriesCUDAExt, :CUDA),
    (:DiSLOUTrajectoriesClusteringExt, :Clustering, :Distances),
    (:DiSLOUTrajectoriesQuantumCumulantsExt, :QuantumCumulants, :ModelingToolkitBase),
)

_report_extension_loaded(name::Symbol) = Base.get_extension(DiSLOUTrajectories, name) !== nothing

function _report_optional_versions(io::IO)
    for (extension, packages...) in _REPORT_EXTENSIONS
        module_ = Base.get_extension(DiSLOUTrajectories, extension)
        module_ === nothing && continue
        for package in packages
            println(io, "  ", rpad("$(package):", 20), Base.pkgversion(getproperty(module_, package)))
        end
    end
    return
end

function _report_blas_library()
    libraries = LinearAlgebra.BLAS.get_config().loaded_libs
    return isempty(libraries) ? "unknown" : join(string.(libraries), ", ")
end

const _CITATION_BIBTEX = raw"""@software{Pacchioni2026DiSLOU,
  author = {Pietro Pacchioni and Fabrizio Minganti},
  title = {DiSLOUTrajectories.jl},
  year = {2026},
  license = {BSD-3-Clause},
  url = {https://github.com/Alice-Bob-SW/DiSLOUTrajectories.jl}
}

@article{Pacchioni2026Diagonal,
  author = {Pietro Pacchioni and Patrick Winkel and Fabrizio Minganti},
  title = {Diagonal, Switching, and Locally Optimal Unraveling efficient quantum trajectories in metastable open quantum systems},
  year = {2026}
}"""

"""
    versioninfo(io::IO = stdout)

Print DiSLOUTrajectories.jl's version, runtime environment, optional-extension status, and
citation reminder to `io`.
"""
function versioninfo(io::IO = stdout)
    backend = backend_info()
    println(io, "DiSLOUTrajectories.jl — v", Base.pkgversion(DiSLOUTrajectories))
    println(io, "Diagonal, Switching, and Locally Optimal Unraveling")
    println(io, "Pietro Pacchioni and Fabrizio Minganti · BSD-3-Clause license")

    println(io, "\nVersions")
    println(io, "  ", rpad("Julia:", 20), VERSION)
    println(io, "  ", rpad("QuantumToolbox:", 20), Base.pkgversion(QuantumToolbox))
    _report_optional_versions(io)

    println(io, "\nSystem")
    println(io, "  ", rpad("OS / architecture:", 20), Sys.KERNEL, " / ", Sys.ARCH)
    println(io, "  ", rpad("CPU:", 20), Sys.CPU_NAME, ", ", Sys.CPU_THREADS, " logical cores")
    println(io, "  ", rpad("Memory:", 20), round(Sys.total_memory() / 2^30; digits = 1), " GiB")

    println(io, "\nExecution")
    println(io, "  ", rpad("Julia threads:", 20), Threads.nthreads())
    println(io, "  ", rpad("Distributed workers:", 20), nworkers())
    println(io, "  ", rpad("BLAS:", 20), _report_blas_library(), ", ", LinearAlgebra.BLAS.get_num_threads(), " threads")

    println(io, "\nOptional features")
    println(io, "  ", rpad("CUDA extension:", 28), backend.cuda_extension_loaded ? "loaded" : "not loaded")
    println(io, "  ", rpad("CUDA preparation:", 28), backend.cuda_enabled ? "enabled" : "disabled")
    println(io, "  ", rpad("Clustering extension:", 28), _report_extension_loaded(:DiSLOUTrajectoriesClusteringExt) ? "loaded" : "not loaded")
    println(io, "  ", rpad("Semiclassical extension:", 28), _report_extension_loaded(:DiSLOUTrajectoriesQuantumCumulantsExt) ? "loaded" : "not loaded")

    println(io, "\nCPU preparation backend: ", uppercase(string(backend.cpu)))
    println(io, "Trajectory propagation runs on the CPU.")
    println(io, "\nDocumentation: https://alice-bob-sw.github.io/DiSLOUTrajectories.jl/")
    println(io, "Repository:    https://github.com/Alice-Bob-SW/DiSLOUTrajectories.jl")
    println(io, "Citation:      run cite() for BibTeX.")
    return nothing
end

"""
    about(io::IO = stdout)

Print the same report as [`versioninfo`](@ref).
"""
about(io::IO = stdout) = versioninfo(io)

"""
    cite(io::IO = stdout)

Print BibTeX entries for DiSLOUTrajectories.jl and its method paper to `io`.
"""
function cite(io::IO = stdout)
    println(io, _CITATION_BIBTEX)
    return nothing
end

# Paper: V, Λ, G, γ_j, and V†OV (Eqs. 15–17, Section 3.3.1).
function _prepare_diagonal_cache_data(
        H::Matrix{ComplexF64},
        C::Vector{Matrix{ComplexF64}}, Z::Vector{Matrix{ComplexF64}}
    )
    if _CUDA_DIAGONALIZATION_ENABLED[]
        try
            return _cuda_prepare_diagonal_data(H, C, Z)
        catch err
            err isa InterruptException && rethrow()
            _disable_cuda_diagonalization!()
            @warn "CUDA preparation failed; disabling CUDA and retrying with LAPACK" exception = (err, catch_backtrace()) maxlog = 1
        end
    end
    return _cpu_prepare_diagonal_data(H, C, Z)
end
