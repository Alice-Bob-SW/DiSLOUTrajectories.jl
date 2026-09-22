# Examples

The examples linked here show how to use `DiSLOUTrajectories.jl` on systems similar to those studied in the [DiSLOUTrajectories paper](./getting_started/cite.md). Start with the Kerr walkthrough,
then apply the same workflow to the more complicated two-mode models:

- [Driven Kerr resonator](examples/kerr_resonator.md): discover two gauges,
  compare trajectories with the master equation, and enable Layer III.
- [Two-mode ideal cat](examples/two_mode_ideal_cat.md): discover the memory
  branches and inspect quadrature relaxation, a trajectory switch, and Wigner
  snapshots.
- [Two-mode detuned cat](examples/two_mode_detuned_cat.md): find three metastable regions, compute the steady state, and compare full-space and reduced
  trajectory propagation.

All quation and section
numbers refer to the companion paper, but the numerical parameters may be different.

## Running the examples

To run an example yourself, clone the repository and start Julia 1.10 or
later from its root, optionally enabling multiple threads:

```sh
julia --threads=auto
```

or start a [Julia Jupyter kernel](https://ijulia.org/stable/).

Run the page's code blocks in order in the same session. Each page starts
by activating `examples/Project.toml`, using the local DiSLOUTrajectories checkout, and
instantiating the environment.
It is best to use a fresh Julia session for each example.

### Detuned-cat dependencies

The detuned cat example includes the helpers in `examples/exact_steady_state.jl` for finding the steady state. Its CPU
steady-state solver uses MUMPS, and its GPU path uses CUDA and CUDSS.
After running the page's environment setup block, install its additional
dependencies once:

```julia
Pkg.add(["CUDA", "CUDSS", "MUMPS", "ModelingToolkitBase"])
```

`exact_steadystate` selects the GPU path when CUDA is functional and the
CPU path otherwise. The example environment is separate from `docs/Project.toml`;
so that these optional dependencies are not needed to build the manual.
