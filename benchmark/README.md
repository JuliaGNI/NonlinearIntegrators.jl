# Benchmarks

## OGA seed variants — moved

The Orthogonal Greedy Algorithm studies now live in `scripts/` (`oga_fit_study.jl`,
`oga_sweep.jl`, `oga_double_pendulum.jl`, sharing `oga_activations.jl` and
`oga_report.jl`), and write into `scripts/results/`. They are studies of one component
rather than benchmarks of the integrator suite, which is what this directory holds. See
`scripts/README.md` and the *Orthogonal Greedy Algorithm* section of the package
documentation.

## Shallow-net solver / precision / config sweep

A systematic, SolverBenchmark-style sweep of `ShallowNet` — **one runnable
file per test problem** — over timestep × precision × quadrature order `R` × network
width `S` × activation × nonlinear-solver strategy × regularization `λ` × initial-guess
strategy. Every case integrates exactly **10 steps**; the interval is adapted per case as
`timespan = (0, 10·dt)`. Purpose: (a) find which configs work well per problem, (b)
surface package issues that hurt performance, (c) identify robust solver strategies.

Problems (from GeometricProblems): harmonic oscillator, pendulum (a degenerate 2-component
IODE — no `lodeproblem` exists), double pendulum, Toda lattice with `N = 16`.

Solver strategies: `Newton` with `Static` / `Backtracking` / `StrongWolfe` line search, and
trust-region `DogLeg`. Initial-guess (trajectory) strategies: `midpoint`
(`IntegratorExtrapolation`, the default), `Hermite` (`HermiteExtrapolation`), and
`previous solution` (`NoExtrapolation`).

### Files
- `shallownet_benchmark_common.jl` — shared sweep engine, presets, builders, per-run measurement, CSV.
- `shallownet_report.jl` — CSV parsing, CairoMakie plots, markdown report.
- `run_harmonic_oscillator.jl`, `run_pendulum.jl`, `run_double_pendulum.jl`,
  `run_toda_lattice.jl` — one per problem.
- `report.jl` — aggregates all `results/*.csv` into a combined report.

### Modes
Each run file takes a mode as `ARGS[1]` or `ENV["SHALLOWNET_BENCH_PRESET"]` (default `quick`):

| axis | `full` | `quick` |
|---|---|---|
| dt | 0.01, 0.1, 1.0, 10.0 | 0.1, 1.0, 10.0 |
| precision | Float16, Float32, Float64 | Float64, Float32, Float16 |
| R | 4, 8, 16 | 8 (16 for double pendulum & Toda) |
| S | 4, 6, 8 | 4 (8 for double pendulum & Toda) |
| activation | relu², relu³, relu⁴, tanh | relu³, tanh |
| solver | Newton/{Static,Backtracking,StrongWolfe}, DogLeg | DogLeg |
| λ (regularization) | 0.0, 1e-7, 1e-5, 1e-3, 16√eps(T) | 16√eps(T) |
| initial guess | midpoint, Hermite, previous | midpoint |
| max_iterations | 10000 | 100 |

The `16√eps(T)` regularization scales the Jacobian-diagonal damping with the working
precision: ≈2.4e-7 at Float64, ≈5.5e-3 at Float32, and **0.5 at Float16** — the last is
large and tends to over-damp half precision (a documented data point, not a bug).

`quick` ≈ 18 cases/problem (seconds–minutes); `full` ≈ 26 000 cases/problem (hours — results
are flushed to CSV per case, so an interrupted run keeps its partial output).

Reference for the accuracy error `ref_err`: for every problem, `Gauss(8)` integrated at
Float64 using the smallest timestep in the sweep, over the same 10-step horizon; the
case's final state is compared against it.

Plots (per problem and combined): convergence success-rate bars and a solver×precision
heatmap, plus metric-vs-timestep scatters (coloured by precision) for accuracy, energy
drift, run time, and nonlinear iterations.

### Running
```
julia --project=benchmark -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=benchmark benchmark/run_harmonic_oscillator.jl          # quick
julia --project=benchmark benchmark/run_toda_lattice.jl full            # full sweep
julia --project=benchmark benchmark/report.jl                           # combined report
```
Each run writes `results/<problem>_<mode>.csv`, a `results/<problem>_<mode>.md` summary, and
PNG plots. `report.jl` writes the combined `results/shallownet_benchmark.md`. (The `results/`
contents are git-ignored.)

### Note

Developing this benchmark surfaced (goal b) and fixed a pre-existing package bug: the
`ShallowNet` Hermite `initial_trajectory!` built a `(t, q, p, v, f)` solution
tuple, but the current `GeometricIntegratorsBase` `HermiteExtrapolation` expects `q̇`/`ṗ`
fields, so the `Hermite` initial-guess strategy failed with a `FieldError`. The tuple field
names are now corrected, so all three initial-guess strategies run.

## Derivative backends — symbolic vs ForwardDiff

`benchmark/compare_derivative_backends.jl` compares the four shallow-net integrators along
the one axis that separates them: how they obtain the derivatives of the ansatz with
respect to the network parameters.

| integrator | backend |
|---|---|
| `ShallowNet`, `ShallowNetReversible` | `basis.dqdθ` / `basis.dvdθ`, compiled once by `SymbolicNeuralNetworks.jl` |
| `ShallowNetAutodiff`, `ShallowNetAutodiffReversible` | `ForwardDiff.gradient` of a hand-written ansatz, on every evaluation |

The symbolic pair is run under two code-generation settings, so the comparison also covers
what `SymbolicNeuralNetworks` 0.4.0 changed:

| codegen | meaning |
|---|---|
| `cse+inplace` | the 0.4.0 defaults — common-subexpression elimination, and a batch evaluated by an in-place kernel writing into one preallocated array |
| `plain` | `cse = false, inplace = false`: the code generation of 0.3.x. Same mathematics, different emitted code. |

Three measurements:

1. **Basis build** — the one-off symbolic compilation, per codegen setting. Zero for the
   autodiff pair, which is handed a `ShallowNetBasis{T}(σ, S; symbolic = false)`.
2. **End-to-end solve, cold and warm** — every case is solved twice on a fresh integrator.
   The first solve carries the specialization of the generated kernels (symbolic) or of the
   ForwardDiff tape (autodiff); the second is the steady-state cost and is what the
   accuracy, drift, iteration and timing columns report.
3. **Derivative kernels in isolation** — `basis.dqdθ` / `basis.dvdθ` under both codegen
   settings, against `∂NN_anstaz_∂params` / `∂VNN_anstaz_∂params`, per call, over `S` and
   precision. The two codegen settings are also compared *numerically* here, which is the
   one place that comparison is meaningful — see below.

Everything else is pinned: `Newton`/`Backtracking`, midpoint initial trajectory,
`λ = 16·√eps(T)`, 10 steps per case, and the per-problem `R`/`S` the sweep above uses.

**Two caveats, repeated in the generated report.** The two backends do not discretize the
same thing — the symbolic pair uses the raw network `q(t) = NN(t; θ)`, the autodiff pair the
boundary-interpolating `q_h(t) = (1-t)q̄ + t·q + t(1-t)·NN(t)`, with a different unknown
layout and a different `update!`. And each integrator keeps its own default OGA seed
(`ShallowNetAutodiff` selects on the normalized inner product, the other three on the raw
one), because those are tuned per-integrator baselines. So accuracy and iteration counts
compare *methods*; only the timings compare backends.

The same rule holds for the two codegen settings, for a different reason. At the kernel
level they agree to machine epsilon (3e-17 at Float64), as they must. End to end they do
not: the residual stalls near the round-off floor, so a last-bit difference in the
derivative decides which iterate Newton accepts, and a `ref_err` already at 1e-13 moves by
orders of magnitude. The report measures that amplification rather than asserting it away.

```
julia --project=benchmark benchmark/compare_derivative_backends.jl          # quick
julia --project=benchmark benchmark/compare_derivative_backends.jl full     # + double pendulum, Float16, gelu
DERIV_BENCH_REUSE=true julia --project=benchmark benchmark/compare_derivative_backends.jl  # re-report only
```

| axis | `quick` | `full` |
|---|---|---|
| problems | harmonic oscillator, pendulum | + double pendulum |
| precision | Float64, Float32 | + Float16 |
| dt | 0.1, 1.0 | 0.1, 1.0, 10.0 |
| activation | tanh | tanh, gelu |
| kernel sweep `S` | 4, 8, 16 | 4, 8, 12, 16 |

Writes `results/derivative_backends_<mode>.csv`, `results/derivative_backends_kernels.csv`,
`results/derivative_backends_codegen_agreement.csv`, `results/derivative_backends.md` and
five PNGs.
