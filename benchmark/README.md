# Benchmarks

## OGA initial-guess comparison

`oga_comparison.jl` compares the two Orthogonal Greedy Algorithm initial-guess
variants of `NonLinear_OneLayer_GML`:

- **`OGA1d`** (default) — the seed is assembled at the working precision `T` and the
  output weights come from a QR fit of the `√w`-scaled design matrix;
- **`OGA1d_Legacy`** — the previous algorithm: the seed is assembled in `Float64`
  (a "double-precision island") and the output weights come from the normal
  equations `Gk \ rhs`, then rounded into the working type.

Both are run end-to-end (OGA seed + Newton solve) on the harmonic oscillator over
`t ∈ [0, 1]`, sweeping problem complexity along two axes — the time-step length `dt`
(hence the number of steps `1/dt`) and the number of neurons `S` (hence the number
of network parameters) — at three working precisions (`Float64`, `Float32`,
`Float16`). The script reports, per case, the solver status, the error of the final
state against the analytic solution, and the wall-clock time (after a warm-up run).

The background: the legacy Gram/normal-equations solve has condition number
`κ(Φ)²`, which forces the `Float64` island and becomes rank-deficient in reduced
precision; the QR reformulation is conditioned on `κ(Φ)`. See the "Orthogonal Greedy
Algorithm" section of the package documentation for the full analysis.

### Running

```
julia --project=benchmark -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=benchmark benchmark/oga_comparison.jl
```

### What to expect

- **`Float64` / `Float32`** — where both algorithms converge they reach essentially
  the same accuracy (the seed is only a warm start; the Newton solve sets the final
  error). This confirms the reformulation is a no-regression change.
- **Higher parameter counts** (`S = 8`) — the legacy seed already tends to go
  `singular` (near-duplicate neurons make the Gram matrix / Newton Jacobian
  rank-deficient), while the QR seed still solves.
- **`Float16`** — the legacy variant fails (`singular` / `diverged`) across the
  board, whereas the QR seed stays finite and lets the solve proceed for the smaller
  networks.

## One-layer GML solver / precision / config sweep

A systematic, SolverBenchmark-style sweep of `NonLinear_OneLayer_GML` — **one runnable
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
- `gml_benchmark_common.jl` — shared sweep engine, presets, builders, per-run measurement, CSV.
- `gml_report.jl` — CSV parsing, CairoMakie plots, markdown report.
- `run_harmonic_oscillator.jl`, `run_pendulum.jl`, `run_double_pendulum.jl`,
  `run_toda_lattice.jl` — one per problem.
- `report.jl` — aggregates all `results/*.csv` into a combined report.

### Modes
Each run file takes a mode as `ARGS[1]` or `ENV["GML_BENCH_PRESET"]` (default `quick`):

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
PNG plots. `report.jl` writes the combined `results/onelayer_gml_benchmark.md`. (The `results/`
contents are git-ignored.)

### Note

Developing this benchmark surfaced (goal b) and fixed a pre-existing package bug: the
`NonLinear_OneLayer_GML` Hermite `initial_trajectory!` built a `(t, q, p, v, f)` solution
tuple, but the current `GeometricIntegratorsBase` `HermiteExtrapolation` expects `q̇`/`ṗ`
fields, so the `Hermite` initial-guess strategy failed with a `FieldError`. The tuple field
names are now corrected, so all three initial-guess strategies run.
