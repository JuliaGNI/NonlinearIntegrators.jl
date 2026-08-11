# Benchmarks

The package ships a benchmark suite (under `benchmark/`) for the one-layer GML
variational integrator `NonLinear_OneLayer_GML`. It runs each of several test
problems through a large grid of integrator configurations and records, for every case,
whether the nonlinear solve converged, how accurate the result is, how much the energy
drifts, how many nonlinear iterations it took, and how long it ran. The results are
written to CSV, summarised in a markdown report, and visualised with a set of plots.

The suite has three goals:

1. find which integrator-parameter configurations work well for each test problem;
2. identify issues in the package that are detrimental to performance;
3. identify robust solver strategies.

Three further studies target the OGA initial guess specifically:
`scripts/oga_fit_study.jl` measures seed quality on its own (no integrator, no Newton
solve), `scripts/oga_sweep.jl` runs the end-to-end harmonic-oscillator sweep over seed
variant × precision × regularization factor × activation, and
`scripts/oga_double_pendulum.jl` repeats a reduced grid on the hardest problem. They live
under `scripts/` rather than here because they study one component rather than the
integrator suite, and they are not part of the documentation build. See the *Orthogonal
Greedy Algorithm* section and `scripts/README.md`.

## What is swept

Each case integrates a problem for exactly **10 time steps**; the time span is adapted
per case as `(0, 10·dt)`. The sweep spans, per problem:

| axis | meaning |
|---|---|
| timestep `dt` | integration step size |
| precision | working floating-point type (`Float16` / `Float32` / `Float64`) |
| `R` | Gauss–Legendre quadrature order |
| `S` | number of hidden neurons (network width) |
| activation | `ReLUᵏ` (`k = 2, 3, 4`), `ELU`, `GELU`, or `tanh` |
| solver strategy | `Newton` with `Static` / `Backtracking` / `StrongWolfe` line search, or trust-region `DogLeg` |
| `λ` | Jacobian regularization (`regularization_factor`) |
| initial guess | `midpoint` (`IntegratorExtrapolation`), `Hermite` (`HermiteExtrapolation`), or `previous` (`NoExtrapolation`) |

The test problems (from
[GeometricProblems.jl](https://github.com/JuliaGNI/GeometricProblems.jl)) are the
harmonic oscillator, the pendulum (a degenerate two-component IODE — it has no
`lodeproblem`), the double pendulum, and the Toda lattice with `N = 16`.

## Modes

Each per-problem run file takes a mode — `quick` (default) or `full` — from its first
command-line argument or from the `GML_BENCH_PRESET` environment variable.

| axis | `full` | `quick` |
|---|---|---|
| `dt` | 0.01, 0.1, 1.0, 10.0 | 0.1, 1.0, 10.0 |
| precision | Float16, Float32, Float64 | Float64, Float32, Float16 |
| `R` | 4, 8, 16 | 8 (16 for double pendulum & Toda) |
| `S` | 4, 6, 8 | 4 (8 for double pendulum & Toda) |
| activation | ReLU², ReLU³, ReLU⁴, ELU, GELU, tanh | GELU, tanh |
| solver | Newton/{Static, Backtracking, StrongWolfe}, DogLeg | DogLeg |
| `λ` | 0.0, 1e-7, 1e-5, 1e-3, 16√eps(T) | 16√eps(T) |
| initial guess | midpoint, Hermite, previous | midpoint |
| `max_iterations` | 10000 | 100 |

`quick` is roughly 18 cases per problem (seconds to minutes each — the Toda lattice is
the slowest because of its `N = 16` state and larger network); `full` is on the order of
tens of thousands of cases per problem (hours). Results are flushed to CSV per case, so an
interrupted `full` run keeps its partial output.

The `16√eps(T)` regularization scales the Jacobian-diagonal damping with the working
precision: ≈2.4e-7 at `Float64`, ≈5.5e-3 at `Float32`, and 0.5 at `Float16`. The last is
large and tends to over-damp half precision; note, however, that at half precision the
`ReLUᵏ` basis is ill-conditioned and diverges independently of `λ`, whereas `tanh` still
converges — the accuracy limit there is the precision, not the regularization.

## Metrics

For each case the suite records:

- **status** — `ok`, or a failure class: `maxiter` (the nonlinear solve exhausted
  `max_iterations` — it still returns a finite state, so this is *not* convergence and is
  counted separately, following the same rule as the OGA studies in `scripts/`),
  `singular`, `diverged`, `nonfinite`, or the short name of any other exception raised;
- **`ref_err`** — the relative max-norm error of the final state against a reference,
  which is a `Gauss(8)` integration at `Float64` using the smallest timestep in the
  sweep (over the same 10-step horizon);
- **`ham_drift`** — the maximum relative drift of the Hamiltonian over the run;
- **`iterations`** — the nonlinear-solver iteration count of the final step;
- **`solve_secs` / `total_secs`** — the summed nonlinear-solve time and the wall-clock
  time of the run.

## Running

Instantiate the benchmark environment (it `dev`s the package):

```
julia --project=benchmark -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
```

Run one or more problems (mode defaults to `quick`):

```
julia --project=benchmark benchmark/run_harmonic_oscillator.jl
julia --project=benchmark benchmark/run_pendulum.jl
julia --project=benchmark benchmark/run_double_pendulum.jl
julia --project=benchmark benchmark/run_toda_lattice.jl full   # full sweep
```

Each run writes `benchmark/results/<problem>_<mode>.csv`, a
`benchmark/results/<problem>_<mode>.md` report, and PNG plots. Finally, aggregate every
CSV present into a combined report:

```
julia --project=benchmark benchmark/report.jl
```

which writes `benchmark/results/onelayer_gml_benchmark.md`. The reporting step reads the
CSVs, so a report can be regenerated (or restyled) without re-running the sweep. All
`benchmark/results/` contents are git-ignored.

## Outputs

The CSV has one row per case with the columns

```
problem, T, dt, steps, R, S, activation, solver, linesearch, initial_guess,
lambda, status, ref_err, ham_drift, iterations, solve_secs, total_secs
```

The markdown report contains a status breakdown, convergence/robustness tables (by solver
strategy, initial-guess strategy, precision, and problem), the best configuration found
per problem, and failure hot-spots. It embeds the plots:

- **convergence** — success-rate bars per solver strategy, and a solver × precision
  heatmap (red = not converged, green = converged); the combined report also draws a
  success-rate bar per problem;
- **accuracy**, **energy drift**, **run time**, and **nonlinear iterations** — each as a
  scatter versus the timestep. A per-problem report colours the dots by precision; the
  combined report colours them by problem, so the four problems stay distinguishable.

## Results

The figures on this page are **not committed**: they are regenerated at documentation-build
time by a fresh `quick` run over the four problems (driven from `docs/make.jl`), so they
track the current package. The narrative and tables are kept as an illustrative reference
from one representative `quick` run — the numbers are not a fixed reference and may not
match the freshly generated figures exactly. That run used the `DogLeg` solver, the
`midpoint` initial guess, and the precision-scaled regularization `λ = 16√eps(T)`.

Results are organised as a summary across all problems followed by one section per problem
(harmonic oscillator, pendulum, double pendulum, Toda lattice). The summary scatters are
coloured by problem; each per-problem section shows that problem's scatters coloured by
precision.

### Summary across all problems

Across the four problems (72 cases in the representative run), 28 met the solver's
convergence criterion and a further 26 produced a finite trajectory without meeting it
(`maxiter`). The two are counted separately, and the medians below are over all 54 runs that
produced a trajectory, since a stalled run's accuracy is still measured.

| precision | cases | converged | success | measured | median `ref_err` | median `ham_drift` | median `iter` |
|---|---|---|---|---|---|---|---|
| Float16 | 24 | 10 | 42% | 10 | 7.11e-03 | 1.07e-02 | 1 |
| Float32 | 24 | 18 | 75% | 22 | 2.30e-04 | 3.11e-04 | 6 |
| Float64 | 24 | 0 | 0% | 22 | 1.64e-04 | 2.01e-04 | 100 |

The `Float64` row needs reading with the preset in mind, and it is the clearest illustration
of why `maxiter` is tracked separately. Convergence is judged against
`f_abstol = max(8, solversize)·eps(T)`, which at `Float64` is ≈1.8e-15 — a target these
problems reach only after thousands of iterations, while `quick` caps at 100. Measured on
the harmonic oscillator with the cap lifted to 10000, the same case converges at **8568**
iterations. So the `Float64` zero is a statement about the preset's iteration budget, not
about the integrator; the accuracy column shows those runs are the most accurate of the
three precisions. `Float32` reaches its (looser) tolerance in a handful of iterations, and
`Float16` in one — which is why its success rate is the *highest* while its accuracy is the
worst by two orders of magnitude. Success rate and accuracy answer different questions here.

Success rate broken down by problem, by solver strategy, and by solver × precision:

![Convergence success rate by problem](figures/onelayer_gml_benchmark_convergence_problem.png)

![Convergence success rate by solver strategy](figures/onelayer_gml_benchmark_convergence_solver.png)

![Convergence success rate by solver and precision](figures/onelayer_gml_benchmark_convergence_heatmap.png)

Accuracy, energy drift, run time and nonlinear-iteration counts versus the timestep, with
all four problems overlaid and **coloured by problem** (each dot is one case that produced a
trajectory, `ok` or `maxiter`). Accuracy and energy conservation degrade sharply as the
timestep grows; at `dt = 10` the 10-step horizon is far too coarse and the relative error is
`O(1)`.

![Accuracy versus timestep](figures/onelayer_gml_benchmark_accuracy_vs_dt.png)

![Energy drift versus timestep](figures/onelayer_gml_benchmark_energy_drift_vs_dt.png)

![Run time versus timestep](figures/onelayer_gml_benchmark_runtime_vs_dt.png)

![Nonlinear iterations versus timestep](figures/onelayer_gml_benchmark_iterations_vs_dt.png)

The most accurate (lowest `ref_err`) configuration found for each problem, over the runs
that produced a trajectory:

| problem | best `ref_err` | T | dt | network | iguess / λ |
|---|---|---|---|---|---|
| harmonic\_oscillator | 1.27e-06 | Float32 | 0.1 | R8 S4 gelu | midpoint, λ=5.5e-3 |
| pendulum | 2.97e-05 | Float32 | 0.1 | R8 S4 gelu | midpoint, λ=5.5e-3 |
| double\_pendulum | 7.08e-08 | Float64 | 0.1 | R16 S8 tanh | midpoint, λ=2.4e-7 |
| toda\_lattice | 5.94e-10 | Float64 | 0.1 | R16 S8 gelu | midpoint, λ=2.4e-7 |

The `quick` preset sweeps only `gelu` and `tanh`; the `ReLUᵏ` powers are in `full`. Outright
failures — as opposed to `maxiter` — concentrate at half precision and at the largest
timestep `dt = 10`, consistent with the accuracy plot.

### Harmonic oscillator

The simplest test problem: a single linear oscillator, and the one with the highest
convergence rate of the four (9 of 18 in the representative `quick` run). The precision
split is the clearest — Float64 and Float32 track each other closely on accuracy
(`ref_err ≈ 1e-6` at `dt = 0.1` with the `quick` preset's smooth activations) while Float16
is limited by the working precision. Each dot below is a case that produced a trajectory,
coloured by precision.

![Accuracy versus timestep — harmonic oscillator](figures/harmonic_oscillator_quick_accuracy_vs_dt.png)

![Energy drift versus timestep — harmonic oscillator](figures/harmonic_oscillator_quick_energy_drift_vs_dt.png)

![Run time versus timestep — harmonic oscillator](figures/harmonic_oscillator_quick_runtime_vs_dt.png)

![Nonlinear iterations versus timestep — harmonic oscillator](figures/harmonic_oscillator_quick_iterations_vs_dt.png)

![Convergence heatmap — harmonic oscillator](figures/harmonic_oscillator_quick_convergence_heatmap.png)

### Pendulum

A *degenerate* two-component IODE (`ϑ`: `p₁ = ml²q₂`, `p₂ = 0`; it has no `lodeproblem`),
included deliberately to stress the nonlinear solve. It is nonlinear and about an order of
magnitude less accurate than the harmonic oscillator (`ref_err ≈ 3e-5` at `dt = 0.1`), and
it needs the most iterations of the four — a median of 100, i.e. the `quick` cap.

![Accuracy versus timestep — pendulum](figures/pendulum_quick_accuracy_vs_dt.png)

![Energy drift versus timestep — pendulum](figures/pendulum_quick_energy_drift_vs_dt.png)

![Run time versus timestep — pendulum](figures/pendulum_quick_runtime_vs_dt.png)

![Nonlinear iterations versus timestep — pendulum](figures/pendulum_quick_iterations_vs_dt.png)

![Convergence heatmap — pendulum](figures/pendulum_quick_convergence_heatmap.png)

### Double pendulum

A four-dimensional chaotic system, and the hardest of the four: it accounts for every
`singular` case in the representative run and has the lowest convergence rate (4 of 18).
Quick mode uses a larger network (`R = 16`, `S = 8`) than the two simple problems; `tanh`
gives the best accuracy here (`ref_err ≈ 7e-8` at `dt = 0.1`, Float64).

![Accuracy versus timestep — double pendulum](figures/double_pendulum_quick_accuracy_vs_dt.png)

![Energy drift versus timestep — double pendulum](figures/double_pendulum_quick_energy_drift_vs_dt.png)

![Run time versus timestep — double pendulum](figures/double_pendulum_quick_runtime_vs_dt.png)

![Nonlinear iterations versus timestep — double pendulum](figures/double_pendulum_quick_iterations_vs_dt.png)

![Convergence heatmap — double pendulum](figures/double_pendulum_quick_convergence_heatmap.png)

### Toda lattice (N = 16)

The largest problem, with a 16-dimensional state and a correspondingly larger network
(`R = 16`, `S = 8` in quick mode). It is the slowest to run — its run-time scatter sits
above the other three problems in the summary — and also the most accurate, reaching
`ref_err ≈ 6e-10` at `dt = 0.1` in Float64.

![Accuracy versus timestep — Toda lattice](figures/toda_lattice_quick_accuracy_vs_dt.png)

![Energy drift versus timestep — Toda lattice](figures/toda_lattice_quick_energy_drift_vs_dt.png)

![Run time versus timestep — Toda lattice](figures/toda_lattice_quick_runtime_vs_dt.png)

![Nonlinear iterations versus timestep — Toda lattice](figures/toda_lattice_quick_iterations_vs_dt.png)

![Convergence heatmap — Toda lattice](figures/toda_lattice_quick_convergence_heatmap.png)

## Extending

To add a problem, copy one of the `run_*.jl` files and supply a `build_prob(T, timespan,
timestep)` closure returning an `AbstractProblemIODE` at element type `T`, plus a
`hamiltonian(t, q, p, params)` closure; then call `run_sweep`. Per-problem axis overrides
(as the double pendulum and Toda lattice use for `R` and `S`) are passed as the `Rs` /
`Ss` keyword arguments to `run_sweep`. The shared engine, presets, and reporting live in
`benchmark/gml_benchmark_common.jl` and `benchmark/gml_report.jl`.
