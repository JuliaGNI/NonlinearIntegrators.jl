# Scripts

Exploratory drivers and studies. Unlike `benchmark/`, nothing here is run by the docs build
or by CI — these are for investigating one question at a time.

The `oga_*.jl` files are the Orthogonal Greedy Algorithm studies; `vise_study.jl` is the VISE
driver; the rest are older manual drivers for individual integrators (`run_*.jl`, `test_*.jl`) and
post-processing helpers. Everything writes into `results/`, which is git-ignored.

## `vise_study.jl`

VISE against the linear variational integrators it is the nonlinear counterpart of — `CGVI` at the
same quadrature order, implicit midpoint, and a `Gauss(8)` reference — on the harmonic oscillator,
the perturbed pendulum and Hénon–Heiles. Prints a table, writes nothing.

```sh
julia --project=scripts scripts/vise_study.jl
julia --project=scripts scripts/vise_study.jl harmonic-oscillator
```

It replaces two files. `test_vise.jl` was 893 lines of which 854 were commented out: it held the
ansätze and the initial weight vectors, which is why it was kept, but it had not been runnable for
a long time. `vise_plot.jl` was 1257 lines that loaded `.jld2` archives from an absolute path on
another machine, and mixed CairoMakie, Plots, MLJ and SymbolicRegression to draw them. The figures
both were reaching for are now `NonlinearIntegrators.Diagnostics.plot_solution`, in the
`NonlinearIntegratorsPlots` extension — see `docs/src/vise/vise.md`.

The harmonic-oscillator row is a check rather than a measurement: its ansatz spans the exact
solution, so anything above the solver's residual floor there is a regression. The Hénon–Heiles
row is included precisely because it does *not* work — a three-term ansatz per coordinate does not
span that trajectory, and the relative error is O(1) well before the end of the run. The original
sweep recorded the same thing (`HenonHeiles_hams_err = 0.94` over `T = 200`), so this is the ansatz
being too small rather than a solver regression, and the row is there to keep that visible.

## OGA seed variants

Three studies cover the Orthogonal Greedy Algorithm seeds of `ShallowNet`,
across working precisions (`Float16`, `Float32`, `Float64`), `ReLUᵏ` powers, smooth
activations, and the `regularization_factor` ladder. They share
`oga_activations.jl` (float-generic activations, the λ ladder) and `oga_report.jl`
(figures and markdown).

The split into two tiers is the point of the design. End-to-end convergence conflates
the quality of the *seed* with the behaviour of the *solve*, and that confound is what
made the reduced-precision failures hard to attribute: a run that fails looks the same
whether the greedy fit went rank-deficient or the Newton Jacobian did.

### Tier A — seed quality, no integrator (`oga_fit_study.jl`)

Calls `oga_fit` directly: no integrator, no Newton solve, no time stepping. Sweeps
dictionary × selection × fit × activation × precision × target and reports

- `fit_err` — the quadrature-weighted L² error of the seed, recomputed in `Float64`
  from the returned parameters so that precisions share one scale;
- `cond` / `sigma_min` — the seed's design matrix, the proxy for whether the Newton
  system it feeds is solvable;
- `neurons` / `rejected` — how many of the requested neurons the greedy loop could
  place, and how many candidates it refused for adding no new direction.

Every case is an `S ≤ 8`, 11-node problem, so the whole grid runs in seconds.

### Tier B — end-to-end sweep (`oga_sweep.jl`)

The harmonic oscillator, ten steps, `S = 4`, `R = 8`, `dt = 0.1`, over seed variant ×
precision × regularization factor × activation, in two stages:

- `relu` — `ReLUᵏ` for `k = 1…4`, where the `{±1} × (bias grid)` dictionary is
  theoretically complete, so anything that goes wrong is numerical. This is the
  reduced-precision question.
- `smooth` — ELU, GELU and tanh against the 2-D and angular dictionaries built for
  them. This is the activation question.

λ is swept as multiples of `√eps(T)` — `2^k √eps(T)` for `k = 1…6` at
`Float16`/`Float32` and `k = 2, 4, …, 12` at `Float64` — plus a `λ = 0` control, so the
Jacobian-diagonal shift is scaled to the precision it protects. An absolute `1e-5` sits
far below `√eps` at anything but `Float64` and cannot lift a near-singular Jacobian in
reduced precision at all.

**The residual tolerance is scaled the same way**, and it has to be. The solver's default
`f_abstol` is `1.78e-15`, an absolute value scaled to `Float64` and unreachable at
`Float32` or `Float16`; a reduced-precision run then sits at its residual floor and burns
the whole iteration budget while parked on the right answer. Measured before the fix,
`ReLU³` at `Float32` reported 1000 iterations at *every* regularization factor with an
accuracy of `1.8e-7`
— which, read as non-convergence, would have made the whole `Float32` column an artefact
of the tolerance. The sweeps pass `f_abstol = 256·eps(T)` (`oga_f_abstol`).

Two other classification points worth knowing when reading the CSVs: a run that exhausts
`max_iterations` is recorded as `maxiter`, not `ok` (it returns a finite state, so the
naive check would call it converged), and a run whose final state has left the working
precision is recorded as `upcast`.

### Tier B′ — double pendulum (`oga_double_pendulum.jl`)

The problem the seed fails hardest on, at **one** λ rather than the whole ladder: the
harmonic-oscillator sweep already answers what λ does. The value is read from
`results/oga_sweep_relu.csv` — the factor that converged most often there — so it is
measured rather than asserted; absent that file it falls back to the documented
`16√eps(T)` and says so.

### Running

```
julia --project=scripts -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=scripts scripts/oga_fit_study.jl
julia --project=scripts scripts/oga_sweep.jl            # both stages
julia --project=scripts scripts/oga_sweep.jl relu       # just the ReLUᵏ stage
julia --project=scripts scripts/oga_double_pendulum.jl  # after oga_sweep.jl relu
```

Each writes a CSV, a markdown report and PNG figures into `results/`. The reports can be
regenerated from the CSVs alone (`write_fit_study_report` / `write_sweep_report` in
`oga_report.jl`) without re-running anything.

### Background

The original formulation solved the fit through the normal equations, whose condition
number is `κ(Φ)²`; that is what forced the `Float64` island and what goes rank-deficient
in reduced precision. See the "Orthogonal Greedy Algorithm" section of the package
documentation for the analysis, the variant taxonomy, and why `±1` weights suit `ReLUᵏ`
but under-serve smooth activations.


### Files

| File | Role |
|---|---|
| `oga_activations.jl` | float-generic activations (`ReLUᵏ`, ELU, GELU), the λ ladder, the precision-scaled `f_abstol` |
| `oga_report.jl` | CSV parsing, CairoMakie figures, markdown reports — shared by all three studies |
| `oga_fit_study.jl` | Tier A: seed quality, no integrator |
| `oga_sweep.jl` | Tier B: end-to-end harmonic oscillator, `relu` and `smooth` stages |
| `oga_double_pendulum.jl` | Tier B′: the hardest problem at a single λ |

Reports regenerate from the CSVs alone — `write_fit_study_report` / `write_sweep_report` in
`oga_report.jl` — so a figure or table can be reworked without re-running a sweep.
