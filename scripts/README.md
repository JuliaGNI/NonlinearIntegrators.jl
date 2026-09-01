# Scripts

Exploratory drivers and studies. Unlike `benchmark/`, nothing here is run by the docs build or by
CI — these are for investigating one question at a time.

**Code here, data in `runs/`, figures in `results/`.** Both output directories are at the
repository root and git-ignored; the rule is in `Packages/CLAUDE.md` and the reasoning in
`Knowledge/AI/Folder-Structure.md`. Every driver takes `--runs-dir` and `--results-dir`, so the
same script can write into a paper's or a talk's figure directory without being copied there.

```sh
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'
```

## The experiment harness

One registry, five drivers that solve and archive, one renderer. Run in this order; each stage
reads what the previous one wrote.

```sh
julia --project=scripts scripts/run_vise.jl          # ~4 min  → runs/<problem>-vise-h*.jld2
julia --project=scripts scripts/run_nvi.jl           # ~5 min  → runs/<problem>-S*-h*.jld2
julia --project=scripts scripts/run_fourier.jl       # ~4 min  → runs/*-fourier-T*.jld2
julia --project=scripts scripts/run_convergence.jl   # ~25 min → runs/*-convergence-*.jld2
julia --project=scripts scripts/run_oga_seeds.jl     # ~10 min → runs/*-oga-seeds-*.jld2
julia --project=scripts scripts/figures.jl           # ~1 min  → the PDFs in results/
```

Each takes arguments, so one figure can be regenerated on its own:

```sh
julia --project=scripts scripts/run_vise.jl harmonic-oscillator 5.0
julia --project=scripts scripts/run_nvi.jl harmonic-oscillator-S4R8Q16relu3-h1.0
julia --project=scripts scripts/run_convergence.jl perturbed-pendulum
julia --project=scripts scripts/run_convergence.jl harmonic-oscillator vise --final-time 50
julia --project=scripts scripts/run_oga_seeds.jl --steps 1.0,2.0
julia --project=scripts scripts/figures.jl harmonic-oscillator
```

**The solves and the figures are deliberately separate stages, with `runs/` between them.** A
figure can be restyled, or a caption argued about, without paying for the solves again, and a
missing figure stays distinguishable from a failed run.

| file | does |
|:--|:--|
| `archives.jl` | where output goes, the archive schema and its reader/writer, the option parser, reporting. Deps: JLD2 and Printf. `include`d by everything; nothing runs at top level. |
| `experiments.jl` | the registry — problems, ansätze, initial weights, quadrature orders, configurations, per-family solver options — and the diagnostics that need `GeometricIntegrators`. Also `include`d, also inert. |
| `basis_fits.jl` | separable trigonometric least squares, an allocation-free periodogram, `odd_harmonic_fit`, `lattice_fit`. Used by `run_fourier.jl`. |
| `run_vise.jl` | the symbolic-ansatz integrator, 3 problems × 4 time steps, plus a summary table |
| `run_nvi.jl` | the network integrators — shallow and dense — at every time step |
| `run_fourier.jl` | global Fourier and lattice ansätze fitted to a whole trajectory; no time step |
| `run_convergence.jl` | Hamiltonian error against time step: 2 problems × 3 integrator families |
| `run_oga_seeds.jl` | does the orthogonal-greedy seed change the answer? six variants |
| `figures.jl` | globs `runs/`, renders, writes `results/`. **The only place `save` is called.** |
| `compare_runs.jl` | compares two directories of archives numerically; the check for any change that is supposed to be a refactoring |

### No figure code lives here

Figures are built by `NonlinearIntegrators.Diagnostics`, in the `NonlinearIntegratorsPlots`
extension, which `using CairoMakie` activates. `figures.jl` reads an archive, hands it to
`Diagnostics.figures`, and saves what comes back. The extension adds only what `GeometricProblems`
does not already have:

| want | use |
|:--|:--|
| several integrators in one figure, with the continuous solution between the steps | `Diagnostics.plot_solution` |
| several error series against the time step, with a slope per family | `Diagnostics.plot_convergence` |
| every figure one archived run earns, named | `Diagnostics.figures` |
| a phase portrait, a trajectory, traces | the per-problem `GeometricProblems` recipes |
| one method against its expected order | `GeometricProblems.Diagnostics.plot_convergence` |

Fonts, line widths and marker sizes come from `Diagnostics.plot_theme()` — the shared theme of this
ecosystem, kept identical to the copy in `GeometricExamples/src/common.jl`, applied once inside
`figures.jl`. Nothing in the extension sets a size of its own, so changing the theme changes every
figure.

**`GeometricProblems.Diagnostics.plot_energy_error` cannot be reused here, and this was found the
hard way.** It does not work on a *partitioned or implicit* solution, which is every solution these
scripts produce. Its `_invariant_error` branches on `sol isa Union{SolutionPODE, SolutionPDAE}` to
decide whether to pass `p` to the invariant, and that test is `false` for a `GeometricSolution` of a
`LODEProblem` — even though `SolutionPODE`'s definition names `LODEProblem`, because the alias binds
`probType` both as a parameter and in its `where` clause, so the constraint does not apply as it
reads. The `q`-only branch is therefore always taken and a Hamiltonian expecting
`(t, q, p, params)` is called with three arguments. Measured on GeometricProblems 0.8.3 /
GeometricSolutions 0.6.5, and guarded by a `@test_broken` in `test/plots_tests.jl` with a second
assertion pinning the cause, so a fix upstream is noticed rather than silently changing figures.

### Naming

**One scheme: `<problem>-<method>-h<timestep>`**, so sorting a directory groups every method of a
problem together and every step of a method. Two extensions of it, for figures that are not one run
at one step: `<problem>-<study>-<variant>` for a sweep, and `…-h<timestep>-t<interval>` for the same
run over several intervals. All five are `figure_stem`, `study_stem`, `window_stem`,
`galerkin_label` and `network_label` in `src/plots.jl` — in the package rather than here, so that
the extension naming a figure and the driver finding that figure's archive share one definition.

`Q` is a label and `R` a constructor argument, with `Q = 2R` always because the quadrature is
Gauss. Worth stating because a published figure was legended `S6R10Q16tanh` at `R = 10`, which is
`Q20`.

### Time steps

**Every method runs at h ∈ {1, 2, 5}**, so the symbolic and the neural integrators are compared on
the same ladder. Two documented departures, both in `experiments.jl`: `VISE_EXTRA_STEPS` and
`NVI_EXTRA_STEPS` add a step where a published figure used one, and `NVI_STEP_OVERRIDES` puts the
double pendulum at `h = 0.5` alone — it *cannot* run at 1, 2 or 5. At those initial conditions the
double-pendulum LODE is singular for the large-step solves: `ImplicitMidpoint` and `Gauss(8)` both
fail with a `SingularException` at `h ≥ 0.5`, and the network completes two steps at `h = 2.0` and
then fails the same way.

### Three things worth knowing before editing a driver

**`regularization_factor` is load-bearing, and only for the network runs.** The first version of
`run_convergence.jl` called `integrate` without the solver options and 22 of the 32 ReLU³ runs died
with a `SingularException` — curves of one point each — while the same configurations had just run
to `T = 1000` in `run_nvi.jl`. Unregularised, the greedy seed's least-squares system is rank
deficient for ReLU³; tanh tolerated it, which is why the failure looked like a property of ReLU³
rather than a missing keyword. The other half of that: it must **not** reach `CGVI`. It is a keyword
of this package's network integrators, not of the solver, and a `GeometricIntegrators` method has
nowhere to put it — which is why `max_hamiltonian_error` takes the options as an argument instead of
reading a constant.

**The orthogonal-greedy seed decides whether the integrator works.** `run_oga_seeds.jl` answers
this and the answer is not subtle: on one configuration, one problem, six seeds, only two work —
and they agree to every digit, differing only in the fit. `OGA1dStable` does not fail; it stagnates
at `1e-1`, which is worse, because nothing says so. Newton restarts from the seed at every step and
the discrete equations are non-convex, so the seed selects which solution each step lands in with no
global solve to correct a bad pick. Changing `initial_guess_method` away from the default is
therefore not a tweak.

**The network solves do not converge.** `run_nvi.jl` silences the per-step solver warnings, and that
is not tidying up: the nonlinear solver stalls with a residual around `1e-4` at every step, and at
1000 steps the warnings bury the output. The stall is itself a finding — the neural variational
integrators stagnate near `1e-3` in the Hamiltonian error while the polynomial Galerkin integrators
converge at their nominal order. The maximum error reached is printed and archived, so the claim
stays measured.

## OGA seed variants

Three studies cover the Orthogonal Greedy Algorithm seeds of `ShallowNet`, across working
precisions (`Float16`, `Float32`, `Float64`), `ReLUᵏ` powers, smooth activations, and the
`regularization_factor` ladder. They share `oga_activations.jl` (float-generic activations, the λ
ladder) and `oga_report.jl` (figures and markdown).

The split into two tiers is the point of the design. End-to-end convergence conflates the quality
of the *seed* with the behaviour of the *solve*, and that confound is what made the
reduced-precision failures hard to attribute: a run that fails looks the same whether the greedy
fit went rank-deficient or the Newton Jacobian did.

### Tier A — seed quality, no integrator (`oga_fit_study.jl`)

Calls `oga_fit` directly: no integrator, no Newton solve, no time stepping. Sweeps
dictionary × selection × fit × activation × precision × target and reports

- `fit_err` — the quadrature-weighted L² error of the seed, recomputed in `Float64` from the
  returned parameters so that precisions share one scale;
- `cond` / `sigma_min` — the seed's design matrix, the proxy for whether the Newton system it feeds
  is solvable;
- `neurons` / `rejected` — how many of the requested neurons the greedy loop could place, and how
  many candidates it refused for adding no new direction.

Every case is an `S ≤ 8`, 11-node problem, so the whole grid runs in seconds.

### Tier B — end-to-end sweep (`oga_sweep.jl`)

The harmonic oscillator, ten steps, `S = 4`, `R = 8`, `dt = 0.1`, over seed variant × precision ×
regularization factor × activation, in two stages:

- `relu` — `ReLUᵏ` for `k = 1…4`, where the `{±1} × (bias grid)` dictionary is theoretically
  complete, so anything that goes wrong is numerical. This is the reduced-precision question.
- `smooth` — ELU, GELU and tanh against the 2-D and angular dictionaries built for them. This is
  the activation question.

λ is swept as multiples of `√eps(T)` — `2^k √eps(T)` for `k = 1…6` at `Float16`/`Float32` and
`k = 2, 4, …, 12` at `Float64` — plus a `λ = 0` control, so the Jacobian-diagonal shift is scaled to
the precision it protects. An absolute `1e-5` sits far below `√eps` at anything but `Float64` and
cannot lift a near-singular Jacobian in reduced precision at all.

**The residual tolerance is scaled the same way**, and it has to be. The solver's default `f_abstol`
is `1.78e-15`, an absolute value scaled to `Float64` and unreachable at `Float32` or `Float16`; a
reduced-precision run then sits at its residual floor and burns the whole iteration budget while
parked on the right answer. Measured before the fix, `ReLU³` at `Float32` reported 1000 iterations
at *every* regularization factor with an accuracy of `1.8e-7` — which, read as non-convergence,
would have made the whole `Float32` column an artefact of the tolerance. The sweeps pass
`f_abstol = 256·eps(T)` (`oga_f_abstol`).

Two other classification points worth knowing when reading the CSVs: a run that exhausts
`max_iterations` is recorded as `maxiter`, not `ok` (it returns a finite state, so the naive check
would call it converged), and a run whose final state has left the working precision is recorded as
`upcast`.

### Tier B′ — double pendulum (`oga_double_pendulum.jl`)

The problem the seed fails hardest on, at **one** λ rather than the whole ladder: the
harmonic-oscillator sweep already answers what λ does. The value is read from
`runs/oga_sweep_relu.csv` — the factor that converged most often there — so it is measured rather
than asserted; absent that file it falls back to the documented `16√eps(T)` and says so.

### Running

```sh
julia --project=scripts scripts/oga_fit_study.jl
julia --project=scripts scripts/oga_sweep.jl            # both stages
julia --project=scripts scripts/oga_sweep.jl relu       # just the ReLUᵏ stage
julia --project=scripts scripts/oga_double_pendulum.jl  # after oga_sweep.jl relu
```

Each writes a CSV into `runs/` and a markdown report and PNG figures into `results/`. The reports
regenerate from the CSVs alone — `write_fit_study_report` / `write_sweep_report` in `oga_report.jl`
— so a figure or a table can be reworked without re-running a sweep.

`oga_report.jl` keeps its own heatmap code rather than routing it through the plotting extension,
and that is deliberate. Its colour ramp is a single-hue sequential scale validated for
colour-vision deficiency, chosen against a red→green default, with every cell carrying its numeric
value and a contrast flip partway up the ramp. Those are explicit per-axis sizes and colours, and
the extension's standing invariant is that **nothing in it sets a font size, colour or line width of
its own** — every one comes from the ambient theme. Moving the heatmap in would either break that
invariant or discard the accessibility work.

### Background

The original formulation solved the fit through the normal equations, whose condition number is
`κ(Φ)²`; that is what forced the `Float64` island and what goes rank-deficient in reduced precision.
See the "Orthogonal Greedy Algorithm" section of the package documentation for the analysis, the
variant taxonomy, and why `±1` weights suit `ReLUᵏ` but under-serve smooth activations.

## Retained files

These are kept for what they record, not because they run. Say so before assuming any of them is a
current entry point.

| file | why it is here |
|:--|:--|
| `vise_plot.jl` | the `SRRegressor` symbolic-regression pipeline that **discovered** the VISE ansätze. It is the provenance of every ansatz in `experiments.jl`, and there is no other record. It cannot run in this environment: it needs `MLJ` and `SymbolicRegression`, deliberately not added, and it loads archives from an absolute path on another machine. |
| `test_vise.jl` | 893 lines of which most are commented, but the last sixty are a live **6-degree-of-freedom Toda lattice** VISE run with six discovered ansätze and their initial weight vectors — an experiment that is in no registry and nowhere else. |
| `test_CGVI_BSpline.jl` | a B-spline/CGVI combination against the retired `obsolete/BSpline/` basis |
| `test_stability_analysis.jl` | a symbolic stability analysis, 506 lines |
| `result_summary_table.jl` | post-processing of a large S/R/k/h/λ grid into tables |

Converting the two `Plots`-based files to CairoMakie is what would let `Plots` leave
`scripts/Project.toml`; it is standing work.
