# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **The experiment suite lives in `scripts/`.** The registry, five drivers that solve and archive,
  and one renderer — `experiments.jl`, `archives.jl`, `basis_fits.jl`, `run_vise.jl`, `run_nvi.jl`,
  `run_fourier.jl`, `run_convergence.jl`, `run_oga_seeds.jl`, `figures.jl`. It was written against
  this package but kept in a talk directory, where nothing else could reach it and where its
  `Project.toml` pointed back here through a `[sources]` path. **It arrived without adding a single
  dependency**; `scripts/README.md` is the description.

- **`Diagnostics.figures(data)`** — the composition layer above `plot_solution` and
  `plot_convergence`. Given one run's archive as a plain dictionary, it returns every figure that
  run earns as `stem => Figure` pairs, dispatching on the archive's `"kind"`. That is what lets a
  renderer be a loop over a directory rather than a second registry of which experiment produces
  which picture, kept in step with the first by hand. It returns figures and does not write them.

- **Figure naming is part of the package**: `figure_stem`, `window_stem`, `study_stem`,
  `galerkin_label`, `network_label` in `src/plots.jl`. Callers who archive a run and then plot it
  need the extension and the script to agree on its name, and these were previously two definitions
  that could drift — `network_label`'s format alone existed in four inline copies. `Q = 2R` always,
  which is asserted, because a published figure was once legended `S6R10Q16tanh` at `R = 10`.

- **`coarse_grid_error(sol, ref_sol, substeps)`** — the relative maximum error against a reference
  on a finer grid, compared at the macro steps the two share. Use it instead of
  `GeometricSolutions.relative_maximum_error` for an oscillator: that one normalises **per step**,
  and the divisor vanishes at every zero crossing, so a bounded absolute error is reported as an
  arbitrarily large relative one. Measured on the harmonic oscillator at `h = 1` over `t ∈ [0, 200]`,
  implicit midpoint came out at `1.05e+02` — a phase error sampled next to a zero crossing, not a
  diverged solution.

- **`relative_invariant_error(sol, invariant, parameters)`** — the three-argument form, gathering the
  series from a solution before reducing it. The composition four scripts wrote out by hand.

- **`scripts/compare_runs.jl`** — compares two directories of archives numerically and exits
  non-zero on a difference above solver tolerance. The check for any change that is supposed to be a
  refactoring. It reports keys present on only one side separately, because a run that silently
  stopped writing a series is the one failure an archive comparison exists to catch and the one it
  cannot express as a number.

- **An archive directory stays readable across revisions of the writer.** `figures.jl` renders a
  directory it did not write, and that directory accumulates: `archive_kind` infers the shape of
  figure from which series an archive carries when it has no `"kind"`, `load_runs` fills in a
  `"stem"` from the filename, and `normalise_schema!` maps the older `figure_window` onto
  `windows`. Without this, redrawing a figure from an archive that already holds every number the
  figure needs would have meant re-running the solves — which is the exact cost the split between
  `runs/` and `results/` exists to avoid. An archive that is genuinely unreadable is named and
  skipped rather than aborting the run: one stale file must not cost the other sixty-one figures.

### Changed

- **Output goes to `runs/` (data) and `results/` (figures), at the repository root**, and every
  driver takes `--runs-dir` and `--results-dir`. Previously each script derived its output path from
  `@__DIR__` as a `const`, which is what forced a caller who wanted the figures elsewhere to copy
  the script. `scripts/results/` is gone; the OGA sweeps' CSVs are data and now sit in `runs/` with
  their reports and figures in `results/`. This is the tree-wide convention in `Packages/CLAUDE.md`.

- **One argument parser for every driver, and it rejects what it does not know.** The six parsers it
  replaces disagreed about an unrecognised argument: three threw a bare `MethodError` from indexing
  a `findfirst` that had returned `nothing`, `run_convergence` pushed the flag onto its list of
  problem names and then, matching nothing, fell back to running the entire twenty-minute sweep, and
  `figures` rendered nothing, printed `done` and exited 0. A mistyped flag now names the valid
  arguments.

- `scripts/vise_study.jl` is folded into `run_vise.jl`, which archives the runs *and* prints the
  summary table. `Infiltrator`, `Distributed` and `Test` leave `scripts/Project.toml` with the files
  that used them.

### Removed

- **Five network drivers that had not run in a long time**: `run_shallownet.jl`,
  `test_shallownet_autodiff.jl`, `test_shallownet_reversible.jl`,
  `test_shallownet_autodiff_reversible.jl`, `test_densenet.jl`. Not "superseded" — *broken*, and
  verified so before deletion: `run_shallownet.jl` used a `k_relu` whose loop was commented out, so
  every one of its nine iterations threw `UndefVarError` into a bare `catch`;
  `test_shallownet_autodiff.jl` commented out `R_list`, `S_list` and `k_list` and then used all
  three at top level; two more wrote into directories that do not exist.

  **Nothing replaces them, because nothing needs to.** The four shallow-net variants are compared
  by `benchmark/compare_derivative_backends.jl`, which sweeps exactly those four; `DenseNet` at
  three time steps is `run_nvi.jl`, whose archives were checked against the old ones and are
  identical; and the `S`/`R`/`k`/λ sweep on the harmonic oscillator is `oga_sweep.jl`. Writing a
  sixth driver would have been a third tool for a job already done twice.

- `scripts/find_optimal_results.jl`, which scanned a `parallel_results/` directory that does not
  exist; `scripts/runtests.jl`, an empty stub distinct from the real `test/runtests.jl`; and
  `scripts/parallel_run.sh`, a mostly-commented driver for the scripts above.

### Notes

**`scripts/test_vise.jl` and `scripts/vise_plot.jl` were *not* deleted**, and the 0.5.0 entry below
saying that what they held "is in the new one" is **wrong**. Both were re-read before the deletion
they were queued for:

- `vise_plot.jl` carries the `SRRegressor` symbolic-regression pipeline that **discovered** the VISE
  ansätze. It is the provenance of every ansatz in `experiments.jl` and there is no other record. It
  cannot run — it needs `MLJ` and `SymbolicRegression`, which are deliberately not in
  `scripts/Project.toml`, and it loads archives from an absolute path on another machine — but that
  makes it a document, not a dead file.
- `test_vise.jl` is mostly commented, but its last sixty lines are a live **six-degree-of-freedom
  Toda lattice** VISE run with six discovered ansätze and their initial weight vectors, an
  experiment that is in no registry.

Both are listed under "Retained files" in `scripts/README.md` with what each records.

### Fixed

- **The error panel of `plot_solution` is a curve again for a series at round-off.** 0.5.0 masked
  every value a logarithmic axis cannot take with `NaN` and passed the full time vector; `NaN`
  breaks the polyline through it, so each masked point cut the line in two. That is invisible while
  only `t₀` is zero — which is what the figures of that release had — and wrong as soon as the
  invariant is conserved to machine precision, because `H(tₙ) - H₀` is then exactly `0` at a good
  fraction of the samples. Measured on the global Fourier fit of the perturbed pendulum: 22 of 101
  samples exactly zero, and an error panel of disconnected fragments and five isolated dots.

  `log_safe` is replaced by `log_points`, which drops the unplottable point from the **time** vector
  as well and so bridges it. Nothing shifts left — the property masking was chosen for — because
  every surviving point keeps its own time. A series with no plottable point at all is now left out
  entirely rather than drawn empty, as `plot_convergence` already left out a configuration that
  failed at every step.

  Every archived series here is finite, so this only ever bridges an exact zero. A run whose
  invariant blows up mid-way would now be bridged rather than broken; there is no such archive to
  say what that should look like, and inventing the distinction before one exists would be guessing.

## [0.5.0] - 2026-08-31

The package can plot its own integrators. It had no `ext/` directory and no `[weakdeps]` at all
until now, so every figure ever made from a `VISE` or `ShallowNet` run was made by a script
carrying its own Makie code — and there are a lot of those scripts, all carrying much the same
code.

### Added

- **`ext/NonlinearIntegratorsPlots.jl`, this package's first extension**, on a `Makie` weakdep.
  **Two** plot functions and a theme, in a `NonlinearIntegrators.Diagnostics` submodule:

  | | |
  |:--|:--|
  | `plot_solution` | *several* integrators of one problem in one figure — `q(t)`, `p(t)` and the relative Hamiltonian error, with the **continuous** solution between the discrete steps drawn through them |
  | `plot_convergence` | *several* error series against the time step, with a reference slope per order |
  | `plot_theme()` | the shared Makie theme, for the caller to `set_theme!` |

  Two and not more, because two is what `GeometricProblems` is missing. A phase portrait, a
  trajectory or a set of traces are the per-problem recipes there; one method's error against its
  expected order is that package's own `plot_convergence`. Reimplementing either here would be a
  second thing to keep right. What is left is the pair those cannot express — no `GeometricProblems`
  recipe takes more than one solution, none knows about `integrate`'s second return value, and its
  `plot_convergence` draws one series against one slope, which cannot show a neural family against a
  polynomial one.

  **The one exception, found by the test written to assert the opposite.**
  `GeometricProblems.Diagnostics.plot_energy_error` and `plot_invariant_error` cannot compute the
  energy of a *partitioned or implicit* solution, which is every solution this package produces, so
  the invariant error is **not** reused — `plot_solution` computes it from `q` and `p` itself, and
  `relative_invariant_error` is exported for callers who want the number.

  `_invariant_error` branches on `sol isa Union{SolutionPODE, SolutionPDAE}` to decide whether to
  pass `p`, and that test is `false` for a `GeometricSolution` of a `LODEProblem` even though
  `SolutionPODE`'s definition names `LODEProblem`: the alias binds `probType` both as a parameter
  and in its `where` clause, so the constraint does not apply as it reads. Verified —
  `sol isa SolutionPODE` is `false` while `typeof(sol.problem) <: LODEProblem` is `true`. The
  `q`-only branch is always taken and a Hamiltonian expecting `(t, q, p, params)` is called with
  three arguments; on the harmonic oscillator that reaches the three-argument method, which expects
  `q = [q, v]`, and raises `BoundsError`. A problem whose three-argument Hamiltonian happened to
  accept a one-dof `q` would return a **silently wrong** energy instead.

  Measured on GeometricProblems 0.8.3 / GeometricSolutions 0.6.5, and held by a `@test_broken` plus
  an assertion pinning the cause, so a fix upstream flips the test rather than passing unnoticed.

  A second, independent trap in the same place, which would still bite after that is fixed: a
  `lodeproblem` built by `EulerLagrange` carries `NullInvariants`, so there is no `:h` key and the
  Hamiltonian has to be passed as `energy = <function>`.

  Each function returns a Makie `Figure` and none saves one, as everywhere else in this ecosystem.
  `Makie` and not `CairoMakie` is the weakdep, so the backend stays the caller's choice. The
  extension sets **no** font size, colour or line width of its own: sizes come from the ambient
  theme, series colours from `Makie.wong_colors()`.

  **A submodule rather than more exports**, which is the one place this departs from the shape of
  `GeometricProblems`' per-problem plot functions. `plot_solution` is a name *every*
  `GeometricProblems` problem submodule exports and `plot_convergence` one that its `Diagnostics`
  does, so exporting them from this package's top level makes both ambiguous in any scope that also
  wrote `using GeometricProblems.HarmonicOscillator` — which is what a script integrating a problem
  naturally writes, and what this package's own `test/testsetup.jl` does. So they sit behind
  `NonlinearIntegrators.Diagnostics`, exactly as `GeometricProblems.Diagnostics` does, and the top
  level stays clear.

- **`plot_theme()`**, the shared Makie theme of this ecosystem — larger fonts and thicker lines than
  the Makie defaults. Kept identical to the copy in `GeometricExamples/src/common.jl` and the
  publication companion packages, and asserted field by field in the tests, because a theme that
  has quietly drifted is the kind of difference nobody notices until two figures sit side by side.
  A function and not a `const`, because a `Theme` is a Makie type and `src/plots.jl` is loaded
  whether or not Makie is.

- **`continuous_solution(internal_values, timestep; dof, t₀)`**, exported. The continuous solution
  *between* the discrete steps, as a `(t, q)` pair.

  This is the post-processing every integrator here needs and none provided, and it had been
  written out by hand at least six times — three copies in one talk's figure scripts, two in
  `scripts/`, one in the docs — each as `vcat(hcat(internal_sol...)[2:end, :]...)` against a
  hand-written `h/40:h/40:TT`. Both halves of that are traps. Row 1 of every step is the step's
  *left* endpoint, so concatenating the records whole duplicates each interior step boundary; and
  the `40` is `record_grid_points - 1`, a method keyword, so every one of those copies was silently
  wrong for any method not built with the default 41.

- **`Trajectory`**, exported, and `relative_invariant_error`. What a figure needs about one run and
  nothing else, as plain vectors — so a result archived for later plotting does not pin the version
  of `GeometricSolutions` that wrote it into the archive, and so a comparison curve that never was
  a solution (a closed-form expression, a digitised reference) is expressible without pretending to
  be one. `Trajectory(label, sol; internal_values, hamiltonian, parameters)` builds one from a
  solution; the plot functions accept solutions and `"label" => solution` pairs wherever they
  accept a `Trajectory`, and convert.

- `test/plots_tests.jl`, with `CairoMakie` in `test/Project.toml`. The `plot_*` tests are smoke
  tests, as in `GeometricProblems`; `continuous_solution` gets real assertions instead, including
  against `record_grid_points = 21` — the case a re-hardcoded 41 fails.

### Changed

- `scripts/test_vise.jl` and `scripts/vise_plot.jl` → **`scripts/vise_study.jl`**. The first was
  893 lines of which 854 were commented out; the second 1257 lines that loaded `.jld2` archives
  from an absolute path on another machine, and mixed CairoMakie, Plots, MLJ and SymbolicRegression
  to draw them. What they held that was worth keeping — the ansätze, the initial weight vectors,
  the solver settings — is in the new one, which runs.

  **This last sentence is not true, and both files are still here.** Re-read before the deletion it
  implied, `vise_plot.jl` turned out to carry the symbolic-regression pipeline that *discovered* the
  ansätze, and `test_vise.jl` a live six-degree-of-freedom Toda lattice run that is in no registry.
  See the `[Unreleased]` notes above. `vise_study.jl` itself has since been folded into
  `run_vise.jl`.
- `docs/src/vise/vise.md`: the Plots.jl example, and its link to a `../figures/HHh2ref.svg` that
  does not exist, replaced by the extension.

### Removed

- `benchmark/Manifest.toml`. It dev-pathed this package to
  `/Users/mkraus/Datashare/Julia/NonlinearIntegrators`, which does not exist, and pinned version
  0.4.0 against a repository at 0.4.3, so `Pkg.instantiate` there could only fail.
  `benchmark/Project.toml` beside it is correct (`[sources] path = ".."`), so deleting the manifest
  is the whole fix.

### Layout

Every figure is **2:1** — `FIGURE_WIDTH / FIGURE_ASPECT`, 1200×600 — whatever its panel count, so
figures with different panel counts sit on a slide the same way and can be included at one width.
For one degree of freedom the panels are **stacked** in a single full-width column sharing one time
axis; for `D > 1` they are a `D`×2 grid of equal-width columns with the error panel spanning
underneath.

`plot_solution` also takes `trace_timespan`, which restricts the `q` and `p` panels while leaving
the error panel on the whole run. A run of a hundred oscillations is a solid block of ink at any
figure width, and what the traces are there to show — that the ansatz tracks the reference — is
settled in the first few periods, while the error panel is the one that needs the full range.

### Notes

Several things found while writing this, recorded because they are the kind of thing that gets
rediscovered:

- **`Legend` and `Label` report their width to the layout.** In a single-column figure the column is
  then sized to whichever of them is widest, and the axes shrink to match: measured on a
  three-panel figure, a 50-character title held the axes to **386 pt of a 900 pt page** — 43%,
  centred, the rest white. `tellwidth = false` on both is what makes the panels full width. Worth
  knowing because the symptom looks like a figure-size problem and is not.
- **A dense comparison must be a line, not markers.** `plot_solution` draws a comparison as scatter
  markers, which is right for one computed at the same step as the primary — and wrong for one on a
  much finer grid: a midpoint solve at `h/20` over 40 time units is 1600 markers, which buries the
  panel. The rule is "more than twice as many points as the primary has steps", which separates the
  two cases without a keyword nobody would remember to set.

- **A logarithmic error axis needs its zeros masked, not dropped.** Every invariant-error series
  starts with an exact `0` — `(H(t₀) - H₀)/H₀` is zero by construction — and one `log10(0)` drags
  the axis limits to `-Inf`, whereupon Makie falls back to a default decade range and the panel
  renders **empty**, with nothing to say why. The first render of the figures this was written for
  did exactly that: three series with real values near `1e-8`, on an axis running `1e0` to `1e3`.
  `log_safe` masks to `NaN`, which Makie skips, and masking rather than deleting keeps each series
  aligned with its own time vector.
- `scripts/oga_report.jl` and `benchmark/shallownet_report.jl` still carry their own palettes and
  their own font sizes, and neither uses `plot_theme()`. They plot a different subject — labelled
  heatmaps and benchmark sweeps, where the palette is doing work the theme cannot — so converging
  them is real work rather than a side effect of this release, and **it has not been done**.
- `plot_theme()` is a fifth hand-kept copy of that theme (`GeometricExamples` and three publication
  companion packages hold the others). Somewhere to put it once would be better; there is no
  package in the dependency graph of all five that it could go in.

## [0.4.3] - 2026-08-30

The de-piracy wave, taken as a set of compat bounds. Nothing in this package's exported surface
changes, and neither does any of its code.

### Changed

- **`NeuralNetworkParameters` 0.2.2 → 0.3, `AbstractNeuralNetworks` 0.7.1 → 0.8,
  `SymbolicNeuralNetworks` 0.7 → 0.8, `GeometricOptimizers` 0.6 → 0.7, `SimpleSolvers`
  "0.12.1, 0.13" → 0.13.2, `GeometricBase` 0.14.8 → 0.14.9.** One constraint again, not six.

  `NeuralNetworkParameters` 0.3.0 removes `ParameterSet`, the `Union{NetworkParameters, NamedTuple}`
  the rest of this ecosystem dispatched on, because a method written on it was a method on
  `Base.NamedTuple`. Every other bound follows: 0.8.0, 0.8.0 and 0.7.0 are the first releases of the
  three packages above that permit the 0.3 container, `GeometricOptimizers` 0.7.0 requires
  `SimpleSolvers` 0.13.2, and 0.3.0 takes `GeometricBase` as a hard dependency at 0.14.9. Any one of
  them alone is unsatisfiable.

  **None of it reaches code here.** This package names only `NetworkParameters`, `params`, `flatten`,
  `unflatten` and `unflatten!` from the container package, and none of those changed; it names neither
  `ParameterSet` nor `EquationSet` nor `ArrayNamedTuple`. `GeometricOptimizers` 0.7.0 turning a bare
  `NamedTuple` away at `Optimizer` costs nothing for the reason 0.4.2 already gave: the `ShallowNet`
  initialisation flattens to a plain `Vector` before `Optimizer` sees it (`src/nvi/shallownet.jl`), so
  no parameter set is ever handed over whole.

- **The `residual!` budgets are re-measured, and two of the four rows moved — downwards.**

  | | 0.4.2 | 0.4.3 |
  |---|---|---|
  | `ShallowNet` | 11 424 | 11 424 |
  | `ShallowNetReversible` | 11 424 | 11 424 |
  | `ShallowNetAutodiff` | 54 656 | **52 096** |
  | `ShallowNetAutodiffReversible` | 54 656 | **52 096** |

  Worth stating rather than quietly re-recording, and worth noting which way: 0.4.2 moved these same
  two rows *up* by 3 072 bytes and left the symbolic pair alone. This release moves them back down by
  2 560 and again leaves the symbolic pair alone — so the autodiff path, which reaches its derivatives
  through `ForwardDiff` rather than a generated kernel, is the one that feels these releases, and the
  generated one is not. The ceilings (17 000 and 78 000) hold with room, so none of them moves.

## [0.4.2] - 2026-08-26

Julia 1.11 becomes the floor and `GeometricOptimizers` moves to 0.6. Nothing in this package's
exported surface changes.

> [!NOTE]
> This entry was written after the fact. 0.4.2 was released without one; its content is taken from
> the commit that made the change and from the figures recorded there.

### Changed

- **`GeometricOptimizers` 0.5 → 0.6.** 0.6.0 takes a whole `NetworkParameters` through the optimizer
  and is breaking: it deletes four `outer!`/`_mul!` methods, one `update!(::BFGSState, …)`, `add!` on
  a parameter set, and two `l2norm` methods that moved upstream to `GeometricBase`. Nothing here hands
  it a container, so the bump is a compat entry and no code.

- **`NeuralNetworkParameters` 0.2.1 → 0.2.2**, which is what `SymbolicNeuralNetworks` 0.7.1 requires
  anyway.

- **Julia 1.11 is the floor**, in step with `NeuralNetworkParameters`, `SimpleSolvers`,
  `GeometricOptimizers`, `AbstractNeuralNetworks`, `SymbolicNeuralNetworks` and
  `GeometricMachineLearning`. `GeometricBase` deliberately stays on 1.10.

- **The `residual!` budgets were re-measured on 1.11.9 rather than carried over**, and two of the four
  rows moved: `ShallowNetAutodiff` and `ShallowNetAutodiffReversible` from 51 584 to 54 656, while
  `ShallowNet` and `ShallowNetReversible` stayed at 11 424. The prediction had been the other way
  round — `NeuralNetworkParameters` 0.2.2 took `SymbolicNeuralNetworks`' single-sample split from 768
  bytes to 560, and the symbolic `residual!` here calls `DQDθ` on a length-one `Vector`, so that is
  the path it takes — and it moved this figure by nothing.

## [0.4.1] - 2026-08-25

Three compat bounds move together, and the two things the releases behind them bring make code
here shorter. Nothing in this package's exported surface changes.

### Changed

- **`GeometricOptimizers` 0.4 → 0.5, `NeuralNetworkParameters` 0.1 → 0.2.1,
  `SymbolicNeuralNetworks` 0.6 → 0.7.** One constraint, not three. `GeometricOptimizers` 0.5.0
  drops the `ParameterHandling` shim and lets `NeuralNetworkParameters` do the flattening, which
  makes the 0.2 container a requirement rather than a choice — 0.4 and earlier want the 0.1
  container — and `SymbolicNeuralNetworks` 0.7.0 is the first release that permits 0.2, where
  0.6.0 allows only 0.1. Any one of the three alone is unsatisfiable.

  Nothing here called what `GeometricOptimizers` 0.5.0 *removed*: `apply_toNT`, the pirated
  `ParameterHandling` methods on Base types, or `NeuralNetworkParametersExt`, whose contents are
  `src/parameter_protocol.jl` there now.

- **`AbstractNeuralNetworks` is bounded at `0.7.1`, not `0.7`.** 0.7.0 pins
  `NeuralNetworkParameters = "0.1"`, so it can never coexist with the 0.2 container this package
  now requires: the wider bound admitted a version that is not satisfiable here, which the
  resolver works out on its own but which makes the bound a false statement about what this
  package supports. `SymbolicNeuralNetworks` 0.7 names 0.7.1 for the same reason.

- **`NeuralNetworkParameters` is bounded at `0.2.1`, not `0.2`.** Not a bound about what loads —
  every name used here is in 0.2.0 — but about what `test/quality/inference_and_allocations.jl`
  measures; see the entry below.

- **The training loops hand `GeometricOptimizers` a flat `Vector`, and
  `optimizer_params`/`network_params` are gone.** Both were `@generated` helpers that flattened
  the layer nesting into `L1_W`-style keys and back, because `Optimizer` takes a *flat*
  `NamedTuple` where network parameters are one level deeper. That shape was never what the
  optimizer works in: handed a `NamedTuple`, `GeometricOptimizers` flattens it itself inside the
  `Gradient` it builds, so the flat `NamedTuple` bought a second, nested unflattening on every
  function and gradient evaluation of every epoch.

  `NeuralNetworkParameters` 0.2's `flatten`/`unflatten`/`unflatten!` replace both. What makes
  that possible now and not before is that a `ParameterLayout` is a *value*: the
  `ParameterHandling.flatten` this package could have called instead returned a chain of nested
  closures, one per level of the tree, which is not type stable — and that instability is exactly
  why the two helpers were `@generated` in the first place. `unflatten` keeps the element type of
  the vector it is given, which is what lets the same closure be called on `ForwardDiff.Dual`s.

  The three call sites — `ShallowNet`'s `TrainingMethod` seed and `DenseNet`'s `TrainingMethod`
  and `LSGD` seeds — lose the aliasing the old flat view gave them, so each gained an explicit
  `unflatten!` where the network's arrays are read back: once after `solve!` for `ShallowNet`, and
  once per epoch for the two hand-rolled `DenseNet` loops, which read the loss off `PNN.params`
  inside the loop. `test/unit/optimizer_params_unit.jl` becomes
  `test/unit/parameter_flattening_unit.jl` and pins the contract this package depends on —
  including that `unflatten` preserves a `Dual` element type, without which every gradient in
  those loops would come out zero.

  `flatten_params!` and `_param_arrays` stay. `NeuralNetworkParameters.flatten!` does the same
  walk, but is allocation-free only when handed a stored `ParameterLayout`, and the four call
  sites in `DenseNet`'s `components!` flatten a freshly built gradient set with nowhere on the
  cache to keep a layout for it. A `@generated` walk needs none.

  Not available yet, and checked: handing a `NetworkParameters` straight to `Optimizer`.
  `GeometricOptimizers` 0.5 makes `NetworkParameters{T}` a member of `OptimizerSolution{T}`, but
  its changelog's *Known issues* records that the swap itself is not in the release — every
  elementwise primitive in its `named_tuple_wrapper.jl` still dispatches on `ArrayNamedTuple`,
  and `default_gradient` has no container method — so a container "gets further in before failing
  rather than being turned away at the door".

- **`benchmark/Project.toml`'s `[sources]` entry takes a relative path.** It carried an absolute
  one, `/Users/mkraus/Datashare/Julia/NonlinearIntegrators`, since [0.2.0], so the benchmark
  environment resolved on exactly one machine. `docs/Project.toml` has always used `".."`.

### Fixed

- **The allocation gate has one ceiling per row again, on every Julia version.** The 1.10-only
  ceiling of 42 000 that [0.4.0] shipped for `ShallowNet` and `ShallowNetReversible` is removed:
  `SymbolicNeuralNetworks` 0.7.0 and `NeuralNetworkParameters` 0.2.1 close
  [SymbolicNeuralNetworks#55](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/55),
  which this repository reported from
  [#86](https://github.com/JuliaGNI/NonlinearIntegrators.jl/pull/86), and the two symbolic rows
  are back under the tight 17 000 on 1.10 — 15 168 bytes per `residual!` call, measured on
  1.10.11 against 28 096 for the same probe under SNN 0.6.0 / NNP 0.1.1, with nothing but those
  two versions changed. The cause was a `map` over a closure that 1.10 does not always elide, on
  the walk that splits a generated function's flat result back into the nesting of the
  parameters; the fix has two independent halves, and both bounds above are what make them
  reachable. See *Open Issues* → *Upstream*, where the entry is now closed.

## [0.4.0] - 2026-08-24

The parameter container moved out from under this package. `AbstractNeuralNetworks` 0.7 gave the
type its own package, and `SymbolicNeuralNetworks` 0.6 followed; this release follows both. Nothing
in this package's own exported surface changes — no integrator, method, basis or constructor is
renamed, and the numerics are untouched.

### Breaking

- **`NeuralNetworkParameters` is `NeuralNetworkParameters.NetworkParameters`.**
  [`AbstractNeuralNetworks` 0.7](https://github.com/JuliaGNI/AbstractNeuralNetworks.jl/pull/33)
  moved the parameter container — the struct, the leaf protocol, the tree walks, the flat form, the
  HDF5 path and the Zygote rule — out to
  [`NeuralNetworkParameters`](https://github.com/JuliaGNI/NeuralNetworkParameters.jl), where the
  type is called `NetworkParameters`, and removed the old name outright rather than leaving an
  alias, so that one type has one name across the ecosystem. It is the same type object, so code
  that only *uses* a parameter set is unaffected; code that *names* the type has to be edited:

  ```julia
  using AbstractNeuralNetworks: NeuralNetworkParameters   # before
  using NeuralNetworkParameters: NetworkParameters        # after
  ```

  Every call site here names it where it now lives — three method signatures and the wrapper
  construction in `src/nvi/utilities.jl`, the per-dimension parameter set in `shallownet.jl`,
  `shallownet_reversible.jl` and `densenet.jl`, and the `L3` merge in `DenseNet`'s LSGD loss. The
  import is selective rather than a bare `using`, because `NeuralNetworkParameters` also exports
  `flatten`/`unflatten` and the flat-vector conversions in `nvi/utilities.jl` —
  `optimizer_params` and `network_params` — are this package's own, over a layout one level
  shallower than a network's. The `params` accessor is taken from the same module, qualified;
  `AbstractNeuralNetworks` re-exports it, so both spellings resolve to one binding, but the type
  and its accessor now read as coming from the same place.

  `NeuralNetworkParameters` enters the tree as a dependency of both `AbstractNeuralNetworks` 0.7
  and `SymbolicNeuralNetworks` 0.6. It is listed here directly, with a `[compat]` entry, because
  this package names the type itself rather than only passing parameter sets through. The test
  environment lists it too, since the unit tests construct parameter sets by hand.

- **`SymbolicNeuralNetworks` moves to 0.6** (was 0.5) and **`AbstractNeuralNetworks` to 0.7** (was
  0.6.4). These are one constraint, not two: SNN 0.6 is the release that tracks ANN 0.7, and SNN
  0.5 pins ANN `0.6.4 - 0.6`, so the pair has to move together or the environment does not resolve.
  Nothing else SNN 0.6 changes reaches here — it renames nothing in its exported surface, and the
  one name it drops, `QPTOAT`, was never used in this package. Its rewrite of `SymbolicPullback`
  into a per-layer composition (an `O(width^depth)` expression becomes a sum over layers) is a
  path this package does not take, since both bases go through `symbolic_parameter_gradient`.

  Unlike the 0.6.4 floor this replaces — recorded under *Open Issues* → *Reviewing the 0.5 update*
  as a bound stating something other than this package's own requirement — `0.7` is this package's
  requirement: `NetworkParameters` is named directly in `src/`.

- **`benchmark/` takes `NeuralNetworkParameters` in place of `AbstractNeuralNetworks`.**
  `compare_derivative_backends.jl` was the only file under `benchmark/` that used
  `AbstractNeuralNetworks` at all, and only for the parameter container, so the dependency is
  replaced rather than added to. The docs CI job instantiates this environment.

### Known issues

- **On Julia 1.10, `residual!` allocates 1.85x what it did** for `ShallowNet` and
  `ShallowNetReversible` — 28 096 bytes per call against 15 168 before. Julia 1.11 and later are
  unaffected, at an unchanged 11 424. The cause is upstream of this package; it is recorded in full
  under *Open Issues* → *Upstream*, and the allocation gate carries a 1.10-only ceiling so that a
  further regression there is still caught.

## [0.3.0] - 2026-08-17

### Breaking

- **`CGVINodal` is removed. It now lives in GeometricIntegrators**, next to `CGVI`, which is
  where a linear variational integrator belongs — this package is for the nonlinear ones, and
  `CGVINodal` was only ever here as the linear reference the network integrators are compared
  against. It arrived there in
  [JuliaGNI/GeometricIntegrators.jl#219](https://github.com/JuliaGNI/GeometricIntegrators.jl/pull/219),
  which also removes the duplication between the two CGVI variants by putting everything they
  share behind one `CGVIMethod` supertype.

  Replace `using NonlinearIntegrators: CGVINodal` with `using GeometricIntegrators: CGVINodal`,
  and require **GeometricIntegrators v0.18.3** or later — that is the release the name resolves
  in. The constructor, the coefficients and the numerics are unchanged, so results are
  identical; upstream additionally rejects a basis whose nodes do not include the interval
  endpoints, which was previously only documented as a requirement.

  Everything `test/unit/cgvi_unit.jl` asserted moved with it — the per-precision accuracy runs,
  the `D = 2` `CoupledHarmonicOscillator` layout regression and the trait sweep are now in
  `test/integrators/galerkin_integrators_tests.jl` upstream, run against both `CGVI` and
  `CGVINodal`, so the coverage is wider there than it was here. This package gains no
  dependency on GeometricIntegrators.

- **`show_status` now defaults to `false`** on all five network integrators. It defaulted to
  `true`, and in `ShallowNetReversible` and `ShallowNetAutodiffReversible` it gated a `println`
  of the full residual vector *inside* `residual!` — which runs once per Newton iteration and
  once per ForwardDiff Jacobian column, with `b` a vector of `Dual`s. The default configuration
  therefore printed O(iterations × unknowns) residual vectors per time step. The two
  non-reversible siblings had already had those lines removed; the reversible copies were
  missed. All such output is now `@debug`.

- **`Zygote` is no longer a dependency.** Its single call site is described under *Fixed* below.

- **`mse_loss` is renamed `mae_loss`.** The name said squared error, the docstring said absolute
  error, and the body computed absolute error. The body is authoritative, so renaming leaves the
  numerics of every `TrainingMethod` seed exactly as they were; switching to a squared error
  would have changed them silently. A dead `μ` keyword was dropped and `λ` made type-generic (it
  defaulted to a `Float64` literal, which promoted a `Float32`/`Float16` loss to `Float64`).

- **`box_init_plain` requires an element type and takes an `rng`.** See *Fixed*.

### Fixed

- **`VISE.integrate!` corrupted its output on any restart.** It sized `internal_values` and
  `each_step_solution` as `n₂-n₁+1` but indexed them by `n`, so every `n₁ > 1` left the first
  `n₁-1` slots `#undef` and ran off the end. This is the same bug that had already been found
  and fixed in the network integrators' `integrate!`, comment and all; the VISE copy of the loop
  was never updated. Both containers also had abstract element types (`Vector{Matrix}`,
  `Vector{Vector}`), now concrete.

- **The Documenter build failed on two unresolvable cross-references.** The new `VISEBasis`
  docstring links `[`VISE`](@ref)` and the new `VNN_ansatz` one links `[`NN_ansatz`](@ref)`,
  and neither target had a docstring for `@autodocs` to pick up — so `makedocs` terminated with
  `encountered an error [:cross_references]` before rendering. Both now have one: `VISE` gains
  the constructor documentation it never had (including the three-element `integrate` return,
  which no page stated), and `NN_ansatz` states the ansatz and why it needs no boundary-point
  parameter gradients.

- **`VISE.record_finer_solution!` replaced its cache buffer instead of filling it.**
  `tem_W[d] = x[start_idx:start_idx+W_size-1]` rebinds the slot to a freshly allocated copy of
  the slice, so the cache ended each step holding different arrays than the ones it was
  constructed with, at one allocation per degree of freedom per step. `copyto!` into the
  existing buffer, as everywhere else.

- **`box_init_plain` reseeded Julia's global RNG as a side effect of a default argument.** The
  keyword defaulted to the *expression* `Random.seed!(1)`, which is evaluated on every call that
  omits it: `DenseNet`'s LSGD seed calls it three times in a row, so it reseeded the global RNG
  three times and drew all three layers from the same freshly-seeded stream — correlated, not
  independent — discarding the `Random.seed!(42)` the caller had just done. It also defaulted to
  `Float32` while every call site omitted the type, so a `Float64` DenseNet drew its weights at
  single precision and converted on assignment, in contradiction of the package's "no silent
  upcast" invariant and invisible to a test that checks the `eltype` of the result.

- **`VISE` and `CGVINodal` trait functions did not reach the framework.** `isexplicit`,
  `isimplicit`, `issymmetric` and `issymplectic` were defined *unqualified*. None of the four is
  imported into this module, so each definition created a new, shadowing
  `NonlinearIntegrators.isexplicit` rather than extending `GeometricIntegratorsBase`'s generic —
  and the framework kept answering `missing` for all of them. It mattered most for
  `CGVINodal`, where `issymplectic = true` is a real claim about the continuous-Galerkin
  construction and was the one property downstream code could have selected the integrator on.
  The network integrators had always qualified theirs.

- **`VISEBasis` no longer builds its compiled functions with `eval`.**
  `Symbolics.eval(Symbolics.build_function(…))` added methods in a newer world age than the
  caller's, so constructing a basis and evaluating it within one function body raised
  `MethodError: … The applicable method may be too new`. It only appeared to work because the
  sole test built the basis at top level, where world age advances between statements.
  `build_function(…; expression = Val(false))` returns a `RuntimeGeneratedFunction` instead —
  no `eval`, no world-age barrier, and a concrete type.

- The OGA *Precision* page said `[compat]` pins GeometricIntegratorsBase 0.5; it has required
  0.6.3 since the SimpleSolvers 0.12 upgrade. The argument for the bound — an `f_abstol` that
  scales with `datatype(problem)`, and defaults that merge with the caller's options rather than
  being substituted away — is unchanged.

- **The documentation deploys again on a tag.** `deploydocs` was called with `devurl = "stable"`,
  publishing `main` at `/stable/`. Its default `versions` is `["stable" => "v^", "v#.#", devurl =>
  devurl]`, so that put two entries under the one name — the symlink to the newest release, and the
  devurl directory — and the `v0.2.0` tag build died with *``link `"stable" => "v0.2.0"` cannot
  overwrite `devurl = stable` with the same name``* after building and cloning `gh-pages`, before
  pushing. Nothing was published.

  The misconfiguration is as old as the `deploydocs` call and could not have shown up earlier: the
  Documenter workflow only deploys a *release* build on a tag push, and the repository had no
  version tags at all until 0.2.0, so `v^` matched nothing and the link was never attempted.
  Making the first release is what exposed it.

  `devurl` is now absent, i.e. Documenter's default of `"dev"`: `main` goes to `/dev/`, each tag to
  `/vX.Y.Z/`, and `/stable/` is the symlink to the newest release. **This moves published URLs** —
  `/stable/` served `main` before and now means the release. `/v0.2.0/` itself will not appear
  retroactively, the tag carrying the `make.jl` it was built with; it lands from the next tagged
  release onwards. The `/stable/` tree currently on `gh-pages` was built from the `main` commit the
  tag points at, so it holds the 0.2.0 documentation already.

### Changed

- **The OGA section gains an at-a-glance comparison and documents the adapter layer.** Three
  tables at the top of *Algorithms* compare the dictionaries, the selection rules and the fits
  on what they add, what they cost and how they behave on a dependent atom — the design-level
  view, next to the measured ranking in *Studies* and the prescriptive list in *Usage*. Two
  implementation decisions that previously reached the manual only through docstrings are now
  on the page: the `rtol` default of `PivotedQR`/`TruncatedSVD` (`eps(T)·max(4, k)`, and why
  not `sqrt(eps(T))` — that is the scale of the rank-gain floor, so truncating there would
  discard the directions selection just admitted), and the power-of-two dictionary rescaling
  that opens the greedy loop.

  *Usage* gains the glue between `oga_fit` and the integrators, which `src/oga/adapters.jl`
  had but no page stated: the seed runs **one independent greedy fit per degree of freedom**,
  its target is the `initial_trajectory_method` estimate rather than the solution, and the node
  count and Simpson weights both come from `extrapolation_substep` — which is where the `M = 11`
  assumed throughout *Algorithms* comes from. Also that `TrainingMethod` is an alternative seed
  for `ShallowNet`, and that `show_status` is what prints the per-component neuron count,
  residual and rejection count — off by default, see *Breaking* above.

- **The Newton hot path no longer allocates gratuitously.** Measured bytes per `residual!` call
  (Float64, S = 4, R = 8, D = 1), before → after:

  | integrator | before | after | |
  |---|---|---|---|
  | `ShallowNet` | 21 344 | 11 424 | 1.9× |
  | `ShallowNetReversible` | 26 176 | 11 424 | 2.3× |
  | `ShallowNetAutodiff` | 211 968 | 51 584 | 4.1× |
  | `ShallowNetAutodiffReversible` | 216 800 | 51 584 | 4.2× |

  The substantive items: six cache arrays in the autodiff integrators were written on every
  residual evaluation and **never read anywhere**, along with the two `ForwardDiff.gradient`
  calls that existed only to fill them; `DVDθ` was evaluated *twice* per quadrature node in the
  symbolic pair, once for `a` and again for `dvdWc`/`dvdbc`, although one call returns both;
  `apply_NN` materialised three slices of its parameter vector plus three broadcast temporaries
  per call, inside a `ForwardDiff.gradient` and again inside a nested gradient-of-a-derivative;
  `ps_vec` was allocated up to `2D+1` times per `components!` call across three loops that each
  recomputed the same gather; the one-element network inputs `[quad_nodes[j]]`, `[zero(ST)]`,
  `[one(ST)]` allocated per node per dimension; and `s̃` was allocated in all seven caches and
  never read or written.

- **`cache(int, ST)` is now inferable.** `CacheType` built its return type from *runtime* fields
  (`method.basis.S`, `method.common.extrapolation_substep`), so it returned
  `ShallowNetCache{Float64, _A, 8, _B} where {_A, _B}` — not concrete. That is the type the
  framework's `CacheDict` lookup asserts against, so every value read out of the cache was `Any`.
  Those parameters were phantom: no field type mentioned them and nothing dispatched on them.
  They are ordinary constructor arguments now. For the autodiff integrators this was the root of
  thirteen runtime dispatches, `ps_vec::Any` poisoning every gradient call downstream.

- **`ForwardDiff.gradient` replaced by `gradient!` on the ansatz derivatives.** `gradient(f, x)`
  chooses its chunk size from `length(x)`, a runtime value here, so its return type is not
  inferable and came back `Any`. The in-place form returns the buffer it was given.

- **The four shallow-net caches are two.** Within each pair they were byte-identical apart from
  the struct name and one line — the length of the unknown vector — which is now a constructor
  argument. `SymbolicShallowNetCache` and `AutodiffShallowNetCache` live in
  `network_integrator_core.jl`. Likewise `DenseNet`'s parameter packing, which appeared verbatim
  in four places as a loop in which four of the five assignments did not depend on the loop
  variable, is now `densenet_pack!`/`densenet_unpack!`.

- **`update!`'s package-local second form is now `update_solution!`.** Splitting the update into
  "run `components!`, then write the cache into `sol`" is this package's own convention — the
  framework has no such form — and defining it as a fourth `update!` method gave it a signature
  ambiguous against the framework's own. Five of the eight ambiguities Aqua reported were this;
  the three that remain come from following the framework's documented extension signature and
  are not reachable.

- **VISE is typed.** `VISE.quadrature` and eight `VISECache` fields were `::Any`, and their
  contents `Vector{Any}` because the builders started from `mat = []`.

- **`VISE` gains `record_grid_points`,** matching the five network integrators. The recording
  grid was a hard-coded 41 in two places that had to agree by hand — the `stage_values` buffer
  in `VISECache` and the loop in `record_finer_solution!` — and the latter built its nodes with
  `collect(0:1/40:1)`, i.e. in `Float64` whatever precision the run was started at. The grid is
  built at the working element type now, the way the network integrators build theirs.

- **Dead code removed:** `default_iparams` (three definitions, no call sites),
  `GaussQuadrature64`/`GaussQuadrature128`, four unused ansatz partial derivatives, the `X`
  field of the autodiff shallow-net cache (allocated per cache, indexed nowhere — `s̃`'s
  counterpart, and only the *symbolic* pair reads its `X`), a duplicate
  `update!` override in `ShallowNetReversible`, and a commented-out second copy of the ansatz in
  `shallownet_autodiff_reversible.jl` that made the file look self-contained when it in fact
  depends on its sibling being included first. `Int(S/2)`-style float arithmetic in index
  expressions became `S ÷ 2`.

### Tests

- **The five per-integrator unit files are one table-driven file.** They were structurally
  identical — each declared its own copy of the same three-element extrapolation list, then
  repeated the same accuracy block and cross-product loop, varying only the constructor and one
  tolerance. Two of them differed in 27 of 44 lines, every one a type-name substitution plus a
  single number. `testsetup.jl` gains the shared `EXTRAPOLATIONS`, `ho_accuracy_problem`,
  `accuracy_guard`, `dispatch_case` and `assert_finite_endpoint`.

- **Bases are memoised.** The suite performed ~72 SymbolicNeuralNetworks code-generation runs to
  obtain about two distinct objects per element type, and `.githooks/pre-push` pays that on
  every push.

- **New gates:** per-integrator `@allocated` budgets on `residual!` and `@inferred`; Aqua's full
  suite; and a `D = 2` layout guard for the network integrators — every layout mistake between
  `components!`, `residual!` and `update!` collapses to the identity at `D = 1`, which is why
  the linear reference integrator's unit file already carried the equivalent guard before both
  moved upstream with `CGVINodal` (see *Breaking*). VISE's test went from one Float64 step
  asserting only finiteness to a five-step accuracy assertion (its ansatz is exact for the
  harmonic oscillator), a restart guard, and trait checks.

- **Test dependencies moved to `test/Project.toml`**, so Aqua and JET stay out of the package's
  own dependency graph. JET's analysis is a developer script,
  `test/quality/jet_residual.jl`, rather than a suite assertion — see the note in
  `test/quality/aqua_jet.jl` for why.

## [0.2.0] - 2026-08-16

### Breaking

- **`integrate` returns `(sol, internal_values)` for every network integrator.**
  `NonLinear_OneLayer_GML` used to return a named tuple carrying per-step instrumentation
  (`sol`, `mse_err_list`, `abs_err_list`, `training_time_list`, `solving_time_list`, …)
  while the other four returned the two-element tuple, so nothing could be written
  generically across them. The instrumentation cache fields are gone with it; the benchmark
  harness therefore reports only `total_secs`, the wall clock around `integrate`, and its
  CSV drops the `solve_secs` column (17 → 16 fields). Callers reading `res.sol` must
  destructure: `sol, internal_values = integrate(...)`.
- **Constructor keywords renamed.** `nstages` → `extrapolation_substep` (it counts
  sub-steps of the warm-start extrapolation, not quadrature stages) and
  `initial_trajectory` → `initial_trajectory_method` (to match `initial_guess_method`).
  Applies to all five network integrators and to `PR_Integrator`. New keyword
  `record_grid_points = 41` replaces the hard-coded 41-point recording grid.
- **`stages_compute!` is renamed `record_finer_solution!`.** It never computed quadrature
  stages; it samples the converged ansatz on a finer grid for plotting.
- Removed the unused `use_hamiltonian_loss` keyword from all one-layer integrators and
  `problem_initial_hamitltonian` from `NonLinear_OneLayer_GML`.
- `issymplectic` now genuinely returns `missing` for every `NetworkIntegratorMethod`.
  The previous per-integrator definitions were *bare*, so — `GeometricIntegratorsBase`
  exporting the name and the module only doing `using` — they defined a shadowing
  `NonlinearIntegrators.issymplectic` that nothing ever called; the trait fell through to
  the framework's `missing`. The definitions are now qualified and therefore live, and the
  value is stated as `missing` rather than the `true` that a refactor would otherwise have
  introduced by accident: symplecticity is not established for an ansatz whose parameters
  are refitted every step.
- **`OGA1d_Legacy` is renamed `OGA1dNormalEquations`.** The new name says what the
  variant *is* — the reference implementation from the original paper, solving the fit
  through the normal equations in a `Float64` island — rather than merely that it came
  first. Callers passing `initial_guess_method = OGA1d_Legacy()` must update; there is no
  deprecation shim: the type only ever existed on `main`, never in a tagged release. The
  argument for a version bump that this entry used to carry — that a downstream `[compat]`
  bound on `0.2` should fail at resolve time rather than at run time with an `UndefVarError`
  — no longer applies: 0.2.0 *is* this release, so there is no earlier tag for such a bound
  to name. Known downstream: SolverBenchmark's `nonlinear_onelayer_method` defaults to this
  seed, and takes it from `main` rather than from a registered version.
- **`Time_reversible_OneLayer` and `Time_Reversible_Hardcode` now reject a basis with an
  odd number of neurons.** Both represent the step with neurons in mirrored pairs and store
  only the `S/2` independent hidden parameters, so an odd `S` was never usable: it
  previously failed at the first time step with an `InexactError` out of `Int(S/2)` in
  `components!`, several call levels from the cause. It is now an `ArgumentError` at
  construction. `oga_fit` enforces the same condition for any caller of the shared greedy
  loop — an odd `nneurons` under a mirrored symmetry would place one neuron fewer than
  asked and leave the last at `(0, 0)`, which is the duplicated-neuron state `fill_unused`
  exists to prevent.
- `initial_params!` is unified on the three-argument form `initial_params!(int, method,
  sol)`. The two boundary-ansatz integrators already needed `sol`; the two `OneLayer` ones
  and `NonLinear_DenseNet_GML` took two arguments, so the seed could not be written
  generically across all of them. Only relevant to code defining its own
  `InitialParametersMethod`.

### Added

- **`ShallowNetBasis{T}(σ, S; symbolic = false)`** builds the network without compiling the
  symbolic derivatives. `ShallowNetAutodiff` and `ShallowNetAutodiffReversible` differentiate
  their ansatz with `ForwardDiff` at run time and never read `dqdθ`, `V_func` or `dvdθ`, so
  for them the build was pure overhead — 15 ms against 29 ns for `tanh` at `S = 8`, Float64,
  once the code generation itself has been compiled. The three integrators that *do* read those
  fields (`ShallowNet`, `ShallowNetReversible`, `DenseNet`) now reject such a basis in their
  constructor instead of failing on a `nothing` call inside `components!`. The new predicate
  `has_symbolic_derivatives(basis)` is exported.

- **`cse` and `inplace` keywords on `ShallowNetBasis` and `DenseNetBasis`**, forwarded to
  `SymbolicNeuralNetworks.build_nn_function`. Both default to `true`, which is also what that
  package uses — pinned here rather than left implicit, so an upstream change cannot silently
  change the code generation — and they exist to be turned *off*:
  `cse = false, inplace = false` re-emits the forward pass shared by the gradient blocks once
  per block and evaluates a batch out of place, which is what the new benchmark measures the
  two settings against each other for. `inplace = false` is also what a caller who wants to
  differentiate the kernels with `Zygote` would need, since the in-place form mutates its
  output.

- **A `ForwardDiff` cross-check of the compiled derivatives**, in
  `test/unit/dispatch_variants_unit.jl`, over both bases and both `TEST_TYPES`. `dqdθ`, `dvdθ`
  and `V_func` are compared against `ForwardDiff` over a flattened parameter vector, which
  reaches the network through `basis.NN` and so shares no code with the symbolic path beyond
  the network itself — and, unlike `NI.∂NN_ansatz_∂params`, does not require the hand-written
  ansatz to match. The flattened lengths are asserted too (`3S`, and `NP` for `DenseNetBasis`),
  which pins the layout `components!` slices out of.

  Until this, the only kernel-level assertion was `cse+inplace` against `plain` — one symbolic
  expression under two code-generation settings, which catches a wrong code path but not a
  wrong expression. That is what made the `symbolic_pullback` → `symbolic_parameter_gradient`
  move riskier than it reads: the scalar return shape changed with the name, and indexing the
  new return value with `[1]` yields the first parameter *leaf* rather than an error, so a
  version of the update that kept the index would have compiled a silently truncated gradient.
  The check that caught that by hand now runs in CI.

- **`benchmark/compare_derivative_backends.jl`**, comparing `ShallowNet` /
  `ShallowNetReversible` (symbolic derivatives) against `ShallowNetAutodiff` /
  `ShallowNetAutodiffReversible` (`ForwardDiff`), with the symbolic pair run under both
  code-generation settings. Three measurements: the one-off basis build, the end-to-end solve
  split into a cold and a warm run, and the derivative kernels timed in isolation. Measured at
  Float64, `tanh`, median time and bytes per call:

  | S | kernel | `cse+inplace` | `plain` | `ForwardDiff` |
  |---|---|---|---|---|
  | 4 | `dqdθ` | 0.042 µs / 528 B | 0.083 µs / 528 B | 1.375 µs / 6448 B |
  | 4 | `dvdθ` | 0.083 µs / 528 B | 0.125 µs / 528 B | 1.500 µs / 8128 B |
  | 8 | `dqdθ` | 0.083 µs / 720 B | 0.125 µs / 720 B | 2.792 µs / 16784 B |
  | 8 | `dvdθ` | 0.083 µs / 720 B | 0.250 µs / 720 B | 3.416 µs / 22928 B |
  | 16 | `dqdθ` | 0.125 µs / 1136 B | 0.792 µs / 2304 B | 7.791 µs / 54096 B |
  | 16 | `dvdθ` | 0.167 µs / 1136 B | 1.042 µs / 2304 B | 9.291 µs / 77136 B |

  The compiled kernels run 18–62× faster than `ForwardDiff` and allocate 12–68× less; within
  the symbolic backend, `cse+inplace` pulls ahead of `plain` as the network widens (1.5–2.0×
  at `S = 4`, 6.2–6.3× at `S = 16`). Timer resolution is ~0.042 µs, so the narrow rows are one
  tick apart and should be read as such.

  Note what the file does *not* claim. The two backends use different ansätze (raw network
  vs. boundary-interpolating) and different default OGA seeds, so accuracy and iteration
  counts compare methods, not backends. And while the two codegen settings agree to machine
  epsilon at the kernel level (≤ 8e-17 at Float64), they do *not* agree end to end: the residual
  stalls near the round-off floor, so a last-bit difference decides which iterate Newton
  accepts. The report measures that amplification rather than asserting it away. See
  `benchmark/README.md`.

- **New dictionaries.** `WeightBiasGrid2d` is a genuine 2-D grid over `(w, b)`, with
  log₂-spaced weight magnitudes crossed with the bias grid. For a positively homogeneous
  activation the weight magnitude is redundant — `σ(wx+b) = |w|ᵏσ(sign(w)x + b/|w|)`, so
  only the sign carries shape information, which is why the `{±1} × (bias grid)` set is
  complete for `ReLUᵏ` — but ELU and GELU are not homogeneous, and for them `|w|` is a real
  length-scale parameter. Restricting the weight axis to `{±1}` recovers the 1-D
  dictionary exactly, so this is a strict generalisation. `AngularGrid` places atoms on
  rays through the origin of `(w, b)` space, which is the dictionary the underlying
  approximation theory is stated for, and samples uniformly in atom space rather than
  uniformly in bias. `Refined` wraps any dictionary and polishes the selected atom off the
  grid by a derivative-free local search, decoupling accuracy from dictionary size.
- **New selection rules.** `NormalizedProjection` scores by
  `|⟨r,g⟩_w| / ‖g‖_w`, the textbook greedy criterion, which is scale-invariant and so
  mandatory for a 2-D dictionary. `OrthogonalProjection` scores against the part of the
  atom orthogonal to those already selected — the actual orthogonal-greedy criterion rather
  than matching pursuit — and refuses any atom whose orthogonal part has collapsed. That
  is the direct fix for the observed reduced-precision failure: an atom adding no new
  direction can no longer be selected, which is the condition that used to surface as
  `SingularException: zero pivot found at index 3` out of four neurons.
- **New fits.** `IncrementalQR` maintains the factorisation across greedy steps
  (`O(k·n)` per step instead of `O(k²·n)`, and its `Q` powers the orthogonal selection
  score for free). `PivotedQR` and `TruncatedSVD` are rank-revealing; both are hand-rolled
  because `qr(A, ColumnNorm())` and `svd` are LAPACK-only and therefore do not exist at
  `Float16`, the precision that needs them. `NormalEquationsFit` exposes the Gram solve
  with the ridge and the `Float64` island as independent switches, so "island vs working
  precision" and "ridge vs no ridge" are ablations on one code path.
- `oga_check_precision`, called once per fit: throws if the activation does not evaluate at
  the working precision. The `max(0.0, x)^k`-instead-of-`max(zero(x), x)^k` trap promotes
  the whole seed to `Float64` and is otherwise visible only as suspiciously good
  half-precision accuracy.
- `test/unit/oga_kernels.jl`: direct coverage of the OGA numerics at `Float16`, `Float32`
  and `Float64` — previously they were exercised only through full integrations. Includes
  the `eltype === T` / `@inferred` no-upcast gate over every dictionary × selection × fit
  combination, the `OGA1d` atom-selection pin, verification of the hand-rolled
  factorisations against LAPACK, and a check that the normalised and orthogonal selection
  rules find the brute-force optimal first atom (which raw projection, by design, does
  not).
- **The Orthogonal Greedy Algorithm documentation is now a six-page section** — Overview,
  Theory, Algorithms, Usage, Precision, Studies — replacing the single page. *Theory* derives
  the selection criterion from the one-step residual reduction (which is also where the
  rank-gain floor comes from), gives the dictionary-completeness argument for positively
  homogeneous activations and shows where it fails for smooth ones, and sets out the
  conditioning analysis. *Algorithms* documents each of the four dictionaries, three selection
  rules, five fits and four guard rails with its mechanism, implementation, cost and when to
  choose it. *Usage* covers presets, composing configurations, per-integrator behaviour,
  reading `OGAResult`'s diagnostics, and extending with a new component. *Precision* states the
  no-implicit-conversion invariant and how it is enforced. *Studies* reports the measurements
  with their methodology and caveats.
- New studies in `scripts/` (not `benchmark/`, which holds the integrator-suite benchmarks and
  is driven by the docs build), replacing `benchmark/oga_comparison.jl`: `oga_fit_study.jl` measures seed
  quality with no integrator and no Newton solve (the two are otherwise confounded),
  `oga_sweep.jl` runs the end-to-end harmonic-oscillator sweep over variant × precision ×
  regularization factor × activation in a `ReLUᵏ` stage and a smooth-activation stage, and
  `oga_double_pendulum.jl` repeats a reduced grid at a single λ on the problem the seed
  fails hardest on. `regularization_factor` is swept as `2^k √eps(T)` rather than as
  absolute values, so the shift is scaled to the precision it protects — and `f_abstol` is
  scaled the same way, at `256·eps(T)`. The latter is not cosmetic: the solver's default
  `f_abstol` is `1.78e-15`, an absolute `Float64`-scaled value that `Float32` and `Float16`
  cannot reach, so a reduced-precision run sits at its residual floor and burns the whole
  iteration budget while parked on the right answer. Measured before the fix, `ReLU³` at
  `Float32` reported 1000 iterations at every regularization factor with an accuracy of
  `1.8e-7` — read as
  non-convergence, that made the entire `Float32` column an artefact of the tolerance rather
  than a fact about the seed. Runs that do exhaust the budget are recorded as `maxiter`
  rather than `ok`, and runs whose final state leaves the working precision as `upcast`.
- Shared OGA numerical helpers, now in `src/oga/numerics.jl`:
  - `weighted_lstsq(Φ, w, y)` — quadrature-weighted least squares via QR on the
    `√w`-scaled design matrix, with a Tikhonov-ridged fallback that only engages
    when the plain solve returns a non-finite result (the genuinely
    rank-deficient `Float16` case).
  - `oga_norm_floor(T, ref) = sqrt(eps(T)) · ref` — precision-scaled floor for
    the dictionary-normalization guard.
  - `oga_tikhonov(G; C = 100) = C · eps(T) · tr(G) / n` — precision-scaled
    Tikhonov floor (used as the ridge in the `weighted_lstsq` fallback).
  - `bias_grid(lo, hi, n, T)` — index-based construction of the bias grid.
- A coherence guard in the greedy selection that blocks atoms whose
  quadrature-weighted L² coherence with an already-selected atom exceeds
  `1 - sqrt(eps(T))`. It is inert at `Float64`/`Float32` and only bites at
  `Float16`, where it keeps the selected neurons linearly independent.
- Documentation section describing the findings, the reformulated algorithm and
  its references, with a self-contained didactic `Float16` example.

### Changed

- **`SimpleSolvers` 0.11 → 0.12.1, and `GeometricOptimizers` comes from the registry.** The
  `[sources]` git pin is **gone**: GeometricOptimizers 0.2.0 is now in General, and its registered
  copy requires SimpleSolvers 0.12, which is what forces the SimpleSolvers bump here. Two of the
  three entries under *Known issues* below close with it — this package can be tagged again, and
  **Julia 1.10 can install it again**, `[sources]` having been the Pkg 1.11 feature that made the
  three 1.10 CI jobs unsatisfiable. The `[sources]` explanations in `benchmark/Project.toml` and
  `scripts/Project.toml`, which existed only to say why the pin was *not* repeated there, go with
  it.

  SimpleSolvers 0.12 is breaking in one way that reaches this package: a `NonlinearSolver` no
  longer emits line-search warnings from inside its iteration, so a rejected line search reports to
  its caller through the returned status and to nobody at all if the caller drops it. See the next
  entry. Its other two breaks do not apply — nothing here constructs a `NonlinearSolverStatus`, and
  the one third-party `LinesearchMethod` in the dependency graph is GeometricOptimizers'
  `DecayingStatic`, which implements `solve_with_status` — the side of the rewritten contract that
  is now the one to implement — and has `solve` derived from it, so it was unaffected.

- **The nonlinear solves use `solve_with_status!`**, and hand the status to
  `check_solver_status`, which GeometricIntegratorsBase 0.6.3 adds as the single place a step's
  solve outcome is acted on. All three sites: the network integrators' shared `integrate_step!`,
  `CGVINodal`'s and `VISE`'s. `check_solver_status` is silent by default — SimpleSolvers remains
  the one voice that reports a failed solve — so nothing changes in what a run prints. It replaces
  the commented-out `print_solver_status` / `check_solver_status` stubs, which named this exact
  function and had never been able to call it.

  `CGVINodal` and `VISE` also move to the **state-taking** form. Both are `GeometricIntegrator`s
  and so carry a persistent `solverstate`, which the three-argument call they used ignored,
  allocating a fresh `NonlinearSolverState` on every time step. The network integrators already
  passed theirs. That form of `solve_with_status!` is new in SimpleSolvers 0.12.1.

- **`ShallowNet`'s `TrainingMethod` seeding runs through `GeometricOptimizers.solve!`** instead of
  a hand-rolled epoch loop, and reads the `OptimizerResult` it returns. The epoch budget moves from
  a `for` range to `max_iterations`, because `solve!` runs its own loop against
  `meets_stopping_criteria`; `warn_iterations = 0` goes with it, since reaching a 50 000-epoch
  budget is the normal outcome here rather than a diagnosis, and at the default of 1000 `solve!`
  would print its warning once per dimension per time step.

  **The trained seed can differ.** The loop makes the same `solver_step!` calls in the same order,
  so a run that uses its whole budget is unchanged; but `solve!` may also stop *early*, on the
  convergence criteria or on a non-finite iterate, where the old loop always burned all `nepochs`.
  Since this only produces the initial guess for the Newton solve that follows, an earlier stop
  costs iterations in that solve rather than accuracy in the result. The debug line now reports the
  epochs actually spent and whether the optimizer converged, instead of asserting the budget.

  In practice it does *not* stop early at these settings, and that is by construction rather than
  by luck: `allow_f_increases` defaults to `true`, so `Adam` overshooting on one step does not end
  the solve, and `DecayingStatic` is still taking steps of `η₂ = 5e-5` at the horizon — eleven
  orders of magnitude above the `x_abstol` of `2eps`, which is the gate that would have to fire.
  What the assessment costs is one extra objective evaluation per epoch: `solve!` builds an
  `OptimizerStatus` after every `solver_step!`, which the hand-rolled loop did not.

  There is no `solve_with_status!` for a GeometricOptimizers optimizer — see *Open Issues*.

- **The two `DenseNet` seeding loops stay hand-rolled**, deliberately. `TrainingMethod`'s breaks
  early on the loss, and `LSGD`'s re-solves the `L3` layer by least squares *inside* every epoch;
  `solve!` has a hook for neither. They therefore still check no optimizer status — recorded under
  *Open Issues*.

- **The hardcoded-ansatz helpers of `ShallowNetAutodiff` / `ShallowNetAutodiffReversible`
  spell "ansatz" correctly.** `NN_anstaz`, `VNN_anstaz`, `VNN_anstaz_zygote`,
  `∂NN_anstaz_∂params`, `∂VNN_anstaz_∂params`, `∂NN_anstaz_∂q̄`, `∂NN_anstaz_∂q`,
  `∂VNN_anstaz_∂q̄` and `∂VNN_anstaz_∂q` became `*_ansatz_*`. Names only; no behaviour
  changed. None of them is exported, and the only call site outside `src/` was
  `benchmark/compare_derivative_backends.jl`, which reaches them through the `NI.` prefix
  and was updated with them.

- **`SymbolicNeuralNetworks` 0.4 → 0.5**, and with it `AbstractNeuralNetworks` ≥ 0.6.4, which
  is where 0.5 takes `input_dimension`/`output_dimension` from. 0.5 is a refactor of the whole
  package below its exported surface, with no deprecation shims, but only one of its breaking
  changes reaches this package: `symbolic_pullback` is now `symbolic_parameter_gradient`, and
  for a *scalar* expression it returns the parameter-shaped gradient directly instead of a
  one-element array holding it. Both bases were differentiating a scalar somewhere and
  unwrapping the array afterwards, so the unwrapping went away with it.

  Everything else the release changes is invisible here. `nn.input` became a `Vector{Num}`
  rather than a `Symbolics.Arr` — this package only ever passes it through, and
  `build_nn_function` accepts either. `build_nn_function` now returns an
  `InPlaceBatchedFunction`/`OutOfPlaceBatchedFunction` instead of a closure, which makes
  `NetworkBasisCore`'s `QWFT`/`VT`/`VWFT` type parameters concrete and inferable at no source
  cost. `Jacobian` flattens a non-vector argument with `vec`, which our one-element case never
  notices. Verified kernel by kernel against `ForwardDiff`: `dqdθ`, `dvdθ` and `V_func` agree
  to round-off for both bases (≤ 5.6e-17 at Float64).

  `DenseNetBasis` now builds `dvdθ` the way `ShallowNetBasis` does, by differentiating the
  scalar entry of the 1×1 Jacobian rather than the whole array. The two bases had drifted
  apart, and under 0.5 the array form wraps a one-element array of equation-set functions in
  an extra indirection for nothing. `DVDθ` in `DenseNet.components!` consequently returns a
  parameter set like `DQDθ` does, and the `[1,1]` that used to unwrap it is gone.

  That left the two constructors' derivative blocks identical up to which network they are
  handed, so they are now one function: `build_shallownet_derivatives` became
  `build_network_derivatives` and moved beside `NetworkBasisCore`, whose four symbolic slots
  it fills. Both bases call it. Unexported, and the only reference outside `src/` was a
  comment in `benchmark/compare_derivative_backends.jl`.

  `benchmark/compare_derivative_backends.jl` was re-run against 0.5 and the tables above are
  its output; the two codegen settings still agree to round-off (≤ 8e-17 at Float64, ≤ 8e-8 at
  Float32) and `cse` still buys about 4× on the `DenseNetBasis` build. The absolute build
  times quoted in the docstrings moved with the release and were re-measured with it.

- **`SymbolicNeuralNetworks` 0.3 → 0.4.** A performance release with a source-compatible API:
  code generation now performs common-subexpression elimination, batches are evaluated by an
  in-place kernel writing into a single preallocated array, and an equation *set* (which is
  what a symbolic parameter gradient is) is generated as one function rather than one per leaf.
  Measured here: `ShallowNetBasis` construction 1.6–2.0× faster, `DenseNetBasis` 4.1×
  (3.22 s → 0.79 s — the deeper network is where re-emitting the shared forward pass hurt
  most). No call site changed; the new `cse` and `inplace` keywords are set to the fast path,
  which is also what they default to upstream, and are now exposed on the bases (see *Added*).

  The build-time and run-time wins come from different halves. For a *shallow* net `cse`
  costs nothing to build and buys nothing at `S = 4`, while generating the in-place kernel is
  the expensive half of the build (0.24 s against 0.017 s for `tanh` at `S = 8`) and is what
  pays it back at evaluation time. For `DenseNetBasis` it is the other way round: `cse` is the
  whole 4.1×.

  Two things to know. The in-place result cannot be differentiated with `Zygote`, which is
  fine here — the only `Zygote.gradient` in the package (`VNN_ansatz_zygote`) differentiates
  a hand-written ansatz, and the generated kernels only ever see `ForwardDiff.Dual`. And the
  output element type is now promoted over the *inputs* rather than inferred from the
  expression, so a `Float32`/`Float16` network whose generated code contains a `Float64`
  constant is rounded rather than widened — what the no-silent-upcast invariant wants, but it
  can shift reduced-precision results. The `ShallowNet` Float64 accuracy guard moved by 2 ulp
  (`0.38012229853795865` → `0.38012229853795837`).

- **BREAKING: every exported type and every source file has been renamed** to line up with
  GeometricIntegrators. The old names are gone; there are no deprecation shims. Nothing about
  the numerics changed — this is purely nomenclature.

  Source files now sit in one lowercase directory per method family. `src/network_basis/` and
  `src/network_integrators/` merge into `src/nvi/` (network variational integrators),
  `src/CGVI_standard/` becomes `src/cgvi/`, and `src/SINDy_methods/` becomes `src/vise/`,
  joining the existing `src/oga/`. Test, script, benchmark and documentation files follow the
  same scheme. Within a directory each file is named after what it defines — `shallownet.jl`,
  `shallownet_basis.jl`, `densenet.jl`, `densenet_basis.jl`, `vise.jl`, `vise_basis.jl`,
  `cgvi.jl` — so the `Linear`/`NonLinear` and `_Int` markers disappear along with the type
  names that carried them.

  Integrators keep the bare family name and carry the variant as a suffix:

  | old | new |
  | --- | --- |
  | `NonLinear_OneLayer_GML` | `ShallowNet` |
  | `Hardcode_int` | `ShallowNetAutodiff` |
  | `Time_reversible_OneLayer` | `ShallowNetReversible` |
  | `Time_Reversible_Hardcode` | `ShallowNetAutodiffReversible` |
  | `NonLinear_DenseNet_GML` | `DenseNet` |
  | `CGVI_standard` | `CGVINodal` |
  | `PR_Integrator` | `VISE` |

  Bases take the `...Basis` names, so each matches the file that defines it, and the per-family
  abstract supertypes gain an `Abstract` prefix to make room:

  | old | new |
  | --- | --- |
  | `OneLayerNetwork_GML` | `ShallowNetBasis` |
  | `DenseNet_GML` | `DenseNetBasis` |
  | `OneLayerNetBasis` (abstract) | `AbstractShallowNetBasis` |
  | `DenseNetBasis` (abstract) | `AbstractDenseNetBasis` |
  | `PR_Basis` | `VISEBasis` |

  `OneLayerMethod` becomes `ShallowNetMethod`, and every `*Cache` follows its method. A call
  site now reads `ShallowNet(ShallowNetBasis{T}(tanh, 8), quad)`.

  Three of the new names say something the old ones did not:

  - `Autodiff` replaces `Hardcode`. The distinction is the derivative backend: these two
    integrators differentiate a hand-written ansatz with `ForwardDiff`, where `ShallowNet` and
    `ShallowNetReversible` use derivatives compiled ahead of time by `SymbolicNeuralNetworks`.
  - `CGVINodal` replaces `CGVI_standard`. It is not a copy of `GeometricIntegrators.CGVI`:
    upstream solves for all `S` basis coefficients plus the endpoint momentum (`D*(S+1)`
    unknowns), whereas this one pins `X[1] = q̄`, reads `q` back off the last coefficient and
    computes `p` explicitly, leaving `D*(S-1)`. That reduction requires an interpolatory basis
    with nodes at both endpoints — the coefficients *are* nodal values. The name also avoids a
    binding clash for anyone loading both packages, as `benchmark/` and `scripts/` do.
  - `VISE` replaces `PR_Integrator`, matching what the documentation has always called it.

  The `_GML` suffixes are dropped throughout: GeometricMachineLearning stopped being a
  dependency earlier in this same release cycle, so they pointed at nothing.

  In `benchmark/`, the environment variables `GML_BENCH_PRESET` and `SKIP_GML_BENCH` become
  `SHALLOWNET_BENCH_PRESET` and `SKIP_SHALLOWNET_BENCH`, and the combined-report figure prefix
  changes from `onelayer_gml_benchmark` to `shallownet_benchmark`. Result CSVs written before
  this change still parse, but the combined figures are written under the new prefix.

  In `scripts/`, the JLD2 result keys are renamed with everything else: `HO_PR_sol_q` becomes
  `HO_vise_sol_q`, and so on for the Pendulum, PerturbedPendulum and HenonHeiles families.
  The writer (`test_vise.jl`) and both readers (`vise_plot.jl`, `find_optimal_results.jl`) were
  changed together, so they remain consistent — but **`.jld2` files produced before this change
  can no longer be read by the plotting scripts**, since the keys they contain no longer exist.
  Re-run the sweep, or rename the keys in place. `scripts/results/` is git-ignored, so nothing
  in the repository is affected.

- **Upgraded to QuadratureRules 0.2 and CompactBasisFunctions 0.3**, along with
  GeometricIntegratorsBase 0.6, GeometricEquations 0.21 and SimpleSolvers 0.11. The source side was
  already done — `basis` and `nnodes` come from `GeometricBase` and `nbasis` from
  `CompactBasisFunctions`, which is what the new versions require. RungeKutta 0.6 is satisfied
  vacuously: it is not a dependency of this package at all any more (see below).
- Zygote compat widened to `0.6, 0.7`; the graph resolves to 0.7.12. Zygote remains a direct
  dependency for `VNN_ansatz_zygote`, which supplies the velocity of the hardcoded ansatz in
  `ShallowNetAutodiff` and `ShallowNetAutodiffReversible`.
- **`ImplicitMidpoint` now comes from `GeometricIntegratorsBase`** for the
  `IntegratorExtrapolation` warm start and `VISE`, and requires 0.6: the warm start
  integrates a LODE sub-problem and reads `p` back out of it to seed the momentum degree of
  freedom, which needs that release's `IODEProblem`/`LODEProblem` methods rather than an ODE-only
  implicit midpoint.

- **The network training loops now use `GeometricOptimizers` instead of
  `GeometricMachineLearning`.** The optimizer functionality has been retired from
  GeometricMachineLearning into GeometricOptimizers, so `Optimizer` /
  `AdamOptimizerWithDecay` / `GradientOptimizer` / `optimization_step!` are replaced by
  `Optimizer(ps, loss; algorithm, linesearch)` plus `solver_step!`/`update!`, with `Adam` and
  `GradientMethod` supplying only a *direction* and the learning rate living in the line
  search. The learning-rate schedule is preserved:
  `DecayingStatic(T; η₁ = 1e-3, η₂ = 5e-5, n = nepochs)` matches
  `AdamOptimizerWithDecay(nepochs, 1e-3, 5e-5)`, both decaying the step size as `γ^t · η₁` with the
  same `γ = exp(log(η₂/η₁)/n)`. The optimizer method and line search are now built at the
  parameter element type, where the old call passed Float64 constants regardless. Trained weights
  are nevertheless not bit-identical to previous releases': the bias-correction algebra inside the
  Adam moment update differs, and gradients now come from the optimizer's own `GradientAutodiff`
  rather than from explicit `Zygote.gradient` calls in the loops.
- New internal helpers `optimizer_params` / `network_params`
  (`src/nvi/utilities.jl`) convert between the nested per-layer parameters of
  `AbstractNeuralNetworks` and the flat `NamedTuple` of arrays that GeometricOptimizers
  accepts. They alias rather than copy, so the optimizer's in-place updates remain visible
  through `PNN.params`. Both are `@generated`: written as ordinary code they build their key
  set with `Symbol(lname, :_, f)` at run time, which inference cannot fold, so
  `optimizer_params` returned an abstract `NamedTuple` and `network_params` — which runs inside
  the differentiated loss on every gradient evaluation — was inferred no better.

- **Shared code for the five network integrators is extracted into
  `NetworkIntegratorCore` and `NetworkBasisCore`.** Each method struct now holds a
  `common::NetworkIntegratorCore` (with `getproperty` forwarding, so `method.basis` and
  friends keep working) plus only the fields that are genuinely its own; the accessors,
  traits, `initial_guess!`, the three `initial_trajectory!` methods, `residual!`, both
  `update!`s, `integrate_step!` and `integrate!` exist once instead of five times. The five
  concrete caches subtype a shared `NetworkIntegratorCache`. This is the second axis of the
  same decomposition as `src/oga/`: that release factored out *which neurons the seed
  picks*, this one factors out *the step machinery around it*.
- `NoExtrapolation` is now available to all five integrators, not just
  `NonLinear_OneLayer_GML`.
- Docstrings added to every exported method, basis and option type; the landing page gains
  a type-hierarchy overview and a runnable example.
- The docs build now fails immediately, naming the files, when the Benchmarks page
  references a figure the sweep did not produce — previously a batch of Documenter
  cross-reference errors half an hour into the build.
- **All OGA code now lives in `src/oga/`, behind one composable seed type.** The three
  axes of the algorithm — which candidate neurons are on offer, how the greedy step ranks
  them, and how the output weights are refit — are independent, and are now fields of a
  single `OGA{Dictionary,Selection,Fit}` rather than a type per combination. The named
  presets `OGA1d`, `OGA1dStable`, `OGA2d` and `OGASphere` are corners of it; `OGA1d()`
  keeps its previous behaviour exactly, including which atoms it selects (pinned by
  `test/unit/oga_kernels.jl`, since normalising before selection steers the Newton solve
  into a different and empirically worse basin).
- **All four network integrators share one greedy implementation.** `oga_fit` is
  integrator-agnostic: it takes a dictionary spec, an activation, quadrature nodes and
  weights, and a target, and returns neuron parameters. Previously each integrator carried
  its own copy of the loop, and the four copies had drifted into four different guard-rail
  policies — `Hardcode_int` normalised for selection, the two `OneLayer` variants
  normalised only for coherence, and `Time_reversible_Hardcode_int` had neither a norm
  floor nor any deduplication. The per-integrator differences that remain are declared
  rather than reimplemented: the `t(1-t)` ansatz modulation, whether neurons come in
  mirrored pairs with shared or independent output weights, and — for `Hardcode_int` — that
  its greedy step ranks candidates by the *normalized* inner product where the other three
  use the raw one. Which of the two an integrator uses changes which neurons get selected
  and hence which basin the Newton solve lands in, so each keeps the rule it was tuned with
  (`OGA1dNormalized()` is `Hardcode_int`'s constructor default) rather than inheriting a
  single shared one.
- The greedy loop rescales the dictionary by a single **power of two** so the largest atom
  has norm ≈ 1. At `Float16` squared quantities overflow long before the values do — a
  `ReLU³` atom of norm 43 squares to 1874, and two of those multiply past the 65504
  ceiling. A power of two is exact in binary floating point, so `Float64`/`Float32` atom
  selection is bit-for-bit unchanged.
- Every fit is now guaranteed to return a finite result with one entry per selected atom,
  enforced once in `oga_solve` rather than per fit. This closed two real gaps: Julia's
  generic `\` *throws* `SingularException` on a rank-deficient matrix rather than returning
  garbage (so the default fit could put a `SingularException` back on the seed path at
  `Float16` — the exact failure the reformulation exists to remove), and the pivoted-QR
  fit could return `Inf`/`NaN` from a division by a pivot that survived truncation.
- **The Orthogonal Greedy Algorithm (OGA) initial guess is now precision-generic.**
  Previously the OGA seed that warm-starts the Newton solve in the network
  integrators was assembled in `Float64` regardless of the solver's working type,
  because the least-squares step used the normal equations `Φ diag(w) Φᵀ`, whose
  condition number is `κ(Φ)²` and which becomes rank-deficient in reduced
  precision. The seed is now built entirely at the working type
  `T = eltype(nlsolution(int))`, so the whole path (dictionary construction,
  greedy selection, least-squares fit) is GPU-portable and consistent with the
  rest of the solver. This affects `NonLinear_OneLayer_GML`, `Hardcode_int`,
  `Time_reversible_OneLayer`, and `Time_reversible_Hardcode_int`. See the
  "Orthogonal Greedy Algorithm initial guess" section of the documentation for
  the full analysis.
- The greedy least-squares fit now uses a **QR factorization of the `√w`-scaled
  design matrix** (conditioned on `κ(Φ)` instead of `κ(Φ)²`) instead of forming
  and solving the Gram matrix. This removes the need for the `Float64` island and
  lets the fit run at `Float32`/`Float16`.
- **`[compat]` now requires GeometricIntegratorsBase 0.5, GeometricIntegrators 0.17,
  SimpleSolvers 0.10 and GeometricProblems 0.8.** The lower bounds are raised deliberately,
  not routinely: under GeometricIntegratorsBase 0.4 the default `f_abstol` was the `Float64`
  constant `8eps() = 1.78e-15` regardless of the working precision — unreachable at `Float32`
  and `Float16` — *and* the whole default set was substituted away as soon as a caller passed
  any option, so `integrate(prob, method; max_iterations = ...)` actually ran with
  `f_abstol = 0` and lost `min_iterations = 1` with it. 0.5 scales the tolerance with
  `datatype(problem)` and merges the defaults with the caller's options, which is what makes
  a reduced-precision run able to converge at all rather than sit at its residual floor
  burning the iteration budget. The documentation now describes that behaviour, so the bound
  makes the description true.
- **The benchmark suite's network width is now measured per problem, and the Toda lattice is
  excluded from the documentation build.** `S` decides the accuracy the shallow-network ansatz
  can represent, and therefore whether the nonlinear solve has a reachable target at all: too
  narrow, and the residual floors above the tolerance while the solve iterates to its cap.
  Measured at `Float64`/`tanh`/`DogLeg`, `quick`'s previous `S = 4` reached `ref_err = 2.8e-06`
  on the harmonic oscillator in 1000 iterations, where `S = 10` reaches `3.2e-14` in about 100.
  The widths are now 10 (harmonic oscillator), 8 (pendulum) and 10 (double pendulum). The
  pendulum's is an optimum rather than a maximum — its degenerate `ϑ` leaves the parameter
  Jacobian singular, so a wider network enlarges the null space and `S = 12` diverges outright.
  The cost is at half precision, and is a deliberate trade rather than an oversight: on the
  harmonic oscillator's full quick grid, `Float16` convergence falls from 17/36 at `S = 4` to
  12/36 at `S = 8` and 9/36 at `S = 10`, while the best `Float64` `ref_err` improves from
  2.8e-06 to 3.4e-14. A wider network puts more nearly dependent columns in the parameter
  Jacobian, and 11 bits of mantissa cannot separate them. Accuracy is what the suite reports and
  half precision is studied on its own terms in the OGA section, so the widths favour `Float64`;
  `S` is the knob if that priority ever inverts. See "The half-precision trade-off" on the
  Benchmarks page. The Toda lattice has no measured width yet, which puts its quick grid at ~5 h against
  ~7 min for the other three, so it is out of the docs build until it has one;
  `benchmark/run_toda_lattice.jl` and the `full` preset still run it.
- `quick` no longer overrides `max_iterations`, using the solver default of 1000; the `maxiter`
  status reads the cap that actually applied off the solver configuration rather than assuming
  it. `SOLVERS_QUICK` gains `Newton`/`Backtracking` alongside `DogLeg`, so every precision has
  at least one strategy that converges.
- **The benchmark suite no longer records a stalled solve as converged.** `integrate`
  returns a finite state after exhausting `max_iterations`, and
  `benchmark/gml_benchmark_common.jl` classified on finiteness alone, so a run that burned
  its whole iteration budget was counted as `ok` — concentrated in exactly the
  reduced-precision rows the suite is read for. Those are now `maxiter`, its own status,
  matching the rule the OGA studies in `scripts/` already used. Accuracy and drift are
  still recorded for them, so a stall can be told apart from a divergence. The reported
  convergence counts on the Benchmarks page drop accordingly; they were measured, not
  estimated, and the earlier ones overstated convergence.

### Removed

- **`GeometricIntegrators` is no longer a dependency**; the package builds on
  `GeometricIntegratorsBase` alone. Every `GeometricIntegrators.Integrators.X` extension point was
  already a `GeometricIntegratorsBase` generic imported into that module, so those call sites are
  simply requalified. `GeometricEquations` replaces it in `[deps]` because GeometricIntegrators was
  re-exporting it — that is where `AbstractProblemIODE`, `StateVariable` and `initial_conditions`
  come from, and GeometricIntegratorsBase does not pass them on. `create_internal_stage_vector` was
  the only genuinely GeometricIntegrators-local name and is now defined in
  `src/nvi/utilities.jl`. Consequences: `RungeKutta` and `GenericLinearAlgebra`
  leave the dependency graph entirely. Runge-Kutta reference integrators such as `Gauss(8)` are
  only ever needed by the `benchmark/` and `scripts/` environments, which declare
  GeometricIntegrators themselves.
- **`GeometricMachineLearning` is no longer a dependency.** Once the optimizer calls moved to
  GeometricOptimizers, its only remaining use was `GeometricMachineLearning.NeuralNetwork`, which
  it `import`s straight from `AbstractNeuralNetworks` — the same object — so the call site now
  names `AbstractNeuralNetworks.NeuralNetwork` directly. The `_GML` suffixes on the basis and
  integrator types, which recorded that provenance, are dropped by the renaming above.
- `ContinuumArrays` is no longer a dependency; it had no use in `src/`. The quasi-array indexing
  and `grid` come from `CompactBasisFunctions`, which carries `ContinuumArrays` itself.
- `GeometricProblems` moved from `[deps]` to the test target — its only use in `src/` was the
  dead `SINDy_methods/PR_Pretraining.jl`, now retired to `obsolete/script/`. That file was
  never `include`d and called into `Flux`, which was never a dependency.
- Unused imports in `src/NonlinearIntegrators.jl`: `relative_maximum_error` (replaced by the
  `GeometricSolution` and `timesteps` the source actually names), `Options`, `NonlinearSolver`,
  `DogLeg`, and a no-op `using Base`.
- `Optimisers` is no longer a dependency. It was carried as a bare `using` with no call site
  anywhere in `src/`; the training loops it once served moved to GeometricMachineLearning long
  before this release and now to GeometricOptimizers.

### Fixed

- **The `residual!` of the autodiff pair no longer accumulates in `Float64` at reduced
  precision.** Three `Float64` literals sat in the `p̄` row of the residual — `(1.0 -
  quad_nodes[j])` and `(-1.0)` in `src/nvi/shallownet_autodiff.jl:390-391`, `(-1.0)` in
  `src/nvi/shallownet_autodiff_reversible.jl:401` — where they stand in for the analytic
  boundary derivatives `∂q_h/∂q̄ = 1-t` and `∂v_h/∂q̄ = -1`. The accumulator `z` starts as
  `zero(ST)`, and during the Newton solve `ST` is a `ForwardDiff.Dual`; multiplying by a
  `Float64` literal promotes it, so at `Float32` the residual was summed as
  `Dual{…,Float64}` and only rounded back on the write into `b::Vector{ST}`. That is a
  silent upcast the suite cannot see — `assert_no_upcast` checks the eltype of the final
  state, which is converted back — and it retypes `z` mid-loop on the Newton path. They are
  now integer literals, `(1 - quad_nodes[j])` and `(-1)`, which is the idiom
  `shallownet_autodiff_reversible.jl:400` already used and which takes the precision of the
  operand instead of imposing one. `Float64` results are bit-identical; `Float32` moves by
  round-off. Found while reviewing the `*_anstaz_*` → `*_ansatz_*` rename, which is what
  drew attention to these hand-written copies of the boundary derivatives.

- **`CGVINodal` (formerly `CGVI_standard`) can integrate problems with more than one degree of
  freedom.** Found while diffing it against `GeometricIntegrators.CGVI` for the rename above, and
  pre-existing rather than introduced by it: a `D = 2` run (Lagrange basis on 4 Lobatto nodes)
  died in the first step with `SingularException: Zero pivot found at index 6`, against
  `D*(S-1) = 6` unknowns.

  Four places index the flat vector of `S-1` free basis coefficients per degree of freedom, and
  two of them — `initial_guess!` (`x[D*(i-1)+k]`) and `residual!` (`b[D + D*(i-1)+k]`) — already
  agreed on *degree of freedom fastest, basis index slowest*. The other two did not:

  - `components!` unpacked the solution as `C.X[s+1][d] = x[D*(d-1)+s]`, which is neither
    convention. At `D = 2, S = 4` that read `x[3]` twice and never read `x[6]`, leaving column 6
    of the Jacobian identically zero — the zero pivot. It now reads `x[D*(s-1)+d]`.
  - `update!` assigned `sol.q .= nlsolution(int)[end]`, broadcasting one scalar across all `D`
    components of `q`. It now takes the last basis coefficient per degree of freedom from `C.X`,
    which the basis being interpolatory at the end of the interval makes the new position.

  Both corrections are the identity at `D = 1`, so the existing `D = 1` guards are unchanged to
  the last bit. `test/unit/cgvi_unit.jl` gains a `D = 2` regression test built on
  `CoupledHarmonicOscillator` with the coupling parameter set to zero, which decouples it into two
  independent oscillators with different frequencies and different initial conditions, so each
  degree of freedom is checked against its own closed-form solution and a layout mistake cannot
  hide behind a merely worse number. Only this reference integrator was affected; the network
  integrators and `VISE` already handled `D > 1`.
- The package loads on Julia 1.13 again. `GenericLinearAlgebra` overwrites a `LinearAlgebra`
  method, which 1.13 forbids during precompilation, and that took `RungeKutta` and hence
  `GeometricIntegrators` down with it. Both have left the dependency graph.

- **`nbasis` resolved to an empty local function.** `CompactBasisFunctions` exports
  `basis`/`nbasis`, and a bare top-level definition under `using` creates a *new* function
  rather than extending the imported one. With definitions split between bare and qualified
  forms, every internal `nbasis(method(int))` call raised `MethodError`. Both names are now
  imported explicitly, so a bare definition anywhere extends the right generic.
- **`HermiteExtrapolation` was unusable, in four separate ways.** Widening the test matrix to
  the full (seed × extrapolation) cross product surfaced all of them; none was reachable
  before, because no test or benchmark drove that combination through these integrators.
  - `default_iguess` returned this package's `IntegratorExtrapolation` for `Hardcode_int`,
    `Time_Reversible_Hardcode` and `Time_reversible_OneLayer`. `iguess` is the *framework's*
    vocabulary — the extrapolation `GeometricIntegratorsBase.solutionstep!` applies — and it
    has methods only for the framework's own types, so all three raised `MethodError` on the
    first step. The two knobs are now kept distinct: `initial_trajectory_method` selects our
    code path, `iguess` is left to the framework.
  - The `soltmp` named tuple passed to `solutionstep!` used the field names `v`/`f`, but the
    `AbstractProblemIODE` method reads `q̇`/`ṗ` — a `FieldError` once dispatch succeeded.
    (`NonLinear_DenseNet_GML`'s override already had the right names, which is why it was the
    only one that got as far as a solve.)
  - `Hardcode_int` and `Time_Reversible_Hardcode` wrote the extrapolated positions into the
    *output-weight* slots of `x` and never populated `network_labels` — which is what the OGA
    seed reads. The seed therefore fitted the boundary ansatz to an all-zero target and
    overwrote the slots the extrapolation had just filled. They now fill `network_labels`.
  - Both also stored `p̃` in `x[D*S+k]`, which for the boundary ansatz is the endpoint
    *position* unknown, not the momentum; they now store `q̃` there, matching their own
    `IntegratorExtrapolation` methods.

  Two failures that looked like independent numerical fragilities were symptoms of the
  third item above: `OGA1dNormalEquations` raising `SingularException` (its Gram matrix was
  rank-deficient because the fit target was all zeros, not because of its κ(Φ)²
  conditioning) and `NonLinear_DenseNet_GML` raising `NaN detected in direction vector!` for
  both of its seeds. Both converge now.

  Note that a real Hermite warm start needs `initialguess = HermiteExtrapolation()` on the
  integrator as well: with the framework default (`NoInitialGuess`) `solutionstep!` is a
  no-op, so `initial_trajectory_method` alone selects the code path but extrapolates nothing.
  This is what the benchmark harness has always done, and the tests now do it too.
- `internal_values` in the shared `integrate!` is indexed from `n₁`, not from 1: a restart
  with `n₁ > 1` previously left the leading slots `#undef` and indexed past the end.
- `Time_reversible_Hardcode`'s `HermiteExtrapolation` override was missing a local binding
  for `network_inputs` (a latent `UndefVarError`).
- **`PR_Integrator` could not take a step.** `integrate_step!` called
  `solve!(solver, x, args)`, which matches no `SimpleSolvers.solve!` method — the argument
  order is `(x, solver, args)`, as in `CGVI_standard` and the shared `integrate_step!`. It
  went unnoticed because `test/unit/pr_integrator_unit.jl` existed but was never `include`d
  by `runtests.jl`; it now is.
- Dropped the `BSplineKit` and `Infiltrator` dependencies; neither was used by `src/`.
- `Project.toml` gained the missing `[compat]` entries for `julia`, `Logging` and `Test`.

- Replaced the hard-coded regularization/guard constants that were silently
  ineffective in reduced precision:
  - the `1e-12` dictionary-norm guard (which sat below `eps(Float32)` and so
    never fired) is now `oga_norm_floor(T, …)`;
  - the `Gk + 1e-12·I` and `Gk + 1e-14·I` Tikhonov ridges (which round away
    entirely below `eps(Float32)`) are replaced by the precision-scaled ridge in
    `weighted_lstsq`.
- The bias grid is built from an integer-indexed range cast to `T`, avoiding the
  `Float16` "`range step cannot be zero`" trap that occurred when a large
  `dict_amount` overflowed `T(dict_amount)` to `Inf`.
- Removed a stray `global xk_low` from the OGA fit in
  `Time_reversible_Hardcode_int.jl`.

### Known issues

- ~~`GeometricOptimizers` 0.2.0 is taken from git via `[sources]`~~ — **closed.**
  GeometricOptimizers 0.2.0 is in General, the `[sources]` section is gone, and this package can
  be tagged again.
- ~~**Julia 1.10 cannot install this package while that `[sources]` pin is present.**~~ —
  **closed** with the entry above. `[sources]` is a Pkg 1.11 feature that 1.10 ignores, which left
  `GeometricOptimizers = "0.2"` with no candidate; resolving from the registry gives it one. The
  three Julia 1.10 CI jobs should go green. Not verified locally — no 1.10 was run for this
  change.
- **The test suite takes hours on Julia 1.12** — 287 minutes in CI against ~20 on 1.13 and ~10
  on `main` before this release. Measured locally, a two-step `ShallowNet` run with
  `initial_guess_method = TrainingMethod()` takes over 30 minutes on 1.12 and 28.8 seconds on
  1.13; the same integrator with `OGA1d()`, which never touches GeometricOptimizers, takes 28.0
  seconds on 1.12. `LSGD` is affected as well, so it is not specific to `Adam`. The process sits
  in `jl_type_infer`, and `--trace-compile` stops emitting — an inference blowup, not numerical
  work.

  Not the same bug as GeometricOptimizers
  [#35](https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/35), which is in the manifest
  and does not help here. Standalone reproductions of the optimizer usage — construction and
  `solver_step!` in one body, `Adam` + `DecayingStatic`, the loss evaluating an
  `AbstractNeuralNetworks.Chain` at `NeuralNetworkParameters` under the default ForwardDiff
  gradient — all run in about 2 seconds on 1.12. The blowup only appears once that call sits
  inside the `integrate` call graph. Unresolved; 1.13 and 1.10 are unaffected.

- ~~**The documentation build cannot resolve its benchmark environment.**~~ — **closed** by
  GeometricIntegrators 0.18.2. `benchmark/Project.toml` depends on `GeometricIntegrators`, and
  every release up to 0.18.1 pinned `SimpleSolvers = "0.11"` against the 0.12.1 this package now
  requires, so `Pkg` reported *"Unsatisfiable requirements detected for package SimpleSolvers"*
  and the *Install benchmark dependencies* step failed. Nothing in this package could close it —
  the step is not removable, `docs/make.jl` running the `quick` benchmark sweep in that
  environment to regenerate the figures the Benchmarks page embeds, and a git `[sources]` pin
  would not have helped either, GeometricIntegrators `main` having carried the same 0.11 bound.

  0.18.2 requires `GeometricIntegratorsBase = "0.6.3 - 0.6"` and `SimpleSolvers = "0.12.1 - 0.12"`,
  which is exactly this package's pair, and the benchmark environment resolves against it. The job
  had been red on `main` too, by way of the git-pinned GeometricOptimizers, so this was inherited
  rather than introduced by the SimpleSolvers bump.

## Open Issues

Not a release section: this is a standing list of known defects that are understood but not
fixed, so that reviewing the same code twice does not rediscover them. Entries move into a
release's `Fixed` when they are dealt with.

### Environment

The first four are described in full under [0.2.0] → *Known issues*; they are listed here only
so that this section is the complete index.

- ~~**The git `[sources]` pin blocks registration in General.**~~ **Closed**: GeometricOptimizers
  0.2.0 is registered and the pin is gone.
- ~~**Julia 1.10 cannot install the package**~~ **Closed** with the entry above, for the same
  reason. Not verified on a 1.10 locally.
- **Julia 1.12 spends hours in type inference** on the GeometricOptimizers-driven initial-guess
  methods. Unlike the two above it does not clear itself, and it is the only one of that trio
  that is a genuine defect rather than a consequence of depending on an unregistered package.
- ~~**The docs CI job cannot resolve `benchmark/Project.toml`**~~ — **closed** by
  GeometricIntegrators 0.18.2, which requires SimpleSolvers 0.12.1 rather than the 0.11 that
  0.18.1 and everything before it pinned.
- **The Julia 1.13 CI test phase roughly doubled**, from about 10 minutes for the whole job on
  `main` to 19m57s of tests here, and nothing found so far explains it. No local baseline is
  available to compare against: `main`'s environment no longer resolves, which is what this
  branch exists to fix, so the only local number is 8m19s for this branch — which matches what
  the branch reports and says nothing about the delta. Worth re-measuring once `main` carries
  these changes and a `main` baseline can be taken again.

### Upstream

- ~~**`SymbolicNeuralNetworks` 0.6 costs 1.85x the allocations in the symbolic `residual!` path,
  on Julia 1.10 only.**~~ **Closed** in [0.4.1] by
  [SymbolicNeuralNetworks#55](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/55).
  Per `residual!` call at Float64, `S = 4`, `R = 8`, `D = 1`, `ShallowNet` and
  `ShallowNetReversible` went from 15 168 bytes under `AbstractNeuralNetworks` 0.6.4 / SNN 0.5 to
  28 096 under 0.7 / 0.6, while 1.11, 1.12 and 1.13 stayed at 11 424 either way.

  This report was right about where it was not — not the `NetworkParameters` rename, not the two
  `*Autodiff` rows, and on the path SNN 0.6 laid out over `NeuralNetworkParameters.ParameterLayout`
  in place of a local `FlatSlice` — and did not isolate the cause. It was a `map` over a closure
  that Julia 1.10 does not always elide, on the walk that splits a generated function's flat result
  back into the nesting of the parameters, and it is fixed in two independent halves:
  `SymbolicNeuralNetworks` 0.7.0 for the batched walk and `NeuralNetworkParameters` 0.2.1 for the
  un-batched one. This package's symbolic `residual!` calls `DQDθ` on a length-one `Vector`, so it
  takes the un-batched path and it is the 0.2.1 half that returns it to 15 168 — confirmed here on
  1.10.11, 15 168 against 28 096 for the same probe under the old pair. Both are named in
  `Project.toml`; the 1.10-only ceiling of 42 000 is gone and 17 000 holds on every version.

- **`GeometricOptimizers.GradientMethod` cannot be used with a searching line search** on
  Euclidean parameters: `_trial_slope` calls `gradient(cache)` while the first-order caches
  expose `gradient_array`, so it throws `MethodError: no method matching
  gradient(::GradientCache)` — including via the `default_linesearch` that `Optimizer` picks
  when none is given. Recorded under *Not fixed here* in
  [GeometricOptimizers#35](https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/35). The
  LSGD loop in `src/nvi/densenet.jl` works only because it passes `Static` explicitly, and there
  is a comment at that call site saying so.

  **Not re-checked against SimpleSolvers 0.12**, which rewrote the very contract this entry lives
  in: a `LinesearchMethod` now implements `solve_with_status` and gets `solve` derived from it, and
  the generic `solve_with_status` raises rather than deriving itself from `solve`. That is a change
  to how a searching line search is *reached*, not to what `_trial_slope` asks a cache for, so the
  entry is expected to still hold — but nothing here exercises it, because `Static` is still passed
  explicitly. Treat as unverified rather than as confirmed.

- **A GeometricOptimizers optimizer has no `solve_with_status!`.** SimpleSolvers grew one for its
  nonlinear solvers in 0.11 and a state-taking form in 0.12.1, and this package now uses it
  throughout; the optimizer side has no counterpart. `solve!(x, state, opt)` does return an
  `OptimizerResult` carrying the outcome, which is what the `ShallowNet` seeding loop now reads,
  so nothing is unreachable — but `OptimizerResult` and `OptimizerStatus`, and every accessor the
  seeding loop reads them through (`status`, `isconverged`, `iteration_number`), are none of them
  exported by GeometricOptimizers, so all four call sites reach past the exported surface
  (`GeometricOptimizers.status(result)` and friends). Either exported accessors or a
  `solve_with_status!` would close it.

- **The two `DenseNet` seeding loops check no optimizer status.** They drive the optimizer by hand
  (`increase_iteration_number!` / `solver_step!` / `update!`) because `solve!` has no hook for
  what each of them does inside its epoch — an early exit on the loss for `TrainingMethod`, a
  least-squares re-solve of the `L3` layer for `LSGD`. So the only thing either of them tests is
  its own loss threshold (`5e-8` and `5e-5`, which are also the un-scaled `Float64` literals
  recorded under *Training loops and losses*), and a non-finite iterate or a diverging optimizer
  is invisible to both. Closing this needs either those hooks upstream or a status assembled by
  hand from the state.

### Training loops and losses

All of these predate the move to GeometricOptimizers; none is a regression.

- **`mse_loss` is not the mean squared error.** It returns `mean(abs, y_pred - y)`, the mean
  *absolute* error — as its own docstring says. Renaming it touches every training call site,
  so it is left alone rather than changed silently; whichever way it is resolved, the name and
  the formula should agree.
- **`mse_loss`'s `μ` keyword is unused**, and its `λ` defaults to `0.0`, which switches off the
  boundary penalty `λ * |NN(x[1], ps) - y[1]|²` that is the only thing `λ` and `μ` are there
  for. No call site passes either, so the penalty is dead code at present.
- **The early-exit thresholds are Float64 literals**: `err < 5e-8` for `TrainingMethod` and
  `err < 5e-5` for `LSGD` in `src/nvi/densenet.jl`. Neither is scaled to the working
  precision, so at `Float32` — where `eps` is 1.2e-7 — the `TrainingMethod` exit is below the
  accuracy a network fit can reach and the loop always runs the full epoch budget. They should
  derive from `eps(PT)` the way the OGA guards now do.
- **`ShallowNet`'s `TrainingMethod` has no early exit at all**, where the `DenseNet` one does.
  That may well be deliberate, but the asymmetry is undocumented.
- **`box_init_plain` defaults to `Float32`** and the three LSGD call sites take that default,
  so a `Float64` DenseNet is seeded from `Float32` random draws that are then widened on
  assignment. The suite's no-silent-upcast gate does not catch it, being a downcast of the
  *seed* rather than of the solution. It should take the working precision, as
  `simpson_quadrature` and the OGA dictionaries do.
- **`DenseNet`'s `TrainingMethod` passes the whole `NeuralNetwork` to `mse_loss`** where
  `ShallowNet` passes the bare model, so the loss closure captures more than it needs. Both
  work; they should agree.

### Loops and allocation

- **Four loop-invariant assignments sit inside `for i in 1:S₁`** in
  `src/nvi/densenet.jl`, in both `initial_params!` methods, in `components!` and in
  `record_finer_solution!`. Only the `ps[k].L2.W[:, i]` line depends on `i`; the other four
  slices are rewritten identically `S₁` times. `components!` runs on every residual evaluation,
  so this is on the Newton path.
- **`flatten_params` accumulates into an untyped `Vector{Any}`** and finishes with
  `vcat(flat_list...)`, a splat whose length is not known to the compiler. `components!` calls
  it `2 + 2R` times per dimension, `R` being the number of quadrature nodes — also on the
  Newton path.

### Derivative evaluation

Surfaced while updating to `SymbolicNeuralNetworks` 0.4 and writing
`benchmark/compare_derivative_backends.jl`; re-checked against 0.5.

- **The compiled kernels are still called once per quadrature node.** `components!` evaluates
  `DQDθ`/`DVDθ` node by node — the loops at `src/nvi/shallownet.jl:274-293`,
  `src/nvi/shallownet_reversible.jl:223-242` and `src/nvi/densenet.jl:406-418` — which is
  `2R + 2` calls per dimension per Newton iteration, sixteen of them at the benchmark's
  `R = 8`. `SymbolicNeuralNetworks` evaluates a whole batch through one in-place kernel and a
  single allocation, so the same work is two calls if the nodes are passed as one batch. The
  kernel benchmark puts the per-call cost at 0.042–0.167 µs and the per-call allocation at
  528–1136 B (Float64, re-measured under 0.5), so the saving is a constant factor on the
  Newton path rather than an order of magnitude. Deliberately out of scope for both updates;
  it needs the derivative bookkeeping in `components!` reindexed, and `unflatten`'s batch
  layout (`m × (n·N)`, column-major) worked into the slicing.

- **The autodiff pair computes the velocity with `Zygote` and its parameter gradient with
  `ForwardDiff`,** for the same expression. `VNN_ansatz_zygote`
  (`src/nvi/shallownet_autodiff.jl:228`) is what fills `V` at the quadrature nodes
  (`shallownet_autodiff.jl:341`, `shallownet_autodiff_reversible.jl:351`), while
  `∂VNN_ansatz_∂params` differentiates the `ForwardDiff` version, `VNN_ansatz`
  (`shallownet_autodiff.jl:230-232`). Both compute `dq_h/dt`. Reverse mode for a
  scalar-in/scalar-out derivative is the wrong tool, and the mismatch is at odds with the
  `Autodiff` name, which the 0.3.0 rename introduced to mean `ForwardDiff`. Switching the
  value to `VNN_ansatz` looks like a one-line change; it is untested and would move the
  numbers, so it is not one to make blind.

### Nonlinear solve conditioning

- **The residual floors above the convergence tolerance, so which iterate Newton accepts
  depends on last-bit differences.** Measured by running the symbolic integrators under both
  code-generation settings, which compute the same derivative to machine epsilon (≤8e-17 at
  Float64, verified in `benchmark/results/derivative_backends_codegen_agreement.csv` and in
  `test/unit/dispatch_variants_unit.jl`): end to end, 3 of 8 paired `ShallowNet` cases and 4
  of 8 `ShallowNetReversible` cases stop after a different number of iterations, and `ref_err`
  moves by up to 200×. On the harmonic oscillator at Float64, `dt = 1`, both autodiff
  integrators run the full 1000-iteration budget to a residual of 5e-12 / 1e-11 and are
  recorded as `maxiter`. This is the same phenomenon `SimpleSolvers` reports in its give-up
  warning — a floor of the discretisation that no eps-scaled tolerance can bound. Two
  consequences worth writing down: accuracy comparisons between configurations that differ
  only in round-off are not meaningful at this level, and a per-problem convergence tolerance
  above the floor would be more honest than burning the iteration budget.

### Dead code and documentation

- **`default_iparams` is defined for three integrators and called nowhere.**
  `src/nvi/shallownet_autodiff.jl:47`, `src/nvi/shallownet_reversible.jl:58` and
  `src/nvi/shallownet_autodiff_reversible.jl:55` each declare it; nothing in `src/`, `test/`,
  `scripts/`, `benchmark/` or `docs/` reads it. The values duplicate the defaults the
  constructors already carry, so it is documentation in code form that nothing keeps honest.
  Either wire it into the constructors as *the* source of the default, or drop it.

- **The four analytic boundary derivatives of the hardcoded ansatz are called nowhere.**
  `∂NN_ansatz_∂q̄`, `∂NN_ansatz_∂q`, `∂VNN_ansatz_∂q̄` and `∂VNN_ansatz_∂q`
  (`src/nvi/shallownet_autodiff.jl:234-238`) return `1-t`, `t`, `-1` and `1` — the exact
  derivatives of `q_h` and `dq_h/dt` with respect to the two endpoint unknowns — and nothing
  in `src/`, `test/`, `scripts/`, `benchmark/` or `docs/` reads them. All four are written out
  by hand elsewhere: `residual!` spells the two `∂/∂q̄` ones into the `p̄` row of the residual
  (`shallownet_autodiff.jl:390-391`, `shallownet_autodiff_reversible.jl:400-401`) and
  `update!` spells the two `∂/∂q` ones into the momentum update
  (`shallownet_autodiff.jl:437-438`, `shallownet_autodiff_reversible.jl:446-447`, where the
  `∂VNN/∂q = 1` factor is left implicit). Neither is `components!`, and neither assembles a
  Jacobian — the solver differentiates one out of `residual!`. Either call the helpers at
  those four sites or drop them. Surfaced while renaming `*_anstaz_*` to `*_ansatz_*`, which
  had to touch all four.

  Two details that bear on which way it goes. Their signature is the reason nothing calls
  them: all four take `(ps, S, activation, t, q̄, q)` and read only `t`, so at the four sites
  above a six-argument call would replace an expression as short as `1 - t`, five of whose
  arguments are there to be discarded. Reviving them means fixing the signature first. And
  `∂NN_ansatz_∂q̄` is written `one(t) .- t`, broadcasting where its three scalar siblings do
  not — harmless on a scalar `t`, but it is the kind of drift a definition nothing exercises
  accumulates.

- **`src/nvi/shallownet_autodiff_reversible.jl:218-244` is a commented-out duplicate** of the
  ansatz definitions that live, uncommented, in `shallownet_autodiff.jl:212-238`. It is
  already stale: it still spells the boundary factors `1.0 - t` where the live copy uses
  `one(t) - t`, so it predates the precision-generic refactor and would silently upcast if
  it were ever uncommented. It also has to be hand-edited to keep it in step — the
  `*_anstaz_*` rename did exactly that, for a block no compiler checks. It should be deleted;
  the reversible integrator gets these functions from the module, not from this block.

- **`ShallowNetAutodiff` and `ShallowNetAutodiffReversible` have drifted apart in two spots
  where they should read identically.** The two integrators are near-copies of each other, so
  every gratuitous difference is a place a reader has to stop and work out whether it is
  meaningful. Neither of these is:

  - `update!` initialises its accumulator as `zero(eltype(sol.p))` in
    `src/nvi/shallownet_autodiff.jl:434` and as `zero(DT)` in
    `src/nvi/shallownet_autodiff_reversible.jl:443`. Same type, two spellings; `zero(DT)` is
    the one that says where the type comes from.
  - The two `show_status ? println(...)` residual dumps at the end of `residual!` are live in
    `shallownet_autodiff_reversible.jl:428-429` and commented out in
    `shallownet_autodiff.jl:419-420`. `show_status` defaults to `false`, so nothing prints
    either way, but the pair should agree on whether the facility exists.

  Both surfaced while reviewing the `*_anstaz_*` → `*_ansatz_*` rename, which read the two
  files side by side.

- **`docs/src/index.md` renders past Documenter's `size_threshold_warn`** (118 KiB against
  100 KiB), warning on every build. Still well under the 200 KiB hard threshold. It wants
  splitting into per-family pages, which is a docs reorganisation rather than a fix.

### Symbolic derivatives

Surfaced while updating to `SymbolicNeuralNetworks` 0.5.

- **The first basis construction in a process costs about 9.3 s**, with `cse`/`inplace` on or
  off (9.31 s against 9.63 s for `DenseNetBasis{Float64}(tanh, 3, 3)`). Practically all of it
  is compiling the code-generation machinery rather than generating code: the same build warm
  is 27 ms with the defaults and 110 ms without them. That latency is upstream and not
  something this package can fix, but it dominates what a user actually pays for the first
  basis, and it is two orders of magnitude above the warm figures the docstrings quote. The
  docstrings now say they are warm measurements; a note in the user-facing documentation
  would be more use than a note here.

- **The 0.5 kernel numbers were measured in `quick` mode only.** That tier sweeps Float64 and
  Float32 over two problems, so 0.5's effect at Float16 and on the double pendulum is
  unmeasured. Float16 is the tier where this package has been bitten before — it has its own
  regression test in `test/integration/` precisely because the OGA/Newton path is
  ill-conditioned there — and 0.5 changed both the emitted code and the allocation of the
  in-place result, whose element type comes from the *inputs*. A `full` run would settle it.

- **The build figures in the *`SymbolicNeuralNetworks` 0.3 → 0.4* entry above (3.22 s → 0.79 s)
  are not comparable to the warm figures now quoted in the docstrings** (110 ms → 27 ms for the
  same basis). The ratio is the same ~4× and that is the claim both make, but the absolute
  numbers differ by a factor of thirty and nothing records how the older pair was measured —
  most likely a first build in a fresh process, i.e. mostly the compile latency of the entry
  above. Left as written, since it is the record of what was measured then; re-stating it would
  be inventing a measurement that was never taken.

- **`V_func` returns a 1×1 matrix that both of its call sites immediately unwrap.** The shape
  is an honest consequence of the Jacobian of a scalar-in/scalar-out network being 1×1, but it
  carries no information: the one integrator that consumes it (`src/nvi/densenet.jl:440`)
  strips it with `[1]`, and so do both kernel testsets in
  `test/unit/dispatch_variants_unit.jl`, one with `vec` and one with `[1]`.
  `ShallowNetBasis` builds the slot as well, and no integrator reads that one at all. Either
  build it from `VNN[1,1]` like the two gradients now are, which would make it return a scalar
  — `build_nn_function` does accept a scalar expression — or leave it and note why; but the
  three derivative slots should not disagree about whether they are scalars.

### Reviewing the 0.5 update

Surfaced while reviewing the `SymbolicNeuralNetworks` 0.5 update, after the fixes that review
did make. None of these is a defect in the update: the compiled kernels were checked against
`ForwardDiff` off the integrator and agree to round-off at both bases, both codegen settings
and both precisions (≤ 7.2e-16 at Float64, ≤ 9.4e-8 at Float32, with `dvdθ` flattening to
exactly `NP`), which is the check now in `test/unit/dispatch_variants_unit.jl`.

- **The `29 ns` that the `symbolic = false` build is measured at is a measurement of nothing.**
  `ShallowNetBasis{Float64}(tanh, 8; symbolic = false)` allocates 0 bytes, and a million
  constructions in a loop take 2.9 µs in total — the compiler elides them. `Dense` carries its
  dimensions in type parameters and stores only the activation, so a derivative-free basis puts
  nothing on the heap; the parameters do not exist until `NeuralNetwork(NN, T)` is called. The
  figure is therefore harness overhead, and it sits an order of magnitude below the ~0.042 µs
  timer resolution the kernel table already flags. What the docstring
  (`src/nvi/shallownet_basis.jl`) and the *Added* entry above should say is that the build costs
  15 ms and the opt-out costs nothing at all — the "pure overhead" claim they make is right, it
  is only the second number that pretends to be a cost. Left as measured rather than silently
  restated.

- **The *0.3 → 0.4* comparability note above names only the `DenseNetBasis` pair.** The same
  applies to the other figures in that entry — `ShallowNetBasis` construction "1.6–2.0× faster",
  and 0.24 s against 0.017 s for generating the in-place kernel at `S = 8` — which are quoted
  with no record of how they were taken either, and which the warm re-measurement puts three
  orders of magnitude away. The entry should either scope its caveat to the whole 0.3 → 0.4
  entry or the figures should be re-taken as a set.

- **`benchmark/results/` is gitignored, so the CSV the *Nonlinear solve conditioning* entry
  cites is not in the repository.** `benchmark/results/.gitignore` is `*` plus `!.gitignore`,
  which is right — those files are generated — but it makes
  `benchmark/results/derivative_backends_codegen_agreement.csv` a citation only the person who
  last ran the benchmark can follow. Either name the run that produces it or quote the number
  and drop the path.

- **The new `ForwardDiff` cross-check is a point check, not a sweep.** One time input
  (`t = 0.37`), one random parameter draw per basis, the default code generation only, and
  `TEST_TYPES`, so no Float16. That is enough for what it was added for — a wrong shape or a
  wrong expression shows up immediately, which is the failure mode the 0.5 rename could have
  produced — but an expression that happens to agree at that one point would pass. Sweeping a
  few nodes, or reusing the quadrature nodes the integrators actually evaluate at, would cost
  almost nothing.

- ~~**`Project.toml`'s `AbstractNeuralNetworks = "0.6.4"` is a floor this package does not need.**~~
  Neither `input_dimension` nor `output_dimension` was called anywhere here; the constraint that
  forced 0.6.4 was `SymbolicNeuralNetworks` 0.5's own, and the resolver applied it whether or not
  the entry existed. **Closed** by [0.4.0]: the bound is now `0.7`, and it is this package's own —
  `NetworkParameters` is named directly in `src/`, so the entry states a requirement rather than
  documenting a coupling.
