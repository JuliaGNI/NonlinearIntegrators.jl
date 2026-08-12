# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-11

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
  version is bumped to `0.3.0` so that a downstream `[compat]` bound on `0.2` fails at
  resolve time with a clear message, rather than at run time with an `UndefVarError`.
  Known downstream: SolverBenchmark's `nonlinear_onelayer_method` defaults to this seed.
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

### Changed

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

### Added

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

### Fixed

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
