# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking

- **`OGA1d_Legacy` is renamed `OGA1dNormalEquations`.** The new name says what the
  variant *is* — the reference implementation from the original paper, solving the fit
  through the normal equations in a `Float64` island — rather than merely that it came
  first. Callers passing `initial_guess_method = OGA1d_Legacy()` must update; there is no
  deprecation shim: the type only ever existed on `main`, never in a tagged release. The
  version is bumped to `0.3.0` so that a downstream `[compat]` bound on `0.2` fails at
  resolve time with a clear message, rather than at run time with an `UndefVarError`.
  Known downstream: SolverBenchmark's `nonlinear_onelayer_method` defaults to this seed.
- `initial_params!` is unified on the three-argument form `initial_params!(int, method,
  sol)`. The two boundary-ansatz integrators already needed `sol`; the two `OneLayer` ones
  and `NonLinear_DenseNet_GML` took two arguments, so the seed could not be written
  generically across all of them. Only relevant to code defining its own
  `InitialParametersMethod`.

### Changed

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
