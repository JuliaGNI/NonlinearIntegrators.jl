# Known issues

## Environment

### K1 · Julia 1.12 spends hours in type inference

- location: —
- evidence: on the GeometricOptimizers-driven initial-guess
  methods. Unlike the two above it does not clear itself, and it is the only one of that trio
  that is a genuine defect rather than a consequence of depending on an unregistered package.
- kind: defect
- found: 2026-08-14; described in full under [0.2.0] → *Known issues* in `CHANGELOG.md`

### K2 · The Julia 1.13 CI test phase roughly doubled

- location: —
- evidence: , from about 10 minutes for the whole job on
  `main` to 19m57s of tests here, and nothing found so far explains it. No local baseline is
  available to compare against: `main`'s environment no longer resolves, which is what this
  branch exists to fix, so the only local number is 8m19s for this branch — which matches what
  the branch reports and says nothing about the delta. Worth re-measuring once `main` carries
  these changes and a `main` baseline can be taken again.
- kind: defect
- found: 2026-08-14

## Upstream

### K3 · `GeometricOptimizers.GradientMethod` cannot be used with a searching line search

- location: `src/nvi/densenet.jl`
- evidence: on
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
- kind: upstream
- found: GeometricOptimizers#35

### K4 · A GeometricOptimizers optimizer has no `solve_with_status!`.

- location: —
- evidence: SimpleSolvers grew one for its
  nonlinear solvers in 0.11 and a state-taking form in 0.12.1, and this package now uses it
  throughout; the optimizer side has no counterpart. `solve!(x, state, opt)` does return an
  `OptimizerResult` carrying the outcome, which is what the `ShallowNet` seeding loop now reads,
  so nothing is unreachable — but `OptimizerResult` and `OptimizerStatus`, and every accessor the
  seeding loop reads them through (`status`, `isconverged`, `iteration_number`), are none of them
  exported by GeometricOptimizers, so all four call sites reach past the exported surface
  (`GeometricOptimizers.status(result)` and friends). Either exported accessors or a
  `solve_with_status!` would close it.
- kind: upstream
- found: 2026-08-16

### K5 · The two `DenseNet` seeding loops check no optimizer status.

- location: —
- evidence: They drive the optimizer by hand
  (`increase_iteration_number!` / `solver_step!` / `update!`) because `solve!` has no hook for
  what each of them does inside its epoch — an early exit on the loss for `TrainingMethod`, a
  least-squares re-solve of the `L3` layer for `LSGD`. So the only thing either of them tests is
  its own loss threshold (`5e-8` and `5e-5`, which are also the un-scaled `Float64` literals
  recorded under *Training loops and losses*), and a non-finite iterate or a diverging optimizer
  is invisible to both. Closing this needs either those hooks upstream or a status assembled by
  hand from the state.
- kind: upstream
- found: 2026-08-16

### K33 · `fatou lint` reports false `parse-error` findings

- location: `test/quality/inference_and_allocations.jl:77-79`
- evidence: Measured with fatou 0.20.0: 6 `parse-error` findings, all in
  `test/quality/inference_and_allocations.jl`. fatou reads `(` followed by a macro call with a
  `for` body, e.g. `(@allocated for _ in 1:n … end) ÷ n`, as an unclosed generator, and one site
  then cascades into several findings ("unclosed comprehension", "trailing tokens after
  statement", "unexpected block keyword", "expected `end`"). Julia parses the file without error
  and the suite runs it. The finding has severity `error` and a `# fatou-ignore` comment does not
  suppress it, so the pre-commit hook shows it on each commit that stages this file. The code is
  correct; do not rewrite it to satisfy the linter.
- kind: upstream
- found: 2026-09-13

### K34 · Revise prints EMFILE errors in the test log

- location: `test/quality/jet.jl`
- evidence: JET 0.12 loads Revise, which stays loaded for the rest of the process, and its file
  watcher runs out of file handles. The full suite, `Pkg.test()` on Julia 1.13.1, logged 8
  `UNHANDLED TASK ERROR` blocks, all `FolderMonitor` EMFILE, with the `quality/jet.jl` testset
  added, and 0 on `origin/main` eb124b7, where JET 0.10 has Revise only as a weak dependency. The
  per-file totals are the same on both trees, except for the new `JET` testset (4 pass) and the
  `Aqua` testset, whose skipped JET test left it (10 pass, 1 broken on the base; 10 pass here).
- kind: upstream
- found: 2026-09-28

## Training loops and losses

All of these predate the move to GeometricOptimizers; none is a regression.

### K6 · `mse_loss` is not the mean squared error.

- location: —
- evidence: It returns `mean(abs, y_pred - y)`, the mean
  *absolute* error — as its own docstring says. Renaming it touches every training call site,
  so it is left alone rather than changed silently; whichever way it is resolved, the name and
  the formula should agree.
- kind: defect
- found: 2026-08-14

### K7 · `mse_loss`'s `μ` keyword is unused

- location: —
- evidence: , and its `λ` defaults to `0.0`, which switches off the
  boundary penalty `λ * |NN(x[1], ps) - y[1]|²` that is the only thing `λ` and `μ` are there
  for. No call site passes either, so the penalty is dead code at present.
- kind: dead code
- found: 2026-08-14

### K8 · The early-exit thresholds are Float64 literals

- location: `src/nvi/densenet.jl`
- evidence: : `err < 5e-8` for `TrainingMethod` and
  `err < 5e-5` for `LSGD` in `src/nvi/densenet.jl`. Neither is scaled to the working
  precision, so at `Float32` — where `eps` is 1.2e-7 — the `TrainingMethod` exit is below the
  accuracy a network fit can reach and the loop always runs the full epoch budget. They should
  derive from `eps(PT)` the way the OGA guards now do.
- kind: defect
- found: 2026-08-14

### K9 · `ShallowNet`'s `TrainingMethod` has no early exit at all

- location: —
- evidence: , where the `DenseNet` one does.
  That may well be deliberate, but the asymmetry is undocumented.
- kind: docs
- found: 2026-08-16

### K10 · `box_init_plain` defaults to `Float32`

- location: —
- evidence: and the three LSGD call sites take that default,
  so a `Float64` DenseNet is seeded from `Float32` random draws that are then widened on
  assignment. The suite's no-silent-upcast gate does not catch it, being a downcast of the
  *seed* rather than of the solution. It should take the working precision, as
  `simpson_quadrature` and the OGA dictionaries do.
- kind: defect
- found: 2026-08-14

### K11 · `DenseNet`'s `TrainingMethod` passes the whole `NeuralNetwork` to `mse_loss`

- location: —
- evidence: where
  `ShallowNet` passes the bare model, so the loss closure captures more than it needs. Both
  work; they should agree.
- kind: defect
- found: 2026-08-16

## Loops and allocation

### K12 · Four loop-invariant assignments sit inside `for i in 1:S₁`

- location: `src/nvi/densenet.jl`
- evidence: in
  `src/nvi/densenet.jl`, in both `initial_params!` methods, in `components!` and in
  `record_finer_solution!`. Only the `ps[k].L2.W[:, i]` line depends on `i`; the other four
  slices are rewritten identically `S₁` times. `components!` runs on every residual evaluation,
  so this is on the Newton path.
- kind: defect
- found: 2026-08-14

### K13 · `flatten_params` accumulates into an untyped `Vector{Any}`

- location: —
- evidence: and finishes with
  `vcat(flat_list...)`, a splat whose length is not known to the compiler. `components!` calls
  it `2 + 2R` times per dimension, `R` being the number of quadrature nodes — also on the
  Newton path.
- kind: defect
- found: 2026-08-14

## Derivative evaluation

Surfaced while updating to `SymbolicNeuralNetworks` 0.4 and writing
`benchmark/compare_derivative_backends.jl`; re-checked against 0.5.

### K14 · The compiled kernels are still called once per quadrature node.

- location: `src/nvi/shallownet.jl:274-293`
- evidence: `components!` evaluates
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
- kind: defect
- found: 2026-08-14

### K15 · The autodiff pair computes the velocity with `Zygote` and its parameter gradient with `ForwardDiff`,

- location: `src/nvi/shallownet_autodiff.jl:228`
- evidence: for the same expression. `VNN_ansatz_zygote`
  (`src/nvi/shallownet_autodiff.jl:228`) is what fills `V` at the quadrature nodes
  (`shallownet_autodiff.jl:341`, `shallownet_autodiff_reversible.jl:351`), while
  `∂VNN_ansatz_∂params` differentiates the `ForwardDiff` version, `VNN_ansatz`
  (`shallownet_autodiff.jl:230-232`). Both compute `dq_h/dt`. Reverse mode for a
  scalar-in/scalar-out derivative is the wrong tool, and the mismatch is at odds with the
  `Autodiff` name, which the 0.3.0 rename introduced to mean `ForwardDiff`. Switching the
  value to `VNN_ansatz` looks like a one-line change; it is untested and would move the
  numbers, so it is not one to make blind.
- kind: defect
- found: 2026-08-14

## Nonlinear solve conditioning

### K16 · The residual floors above the convergence tolerance, so which iterate Newton accepts depends on last-bit differences.

- location: `benchmark/results/derivative_backends_codegen_agreement.csv`
- evidence: Measured by running the symbolic integrators under both
  code-generation settings, which compute the same derivative to machine epsilon (≤8e-17 at
  Float64, verified in `benchmark/results/derivative_backends_codegen_agreement.csv` and in
  `test/nvi/dispatch_variants_unit.jl`): end to end, 3 of 8 paired `ShallowNet` cases and 4
  of 8 `ShallowNetReversible` cases stop after a different number of iterations, and `ref_err`
  moves by up to 200×. On the harmonic oscillator at Float64, `dt = 1`, both autodiff
  integrators run the full 1000-iteration budget to a residual of 5e-12 / 1e-11 and are
  recorded as `maxiter`. This is the same phenomenon `SimpleSolvers` reports in its give-up
  warning — a floor of the discretisation that no eps-scaled tolerance can bound. Two
  consequences worth writing down: accuracy comparisons between configurations that differ
  only in round-off are not meaningful at this level, and a per-problem convergence tolerance
  above the floor would be more honest than burning the iteration budget.
- kind: defect
- found: 2026-08-14

### K35 · The Newton Jacobian of the ShallowNet integrators is exactly rank deficient at every converged point, by construction of the ansatz.

- location: `src/nvi/shallownet.jl`
- evidence: `julia --startup-file=no --project=scripts scripts/newton_jacobian_rank.jl` reports
  rank 5 of 13 unknowns at `S = 4`, with a gap of fourteen orders between the fifth and sixth
  singular values, and deficiencies of 8, 9 and 10 on the converged `OGA1d` rows at
  `S = 4, 5, 6`. The script prints the kinks `τᵢ = −bᵢ/wᵢ` and the weights `cᵢ` of each row,
  and they show three mechanisms.

  - **The scaling gauge.** `relu_k` is positively homogeneous, so `(wᵢ, bᵢ, cᵢ) ↦ (λwᵢ, λbᵢ,
    λ⁻ᵏcᵢ)` leaves the ansatz unchanged: `S` null directions.
  - **The polynomial collapse.** Where every pre-activation keeps one sign across the element,
    `max(0, z)ᵏ = zᵏ` and the neuron is a cubic. `OGA1d` at `S = 4` has its kinks at −3.14,
    3.14, −1.65 and 1.35, all outside `[0, 1]`: rank 5 is the cubic's 4 coefficients and `p̃`,
    and the step is the cubic Galerkin step.
  - **Dead neurons.** `OGA1d` at `S = 5, 6` keeps the same four outside kinks and adds neurons
    with kinks inside at 0.499 and 0.314 but `cᵢ` of 1e-25 … 1e-27, which leaves their `wᵢ` and
    `bᵢ` free.

  The converged `OGA1dNormalized` rows at `S = 5, 6` have three and five live kinks inside the
  element (`cᵢ` up to 1.2e4) and are deficient by 4 and 7, close to the gauge alone. In this
  configuration `OGA1dStable` does not converge at any `S`, and `OGA1dNormalized` does not at
  `S = 4`; those rows are not evidence.

  The solve is not affected: `initsolver` gives every network integrator
  `SimpleSolvers.PivotedQR`, which returns the minimum-norm step and cannot raise. What the
  deficiency costs is that `3S + 1` unknowns carry far fewer degrees of freedom; on the `OGA1d`
  rows, the 5 of a cubic step.

  `scripts/free_knot_rank.jl` tests the design that would remove all three: `w = +1`, an explicit
  cubic part, and the kinks as free knots inside the element. With Gauss 8 on the element, which
  ShallowNet uses, the knots die (`cᵢ ≤ 1e-12`) or fit the quadrature error, and the step is the
  cubic's or worse. With a Gauss rule on each piece between knots there is an accurate,
  nondegenerate stationary point — `S = 4`, endpoint error 1e-13 against 6e-10 for the cubic, no
  gap in the spectrum — but Newton reaches it only from uniform knots. From the OGA-like greedy
  seeds the knots merge, sit at 0 or 1, or leave the element; at best one interior knot survives
  near τ ≈ 0.5, with error 1.6e-11 and the rank still lost. A fix needs that piecewise
  quadrature, the explicit cubic, the gauge, and a solve or a seed that reaches the accurate
  point — a globalised step, descent on the action for example.
- kind: defect
- found: 2026-10-16

## Dead code and documentation

### K18 · `default_iparams` is defined for three integrators and called nowhere.

- location: `src/nvi/shallownet_autodiff.jl:47`
- evidence: `src/nvi/shallownet_autodiff.jl:47`, `src/nvi/shallownet_reversible.jl:58` and
  `src/nvi/shallownet_autodiff_reversible.jl:55` each declare it; nothing in `src/`, `test/`,
  `scripts/`, `benchmark/` or `docs/` reads it. The values duplicate the defaults the
  constructors already carry, so it is documentation in code form that nothing keeps honest.
  Either wire it into the constructors as *the* source of the default, or drop it.
- kind: dead code
- found: 2026-08-14

### K19 · The four analytic boundary derivatives of the hardcoded ansatz are called nowhere.

- location: `src/nvi/shallownet_autodiff.jl:234-238`
- evidence: `∂NN_ansatz_∂q̄`, `∂NN_ansatz_∂q`, `∂VNN_ansatz_∂q̄` and `∂VNN_ansatz_∂q`
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
- kind: dead code
- found: 2026-08-14

### K20 · `src/nvi/shallownet_autodiff_reversible.jl:218-244` is a commented-out duplicate

- location: `src/nvi/shallownet_autodiff_reversible.jl:218-244`
- evidence: of the
  ansatz definitions that live, uncommented, in `shallownet_autodiff.jl:212-238`. It is
  already stale: it still spells the boundary factors `1.0 - t` where the live copy uses
  `one(t) - t`, so it predates the precision-generic refactor and would silently upcast if
  it were ever uncommented. It also has to be hand-edited to keep it in step — the
  `*_anstaz_*` rename did exactly that, for a block no compiler checks. It should be deleted;
  the reversible integrator gets these functions from the module, not from this block.
- kind: dead code
- found: 2026-08-14

### K21 · `ShallowNetAutodiff` and `ShallowNetAutodiffReversible` have drifted apart in two spots where they should read identically.

- location: `src/nvi/shallownet_autodiff.jl:434`
- evidence: The two integrators are near-copies of each other, so
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
- kind: defect
- found: 2026-08-14

### K22 · `docs/src/index.md` renders past Documenter's `size_threshold_warn`

- location: `docs/src/index.md`
- evidence: (118 KiB against
  100 KiB), warning on every build. Still well under the 200 KiB hard threshold. It wants
  splitting into per-family pages, which is a docs reorganisation rather than a fix.
- kind: docs
- found: 2026-08-14

## Symbolic derivatives

Surfaced while updating to `SymbolicNeuralNetworks` 0.5.

### K23 · The first basis construction in a process costs about 9.3 s

- location: —
- evidence: , with `cse`/`inplace` on or
  off (9.31 s against 9.63 s for `DenseNetBasis{Float64}(tanh, 3, 3)`). Practically all of it
  is compiling the code-generation machinery rather than generating code: the same build warm
  is 27 ms with the defaults and 110 ms without them. That latency is upstream and not
  something this package can fix, but it dominates what a user actually pays for the first
  basis, and it is two orders of magnitude above the warm figures the docstrings quote. The
  docstrings now say they are warm measurements; a note in the user-facing documentation
  would be more use than a note here.
- kind: upstream
- found: 2026-08-15

### K24 · The 0.5 kernel numbers were measured in `quick` mode only.

- location: `test/integration/`
- evidence: That tier sweeps Float64 and
  Float32 over two problems, so 0.5's effect at Float16 and on the double pendulum is
  unmeasured. Float16 is the tier where this package has been bitten before — it has its own
  regression test in `test/integration/` precisely because the OGA/Newton path is
  ill-conditioned there — and 0.5 changed both the emitted code and the allocation of the
  in-place result, whose element type comes from the *inputs*. A `full` run would settle it.
- kind: missing test
- found: 2026-08-15

### K25 · The build figures in the *`SymbolicNeuralNetworks` 0.3 → 0.4* entry above (3.22 s → 0.79 s) are not comparable to the warm figures now quoted in the docstrings

- location: —
- evidence: (110 ms → 27 ms for the
  same basis). The ratio is the same ~4× and that is the claim both make, but the absolute
  numbers differ by a factor of thirty and nothing records how the older pair was measured —
  most likely a first build in a fresh process, i.e. mostly the compile latency of the entry
  above. Left as written, since it is the record of what was measured then; re-stating it would
  be inventing a measurement that was never taken.
- kind: docs
- found: 2026-08-15

### K26 · `V_func` returns a 1×1 matrix that both of its call sites immediately unwrap.

- location: `src/nvi/densenet.jl:440`
- evidence: The shape
  is an honest consequence of the Jacobian of a scalar-in/scalar-out network being 1×1, but it
  carries no information: the one integrator that consumes it (`src/nvi/densenet.jl:440`)
  strips it with `[1]`, and so do both kernel testsets in
  `test/nvi/dispatch_variants_unit.jl`, one with `vec` and one with `[1]`.
  `ShallowNetBasis` builds the slot as well, and no integrator reads that one at all. Either
  build it from `VNN[1,1]` like the two gradients now are, which would make it return a scalar
  — `build_nn_function` does accept a scalar expression — or leave it and note why; but the
  three derivative slots should not disagree about whether they are scalars.
- kind: defect
- found: 2026-08-15

## Reviewing the 0.5 update

Surfaced while reviewing the `SymbolicNeuralNetworks` 0.5 update, after the fixes that review
did make. None of these is a defect in the update: the compiled kernels were checked against
`ForwardDiff` off the integrator and agree to round-off at both bases, both codegen settings
and both precisions (≤ 7.2e-16 at Float64, ≤ 9.4e-8 at Float32, with `dvdθ` flattening to
exactly `NP`), which is the check now in `test/nvi/dispatch_variants_unit.jl`.

### K27 · The `29 ns` that the `symbolic = false` build is measured at is a measurement of nothing.

- location: `src/nvi/shallownet_basis.jl`
- evidence: `ShallowNetBasis{Float64}(tanh, 8; symbolic = false)` allocates 0 bytes, and a million
  constructions in a loop take 2.9 µs in total — the compiler elides them. `Dense` carries its
  dimensions in type parameters and stores only the activation, so a derivative-free basis puts
  nothing on the heap; the parameters do not exist until `NeuralNetwork(NN, T)` is called. The
  figure is therefore harness overhead, and it sits an order of magnitude below the ~0.042 µs
  timer resolution the kernel table already flags. What the docstring
  (`src/nvi/shallownet_basis.jl`) and the *Added* entry above should say is that the build costs
  15 ms and the opt-out costs nothing at all — the "pure overhead" claim they make is right, it
  is only the second number that pretends to be a cost. Left as measured rather than silently
  restated.
- kind: docs
- found: 2026-08-15

### K28 · The *0.3 → 0.4* comparability note above names only the `DenseNetBasis` pair.

- location: —
- evidence: The same
  applies to the other figures in that entry — `ShallowNetBasis` construction "1.6–2.0× faster",
  and 0.24 s against 0.017 s for generating the in-place kernel at `S = 8` — which are quoted
  with no record of how they were taken either, and which the warm re-measurement puts three
  orders of magnitude away. The entry should either scope its caveat to the whole 0.3 → 0.4
  entry or the figures should be re-taken as a set.
- kind: docs
- found: 2026-08-15

### K29 · `benchmark/results/` is gitignored, so the CSV the *Nonlinear solve conditioning* entry cites is not in the repository.

- location: `benchmark/results/`
- evidence: `benchmark/results/.gitignore` is `*` plus `!.gitignore`,
  which is right — those files are generated — but it makes
  `benchmark/results/derivative_backends_codegen_agreement.csv` a citation only the person who
  last ran the benchmark can follow. Either name the run that produces it or quote the number
  and drop the path.
- kind: docs
- found: 2026-08-15

### K30 · The new `ForwardDiff` cross-check is a point check, not a sweep.

- location: —
- evidence: One time input
  (`t = 0.37`), one random parameter draw per basis, the default code generation only, and
  `TEST_TYPES`, so no Float16. That is enough for what it was added for — a wrong shape or a
  wrong expression shows up immediately, which is the failure mode the 0.5 rename could have
  produced — but an expression that happens to agree at that one point would pass. Sweeping a
  few nodes, or reusing the quadrature nodes the integrators actually evaluate at, would cost
  almost nothing.
- kind: missing test
- found: 2026-08-15

## Entries about the form of this file

### K31 · Several moved entries refer to text that is not in this file.

- location: `KNOWN_ISSUES.md`
- evidence: K1 says "Unlike the two above … that trio", but the two entries it means are closed and
  are not in this file. K25 cites "the *`SymbolicNeuralNetworks` 0.3 → 0.4*
  entry above", K28 cites "the whole 0.3 → 0.4 entry", K27 cites "the *Added* entry above", and
  K27 also cites "the kernel table". These are in `CHANGELOG.md`, and `[0.2.0]` has
  no link definition in this file. The text is word for word, so it stays as written.
- kind: docs
- found: 2026-09-26

