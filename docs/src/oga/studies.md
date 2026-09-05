# Studies

Three studies measure the variants, in `scripts/` (see `scripts/README.md` for how to run
them). They are studies of one component rather than benchmarks of the integrator suite,
which is what `benchmark/` holds, and they are not part of the documentation build.

## Why two tiers

End-to-end convergence conflates the quality of the **seed** with the behaviour of the
**solve**, and that confound is what made these failures hard to attribute in the first
place: a run that fails looks identical whether the greedy fit went rank-deficient or the
Newton Jacobian did. So the measurement is split.

**Tier A — seed quality** (`oga_fit_study.jl`) calls [`oga_fit`](@ref) directly: no
integrator, no Newton solve, no time stepping. It sweeps dictionary × selection × fit ×
activation × precision × target and reports

- `fit_err` — the quadrature-weighted ``L^2`` error of the seed, recomputed in `Float64`
  *after* the fit from the returned parameters, so precisions share one scale;
- `cond` / `sigma_min` — of the seed's design matrix (all `S` neurons, including any
  zero-weight placeholders, since those are part of what the Newton solve sees). This is the
  proxy for whether the system the seed feeds is solvable;
- `neurons` / `rejected` — how many of the requested neurons the greedy loop could place, and
  how many candidates it refused for adding no new direction.

Every case is an ``S \le 8``, 11-node problem, so ~6000 cases run in seconds. This is the
tier that actually separates the variants.

**Tier B — end-to-end** (`oga_sweep.jl`) integrates the harmonic oscillator for ten steps at
`S = 4`, `R = 8`, `dt = 0.1` with `DogLeg`, over seed variant × precision × regularization
factor × activation, in two stages matching the two questions:

- `relu` — `ReLUᵏ` for `k = 1…4`, where the `{±1}` dictionary is theoretically complete, so
  anything that goes wrong is numerical (the reduced-precision question);
- `smooth` — ELU, GELU and tanh against the 2-D and angular dictionaries built for them (the
  activation question).

**Tier B′ — the hardest problem** (`oga_double_pendulum.jl`) repeats a reduced grid on the
double pendulum at a *single* λ, read from the harmonic-oscillator sweep's CSV — the factor that
converged most often there — so the choice is measured rather than asserted.

## Methodology

Three points that decide whether the numbers mean anything.

### Everything absolute must be scaled to the precision

`regularization_factor` is swept as **multiples of** ``\sqrt{\varepsilon(T)}``, not as absolute
values, so the Jacobian-diagonal shift is scaled to the precision it protects. Each value is
identified by that multiple, which is also how the reports label it — ``\lambda =
16\sqrt{\varepsilon(T)}`` says what the shift is:

| precision | multiples swept | ``\lambda`` range |
|---|---|---|
| `Float16` | 2, 4, 8, 16, 32, 64 | `6.3e-2` … `2.0` |
| `Float32` | 2, 4, 8, 16, 32, 64 | `6.9e-4` … `2.2e-2` |
| `Float64` | 4, 16, 64, 256, 1024, 4096 | `6.0e-8` … `6.1e-5` |

plus a ``\lambda = 0`` control (multiple 0). The `Float64` set is stretched so that all three
ladders span a comparable dynamic range; ``16\sqrt{\varepsilon(T)}`` appears in each and is the
value the package documents as its default.

An absolute `1e-5` sits far below ``\sqrt{\varepsilon}`` at anything but `Float64` and so cannot
lift a near-singular Jacobian in reduced precision at all — which is why the sweep is over
multiples rather than over fixed numbers.

The same applies to `f_abstol`, and there getting it wrong invalidates a whole column rather
than degrading it — see [Precision](@ref). The studies pass
``f_{\text{abstol}} = 256\,\varepsilon(T)``.

!!! warning "How to read the precision columns"
    Because the tolerance is scaled per precision, each column answers *"did it reach its own
    precision's floor?"* — not *"did it reach the same absolute accuracy?"*. The three targets
    are

    | precision | ``256\,\varepsilon(T)`` |
    |---|---|
    | `Float16` | `0.25` |
    | `Float32` | `3.05e-5` |
    | `Float64` | `5.68e-14` |

    which differ by twelve orders of magnitude. Two consequences, and both matter for reading
    the tables below.

    A `Float32` or `Float16` convergence count can legitimately exceed the `Float64` one, and
    does; that is the looser target, not greater robustness. And at `Float16` the target is so
    loose — a residual of `0.25` — that "converged" carries little information on its own:
    the column separates *outright failures* (a thrown `SingularException`, a divergence)
    from everything else, and not much more. Compare variants *within* a column, weight the
    `Float64` column most heavily, and read the accuracy figures alongside the counts.

    This is the unavoidable tension in a precision sweep: a tolerance tight enough to be
    meaningful at `Float64` measures only the tolerance at `Float16`, and one loose enough to
    be reachable at `Float16` measures almost nothing there. Scaling it at least makes the
    failure mode explicit rather than silent.

### A finite result is not a converged one

`integrate` returns a finite state after exhausting `max_iterations`, so classifying on
`isfinite` alone records stalls as successes — the "finite-but-poor result slips under a
relaxed tolerance" hazard. Runs that exhaust the budget are recorded as `maxiter`, not `ok`,
with their accuracy still reported so a stall can be told apart from a divergence. A run whose
final state has left the working precision is recorded as `upcast`.

### Diagnostics are quarantined

`cond`, `sigma_min` and the comparable `fit_err` are computed in `Float64` after
[`oga_fit`](@ref) returns, purely for reporting. Nothing derived from them re-enters the fit.

## Figures

`scripts/oga_report.jl` writes the CSVs, markdown reports and PNGs, and the reports regenerate
from the CSVs alone — so a table or figure can be reworked without re-running a sweep.

Three conventions in the figures, and they are not stylistic:

- **magnitude on a single-hue sequential ramp**, light to dark. Specifically *not* the
  red→green ramp that success-rate grids usually reach for: red↔green is the one pair
  red–green colourblind readers cannot separate, which is about 8% of men;
- **every heatmap cell also carries its numeric value**, so nothing is encoded by colour alone
  and the figure doubles as a table;
- **a fixed categorical order for precision**, so a given precision keeps its colour across
  every figure regardless of which ones a particular sweep produced, with lines direct-labelled
  as well as legended.

One trap worth recording: the colour range must come from the *plotted medians*, not from the
raw rows. Ranging over the raw data stretches the scale to cover per-activation spread that no
cell displays, flattening every median into one indistinguishable shade.

## What the studies found

### Tier A — seed quality

Across 6048 cases at `Float16`/`Float32`/`Float64` there were **no failures**: every
(dictionary, selection, fit) combination returned a finite seed at every precision. That is
the guarantee [`oga_solve`](@ref NonlinearIntegrators.oga_solve) enforces, and it holds.

The ``\kappa^2`` penalty is directly visible. At `Float16`, the working-precision Gram solve
(`normaleq`) is consistently three to four times worse than every QR-family fit — e.g. median
fit error `2.10e-01` against `5.67e-02` on the angular dictionary with normalised selection —
while the *same* Gram solve with the `Float64` island restored (`normaleq+f64`) recovers to
`5.67e-02`, matching QR. That is the island's contribution isolated in one row pair: it was
buying robustness of the squared-condition-number solve, exactly as the analysis predicts, and
nothing else.

!!! note "The `Float16`/`normaleq` numbers are stable only to the figures quoted"
    These are the only cells of the study that do not reproduce exactly across dependency sets.
    Re-running Tier A moved `fit_err` in 197 of 6049 rows, every one of them `Float16` with the
    `normaleq` fit and no other column: at ``\kappa(\Phi)^2 \approx 4.7\times10^{17}`` the
    working-precision Gram solve is past the point where the last digits mean anything, which is
    the regime [Precision](@ref) describes. The median above moved from `2.1036e-01` to
    `2.1032e-01` — the same `2.10e-01` to the three significant figures stated. Read these cells
    at the precision they are quoted at, not beyond it.

Two patterns in the best-variant-per-cell table:

- `orthogonal` selection wins the large majority of cells, at every precision;
- for the **smooth** activations (`elu`, `gelu`, `tanh`) the winner is a 2-D dictionary
  (`grid2d` or `angular`) in *every* case — never the 1-D bias grid. The angular grid also wins
  several `ReLUᵏ` cells, which the non-uniform-coverage argument in [Theory](@ref) predicts.

The rank statistics confirm both guard-rail claims quantitatively. Out of 672 cases per
(precision, selection) group:

| precision | selection | short of full width | cases with rejections | most rejected | non-finite |
|---|---|---|---|---|---|
| `Float16` | `raw` | 49 | 22 | 41 | 0 |
| `Float16` | `normalized` | 49 | 31 | 40 | 0 |
| `Float16` | `orthogonal` | **210** | **199** | 41 | 0 |
| `Float32` | any | 0 | 0–3 | ≤4 | 0 |
| `Float64` | any | 0 | 0–2 | ≤20 | 0 |

At `Float16`, [`OrthogonalProjection`](@ref) refuses a rank-deficient atom in 199 of 672 cases
and consequently places fewer than four neurons in 210 — it is doing exactly the job it exists
for, and the cost is visible as reduced width rather than hidden as a bad fit. At `Float32` and
`Float64` the guards are essentially inert, which is the claim made for them: distinct atoms
are well separated there, so nothing gets blocked. And `non-finite` is **0** in all nine
groups, across all 6048 cases — the [`oga_solve`](@ref NonlinearIntegrators.oga_solve)
finiteness guarantee holding under measurement rather than assertion.

### Tier B — `ReLUᵏ` end-to-end

Converged runs out of 28 per precision (four ReLU powers × seven regularization factors).
Because the sweep pins its own residual tolerance, these counts are independent of the
integrator's default and reproduce exactly across solver versions:

| seed | `Float16` | `Float32` | `Float64` | median err (converged) |
|---|---|---|---|---|
| `reference` ([`OGA1dNormalEquations`](@ref)) | **0/28** | 18/28 | 15/28 | 4.47e-07 |
| `oga1d` ([`OGA1d`](@ref)) | 19/28 | 25/28 | 17/28 | 4.78e-05 |
| `oga1d-stable` ([`OGA1dStable`](@ref)) | 18/28 | 26/28 | 21/28 | 4.78e-05 |
| `oga1d-pivqr` (orthogonal + [`PivotedQR`](@ref)) | 18/28 | 26/28 | 20/28 | 6.96e-05 |
| `oga1d-tsvd` (orthogonal + [`TruncatedSVD`](@ref)) | 18/28 | 26/28 | **23/28** | 4.78e-05 |
| `oga1d-refined` (`Refined` + normalised + incremental QR) | **22/28** | 26/28 | 20/28 | 6.95e-05 |

Three things to read out of it.

**The reference implementation converges nowhere at `Float16`** — 0 out of 28, at every ReLU
power. It fails by *throwing* (`SingularException` from the Gram solve), so unlike the counts
around it this zero is not an artefact of the residual tolerance: no choice of target rescues
it. It is also the *worst* variant at `Float32` and `Float64` (18 and 15, against 25–26 and
17–23). This answers the reduced-precision question.

**The rank-revealing fits are the most robust, and by a clear margin at `Float64`.**
`oga1d-tsvd` reaches 23/28 there against the default's 17 and the reference's 15;
`oga1d-stable` and `oga1d-pivqr` sit just behind at 21 and 20. Off-grid refinement is the
strongest at `Float16` (22/28). The ordering is consistent enough across the three columns to
be a real effect rather than noise — every orthogonal-selection variant beats the default at
`Float64`, and every working-precision variant beats the reference everywhere.

**The reference nevertheless produces the most accurate seed where it works at all** — median
`4.47e-07`, two orders of magnitude better than the `4.8e-05` to `7.0e-05` of the rest. Its
`Float64` seed genuinely is a better warm start. That is the one column where the island pays
for itself, and it is worth stating plainly: the working-precision variants buy robustness, not
accuracy.

Per activation, `ReLU⁴` is the hard case — 4/42 at `Float16` and 13/42 at `Float64`, against
28–39/42 for the lower powers — which is unsurprising, since `σ(b)⁴` over `[-π, π]` spans eight
orders of magnitude and overflows `Float16` outright.

### The regularization factor

The sweep reproduces, independently and with the seed variant as an extra axis, what
SolverBenchmark found: **λ is a threshold, not a tuned value.**

| precision | ``\lambda = 0`` | ``\lambda > 0`` | median iterations, ``\lambda = 0`` → ``\lambda > 0`` |
|---|---|---|---|
| `Float16` | 1/24 | 15–16/24 | 4 → 1 |
| `Float32` | 11/24 | 21–23/24 | 32 → 3–12 |
| `Float64` | 4/24 | 16–22/24 | 4.5 → 5–9 |

At ``\lambda = 0`` convergence collapses; every nonzero factor recovers it, and the six
factors are barely distinguishable from one another across a range spanning a factor of 32
(`Float16`/`Float32`) or 1024 (`Float64`). The `Float32` iteration count is the clearest
signal: 32 iterations per step undamped against 3–12 with any damping.

One caveat on the `Float64` row: convergence rises with the factor (17 → 22 from `4√eps` to
`4096√eps`) but median accuracy *degrades* with it (`4.0e-09` → `1.3e-05`), because the shift
that makes the Jacobian solvable also biases the solution. The largest factor is not the best
choice; it is the one that converges most often.

### Tier B — smooth activations

This is the smooth-activation question. Converged runs out of 21 per precision, for ELU, GELU and
tanh:

| seed | dictionary | `Float16` | `Float32` | `Float64` |
|---|---|---|---|---|
| `oga1d` ([`OGA1d`](@ref)) | 1-D bias grid | 18/21 | 20/21 | 6/21 |
| `oga1d-stable` ([`OGA1dStable`](@ref)) | 1-D bias grid | 18/21 | 20/21 | 7/21 |
| `oga-sphere` ([`OGASphere`](@ref)) | angular grid | 12/21 | 20/21 | 7/21 |
| `oga2d` ([`OGA2d`](@ref)) | 2-D `(w, b)` grid | 18/21 | 21/21 | 9/21 |
| `oga2d-refined` | 2-D grid + off-grid polish | 18/21 | 20/21 | **10/21** |

**Read the `Float64` column** — it is the only one where the residual target is tight enough to
discriminate (see the warning above; at `Float16` the target is a residual of `0.25`). There the
2-D dictionaries lead: 10/21 and 9/21 against 6/21 and 7/21 for the 1-D grid, with the angular
grid at 7/21.

So the weight axis helps, in the predicted direction and for the predicted reason — but the
effect is a factor of about 1.5, not the categorical difference the theory might suggest.
Notably, changing the *selection rule and fit* on the 1-D grid (`oga1d` → `oga1d-stable`) moves
it 6 → 7, while changing the *dictionary* moves it 6 → 9–10; so the dictionary is the more
effective lever, which is the structural claim. Absolute rates stay low — 6–10 out of 21 — so
something beyond the dictionary is also limiting smooth activations at double precision.

### Tier B′ — double pendulum

The grid here is deliberately small: 3 activations × 6 seeds × 1 regularization factor per
precision, 54 runs total. That is enough to check whether the reduced-precision failure
reproduces on a harder problem, and **not** enough to rank the variants — the
working-precision seeds land on 2/3 at `Float16` and `Float64` and on 1/3 at `Float32`
(`oga-sphere` on 2/3 throughout), so the whole spread between variants is one or two cases.

What it does show:

- **The reference fails at `Float16` here too** (0/3, by `SingularException`), while every
  working-precision variant manages 2/3. Same failure, different problem. At `Float32` and
  `Float64` the reference converges as often as the rest (1/3 and 2/3), so it is half
  precision specifically that defeats it.
- **The accuracy ordering matches the harmonic oscillator's**: the reference has the best
  median error where it converges (`2.57e-05`), then `oga1d` (`9.15e-04`), then the 2-D
  dictionaries (`9.5e-03`–`1.0e-02`), then `oga1d-stable` and `oga1d-refined`
  (`2.1e-02`–`2.7e-02`). The robustness/accuracy trade-off is consistent across both problems.
- The chosen factors, taken from the harmonic-oscillator sweep, were `2√eps(T)` at `Float16`,
  `4√eps(T)` at `Float32` and `4096√eps(T)` at `Float64`.

## Where measurement disagreed with the design

The remedies implemented here were laid out in advance, in rough order of expected impact:
switch the fit to weighted QR and add normalisation plus a coherence guard first; add a
truncated SVD "only if you find a residual case that still degrades"; and reserve the
incremental QR "for when the per-step re-solve shows up in profiling".

Two parts of that guidance did not survive contact with the numbers.

**The rank-revealing fits were not a last resort.** They are the most robust variants at
`Float64` — `TruncatedSVD` at 23/28 against the QR default's 17 — so the ordering that put
them behind the main fix was wrong. The cost reasoning that motivated deferring them was also
beside the point: as the note at the top of [Algorithms](@ref) records, the fit is free at this
problem size, so there was never a performance reason to prefer a cheaper factorisation.

**Normalising the dictionary before selection was expected to be a straightforward
improvement.** It is not: it changes which atoms are chosen, and at `Float64` with `ReLUᵏ` that
steers the Newton solve into a worse basin. That is why [`RawProjection`](@ref) remains the
default and normalised selection is an explicit opt-in, and why `OGA1d`'s selected atoms are
pinned by a regression test rather than left free to drift.

The parts that held: the ``\kappa^2`` diagnosis and the QR reformulation, the precision-scaled
guard rails replacing the absolute constants, and the incremental QR being both cheaper and
more stable.

## Caveats

- Tier B fixes `S = 4`, `R = 8`, `dt = 0.1` and `DogLeg`, so it says nothing about how the
  ordering varies with network width, quadrature order, step size or solver strategy.
- The regularization-factor counts are aggregated over seeds and activations, so a cell of
  12/24 mixes
  variants with very different behaviour; the per-seed table is the one to compare within.
- `ReLU¹` is not ``C^1``, so a Newton stall there is expected rather than informative; it is
  included as the ``k = 1`` end of the `ReLUᵏ` axis.
- Tier A's targets stand in for the integrator's label sets rather than being sampled from
  them, so its fit errors are comparable to each other but not to an integrator's accuracy.
