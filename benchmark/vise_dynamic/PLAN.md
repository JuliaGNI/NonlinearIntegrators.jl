# Plan: dynamic VISE — adaptive symbolic expressions, general residuals, agent-driven re-regression

Status: draft for discussion (2026-10-01). Nothing below is implemented yet. Every claim about the
current code cites the line it was read from; every claim about the method is either a standard
theorem (cited) or marked **hypothesis** and paired with the test that will decide it.

---

## 0. What the current code does (read from `src/vise/`)

| # | Fact | Where |
|---|---|---|
| F1 | Only Lagrangian/implicit problems: `Cache{ST}(::AbstractProblemIODE, ::VISE)`; the residual uses `ϑ` and `f`. No `ODEProblem` path. | `vise.jl` 169, 311–314 |
| F2 | Coefficients are *already* updated every step: unknowns `x = (θ, p_{n+1})`, size `S + D`; equations = DEL (S rows) + continuity `q(t_n; θ) = q_n` (D rows). | `vise.jl` 337–356 |
| F3 | The expression is evaluated at **global** time `t_n + c_j h`, so e.g. `∂q/∂ω = −A t sin(ωt+φ)` grows with `t`. | `vise.jl` 271, 290, 303 |
| F4 | Each step's initial guess for `p_{n+1}` integrates the problem with `ImplicitMidpoint` on `[0, h]` at step `h/100`, i.e. 100 implicit solves per VISE step. Share of total cost: **unmeasured**. | `vise.jl` 214–220 |
| F5 | The only "recovery" is: reset θ to `init_w` when `‖θ_prev − init_w‖ > 1`. The ansatz structure never changes. | `vise.jl` 205 |
| F6 | The basis is compiled once (`build_function`, `RuntimeGeneratedFunction`) and is a type parameter of `VISE`, so changing the expression means building a new method/integrator. | `vise_basis.jl` 47–60, `vise.jl` 40 |
| F7 | Known failure: Hénon–Heiles, rel. q error 4.8 / 13.7 / 9.4 at h = 1 / 2 / 5 (VISE paper Table 4). Obsidian note lists hypotheses: no amplitude modulation in the ansatz; global-time ill-conditioning (F3). | paper App. D; `benchmark/Claude outputs/VISE Hénon–Heiles…md` |

## 1. Goal and the three pieces

1. **General residuals.** Keep the DEL residual for Lagrangian problems; add residuals for a general
   first-order ODE `ẋ = v(t, x)`, so dissipative / non-Hamiltonian systems (most of ODEBench, all
   LSR-Synth damped oscillators, kinetics, population models) can be integrated.
2. **Dynamic description.** Coefficients per step (exists, F2) **plus** structural changes of the
   expression during the run, triggered by reference-free error indicators.
3. **Automation.** A driver that, when an indicator crosses a threshold, rolls back to the last
   accepted step and asks proposer agents (symbolic regression and LLM) for a new expression, validates
   candidates deterministically, swaps the ansatz, and continues — no human in the loop.

## 2. Mathematical formulation

Per step `[t_n, t_n + h]`, local time `τ ∈ [0,1]`, ansatz `x(τ; θ_n)`, `ẋ = (1/h) ∂_τ x`,
quadrature nodes/weights `(c_r, b_r)`, `r = 1..R`.

**R1 — DEL (existing, LODE).** Unknowns `(θ, p_{n+1})`:
`Σ_r h b_r [ f_r·∂_θᵢ q(c_r) + ϑ_r·∂_θᵢ q̇(c_r) ] − [ ∂_θᵢ q(1) p_{n+1} − ∂_θᵢ q(0) p_n ] = 0`, `i = 1..S`;
`q(0; θ) = q_n`. Then `q_{n+1} = q(1; θ)`.

**R2 — Collocation (ODE).** `x(0; θ) = x_n`, `ẋ(c_r; θ) = v(t_n + c_r h, x(c_r; θ))`, `r = 1..R`.
Square iff every component has `S_d = R + 1` weights.

**R3 — Constrained discrete least squares (ODE, any `S ≤ D(R+1)`).**
`min_θ ½ Σ_r b_r ‖ẋ(c_r; θ) − v(·, x(c_r; θ))‖²` s.t. `x(0; θ) = x_n`.
KKT system in `(θ, λ)`, size `S + D`, square, solved by the same Newton solver.

Continuity of the piecewise solution holds by construction in all three (`x(0;θ_n) = x_n`,
`x_{n+1} := x(1;θ_n)`), also across an ansatz switch.

**Sanity theorems used as tests (standard results, not hypotheses):**

- T1: R2 with a polynomial ansatz of degree `s` and `R = s` Gauss nodes is the `s`-stage Gauss
  collocation method = Gauss–Legendre RK (Hairer–Nørsett–Wanner I, §II.7).
- T2: R1 with a Lagrange-polynomial ansatz and Gauss quadrature is the Galerkin variational integrator
  `CGVI` (VISE paper §2, App. B).
- T3 (ODE case, `v` Lipschitz with constant `L`): with defect `δ(t) = ẋ̂(t) − v(t, x̂(t))` of the
  continuous, piecewise-C¹ solution `x̂`, `‖x(t) − x̂(t)‖ ≤ e^{Lt}‖x(0) − x̂(0)‖ + ∫₀ᵗ e^{L(t−s)}‖δ(s)‖ ds`
  (Gronwall). This is what makes the defect a meaningful, reference-free indicator.

**Structure caveat (precise).** A composition of symplectic maps is symplectic, so switching the
ansatz between steps cannot by itself destroy symplecticity of the step map. But backward error
analysis (bounded energy error over exponentially long times) assumes *one* method throughout; after a
switch the modified Hamiltonian changes, so a jump in the energy error at switches is expected.
Whether a single nonlinear-ansatz DEL step is symplectic is not established (`issymplectic = missing`,
`vise.jl` 82) — **hypothesis**, tested numerically by `‖DΦᵀ J DΦ − J‖`. R2/R3 on non-Hamiltonian
problems carry no structure claim.

## 3. Error indicators (all computable without a reference solution)

| Id | Indicator | Applies to | Cost |
|---|---|---|---|
| I1 | Newton status: not converged, or iterations > `k_max` | all | free |
| I2 | Relative defect at `M` check points `τ_m` off the quadrature nodes: `max_m ‖ẋ − v‖ / (‖v‖ + ε)`; for LODE the Euler–Lagrange defect `‖∂L/∂q − d/dt ϑ‖` (needs compiled `q̈`) | all | M evaluations of `v` |
| I3 | Legendre mismatch `‖p_{n+1} − ϑ(q(1), q̇(1))‖ / ‖p_{n+1}‖` | LODE | 1 evaluation |
| I4 | Invariant drift `|H_n − H_0| / |H_0|` (only when an invariant is supplied) | conservative | 1 evaluation |
| I5 | `cond(J)` of the Newton Jacobian at convergence | all | one SVD of `(S+D)²` |

Two thresholds per indicator: **soft** (start a background search, keep integrating) and **hard**
(reject the step, roll back). Thresholds are *calibrated*, not guessed: Phase 0/3 records each
indicator next to the true local error (from a reference) and reports the Spearman correlation; an
indicator that does not predict the error is dropped.

## 4. Recovery ladder (cheapest first)

| Level | Action | Changes |
|---|---|---|
| L1 | Re-solve with other initial guesses: previous θ shifted, `init_w`, θ fitted to the previous step's continuous solution | nothing |
| L2 | Refit coefficients of the *same* expression to a short reference window started at `(x_n)` | θ only |
| L3 | Structural change: proposer agents return new expressions; validator picks one | expression |
| L4 | Fallback: one step of `Gauss(s)` / `CGVI`, flagged and counted as a failure | — |

L1–L2 exist because the thesis figures show sharp error spikes attributed to Newton convergence,
not to the ansatz (paper, Fig. 3 caption); those should not cost a symbolic regression.

## 5. Multi-agent pipeline

```
 integrate (Julia) ──step──▶ Monitor (I1–I5, deterministic)
        ▲                         │ soft: spawn search (async)   hard: roll back to n*
        │                         ▼
        │                  Orchestrator (Julia state machine, JSONL log of every event)
        │                         │ context: equations, last accepted state, window data,
        │                         │ current expression + θ history, indicator history
        │          ┌──────────────┼──────────────────┬─────────────────────┐
        │          ▼              ▼                  ▼                     ▼
        │   Library proposer   SymbolicRegression.jl   LLM proposer(s)      (more proposers)
        │   (deterministic:    (equation_search on     (LLM-SR style: skeleton
        │   add τ·cos, τ·sin,  window data, operators  with free params, physics
        │   poly correction)   seeded from current)    context in the prompt)
        │          └──────────────┴─────────┬────────┴─────────────────────┘
        │                                   ▼
        │                     Validator (deterministic): whitelist-parse → Symbolics →
        │                     VISEBasis → fit θ on window → trial VISE on k steps →
        │                     accept iff all steps converge and indicators < hard;
        │                     score = window error + λ·complexity
        └──────── swap ansatz (new method/integrator, start from x_{n*}) ◀──┘
```

Rules that keep it solid:

- Only proposers are agents. Monitoring, validation, acceptance and the integration are deterministic
  Julia code, so every accepted expression has a reproducible numerical justification.
- LLM output is never `eval`'d: it is parsed against a whitelisted grammar (`+ − * / ^`, `sin cos exp
  log sqrt tanh`, parameters, `τ`) into Symbolics.
- Window data for proposers: the accepted VISE continuous solution before `t_{n*}` plus a short
  high-fidelity integration from `x_{n*}` (the equations are known). Its cost is reported, see §8.
- Every run logs: trigger times, indicator values, level reached, all candidates with scores, the chosen
  expression, wall time, and LLM tokens.

## 6. Code layout (GeometricIntegratorsBase style; small files, dispatch on problem type)

```
NonlinearIntegrators.jl/
  src/vise/vise.jl            # existing; local-time option, cheap p-guess (Phase 1)
  src/vise/vise_basis.jl      # existing; add compiled q̈ for I2 on LODE
  src/vise/vise_ode.jl        # NEW: Cache{ST}(::ProblemODE, ::VISE), R2/R3 components!/residual!
  src/vise/vise_monitor.jl    # NEW: indicators I1–I5, thresholds
  src/vise/vise_adaptive.jl   # NEW: segment driver with rollback; returns segments
                              #      (t_start, t_end, expression, θ per step)
  ext/NonlinearIntegratorsSymbolicRegressionExt.jl   # NEW: SR.jl proposer (weak dependency)
  agents/                     # NEW: orchestrator + LLM proposer (backend: decision D2)
  test/vise/vise_ode_unit.jl  # T1, T2, exact-solution ansätze, switch/rollback tests
  benchmark/vise_dynamic/     # this plan, loaders, runners, report
```

## 7. Phases, each with a fixed acceptance check

| Phase | Content | Done when |
|---|---|---|
| 0 | Baseline: rerun `run_vise.jl`; profile cost split (F4 initial guess vs Newton); log Newton iterations and `cond(J)` per step; record I2/I3 vs true local error on HO, pendulum, Hénon–Heiles | Paper Table 2–4 numbers reproduced; cost split and indicator–error correlations in `results/baseline.md` |
| 1 | (a) local-time ansatz + warm start by shifting/refitting the previous expression; (b) momentum guess `ϑ(q(1), q̇(1))` of the previous expression instead of F4 | HO exact test still `< 1e-12`; each change kept only if it lowers wall time or error on all three problems without failures |
| 2 | R2 and R3 for `ODEProblem` | T1 to `1e-12` rel. vs `Gauss(s)`; T2 vs `CGVI`; exact ansätze reproduce `x = a e^{bt}` (linear decay) and the logistic solution to the Newton floor |
| 3 | Monitor + segment driver + ladder L1, L2, L4 (no agents yet) | Deliberately wrong ansatz triggers and recovers; HO never triggers; segment boundaries continuous to round-off |
| 4 | L3 with deterministic proposers: library proposer, SymbolicRegression.jl; validator | Hénon–Heiles: ≥ 10× lower max rel. q error than static VISE at h = 1, 2, 5, with ΔH reported |
| 5 | LLM proposer(s) and asynchronous spawn on soft thresholds | Same problems: compare proposer mixes (library / SR / LLM) by error, #triggers, wall time; ≥ 3 seeds |
| 6 | Benchmarks (§8) | `results/vise_dynamic.md` with tables and figures |
| 7 | Sub-agent code review after each phase (project rule); deferred items to Obsidian | review findings resolved or logged |

## 8. Benchmarks and fairness

ODEBench (63 ODEs, 1D 23 / 2D 28 / 3D 10 / 4D 2, two initial conditions each, 4 chaotic; ODEFormer
§6, App. A) and LLM-SRBench LSR-Synth dynamical sets (chemistry kinetics, biology population growth,
physics damped oscillators; `t ∈ [0, 60]`, 5000 points, last 500 = OOD; material science is static and
excluded) are **equation-discovery** benchmarks: methods receive trajectories and must recover `v`.
VISE is a solver: it receives `v` (or `L`) and produces a piecewise symbolic trajectory. The plan
therefore uses them as follows (decision D1):

- **Track A (default, forward problem).** Use the benchmark systems and initial conditions with the
  ground-truth equations. Initial expression from SR on the ID window only; VISE integrates through
  the OOD window. Metrics as defined by the benchmarks — R² and accuracy `R² > 0.9` (ODEBench), NMSE
  and `Acc_0.1` on ID and OOD (LLM-SRBench) — plus max rel. error, invariant error where one exists,
  wall time, Newton iterations, #triggers, SR/LLM time. Baselines at equal `h`: implicit midpoint,
  `Gauss(s)`, RK4, `CGVI`, static VISE, and the global static SR trajectory expression.
- **Track B (optional, inverse problem).** Discover `v` with an SR method on ID data, then integrate
  it with VISE vs RK45 into OOD. Only this track is comparable to the published leaderboards, and the
  gain there would come from the discovered `v`, not from VISE.
- Plus the original VISE problems and `hard_problems` P1–P4 (Lagrangian, R1).

Fairness rules: report cost both with and without proposer time; compare against the high-fidelity
integrator used to make window data at equal total cost; for chaotic systems, score pointwise error
only up to a fixed number of Lyapunov times.

## 9. Risks

- Circularity/cost: if triggers are frequent, window references dominate and a classical integrator is
  simply cheaper. Measured by #triggers per 1000 steps and the equal-cost comparison.
- Frequency aliasing `ω ↔ ω + 2πn/h` gives several stationary points (Obsidian C7); validator must
  restrict the jump of θ across a switch.
- Energy jumps at switches (§2 caveat) on conservative problems.
- Stiff ODEBench members: Newton on R2/R3 may need damping; failures reported, not tuned away.
- LLM non-determinism: fixed prompts, logged proposals, ≥ 3 runs, LLM-free ablation always reported.

## 10. Open decisions

- **D1** Benchmark meaning: Track A only, or A + B.
- **D2** LLM backend: Claude API called from Julia (HTTP.jl + JSON3), or a Python Claude Agent SDK
  process the Julia orchestrator talks to, or SR.jl only for now.
- **D3** First target: Hénon–Heiles (fix the known failure, R1) or ODEBench 1D (exercise R2/R3).
