# Plan: comparing nonlinear NVI with linear methods on ODE problems that are hard for linear methods

Goal: test the conclusion of `relu_cgvi_equivalence.md` §7, "a nonlinear NVI can only win on problems with fast local variation within one step, with an unknown and varying frequency, or that need adaptivity at a fixed step size". All four problems are Lagrangian ODEs and can be written as an `LODEProblem`.

**Criterion (fixed in advance).** A nonlinear method is "better" in a parameter range if and only if

1. in the error–cost plot, at equal cost its error is more than 3 times smaller than that of **all** linear baselines (L1, L2);
2. it is robust: the failure rate is at most 5%, and fallback steps are reported separately.

Cost is reported as wall time, number of Newton iterations, and (for the reference implementations) FLOP count.

---

## 1. The four problems

| Problem | Lagrangian | Dimension | Conserved quantities / invariants | Difficulty for linear methods | Source |
|---|---|---|---|---|---|
| Frequency-modulated oscillator | ½q̇² − ½ω(εt)²q², ω(s) = 1 + ½ sin s | 1, **non-autonomous** | H not conserved; adiabatic invariant J = H/ω | polynomials under-resolve once h exceeds the period 2π/ω; a trigonometric basis with fixed frequency is mismatched | new branch of GeometricProblems.jl |
| Charged particle in a strong non-uniform magnetic field (2D, B along z; the field is static and given, so the particle orbit is an ODE, not a PDE) | ½m\|ẋ\|² + A(x)·ẋ − φ(x), with A, φ as defined in `MasslessChargedParticle` | 2 | H; magnetic moment μ = m v⊥²/(2B) (adiabatic) | needs h ≪ 2π m/B(x), while B(x) changes along the orbit | new branch of GeometricProblems.jl (for m → 0 it reduces to the existing massless problem, i.e. the guiding-centre limit) |
| Kepler problem with high eccentricity | ½\|q̇\|² + 1/\|q\| | 2 | H, angular momentum L, Runge–Lenz vector (precession error) | rapid change within one step near the perihelion; a fixed step is wasted near the aphelion | new branch of GeometricProblems.jl, with an analytic reference solution (Kepler's equation) |
| Hénon–Heiles (quasi-periodic regime, E < 1/6) | GeometricProblems `HenonHeilesPotential.lodeproblem` | 2 | H | slowly varying amplitudes, energy exchange between the two components | existing |

Parameters (quick / full):

- **Frequency-modulated oscillator**: ε ∈ {1e-2, 1e-3}, T = 2π/ε (one modulation period), q₀ = 1, p₀ = 0.
- **Charged particle in 2D**: A₀ = 1, E₀ ∈ {0, 0.01}, m ∈ {1e-1, 1e-2} (Ω about 10 and 100), T about 2 drift periods.
- **Kepler problem**: a = 1, e ∈ {0.9, 0.99}, T = 10 orbital periods (2π each).
- **Hénon–Heiles**: two initial conditions, (q, p) = (0.1, 0.1, 0.1, 0.1) (Table 4.4 of the thesis, E ≈ 0.02) and (0.2, 0.2, 0.3, 0.3) (package default, E ≈ 0.135). The q error is computed only up to T = 100 (chaotic sensitivity), ΔH up to T = 1000.
- **Step sizes**: normalised by the physical time scale, h·Ω_max ∈ {0.1, 0.3, 1, 3, 10}. The Kepler problem is normalised by the mean motion; the frequency-modulated oscillator and Hénon–Heiles by ω ≈ 1.

Reference solutions: the analytic solution for the Kepler problem; Gauss(8) for the frequency-modulated oscillator, the charged particle and Hénon–Heiles, with dt = 1/50 of the smallest step and a self-check with dt/2.

## 2. Methods compared

| Label | Method | Type | Implementation |
|---|---|---|---|
| L1 | CGVI(P_s, R = s+1), s = 2..6, with m = 1..4 substeps | linear, symplectic | package |
| L2 | Gauss(s) Runge–Kutta | linear, symplectic | package |
| N1 | ShallowNet ReLU³ with kink-free bias | equivalent to CGVI(P₃) in theory (Theorem 1) | package; sanity check on the new problems |
| N2 | ShallowNet tanh / ReLU³ with the default bias (OGA picks interior kinks) | nonlinear | package |
| N3 | VISE with the ansatz of the thesis (Hénon–Heiles only, reproducing Table 4.4) | nonlinear | package |
| N4 | free-knot spline VI | nonlinear (knots), symplectic | package: new `src/spline/free_knot_spline.jl` (D ≥ 1, same interface as `ShallowNet`); checked against `theory/free_knot_vi.jl` in 1D; mainly for the Kepler problem |
| N5 | free-frequency VI: ansatz Σⱼ (aⱼ cos ωhτ + bⱼ sin ωhτ) τʲ plus a polynomial, with the additional condition ∂L_d/∂ω = 0 | nonlinear (frequency), symplectic | `theory` (new, with the same outer/inner structure as the free-knot VI) |

The key comparisons are **N5 against L1/L2** and **N4 against L1 with substeps**.

The modulated trigonometric Galerkin VI with a fixed frequency ω (the same ansatz with frozen parameters, a linear method) is not part of this plan: it belongs to GeometricIntegrators.jl, not to NVI, and is recorded in Obsidian (`GeometricIntegrators/`). This plan can therefore only show whether N5 beats the standard linear methods; attributing a gain to the nonlinearity itself has to wait until that linear version is implemented and compared.

## 3. Metrics

- q error: maximal relative error on the grid (for Hénon–Heiles only up to T = 100).
- Invariants:
  - charged particle, Kepler problem, Hénon–Heiles: ΔH = maxₙ |Hₙ − H₀|/|H₀|;
  - frequency-modulated oscillator: ΔJ; charged particle: Δμ (against the reference solution, since μ itself is only approximately conserved);
  - Kepler problem: ΔL and the error of the perihelion precession angle.
- Cost: wall time, Newton iterations, failure rate; FLOP count for the reference implementations.
- Diagnostics: every ShallowNet step is classified with `classify_step` (P_k-equiv / spline / degenerate); N4 and N5 report the number of fallback steps.
- Figures: two per problem.
  - error–cost Pareto plot: q error against FLOP count or wall time;
  - invariant error against h: same style as the `relu_k_sweep` figures, legend below the plot.

## 4. Code structure (as simple as possible)

The frequency-modulated oscillator, the charged particle and the Kepler problem are implemented directly in a **new branch of GeometricProblems.jl** (e.g. `nvi-hard-problems`), in the style of the existing modules of that package (see `harmonic_oscillator.jl`, `massless_charged_particle.jl`, `henon_heiles_potential.jl`):

```
GeometricProblems.jl/                     # new branch
  src/frequency_modulated_oscillator.jl   # lagrangian, hamiltonian (with t), adiabatic_invariant, default_parameters, lodeproblem / hodeproblem
  src/charged_particle_2d.jl              # A, φ as in MasslessChargedParticle, plus mass m; hamiltonian, magnetic_moment, lodeproblem / hodeproblem
  src/kepler_problem.jl                   # lagrangian, hamiltonian, angular_momentum, runge_lenz_vector, exact_solution (Kepler's equation), lodeproblem / hodeproblem
  src/GeometricProblems.jl                # include and export the new modules
  test/<problem>_tests.jl                 # per problem: lodeproblem and hodeproblem agree when integrated with Gauss(8); Kepler problem agrees with exact_solution to 1e-10
```

On the NonlinearIntegrators.jl side, the `benchmark` environment points to that branch with `Pkg.develop(path = "../../GeometricProblems.jl")` (assuming that repository sits next to NonlinearIntegrators.jl under `GNI/`):

```
benchmark/hard_problems/
  setup.jl           # common interface for constructing the four problems, reference solutions and invariants (only calls GeometricProblems, no problem is redefined)
  run_baselines.jl   # L1, L2 → results/<problem>_baselines.csv
  run_nvi.jl         # N1–N4 (step-by-step integration + classify_step, as in relu_k_sweep.jl) → results/<problem>_nvi.csv
  report.jl          # reads the CSVs, writes Pareto plots, invariant plots and md summaries
benchmark/theory/
  free_frequency_vi.jl  # N5, standard library only, with FLOP counting; self-checks: symplecticity |det − 1|, ∂L_d/∂ω against finite differences
src/spline/
  free_knot_spline.jl   # N4: FreeKnotSpline method, outer root-finding for the knots, inner fixed-knot Galerkin, composite quadrature following the knots;
                        # method / cache / integrate_step! in the style of GeometricIntegratorsBase, included and exported in NonlinearIntegrators.jl
test/
  free_knot_spline_smoke.jl  # in 1D, step-by-step agreement with theory/free_knot_vi.jl; symplecticity |det − 1|; equal to CGVI(P_k) for m = 0
```

Once the tests of the GeometricProblems.jl branch pass, it can be submitted upstream as a PR.

## 5. Phases and order

| Phase | Content | Deliverable / check |
|---|---|---|
| 0 | New branch of GeometricProblems.jl: the three new problems, reference solutions, invariants and their tests; the benchmark environment develops that branch | Kepler problem: numerical and analytic solution agree to 1e-10; frequency-modulated oscillator: the drift of J scales as O(ε); charged particle: for small m the guiding-centre orbit agrees with `MasslessChargedParticle` |
| 1 | Linear baselines L1, L2 (the Boris pusher and the Sundman transformation are left out for now, see Obsidian) | linear Pareto front for every problem |
| 2 | NVI already in the package (N1–N3) | N1 still equals CGVI(P₃) on the new problems (check of the generalisation of Theorem 1); results and failure rates of N2, N3 |
| 3 | N5 reference implementation (frequency-modulated oscillator → charged particle → Hénon–Heiles); N4 in src (Kepler problem) | all self-checks pass; N5 against L1/L2, N4 against L1 with substeps |
| 4 | Summary | `results/hard_problems.md` with the Pareto plots of every problem; conclusions in a new section C8 of `relu_cgvi_equivalence.md` |

Suggested order of the problems: **frequency-modulated oscillator → charged particle → Kepler problem → Hénon–Heiles**.

- The frequency-modulated oscillator is one-dimensional, has a known structure, is the cheapest, and tests the "free frequency" idea directly.
- The charged particle is the most relevant for the applications at IPP.
- The Kepler problem tests the free knots.
- Hénon–Heiles is chaotic and the hardest to assess, so it comes last.

## 6. Risks

- **Aliasing in N5**: both ω and ω + 2πn/h can satisfy the stationarity condition. Remedy: warm-start ω from the previous step and limit its change per step, as for the free knots.
- **ShallowNet near the perihelion of the Kepler problem**: Newton may not converge. Report the failure rate as it is; do not tune parameters to hide it.
- **Δμ of the charged particle is only approximately conserved**: it must be compared with μ(t) of the reference solution, not with a constant.
- **Fairness**: wall times of the package methods and of the reference implementations are not directly comparable. FLOP counts are compared only within one implementation family; across families only error against step size is compared.
