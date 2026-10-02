---
tags: [NVI, VISE, todo, henon-heiles]
created: 2026-10-01
status: later
---

# VISE on Hénon–Heiles: why it fails with slowly varying amplitude

**Context.** Dissertation §4.2, Fig. 4.35 / Table 4.4. Ansatz per component `A cos(ω t + φ) + c` (8 DOFs total).
VISE errors 4.8 / 13.7 / 9.4 (h = 1 / 2 / 5), ΔH ≈ 0.85 / 2.2 / 1.5 — worse than GVI with 4 DOFs at h = 1 (0.023 / 2e-5).
The text ("comparable to implicit midpoint at h = 1") is too optimistic.

## Hypotheses (to check)
1. **No modulation in the ansatz.** `A cos(ωt+φ)` has constant amplitude/frequency inside a step; the test space
   span{cos, t·sin, sin} has no τ·cos, τ·sin directions → modulation error accumulates; HH also exchanges energy between q₁, q₂.
2. **Global time in the expression.** `src/vise/vise.jl` evaluates the expression at `sol.t − h + c·h` (global t), so
   ∂q/∂ω = −A t sin(ωt+φ) grows with t and couples ω and φ → Jacobian gets ill-conditioned over time
   (matches the "Newton convergence issue" jumps in the thesis figures).
3. **Fairness.** Comparison only at equal DOFs, not equal FLOPs.

## Idea to try later
Ansatz linear in the amplitudes, nonlinear only in one frequency, in local time τ ∈ [0,1]:
q(τ) = Σⱼ (aⱼ cos ωhτ + bⱼ sin ωhτ) τʲ (+ polynomial).
- ω frozen → linear modulated-trigonometric Galerkin VI (Theorem 2 logic: symplectic, no gauge, no spurious solutions).
- ω free → extra condition ∂L_d/∂ω = 0, same structure as Theorem 3 (free knots) → symplectic by the envelope argument;
  reuse the outer/inner scheme of `benchmark/theory/free_knot_vi.jl`.
- Must beat: CGVI substeps at equal FLOPs, and the same ansatz with ω fixed (linear). Only "free ω ≫ fixed ω" shows the nonlinearity matters.
- Caveat (C7): the stationary ω is not the error-optimal ω; aliasing ω ↔ ω + 2πn/h gives many stationary points.

Related: [[relu_cgvi_equivalence]] §5–§7, C7; `benchmark/hard_problems/PLAN.md`.
