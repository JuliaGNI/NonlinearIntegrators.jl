# Order and Linear Stability of B-Spline Galerkin Variational Integrators (ODE) and of the Space-Time Spline Integrator (PDE)

Status: 2026-10-20, first version. Every numbered statement carries the script that checks it.
Statements marked **Theorem/Proposition** are proved here; **Observation** means numerical only.

| File | Content | Cost (M-series laptop) |
|---|---|---|
| `spline_vi_order.jl` (this folder) | ODE part: one-step map G(z) of the harmonic oscillator from the CGVI coefficients (BigFloat), a/b decomposition, stability bands, identity A1, global errors HO and Hénon–Heiles | ≈ 3.3 min incl. compilation |
| `MultisymplecticIntegrators.jl/scripts/spline_wave_stability.jl` | PDE part: slab map of `SpaceTime_Spline_Wave_Integrator`, modal decomposition B1, CFL ν*, long runs | ≈ 50 s incl. compilation |

```
julia --project=benchmark benchmark/theory/spline_vi_order.jl              # NonlinearIntegrators.jl
julia --project=. scripts/spline_wave_stability.jl                          # MultisymplecticIntegrators.jl
```

Results: `benchmark/results/spline_vi_order_{local,ab,stability,identity,global}.csv`,
`MultisymplecticIntegrators.jl/scripts/results/spline_wave_stability.{csv,md}`.

---

## 0. Setting and notation

* Lagrangian L(q, q̇) = ½ q̇ᵀ𝕄q̇ − V(q); harmonic oscillator (HO): V = ½ω² q² (scalar after diagonalisation).
* One step [0, h], local time τ = t/h ∈ [0, 1]. Space S = S_p(Z_M): splines of degree p on M uniform cells,
  C^{p−1} at the interior knots, clamped (open) knot vector. Δt = h/M is the cell width.
* Quadrature: composite Gauss–Legendre, r points per cell. For the HO the integrand has degree 2p per cell, so the
  rule is **exact iff r ≥ p + 1**; r = p is the "reduced" rule.
* Integrator (code): `CGVI(BSplineBasis(UniformMesh(M, 0 .. 1), p), CompositeQuadrature(GaussLegendreQuadrature(r), M))`.
  Its residual is the type-II form of the discrete Lagrangian
  L_d(q₀, q₁) = ext { h Σ_j b_j L(q(c_j), q′(c_j)/h) : q ∈ S, q(0) = q₀, q(1) = q₁ },  p₀ = −D₁L_d, p₁ = D₂L_d.
* z = ωh. G(z): one-step map (q₀, p₀) ↦ (q₁, p₁) of the HO; exact map E(z) = rotation by z (ω = 1).
* PDE: `SpaceTime_Spline_Wave_Integrator`, a u_tt − b u_xx = 0, BSplineKit order k = degree + 1 in space and time,
  M_t cells per slab, r Gauss points per cell (`quadrature_order`, default k + 2 ⇒ exact).

## 1. Identity for linear splines

**Proposition 1 (A1).** For p = 1 one step of size h on M cells equals M steps of CGVI(P₁) with step h/M and the same per-cell rule.

*Proof.* S₁(Z_M) is the space of continuous piecewise linear functions; its degrees of freedom are the nodal values
q(k/M). The composite rule splits the discrete action into the sum of the M one-cell P₁ actions. Stationarity with respect
to an interior nodal value is the discrete Euler–Lagrange equation of CGVI(P₁) at that node, and the end relations
p₀ = −D₁, p₁ = D₂ involve only the first and last cell. ∎

*Check:* max |Δq|, |Δp| ≤ 1.5e-15 over T = 16, M ∈ {2, 4}, r ∈ {1, 2}, h ∈ {1, 0.25} (`spline_vi_order_identity.csv`).

**Corollary.** G_{1,M}(z) = G_{1,1}(z/M)^M. With the exact rule (r = 2), tr G/2 = (1 − z²/3)/(1 + z²/6); with the midpoint rule
(r = 1), tr G/2 = (1 − z²/4)/(1 + z²/4) (both formulas match the integrator's map to round-off, `spline_vi_order_trace.csv`). Hence the exact-rule P₁ integrator is stable for
z < M√12 and unstable for z > M√12; the midpoint-rule one is stable for every z > 0 (|tr G|/2 < 1).

## 2. Structure of the HO map

Uniform knots and a symmetric rule make L_d symmetric, L_d(q₀, q₁) = L_d(q₁, q₀). As a quadratic form

  L_d = [ a(z)(q₀² + q₁²) − 2 b(z) q₀q₁ ] / 2h = [ (a − b)(q₀ + q₁)² + (a + b)(q₁ − q₀)² ] / 4h,

and p₀ = −D₁L_d, p₁ = D₂L_d give

  G = [ a/b , h/b ; (a² − b²)/(b h) , a/b ],   det G = 1,   tr G / 2 = a/b.                         (2.1)

Exact: a_E = z cot z, b_E = z / sin z (so a_E/b_E = cos z). *Check:* |G₁₁ − G₂₂| ≤ 3e-76, |det G − 1| ≤ 1.1e-76 (BigFloat, 256 bit; `spline_vi_order_ab.csv`, `_local.csv`).

**Stability.** A real 2×2 map with det 1 is power bounded iff |tr| < 2, or tr = ±2 with G = ±I. By (2.1): stable ⇔ |a(z)| < |b(z)| (up to the boundary case).
The band edges |a| = |b| are where G has an eigenvalue ±1, i.e. the finite positive roots λ = z² of
  (K_a − λ M_q) X + (r₀ ∓ r₁) y = 0,  (r₁ ∓ r₀)ᵀ X = 0,
with M_q = mᵀ diag(b) m, K_a = aᵀ diag(b) a the quadrature mass and stiffness matrices of the basis (a generalised eigenproblem, used in both scripts).

## 3. Order (harmonic oscillator, exact quadrature): proof

Let E(q₀, q₁) = L_d − L_d^E and Δa = a − a_E, Δb = b − b_E, so E = [(Δa − Δb)(q₀ + q₁)² + (Δa + Δb)(q₁ − q₀)²]/4h.

**Lemma 1.** Let r ≥ p + 1 and z < π. For every exact solution q with q(0) = q₀, q(h) = q₁,

  0 ≤ E(q₀, q₁) ≤ ½ ‖(q − Πq)′‖²_{L²(0,h)} ≤ C h Δt^{2p} ‖q^{(p+1)}‖²_∞,

where Π is a local quasi-interpolant onto S_p(Z_M) that reproduces P_p and interpolates at both clamped end points
(de Boor–Fix type; reference to be checked: Schumaker, *Spline Functions: Basic Theory*, 3rd ed., Ch. 4–6).

*Proof.* With exact quadrature the discrete action is the action S[q] = ∫₀ʰ ½(q̇² − ω²q²) dt restricted to S.
For z < π, S is strictly convex on {q(0) = q₀, q(h) = q₁} (Wirtinger: ∫ė² ≥ (π/h)² ∫e² for e(0) = e(h) = 0), so the critical
values are minima: L_d^E = min over H¹, L_d = min over the subset S ⇒ E ≥ 0. Upper bound: L_d ≤ S[Πq]; with e = Πq − q,
e(0) = e(h) = 0, the first variation of S at q vanishes, so S[Πq] − S[q] = ½∫(ė² − ω²e²) ≤ ½∫ė². The last inequality is the
approximation order of S_p in W^{1,∞} on cells of width Δt. ∎

**Theorem 1 (order 2p, HO).** Let r ≥ p + 1. Then Δa − Δb = O(z^{2p+2} M^{−2p}) and Δa + Δb = O(z^{2p} M^{−2p}), the local error is
‖G(z) − E(z)‖ = O(z^{2p+1}) (ω = 1) and the global error on [0, T] is ≤ C T ω (ωΔt)^{2p}.

*Proof.* Apply Lemma 1 to q = A cos ω(t − h/2) (q₀ = q₁ = A cos(z/2)): E = (Δa − Δb) A² cos²(z/2)/h ≤ C h Δt^{2p} ω^{2p+2} A²,
so 0 ≤ Δa − Δb ≤ C′ z² (ωΔt)^{2p}. Apply it to q = A sin ω(t − h/2) (q₁ = −q₀ = A sin(z/2)): E = (Δa + Δb) A² sin²(z/2)/h, so
0 ≤ Δa + Δb ≤ C″ (ωΔt)^{2p}. In (2.1) with h = z:
  Δ(a/b) = [Δ(a − b) b_E − (a_E − b_E) Δb]/(b b_E) with a_E − b_E = O(z²), Δb = O(z^{2p}), b, b_E = 1 + O(z²) ⇒ O(z^{2p+2});
  Δ(h/b) = −z Δb/(b b_E) = O(z^{2p+1});  Δ((a − b)(a + b)/(b z)) = O(z^{2p+1}).
The constants contain 1/cos²(z/2) and z²/sin²(z/2), so the statement is asymptotic (z bounded away from π), and it uses that the
W^{1,∞} constant of Π does not depend on M (open point §6).
So G − E = O(z^{2p+1}). G is a symplectic perturbation of a rotation with |tr G| < 2 (Theorem 2), hence G = P R(θ) P⁻¹ with
cond(P) = 1 + O(z²), and the global error after T/h steps is ≤ cond(P) Σ ‖local error‖ = O(T h^{2p}). ∎

*Checks* (`spline_vi_order_ab.csv`, BigFloat, z = 2⁻¹ … 2⁻⁸, all p ≤ 4, M ∈ {1, 2, 4}):
- signs: Δa − Δb ≥ 4e-42 > 0 and Δa + Δb ≥ 1e-30 > 0 for every exact-rule case, as Lemma 1 predicts;
- slopes of Δa − Δb: 4, 8, 8, 12 for p = 1…4 (≥ 2p + 2); of Δa + Δb: 4, 4, 8, 8 (≥ 2p);
- M-dependence of the leading constants: p = 1, (Δa − Δb)/z⁴ = 1/24, 1/96, 1/384 for M = 1, 2, 4 (∝ M⁻²);
  p = 2, (Δa + Δb)/z⁴ = 1/360, 1/5760, 1/92160 (∝ M⁻⁴).
- local error slope 2p + 1 (2.98, 4.98, 6.98, 8.98) for all M and both r ∈ {p, p+1} (`spline_vi_order_local.csv`);
- global error slope ≈ 2p for HO **and Hénon–Heiles** (T = 16, fit over h = 1/2 … 1/32 and errors > 1e-12): HO 1.99–2.00,
  3.995–3.999, 5.99–6.00 for p = 1, 2, 3; Hénon–Heiles 1.97–2.00, 3.98–4.01, 5.98–6.04, 7.96–8.10 for p = 1…4; NaN (fewer than two
  points above 1e-12, round-off) for HO p = 3, M = 4 and p = 4, and Hénon–Heiles p = 4, M ≥ 2 except (M = 2, r = 4), and err(h = 1/8)·M^{2p} ≈ constant: HO p = 2, r = 3: 2.337e-7, 2.339e-7, 2.338e-7
  for M = 1, 2, 4 (`spline_vi_order_global.csv`).

**General Lagrangians (sketch).** For L = ½q̇ᵀ𝕄q̇ − V(q), Lemma 1 holds verbatim with ω² replaced by ‖V″‖ on a neighbourhood
and z < π replaced by h²‖V″‖ < π² λ_min(𝕄) (convexity), giving 0 ≤ L_d − L_d^E ≤ C h Δt^{2p} along exact solutions.
Marsden–West (2001), Thm 2.3.1, then gives order 2p for the discrete Hamiltonian map. **Gap:** that theorem needs the
estimate for L_d − L_d^E together with its first derivatives in (q₀, q₁) (Patrick–Cuell 2009 make this precise); Lemma 1 bounds only
the values. For the HO the quadratic structure closes the gap (proof above). Hénon–Heiles confirms order 2p numerically.

**Observation (A3).** With the reduced rule r = p the order is still 2p (same slopes), and the error is ≈ 2× the exact-rule
error for p = 1, 2 (HO 3.588e-3 vs 1.794e-3 for p = 1; Hénon–Heiles 4.98e-6 vs 2.36e-6 for p = 2). Not proved here (the rule is not exact,
so Lemma 1 does not apply).

## 4. Stability of the ODE scheme (harmonic oscillator)

**Theorem 2.** Let r ≥ p + 1. For every p, M and z < π: |tr G(z)| < 2, i.e. the scheme is stable.

*Proof.* Evaluate the form at (1, 1) and (1, −1): L_d(1, 1) = (a − b)/h, L_d(1, −1) = (a + b)/h. For z < π both are minima (Lemma 1).
(i) The constant path q ≡ 1 lies in S, so L_d(1, 1) ≤ S[1] = −ω²h/2 < 0 ⇒ a − b < 0.
(ii) S ⊂ H¹, so L_d(1, −1) ≥ L_d^E(1, −1) = (a_E + b_E)/h = z cot(z/2)/h > 0 ⇒ a + b > 0.
Together b > |a|, so |tr G|/2 = |a/b| < 1. ∎

So the first unstable z, z₁* (lower edge of the first band), is ≥ π. *Check* (`spline_vi_order_stability.csv`, exact edges):

| p | M = 1, r = p+1 | M = 2, r = p+1 | M = 4, r = p+1 | M = 1, r = p | M = 2, r = p | M = 4, r = p |
|---|---|---|---|---|---|---|
| 1 | √12 = 3.4641 | 6.9282 | 13.8564 | none ≤ 100 | none | none |
| 2 | √10 = 3.1623 | 3.1534 | 3.1425 | none | 3.1652 | 3.1433 |
| 3 | 3.1425 | 3.1420 | 6.2859 | none | 3.1422 | 6.2866 |
| 4 | 3.1416 (3.141616) | 6.2836 | 6.2833 | none | 6.2850 | 6.2834 |

All are ≥ π, also for the reduced rule (not covered by Theorem 2). r = p, M = 1 has no band in (0, 100] for p ≤ 4 (it is
Gauss collocation; Ober-Blöbaum–Saake report A-stability of these integrators). For p ≥ 2 the first band starts at or just above
π, where the exact L_d^E becomes singular (conjugate point); later bands start at various points (e.g. √12, 7.746, 13.856).
Many bands are narrow, e.g. p = 2, M = 4, r = 2: [3.14334, 3.14341] with |tr|/2 − 1 = 6.2e-10 at the midpoint (BigFloat column of
`spline_vi_order_stability.csv`), but the spectral radius is then 1 + √(2·6.2e-10) ≈ 1 + 3.5e-5 per step.

## 5. Space-time spline integrator (PDE)

**Proposition 3 (B1).** The slab action matrix of `SpaceTime_Spline_Wave_Integrator` (V = 0) is
a kron(Mx, Kt/h) − h b kron(Kx, Mt) (read off the code), with Mx, Kx, Mt, Kt the quadrature mass and stiffness matrices.
Let Mx be positive definite (true for an exact spatial rule). With Kx V = Mx V Λ, VᵀMxV = I, the change of variables c = (V ⊗ I) α,
p = Mx V β turns the slab map S: (c_bottom, p₀) ↦ (c_top, p₁) into the direct sum of the one-mode maps of the time scheme
for a α̈ = −b λ_j α, i.e. G(z_j) with z_j = h √(b λ_j / a).

*Proof.* (V ⊗ I)ᵀ A (V ⊗ I) = kron(I, a Kt/h) − h b kron(Λ, Mt) is block diagonal per mode, the equations and the momentum
p₁ = (A c)[top] transform covariantly (β = Vᵀp). ∎ *Check:* ‖T⁻¹ S T − ⊕ G_j‖_max ≤ 5.7e-13 in all 18 configurations.

**Corollary (CFL, B3).** On a uniform spatial mesh z_j = ν ω̂_j, ν = h√(b/a)/Δx, ω̂_j = Δx √λ_j. The scheme is stable iff every
ν ω̂_j avoids the unstable bands; the largest stable ν from 0 is ν* = z₁*/ω̂_max. With exact rules (the default) Theorem 2 gives the
sufficient condition **ν < π / ω̂_max**.

*Checks* (`spline_wave_stability.md`, c = 0.5, Δx = 1/32; ρ(S) without the λ = 0 Jordan block):

| boundary | k | M_t | r | ω̂_max | ν* = z₁*/ω̂_max | π/ω̂_max | max ρ(S) − 1 below ν* | ρ(S) − 1 in first band |
|---|---|---|---|---|---|---|---|---|
| periodic | 3 | 1 | 5 | 3.1623 | 1.0000 | 0.9934 | 8e-15 | 1.3e-1 |
| periodic | 3 | 2 | 5 | 3.1623 | 0.9972 | 0.9934 | 8e-15 | 4.3e-3 |
| periodic | 4 | 2 | 6 | 3.1436 | 0.9995 | 0.9994 | 2e-14 | 2.3e-4 |
| natural | 2 | 1 | 4 | 3.4641 | 1.0000 | 0.9069 | 5e-15 | 2.7 |
| natural | 2 | 2 | 4 | 3.4641 | 2.0000 | 0.9069 | 6e-15 | 13 |
| natural | 3 | 2 | 5 | 4.6451 | 0.6789 | 0.6763 | 5e-15 | 4.3e-3 |
| natural | 4 | 2 | 6 | 6.8706 | 0.4573 | 0.4573 | 5e-15 | 2.3e-4 |
| periodic | 3 | 1 | 2 (reduced) | 3.4641 | none ≤ 20 | — | 1e-14 | — |
| natural | 4 | 1 | 3 (reduced) | 7.0075 | none ≤ 20 | — | 1e-14 | — |

Long runs with `integrate` itself (periodic, k = 3, r = 5, exact data, max |u| on the last slab): M_t = 1: ν = 0.95, 400 steps → 1.000;
ν = 1.05 → 1.06e5. M_t = 2: ν = 0.95, 10⁴ steps → 1.000; ν = 0.998 (inside [ν*, 1]) → 1.21e2. (At the knots the zigzag mode of
quadratic splines vanishes, so the growth is invisible in `sol.u` at xstep = Δx; it shows at the quadrature points.)

Consequences:
* With the default exact quadrature the space-time spline integrator is **conditionally stable**, ν* ≈ 1 for periodic data and
  ν* ≈ 0.68 (k = 3), 0.46 (k = 4) with the natural (clamped) basis, whose boundary outlier modes raise ω̂_max.
* The reduced rule r = k − 1 (= degree) with M_t = 1 removes the time-step restriction in all tested cases, but the integrator uses the same
  rule in space: for k = 2 the one-point rule makes Mx singular, and for k = 3 it changes ω̂_max (3.4641 instead of 3.1623).
  Conjecture: a reduced rule in time only (exact in space) with M_t = 1 is unconditionally stable (by Prop. 3 it reduces to the
  observation "no band for r = p, M = 1", which would follow from CGVI(P_p) with p Gauss points = p-stage Gauss collocation) — not implemented.
* The commented-out assertion `timestep^2 < c * xstep^2` in `Wave.lpdeproblem` is not the right condition: the stability bound is
  ν = c h/Δx < ν* with ν* from the table.

## 6. Open points
* Check and cite the quasi-interpolant estimate (Lemma 1) and Patrick–Cuell for the nonlinear sketch.
* Prove A3 (r = p) and the absence of bands for r = p, M = 1 for all z (Gauss collocation ⇒ diagonal Padé, |R(iz)| = 1).
* Dirichlet boundary (prescribed columns) and sine-Gordon (V″ = cos u) not run yet.
