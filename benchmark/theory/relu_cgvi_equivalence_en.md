# Equivalence of the ReLUᵏ Shallow-Network Variational Integrator (ShallowNet) and CGVI: Proofs, Generalizations and Numerical Verification

This note turns the observation "ShallowNet(ReLU³, S = 4, R = 4) and CGVI(S = 4, R = 4) have almost identical errors" into theorems that can be checked step by step, and generalizes them to arbitrary width S, arbitrary power k, and to the case where kinks lie inside the time step.

**Every proof step carries a verification tag**, e.g. **[V2.4]**. It refers to the numerical check in `verify_proof_steps.jl` that prints the same tag. All predictions (conjectures) are tagged **[C1]–[C7]** and correspond to `verify_conjectures.jl` (C1–C6) and `free_knot_vi.jl` (C7).

| File | Content | Dependencies |
|---|---|---|
| `relu_cgvi_equivalence.md` | Chinese version of this note | — |
| `relu_cgvi_equivalence_en.md` | This note | — |
| `theory_common.jl` | Reference implementation written from the definitions: discrete action, weak-form residual, Lagrange / truncated-power / B-spline / frozen-neuron bases, network residual and tangent vectors, Newton, LM, type-I solve | standard library only |
| `verify_proof_steps.jl` | Numerical check of every proof step V1.1–V5.6; prints PASS/FAIL per check and exits with code 1 on any failure | standard library only |
| `verify_conjectures.jl` | Part A: the actual `ShallowNet` and `CGVI` (C1–C3), reading θ at every step and classifying it as in this note; Part B: C4–C6 on the reference implementation | `benchmark` project |
| `free_knot_vi.jl` | Free-knot spline variational integrator (outer root-finding on the knots, inner fixed-knot Galerkin, composite quadrature following the knots), compared with CGVI, CGVI with substeps and fixed-knot splines in parameters, FLOPs and error (C7) | standard library only |
| `relu_k_sweep.jl` | ShallowNet(ReLUᵏ / tanh, S) vs CGVI(P_k) over k, S and h, with error in q and relative Hamiltonian error | `benchmark` project |

```
julia --project=benchmark benchmark/theory/verify_proof_steps.jl
julia --project=benchmark benchmark/theory/verify_conjectures.jl                 # A + B
julia --project=benchmark benchmark/theory/verify_conjectures.jl --reference-only  # B only
julia --project=benchmark benchmark/theory/verify_conjectures.jl C1 C3a --final-time=40
julia benchmark/theory/free_knot_vi.jl                                         # C7
```

> **About the numbers.** All scripts have been run in Julia: `verify_proof_steps.jl` passes completely; Part B of `verify_conjectures.jl` (C4–C6) agrees digit by digit with the original Python prototype of the same algorithms, so the numbers labelled "prototype" in C4–C6 of §6 are also the Julia results; the C7 numbers are taken directly from `results/free_knot_vi.md`.

---

## 0. Summary of results

Let σ(x) = max(0, x)ᵏ. The network is q_θ(τ) = Σᵢ aᵢ σ(wᵢτ + bᵢ), identical to `ShallowNetBasis`: `Dense(1,S,σ) → Dense(S,1)`, with no output bias. The kink of neuron i is zᵢ = −bᵢ/wᵢ.

1. **Theorem 1 (no interior kink ⇒ equivalent to CGVI(P_k)).** If no neuron has its kink in (0,1), and the tangent vectors of these neurons span P_k, then the one-step equations of ShallowNet and of CGVI(P_k, R) are **equivalent**: the numbers of equations differ (3S+1 for the network, k+2 for CGVI), but each set of equations is a linear combination of the other and both determine the same (q, pₙ₊₁). The two methods have the same one-step map, the same trajectory and the same error. This holds for every S. Hence, as long as all kinks stay outside the interval, adding neurons does not reduce the error.
2. **Corollary:** "degree of freedom" here means a coordinate component q^d of the physical system (the package uses a separate network for each component), not a network parameter. For each component the one-step equations have 3S+1 unknowns: a, w, b for each of the S neurons, plus pₙ₊₁. But the Jacobian of these equations has rank only k+2; the remaining 3S−k−1 directions form the **null space** of the Jacobian, i.e. directions with Jδx = 0, along which Newton's method cannot determine a step. These directions are exactly the **gauge freedom** (parameter symmetry): directions that change the parameters but not the function q_θ (and hence not pₙ₊₁). Examples: rescaling one neuron (a, w, b) → (a/λᵏ, λw, λb), or moving a kink outside the interval while readjusting the output weights a. Because the Jacobian is singular, the plain Newton linear system has no unique solution, so `regularization_factor` is needed to make every linear solve unique. It only selects which of the equivalent parameter sets is reached; it does not change q_θ or pₙ₊₁. ReLU² is equivalent to CGVI(P₂), not to CGVI(P₃). The step is symplectic.
3. **Propositions 4.1 / 4.2:** the condition "the kinks are distinct" cannot be dropped: there are spurious solutions with coalesced kinks that are not CGVI solutions. On the other hand, neurons with zero amplitude and an interior kink are harmless: q is still the CGVI solution.
4. **Theorem 2 (fixed interior kinks):** with (w, b) frozen, ShallowNet is the Galerkin variational integrator on the spline space S_k(Z), and the result does not depend on the basis used (truncated powers or B-splines).
5. **Theorem 3 (free kinks):** a full ShallowNet step equals "the spline Galerkin step at the knots Z*", where Z* is a stationary point of the discrete Lagrangian L_d^Z(qₙ, qₙ₊₁) with respect to Z. It is therefore still generated by a discrete Lagrangian and is symplectic.
6. **The right direction for generalization (C4):** add m fixed knots inside the interval. Here m concerns both the function space and the quadrature. On one hand, it is the number of neurons whose kinks lie in (0,1) (m extra neurons on top of k+1 neurons without interior kinks); they enlarge the trial space from P_k to the m-knot spline space S_k(Z), raising its dimension from k+1 to k+1+m. On the other hand, the quadrature must also be split at these m knots (composite quadrature). With both in place, the order stays 2k and the error is roughly that of CGVI(P_k) with the step size reduced by a factor m+1. Adding neurons while keeping a single global Gauss rule lets the quadrature error wipe out the benefit of the spline; splitting the quadrature without adding interior-kink neurons leaves the trial space at P_k and the error unchanged.

---

## 1. The one-step equations

### 1.1 Notation

Within one step let t = tₙ + hτ, τ ∈ [0,1]. For brevity take one degree of freedom (D = 1). `ShallowNet` uses an independent network for each degree of freedom (`cache.ps[d]`); the trial set is the product of these networks, so the results below hold component by component.

Let the Lagrangian be L(q, v), with ϑ = ∂L/∂v and f = ∂L/∂q. The quadrature nodes are cⱼ and the weights bⱼ (R-point Gauss, j = 1..R). The discrete action is

$$A_h(q)=h\sum_{j=1}^R b_j\,L\big(q(c_j),\,q'(c_j)/h\big),\qquad q'=\tfrac{dq}{d\tau}.$$

Its variation in the direction δq is

$$\delta A_h(q)[\delta q]=\sum_j b_j\Big(h\,f(Q_j,V_j)\,\delta q(c_j)+\vartheta(Q_j,V_j)\,\delta q'(c_j)\Big),\quad Q_j=q(c_j),\ V_j=q'(c_j)/h.$$

### 1.2 Type-II one-step equations

Given (qₙ, pₙ), the unknowns are the trial function q (or its parameters) and pₙ₊₁. Define the linear functional

$$\rho_{q,p_{n+1}}(\delta q)\;:=\;\delta q(1)\,p_{n+1}-\delta q(0)\,p_n-\delta A_h(q)[\delta q].$$

The one-step equations are

$$\text{(E1)}\quad q(0)=q_n,\qquad\text{(E2)}\quad \rho_{q,p_{n+1}}(\delta q)=0\quad\forall \delta q\in\mathcal T.$$

Two basic notions of Galerkin methods are used here:

- **Trial space**: the set of functions in which we look for the approximate solution q, i.e. all functions the ansatz can represent. For CGVI(P₃) it is the space of all cubic polynomials; for ShallowNet the trial set is all functions of the form Σᵢ aᵢσ(wᵢτ + bᵢ).
- **Test space** 𝒯: the set of directions δq against which the equation is "tested". The exact solution satisfies ρ(δq) = 0 for **all** variations δq; this is the continuous Euler–Lagrange equation. A discrete method cannot achieve this for every δq and only requires it for δq ∈ 𝒯; each linearly independent direction in 𝒯 yields one equation.

In short, the trial space determines "what the solution looks like", and the test space determines "which equations it must satisfy". Methods in which both are the same space are called (Bubnov–)Galerkin methods. The central observation of this note (Observation 1.2) is that to compare two methods it suffices to compare their trial and test spaces.

- **CGVI**: the trial space is a linear space V and 𝒯 = V. The documentation of `CGVI` in GeometricIntegrators states that continuity q_h(tₙ) = qₙ is imposed weakly through a multiplier, and the unknowns are the S coefficients plus pₙ₊₁ — that is, (E1)(E2).
- **ShallowNet**: `residual!` in `src/nvi/shallownet.jl` has four groups of rows.
  - a rows: `r₁ p̃ − r₀ p̄ − Σ b (h m F + a P)`, i.e. ρ(σᵢ).
  - constraint row: `q̄ − r₀·X`, i.e. (E1).
  - W rows: `dqdWr₁ p̃ − Σ …`, i.e. ρ(∂q/∂Wᵢ). There is no pₙ term because ∂q(0)/∂Wᵢ = aᵢ σ′(bᵢ)·0 = 0.
  - b rows: ρ(∂q/∂bᵢ).

  So ShallowNet solves exactly (E1)(E2), with

  $$\mathcal T=T_\theta\mathcal M:=\operatorname{span}\{\partial q_\theta/\partial\theta_i\}.$$

**Lemma 1.1 (variational formulation).** Let 𝒢(θ, pₙ₊₁) = A_h(q_θ) − pₙ₊₁ q_θ(1) + pₙ q_θ(0). Then row i of the network residual equals −∂𝒢/∂θᵢ = ρ(∂q_θ/∂θᵢ).

*Proof.* By the chain rule, ∂A_h(q_θ)/∂θᵢ = δA_h(q_θ)[∂_{θᵢ}q_θ]; the derivatives of the boundary terms are pₙ₊₁ ∂_{θᵢ}q_θ(1) and pₙ ∂_{θᵢ}q_θ(0). ∎ **[V1.1a]** checks the network case, **[V1.1b]** the CGVI case.

**Key observation 1.2.** Row i of the residual equals ρ_{q_θ,p}(tᵢ) with tᵢ = ∂q_θ/∂θᵢ. So θ enters the equations only through two objects: the trial function q_θ and the test space T_θℳ. **Whenever two parameterizations produce the same q and the same test space, their equations are equivalent.** All proofs below reduce to computing these two objects.

**Lemma 1.3 (type I ⇔ type II, discrete Legendre transform).** Suppose the evaluation map δq ↦ (δq(0), δq(1)) on a linear space V is surjective. Define

$$L_d^V(q_a,q_b):=A_h(q^*)$$

where q* is the stationary point of A_h on {q ∈ V : q(0) = q_a, q(1) = q_b}. Then the type-II step (qₙ, pₙ) ↦ (qₙ₊₁, pₙ₊₁) satisfies

$$p_n=-\partial L_d/\partial q_a,\qquad p_{n+1}=\partial L_d/\partial q_b.$$

*Proof.* The stationarity condition is ∇A = λ₀ ∇q(0) + λ₁ ∇q(1). For every δq ∈ V, δA[δq] = λ₀δq(0) + λ₁δq(1). Comparing with (E2) gives λ₁ = pₙ₊₁ and λ₀ = −pₙ. By the envelope theorem, ∂L_d/∂q_a = λ₀ and ∂L_d/∂q_b = λ₁. ∎ **[V1.2a–d]**

Hence CGVI is the variational integrator generated by L_d, and is therefore symplectic.

---

## 2. Lemmas on ReLUᵏ neurons

**Lemma 2.1 (restriction of a neuron with an exterior kink).** If z = −b/w ∉ (0,1), then on [0,1]:
- if wτ + b > 0 (active), σ(wτ + b) = (wτ + b)ᵏ;
- otherwise (inactive), σ(wτ + b) ≡ 0.

*Proof.* wτ + b does not change sign on [0,1]. ∎ **[V2.1]**

**Lemma 2.2 (tangent vectors).**

$$\partial_{a_i}q=\sigma(s_i),\qquad \partial_{b_i}q=a_i\sigma'(s_i),\qquad \partial_{w_i}q=a_i\,\tau\,\sigma'(s_i),\qquad s_i=w_i\tau+b_i.$$

The tangent vectors of q′ are:

$$\partial_{a_i}q'=w_i\sigma'(s_i),\qquad \partial_{w_i}q'=a_i\big(\sigma'(s_i)+\tau w_i\sigma''(s_i)\big),\qquad \partial_{b_i}q'=a_iw_i\sigma''(s_i).$$

**[V2.2]** compares these closed forms with finite differences.

**Lemma 2.3 (apolar pairing).** On P_k define

$$B(f,g)=\sum_{m=0}^k(-1)^m f^{(m)}(0)\,g^{(k-m)}(0).$$

Then B is nondegenerate, and for every f ∈ P_k and z ∈ ℝ:

$$B\big(f,(\tau-z)^k\big)=k!\,f(z),\qquad B\big(f,(\tau-z)^{k-1}\big)=-(k-1)!\,f'(z).$$

*Proof.*

1. Take g = (τ − z)ᵏ; then g^{(k−m)}(0) = k!/m! · (−z)^m. Substituting,

   $$B=\sum_m (-1)^m f^{(m)}(0)\,\frac{k!}{m!}(-z)^m = k!\sum_m \frac{f^{(m)}(0)}{m!}z^m = k!\,f(z)$$

   where the last step is Taylor's formula.
2. Take g = (τ − z)ᵏ⁻¹; then for m ≥ 1, g^{(k−m)}(0) = (k−1)!/(m−1)! · (−z)^{m−1}, so B = −(k−1)! f′(z).
3. Nondegeneracy: in the monomial basis, B(τⁱ, τʲ) is nonzero only for i + j = k, with value ±i!(k−i)!, which is a full-rank anti-diagonal matrix. ∎

**[V2.3a, V2.3b]** check the two identities, **[V2.3c]** checks nondegeneracy.

**Lemma 2.4 (spanning lemma).** Let z₁, …, z_d be distinct. Then

$$\text{(a)}\ \dim\operatorname{span}\{(\tau-z_j)^k\}_{j=1}^d=\min(d,k+1),$$

$$\text{(b)}\ \dim\operatorname{span}\{(\tau-z_j)^k,(\tau-z_j)^{k-1}\}_{j=1}^d=\min(2d,k+1).$$

*Proof.* Let W be one of these spans. Since B is nondegenerate, dim W = k+1 − dim W^⊥, where W^⊥ = {f ∈ P_k : B(f, g) = 0 ∀ g ∈ W}.

- For (a): by Lemma 2.3, W^⊥ = {f ∈ P_k : f(zⱼ) = 0, j = 1..d}. It consists of the multiples of ∏(τ − zⱼ) and has dimension max(0, k+1−d).
- For (b): W^⊥ = {f : f(zⱼ) = f′(zⱼ) = 0}, the multiples of ∏(τ − zⱼ)², of dimension max(0, k+1−2d). ∎

For d = k+1 the determinant of the coefficient matrix can also be written explicitly. Arrange the monomial coefficients of (τ − zᵢ)ᵏ in a matrix C whose column i, row j entry is C(k,j)(−zᵢ)^{k−j}; then

$$|\det C|=\prod_{j}\tbinom{k}{j}\prod_{i<l}|z_i-z_l|.$$

**[V2.4a]** checks (a) for k = 2..6, d = 1..k+2; **[V2.4b]** checks (b); **[V2.4c]** checks the determinant formula; **[V2.4d]** checks that the rank drops when kinks coalesce.

**Lemma 2.5 (Euler homogeneity).** Since σ is homogeneous of degree k, sσ′(s) = kσ(s). Hence

$$\partial_{w_i}q=\frac{k\,a_i\,\partial_{a_i}q-b_i\,\partial_{b_i}q}{w_i}.$$

The same holds for the corresponding residual rows: r_w = (k a r_a − b r_b)/w.

*Proof.*

$$\tau\sigma'(w\tau+b)=\frac{(w\tau+b)\sigma'(w\tau+b)-b\sigma'(w\tau+b)}{w}=\frac{k\sigma(w\tau+b)-b\sigma'(w\tau+b)}{w}.$$

The identity for the residual rows follows from the linearity of ρ. ∎ **[V2.5a]** checks the tangent-vector identity, **[V2.5b]** the residual identity.

So only two of the three rows of each neuron are independent: the a row and the b row.

**Lemma 2.6 (gauge symmetry).** For λᵢ > 0, (aᵢ, wᵢ, bᵢ) ↦ (aᵢ/λᵢᵏ, λᵢwᵢ, λᵢbᵢ) leaves q_θ unchanged; so does permuting the neurons. **[V2.6a, V2.6b]**

---

## 3. Theorem 1: a network without interior kinks is CGVI(P_k)

**Theorem 1.** Suppose θ satisfies:

- **(H1)** No neuron has its kink in (0,1). By Lemma 2.1 each neuron is then either active or inactive.
- **(H2)** T_θℳ = P_k. By Lemma 2.4, (H2) holds as soon as there are k+1 active neurons with distinct kinks. More precisely, it also holds whenever the active neurons have d distinct kinks with 2d ≥ k+1 and each group contains a neuron with aᵢwᵢ ≠ 0.

Then (θ, pₙ₊₁) satisfies the ShallowNet equations **if and only if** (q_θ, pₙ₊₁) satisfies the CGVI(P_k, R) equations (with the same R-point quadrature).

*Proof.*

- **Step 1: q_θ ∈ P_k.** By Lemma 2.1 each neuron is on [0,1] either a polynomial of degree k or identically zero. **[V3.1a]**
- **Step 2: T_θℳ ⊂ P_k.** By Lemma 2.2, for an active neuron ∂_a q = sᵏ, ∂_b q = k a sᵏ⁻¹, ∂_w q = k a τ sᵏ⁻¹, all in P_k; for an inactive neuron all three vectors vanish. **[V3.1a]**
- **Step 3: T_θℳ = P_k.** This is (H2).
  - For S = k+1, 6, 8 with distinct kinks, **[V3.1b]** checks that the rank equals k+1;
  - with kinks coalesced in groups, **[V3.1c]** checks that the rank equals min(2d, k+1).
- **Step 4: the two sets of equations are equivalent.** By Observation 1.2, the network equations are (E1) together with (E2) for δq ∈ T_θℳ = P_k. The CGVI(P_k) equations are the same (E1) together with (E2) for δq ∈ P_k. The trial function q_θ lies in P_k, both test spaces are P_k and the quadrature is the same, so the two sets of equations are **equivalent**, i.e. they have the same solution set (in terms of (q, pₙ₊₁)).
  - Note that the numbers of equations and unknowns differ: the network has 3S+1 unknowns (θ and pₙ₊₁) and 3S+1 equations; CGVI has k+2 unknowns and k+2 equations. Equivalence means the following. The i-th network equation is ρ(tᵢ) = 0 with tᵢ = ∂q_θ/∂θᵢ ∈ P_k. Expanding tᵢ in the CGVI basis functions φⱼ as tᵢ = Σⱼ Cᵢⱼ φⱼ, linearity of ρ shows that each network equation is a linear combination of the CGVI equations ρ(φⱼ) = 0; conversely, by (H2) the tᵢ span P_k, so each CGVI equation is a linear combination of network equations. Hence both sets constrain the same (q, pₙ₊₁).
  - The 3S − (k+1) extra unknowns of the network are exactly the gauge freedom (Corollary 3.1): they do not change q_θ and are not constrained by the equations, so the solutions of the network equations form a whole family of parameters θ, all representing the same q. ∎

**Converse construction (existence).** Given any CGVI solution (q*, p*), choose k+1 distinct kinks ∉ [0,1] and the corresponding activation signs. By Lemma 2.4(a), the functions (zᵢ − τ)ᵏ or (τ − zᵢ)ᵏ form a basis of P_k, so there is a unique a with q_θ = q*. By Theorem 1 this θ solves the network equations.

- **[V3.2a]** checks the fitting error;
- **[V3.2b]** checks that the residual is ≤ 1e-10 for the harmonic oscillator and the pendulum at h ∈ {0.1, 1, 5}.

**Uniqueness direction.** If the CGVI one-step solution is locally unique (its Jacobian is nonsingular, which the implicit function theorem guarantees for sufficiently small h), then every solution of the network equations in the (H1)(H2) region gives the same (qₙ₊₁, pₙ₊₁). **[V3.3a/b]** solve with LM from perturbed initial guesses and check that the kinks stay outside the interval and the result agrees with CGVI to 1e-10.

By induction on the steps: as long as every network step lands in the (H1)(H2) region, the whole trajectory coincides with CGVI. **[V3.5]** checks a 20-step trajectory. The warm start uses only a polynomial extrapolation of the previous network step, never the answer of the current step.

**Corollary 3.1 (structure of the Jacobian).** At a solution in the (H1)(H2) region, the Jacobian ((3S+1)×(3S+1)) of the network equations has rank k+2, and its null space consists exactly of the gauge directions (along which q_θ and pₙ₊₁ do not change).

*Proof.* The residual of row i is rᵢ(θ, p) = ρ_{q_θ,p}(tᵢ(θ)). Differentiating,

$$dr_i=(d\rho)[\delta q,\delta p](t_i)+\rho(dt_i).$$

1. For neurons with exterior kinks, dtᵢ still belongs to P_k (the second parameter derivatives are still polynomials of degree ≤ k). At the solution ρ vanishes on P_k, so the second term drops out.
2. Hence J = Φ∘Ψ, where Ψ : (δθ, δp) ↦ (δq = Σ tᵢδθᵢ, δp) ∈ P_k × ℝ. By (H2), Ψ is surjective with rank k+2.
3. Φ reads off the values of dρ on the tᵢ together with δq(0). Since {tᵢ} spans P_k, Φ carries the same information as the CGVI Jacobian, which is nonsingular, so Φ is injective.
4. Therefore rank J = k+2 and ker J = ker Ψ = {δq = 0, δp = 0}. ∎

**[V3.4a]** checks the rank, **[V3.4b]** checks that the null directions change neither q nor p.

Practical meaning: of the 3S+1 unknowns only k+2 are "real". The Jacobian of plain Newton is singular, so regularization (`regularization_factor`) or LM is required. The larger S, the larger the null space, of dimension 3S−k−1.

**Corollary 3.2 (width and power).**
- A ReLUᵏ network with any S ≥ k+1 is equivalent to CGVI(P_k), not CGVI(P_{S−1}), as long as all kinks are outside the interval. **[V3.7a/b]** check S = 6, 8.
- In particular, ReLU² with S = 4 is equivalent to CGVI(P₂, R = 4) and differs from CGVI(P₃). **[V3.6a/b/c]**

**Corollary 3.3 (symplecticity).** In the (H1)(H2) region the network's one-step map is the CGVI one-step map, hence generated by L_d^{P_k} (Lemma 1.3) and therefore symplectic. **[V3.8]** checks |det ∂(qₙ₊₁, pₙ₊₁)/∂(qₙ, pₙ) − 1| ≤ 1e-7 for the one-dimensional case.

The package currently has `issymplectic(::NetworkIntegratorMethod) = missing`. This corollary provides a proof under explicit conditions.

**Corollary 3.4 (order of convergence).** The error of the network equals the error of CGVI(P_k, R) step by step. Numerically, CGVI(P_k, R) has order 2k when R ≥ k (**[C5]**). So ReLU³ with R = 4 has order 6, consistent with the 6.00 measured for CGVI(4) by `run_convergence.jl`.

---

## 4. Other stationary points of the network equations: why Theorem 1 needs (H2)

**Proposition 4.1 (spurious solutions with coalesced kinks).** If all active neurons share the same kink z (d = 1), then q_θ = C(z − τ)ᵏ (or C(τ − z)ᵏ) and T_θℳ = span{(τ−z)ᵏ, (τ−z)ᵏ⁻¹}, of dimension 2 < k+1. The network equations then reduce to 3 independent conditions: ρ vanishes on this two-dimensional space, plus (E1). The effective unknowns are also 3: C, z, pₙ₊₁. So there are generically isolated solutions, and they are **not** CGVI solutions.

**[V4.1a/b/c]**:
1. solve the reduced equations by scanning and bisection over w = −1, z > 1;
2. embed the solution into 4 neurons with different weights but the same kink and check that the full network residual is ≤ 1e-9;
3. check that the tangent-space rank equals 2;
4. check |qₙ₊₁ − qₙ₊₁^{CGVI}| ≥ 1e-5.

In the prototype, for the harmonic oscillator at h = 1, z* ≈ 6.66 (about 6.8–7 for the pendulum), and qₙ₊₁ differs from CGVI by about 4e-2. The prototype's LM indeed converged on its own to such a solution from an ill-conditioned initial guess: all four kinks moved to 6.66. Diagnostics must therefore count the number of *distinct* kinks.

**Proposition 4.2 (zero-amplitude interior neurons are harmless).** Let (q*, p*) be a CGVI solution represented by some θ without interior kinks. If z ∈ (0,1) satisfies

$$\rho^*\big((\tau-z)_+^k\big)=0,$$

then adding to θ a neuron with a = 0, w = 1, b = −z gives a θ′ that still solves the network equations, and q_{θ′} = q*.

*Proof.* The a row of the new neuron is ρ*((τ−z)₊ᵏ), which vanishes by the choice of z; its w and b rows are multiplied by a = 0 and vanish; all other rows are unchanged. ∎

Conversely, if a converged network contains an interior neuron with a = 0, then ρ((τ − zᵢ)₊ᵏ) = 0 is enforced.

**[V4.2a]** checks that the residual is ≤ 1e-9 at a root z* of ρ (in the prototype the first root is z* ≈ 0.069); **[V4.2b]** checks that the residual is ≥ 1e-6 at z* + 0.05, showing the condition is not trivial.

**Step classification.** From these results, each step (for each degree of freedom) falls into one of three classes:

- **P_k-equiv**: only neurons without interior kinks and zero-amplitude neurons, and the tangent vectors of the former span P_k. The step is exactly CGVI(P_k).
- **degenerate**: no interior kinks, but the tangent space is smaller than P_k. Possibly a spurious solution.
- **spline**: there is an interior kink with nonzero amplitude. See §5.

Part A of `verify_conjectures.jl` performs this classification step by step for the actual `ShallowNet`.

---

## 5. Generalization: kinks inside the interval

**Lemma 5.1.** x₊ᵏ + (−1)ᵏ(−x)₊ᵏ = xᵏ, equivalently (−x)₊ᵏ = (−1)ᵏ(xᵏ − x₊ᵏ). **[V5.1]**

*Proof.* For x ≥ 0 the second term is 0; for x < 0 the first term is 0 and the second is (−1)ᵏ(−x)ᵏ = xᵏ. ∎

So a neuron with w < 0, (z − τ)₊ᵏ, lies in P_k ⊕ span{(τ − z)₊ᵏ}.

**Proposition 5.2 (function space and tangent space).** Let Z = {zᵢ ∈ (0,1)} be m interior kinks. Define

$$S_k(Z):=P_k\oplus\operatorname{span}\{(\tau-z)_+^k\}_{z\in Z}$$

the space of splines of degree k with knots Z and C^{k−1} continuity at the knots. Then q_θ ∈ S_k(Z), and

$$T_\theta\mathcal M\subset S_k(Z;2):=S_k(Z)\oplus\operatorname{span}\{(\tau-z)_+^{k-1}\}_{z\in Z}$$

i.e. every knot becomes a double knot. If there are k+1 neurons with distinct exterior kinks and m interior neurons with aᵢ ≠ 0, then dim T_θℳ = k+1+2m.

*Proof.* The three tangent vectors of an interior neuron (w > 0) are ∂_a q = wᵏ(τ−z)₊ᵏ, ∂_b q = k a wᵏ⁻¹(τ−z)₊ᵏ⁻¹ and ∂_w q = τ·∂_b q. Since τ(τ−z)₊ᵏ⁻¹ = (τ−z)₊ᵏ + z(τ−z)₊ᵏ⁻¹, ∂_w q also lies in S_k(Z;2). The case w < 0 reduces to this via Lemma 5.1. The dimension count follows from the linear independence of truncated powers. ∎

**[V5.2a]** checks q_θ ∈ S_k(Z) and T ⊂ S_k(Z;2) via projection residuals; **[V5.2b]** checks the dimension for k = 2..4, m = 1..3.

This is the exact variational-integrator version of "a one-dimensional shallow ReLUᵏ network = a free-knot spline" (cf. DeVore–Hanin–Petrova, *Acta Numerica* 2021): the trial set is free-knot splines, and the test space doubles every knot.

**Theorem 2 (fixed knots).** Freeze (w, b) and seek stationarity in a only (keep only the a rows and (E1)). The resulting equations are the CGVI equations on the linear space V = span{σ(wᵢτ + bᵢ)}. If V = S_k(Z), this is the Galerkin variational integrator on the spline space. Since (E1)(E2) depend only on the space, the result is independent of the choice of basis: the truncated-power basis, the B-spline basis and the frozen-neuron basis give the same solution. It is also symplectic (Lemma 1.3).

**[V5.3a]** checks the B-spline partition of unity; **[V5.3b]** checks that the three bases give the same solution (≤ 1e-11) both with composite quadrature and with a single global R = 8 rule.

**Theorem 3 (free knots = discrete Lagrangian stationary in the knots).** Suppose every neuron of θ has aᵢ ≠ 0 and wᵢ ≠ 0, the part without interior kinks spans P_k, and the set of interior kinks is Z. Suppose also that S_k(Z) satisfies the hypothesis of Lemma 1.3. Then (θ, pₙ₊₁) satisfies the network equations **if and only if** both

- **(i)** q_θ is the fixed-knot Galerkin step on S_k(Z) (Theorem 2), and
- **(ii)** for every interior knot,

$$\frac{\partial}{\partial z_i}L_d^{S_k(Z)}(q_n,q_{n+1})=0.$$

*Proof.*

1. By Lemma 2.5, each neuron's w row is a linear combination of its a row and b row (with coefficients k a/w and −b/w). So the system is equivalent to "all a rows + all b rows + (E1)".
2. The a rows plus (E1) say that ρ vanishes on span{σᵢ} = S_k(Z), which is (i).
3. For an interior neuron, ∂_b q = −(1/w) ∂_z q, where ∂_z q is the derivative of q with respect to z at fixed coefficients c. So the b row equals −(1/w) ρ(∂_z q). For neurons without interior kinks, ∂_z q ∈ P_{k−1} ⊂ S_k(Z) and the row vanishes automatically by (i).
4. **Envelope formula.** View q_Z = Σ cⱼφⱼ(·; Z) as the stationary point of the type-I problem on {q ∈ S_k(Z) : q(0) = q_a, q(1) = q_b} with multipliers λ₀, λ₁. By the envelope theorem,

   $$\frac{dL_d}{dz}=\frac{\partial A}{\partial z}\Big|_c-\lambda_1\frac{\partial q(1)}{\partial z}\Big|_c-\lambda_0\frac{\partial q(0)}{\partial z}\Big|_c.$$

   Substituting λ₁ = pₙ₊₁, λ₀ = −pₙ (Lemma 1.3) gives

   $$\frac{dL_d}{dz}=\delta A[\partial_zq]-p_{n+1}\partial_zq(1)+p_n\partial_zq(0)=-\rho(\partial_zq).$$

5. Hence b row = (1/w)·dL_d/dz, and the b row vanishes ⇔ (ii). ∎

**[V5.4a]** checks the envelope formula (finite differences vs closed form); **[V5.4b]** checks that the type-I solution satisfies all a rows; **[V5.4c]** checks that the b row equals (1/w)·dL_d/dz; **[V5.4d]** finds the stationary knot Z* (Z* ≈ 0.43849 in the prototype) and checks that the full network residual is ≤ 1e-8.

**Corollary 5.3 (symplecticity).** Near a nondegenerate stationary point Z*(q_a, q_b) (a smooth branch by the implicit function theorem), the network step is generated by

$$L_d^{NN}(q_a,q_b)=L_d^{S_k(Z^*(q_a,q_b))}(q_a,q_b).$$

Applying the envelope theorem once more, (ii) ensures ∂L_d^{NN}/∂q = ∂L_d^Z/∂q. So the free-knot network step is also a variational integrator and hence symplectic. **[V5.6]** checks |det − 1| ≤ 1e-7 (6e-11 in the prototype).

**Remark 5.4 (quadrature).** The integrand L(q_θ, q′_θ) is only C^{k−2} smooth at the kinks (in q′). A single global R-point Gauss rule is not exact for it, whereas composite Gauss quadrature split at the kinks is exact. **[V5.5a/b]**

Consequences:
- With a global rule the discrete action itself carries an O(1)-type error independent of h, which wipes out the benefit of the spline space (see C4).
- When the number of interior knots m satisfies k+1+m > R, there are too few quadrature points and the equations may even become singular (the case m = 3 with a global R = 4 rule in C4).
- The composite quadrature nodes move with θ, so in the free-knot problem A_h is only piecewise smooth in θ.

---

## 6. Conjectures and numerical tests

### C1–C3 (Part A, the actual package)

**C1.** Every step of ShallowNet(ReLU³, S = 4, R = 4) should be **P_k-equiv** and equal to CGVI(P₃, R = 4): diff ≤ max(1e-9, 0.01·err_CGVI). If a step is classified as spline or degenerate, the theory makes no prediction for it and the script reports it as is; the "almost identical errors" seen by the original comparison script should fall in the P_k-equiv class.

**C2.** For S = 6, 8, the theory predicts that every run whose steps are all P_k-equiv equals CGVI(P₃) and is no better.
- `bias = [1.1, π]` puts the kinks of all atoms of the OGA dictionary (w = ±1) outside the interval and in the active state, so the initial guess satisfies (H1);
- `bias = [−π, π]` (the default) allows OGA to pick interior kinks.

**C3.** ReLU² with S = 4 should equal CGVI(P₂, R = 4), and its difference from CGVI(P₃) should be at least 10 times larger.

Decision rules:
- Only when all steps are P_k-equiv and no solve hits maxiter does the theory make the hard prediction "must be equal". If they are then not equal the run is marked **VIOLATED** (check fails); if equal, **CONFIRMED**.
- In all other cases the result is only recorded ("equal anyway" or "differs") and not counted as a failure.

### C4 (Part B): can fixed interior knots reach a smaller error with fewer unknowns?

Prototype results: harmonic oscillator, T = 10, maximum relative error in q.

| Method | h=0.1 | 0.2 | 0.5 | 1 | 2 | 5 |
|---|---|---|---|---|---|---|
| CGVI P₃ R4 | 3.93e-11 | 2.51e-09 | 6.08e-07 | 3.79e-05 | 2.18e-03 | 1.43e-01 |
| spline m=1, global R4 | 5.54e-06 | 2.18e-05 | 1.23e-04 | 2.92e-04 | 2.14e-03 | 1.01e-01 |
| spline m=1, global R8 | 2.72e-08 | 8.34e-08 | 6.00e-07 | 1.93e-05 | 4.13e-04 | 2.72e-02 |
| **spline m=1, composite R4** | **1.09e-12** | 7.00e-11 | 1.74e-08 | 1.17e-06 | 8.85e-05 | 1.98e-02 |
| CGVI P₃ R4, 2 substeps | 6.12e-13 | 3.93e-11 | 9.56e-09 | 6.08e-07 | 3.79e-05 | 5.20e-03 |
| spline m=3, global R4 | singular | | | | | |
| spline m=3, global R16 | 2.83e-05 | 2.84e-05 | 2.92e-05 | 3.14e-05 | 3.05e-05 | 7.20e-04 |
| **spline m=3, composite R4** | **2.58e-14** | 1.64e-12 | 4.02e-10 | 2.63e-08 | 1.84e-06 | 5.05e-04 |
| CGVI P₃ R4, 4 substeps | 2.10e-14 | 6.12e-13 | 1.50e-10 | 9.56e-09 | 6.08e-07 | 9.74e-05 |

The script performs four automatic checks:
- **[C4a]** the frozen-neuron basis and the truncated-power basis give the same error (Theorem 2);
- **[C4b]** with composite quadrature the order stays 6 (|p − 6| ≤ 0.4);
- **[C4c]** the spline error is of the same magnitude as "CGVI on m+1 substeps" (|log₁₀ ratio| ≤ 0.7; the prototype ratios are 1.8 and 2.7);
- **[C4d]** with a global R4 rule the order drops to ≤ 3 (2.0 in the prototype).

Conclusion: with m fixed interior knots and composite quadrature, one solve of dimension k+2+m (coefficients plus pₙ₊₁) reaches nearly the error of CGVI with the step size reduced m+1 times, which needs m+1 sequential solves of dimension k+2. The order is unchanged (2k); what decreases is the error constant, by roughly a factor (m+1)^{2k}. C7, which counts FLOPs, further shows that fewer parameters do not mean less computation: at equal FLOPs, fixed-knot splines and CGVI substeps are roughly on par, with substeps slightly ahead.

### C5: the order of CGVI(P_s, R)

Observed orders in the prototype, h = 0.2 → 0.5:

- (s, R) = (2,3): 3.99
- (2,4): 3.99
- (3,3): 5.99
- (3,4): 5.99
- (4,4): 7.99
- (4,5): 7.98

Conjecture: **the order is 2s when R ≥ s.**

Outside this hypothesis, (4,3) agrees digit by digit with CGVI(P₂, 3) (order 4), and (3,2) has order about 2: with too few quadrature points the higher-degree terms become "invisible". These cases are only printed, not checked.

### C6: what does LM converge to with free knots?

Setting: k = 3, S = 5 (4 neurons without interior kinks plus 1 neuron whose initial kink z₀ is interior), global R = 8, pendulum, h = 1.

- If the solve converges and is **P_k-equiv**, check that it equals CGVI(P₃).
- If it is **spline**, check Theorem 3: the type-I solve reproduces the step and ∇_Z L_d = 0.

The 8 prototype runs:
- (z₀, a₅) = (0.8, 1e-3): converges to a zero-amplitude neuron (Proposition 4.2), |Δq| = 0;
- (0.2, 0.1): converges to a genuine free-knot spline solution, z = 0.435, |Δq₁| = 1.8e-5, Theorem 3 check value 3.5e-11;
- the other 6 runs do not converge within 500 LM iterations: the kinks drift towards 0 and 0.97.

This is consistent with Remark 5.4 and with "the stationary point is not a minimum": the free-knot problem is ill-conditioned.

### C7: free-knot spline VI vs fixed knots vs CGVI: parameters, FLOPs, error

**Method (`free_knot_vi.jl`).** The solution of Theorem 3 is computed in two levels:
- **Inner level**: for given knots Z, take one step of the Galerkin variational integrator on S_k(Z), with composite Gauss quadrature split at Z; the unknowns are the spline coefficients and pₙ₊₁, solved by Newton with an analytic Jacobian. There is no gauge freedom at this level and no regularization is needed.
- **Outer level**: root-finding in the knots, g(Z) = ∂L_d^Z(qₙ, qₙ₊₁(Z))/∂Z = 0. g is computed analytically and, besides the envelope formula, includes a term for the motion of the composite quadrature nodes with Z; its Jacobian uses forward differences (m extra inner solves per outer iteration). The knots are kept in (0.02, 0.98) with spacing at least 0.02, each move is at most 0.1, and each step starts from the previous step's knots. If the outer level does not converge, the knots with the smallest |g| are used and the step is counted as a fallback; such steps are no longer strictly variational.

Self-checks: the analytic g agrees with finite differences of L_d; the free-knot one-step map satisfies |det − 1| ≤ 1e-7 (symplectic); CGVI in the monomial basis gives the same result as CGVI in the Lagrange basis.

**Accounting.** Parameters per step: CGVI k+1; m+1 substeps (m+1)(k+1); fixed-knot spline k+1+m; free-knot spline k+1+2m (the extra m are the knot positions). A ShallowNet of the same function class has 3(k+1+m) parameters, of which only k+1+2m are effective. FLOPs are counted by the code with an explicit model (residual, analytic Jacobian, LU, basis evaluation, knot gradient; sin/cos count as 10 operations); they represent the operation count of an efficient implementation, not measured wall time.

**Results** (k = 3, R = 4, t ∈ (0, 20), (q₀, p₀) = (0.5, 0); q error is the maximum relative error on the grid; full tables in `results/free_knot_vi.md`):

| Problem | h | Method | params/step | q error | H error | FLOP/step | fallbacks |
|---|---|---|---|---|---|---|---|
| harmonic osc. | 1 | CGVI P₃ | 4 | 8.74e-05 | 1.13e-05 | 2.64e+03 | — |
| | | fixed spline m=1 | 5 | 2.70e-06 | 2.61e-07 | 4.42e+03 | — |
| | | free spline m=1 | 6 | 2.33e-06 | 2.24e-07 | 2.36e+04 | 0 |
| | | CGVI 2 substeps | 8 | 1.40e-06 | 1.60e-07 | 5.26e+03 | — |
| | | fixed spline m=2 | 6 | 3.28e-07 | 3.64e-08 | 6.85e+03 | — |
| | | free spline m=2 | 8 | 6.83e-07 | 2.18e-07 | 2.61e+05 | 8 |
| | | CGVI 3 substeps | 12 | 1.24e-07 | 1.38e-08 | 7.88e+03 | — |
| | | fixed spline m=3 | 7 | 6.07e-08 | 6.79e-09 | 1.00e+04 | — |
| | | free spline m=3 | 10 | 2.32e-07 | 3.09e-08 | 1.11e+06 | 18 |
| | | CGVI 4 substeps | 16 | 2.21e-08 | 2.44e-09 | 1.05e+04 | — |
| harmonic osc. | 5 | CGVI P₃ | 4 | 4.36e-01 | 2.50e-01 | 2.71e+03 | — |
| | | fixed spline m=1 | 5 | 6.71e-02 | 3.04e-02 | 4.57e+03 | — |
| | | free spline m=1 | 6 | 1.02e-01 | 1.46e-01 | 4.19e+04 | 0 |
| | | fixed spline m=3 | 7 | 1.70e-03 | 9.16e-06 | 1.04e+04 | — |
| | | free spline m=3 | 10 | 1.03e-03 | 1.40e-03 | 7.50e+05 | 2 |
| | | CGVI 4 substeps | 16 | 3.27e-04 | 4.28e-05 | 1.06e+04 | — |
| pendulum | 1 | CGVI P₃ | 4 | 6.26e-05 | 9.95e-06 | 3.88e+03 | — |
| | | fixed spline m=2 | 6 | 2.41e-07 | 2.78e-08 | 1.26e+04 | — |
| | | free spline m=2 | 8 | 1.63e-06 | 4.03e-07 | 5.08e+05 | 17 |
| | | CGVI 3 substeps | 12 | 9.01e-08 | 1.00e-08 | 1.01e+04 | — |
| pendulum | 5 | CGVI P₃ | 4 | 3.90e-01 | 2.36e-01 | 5.34e+03 | — |
| | | fixed spline m=3 | 7 | 1.61e-03 | 7.66e-04 | 2.85e+04 | — |
| | | free spline m=3 | 10 | 4.12e-04 | 3.15e-04 | 5.46e+05 | 0 |
| | | CGVI 4 substeps | 16 | 2.25e-04 | 2.72e-05 | 1.60e+04 | — |

**Observations.**
1. **Fixed-knot splines improve a lot on CGVI at the same step size, at small cost.** For the harmonic oscillator at h = 1, m = 1 reduces the error 32-fold at 1.7× the FLOPs; m = 3 reduces it about 1400-fold at 3.8× the FLOPs. The energy error is also small, e.g. harmonic oscillator, h = 5, m = 3: q error 1.7e-3, H error only 9.2e-6.
2. **But fixed-knot splines are not more efficient than CGVI substeps.** For the same m their error is about 2–3 times that of m+1 substeps; at equal FLOPs the two are roughly on par, with substeps slightly ahead (harmonic oscillator, h = 0.5: spline m = 3 at about 1.0e4 FLOP/step with error 9.3e-10; 4 substeps at about 1.05e4 FLOP/step with error 3.5e-10). Their only advantage is the parameter count (7 vs 16 for m = 3). The larger m, the worse for the spline: its Jacobian grows like (k+1+m)², while substeps are m+1 small problems of size (k+1)².
3. **Free knots bring no gain on these two problems.** They cost 5–110× the FLOPs of fixed knots (about two orders of magnitude for m = 3), and for m ≥ 2 the outer level often fails to converge (up to 22 fallback steps); the error is mostly on par with or worse than fixed knots (7× worse for the pendulum at h = 1, m = 2), and the energy error is usually larger (harmonic oscillator, h = 5, m = 1: 0.146 vs 0.030). The only clear win is h = 5, m = 3 (4× better for the pendulum, 1.7× for the harmonic oscillator), but there CGVI with 4 substeps is even more accurate at about 1/34 of the FLOPs.
4. **Reason.** The variational principle selects knots that make the discrete action stationary, not knots that minimize the error. This guarantees symplecticity (Corollary 5.3) but not that the knots sit in "good" places. The free knots can be clearly better on the error of a single step (first step at h = 5: 8.7e-4 with free knots vs 9.0e-3 with fixed z = 0.5), but this advantage largely disappears over many steps.

**Conclusion (C7).** For smooth problems, what is really useful in the ReLUᵏ network function class is "a larger trial space + compatible quadrature". Its effect is essentially equivalent to subdividing the step and is not cheaper than simply using CGVI substeps; letting the knots move (the genuinely nonlinear part) brings no gain here. Free knots may still help for problems with fast local variation within one step (e.g. the pericenter passage of a highly eccentric Kepler orbit); this is an untested conjecture.

---

## 7. Answer to "generalize to many neurons and reach a similar error"

1. **The equivalence that can be proven rigorously is "same trial function + same test space".** For ReLUᵏ the test space is the tangent space of the network with respect to its parameters. When all kinks are outside the interval it is exactly P_k (Theorem 1), so a network of any width is **exactly equal** to CGVI(P_k): same error, and no better. Extra neurons only enlarge the gauge null space (Corollary 3.1) and add spurious solutions (Proposition 4.1).
2. **For more neurons to reduce the error, kinks must enter the interval.** The method then becomes a spline Galerkin variational integrator: Theorem 2 for fixed knots, Theorem 3 for free knots. In both cases it can be proven to be a variational integrator generated by a discrete Lagrangian, hence symplectic (Corollary 5.3). But **being provably symplectic does not mean a smaller error**: in the measurements (C7), free knots are not more accurate than fixed knots on the harmonic oscillator and the pendulum, while costing one to two orders of magnitude more computation.
3. **Practical recommendations (revised according to the C4 and C7 measurements).**
   - If the goal is a variational integrator built from a ReLUᵏ network: pick the kinks with OGA (or simply a uniform grid), freeze (w, b), solve only for the coefficients a (Theorem 2), and use composite quadrature split at the kinks (Remark 5.4). This has a rigorous equivalence proof and symplecticity, no gauge null space and no spurious solutions; the order stays 2k, the error is about that of CGVI(P_k) with step h/(m+1), and the energy error is small.
   - But be clear about its place: measured in FLOPs it is roughly on par with, or slightly worse than, plain CGVI substeps; its only advantage is fewer parameters per step. For smooth problems this route is "another way of subdividing the step" and does not surpass linear methods.
   - **Free knots (Theorem 3) are not recommended for smooth problems.** They preserve symplecticity, but in C7 their error is mostly no better than fixed knots, their energy error is usually larger, they cost 5–110× the FLOPs, and for m ≥ 2 the outer root-finding often fails. Moving the knots according to the variational principle does not place them where the error is small. The earlier claim that "free knots are theoretically better" holds only for the single-step error, not over many steps.
   - Free knots may be worth trying for problems with fast local variation within a step (highly eccentric Kepler orbits, near-collisions, multiscale problems), where uniform knots are wasted on the smooth parts; this remains to be tested.
4. **Diagnostics.** At every step read out θ and count: the number and amplitudes of interior kinks, the number of distinct kinks in the part without interior kinks, and the tangent-space rank. This determines whether the step is P_k-equiv, spline or degenerate (`classify_step` in `verify_conjectures.jl`). We suggest adding this diagnostic to the package's `record_finer_solution!` or to the benchmarks.
