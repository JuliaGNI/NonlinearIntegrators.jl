# Equivalence of NVI with Shallow ReLUᵏ Networks and Continuous Galerkin Variational Integrators: Proofs, Generalizations and Numerical Verification

This note turns the observation

> NVI with a shallow network in the configuration $S4R4Q8\,\mathrm{ReLU}^3$ and the continuous Galerkin variational integrator $P3R4Q8$ produce almost identical errors

into theorems that can be checked step by step. It then generalizes them to arbitrary width $S$, arbitrary power $k$, and to the case where kinks lie inside the time step. Notation follows the dissertation (Chapter "Continuous Galerkin Variational Integrators", §`CGVI`, and Chapter "Nonlinear Variational Integrators", §`sec:nvi_action` and §`sec:nvi_shallow_plain`); §0 lists the correspondence with the symbols used in the code.

**Every proof step carries a verification tag**, e.g. **[V2.4]**. It refers to the numerical check in `verify_proof_steps.jl` that prints the same tag. All predictions (conjectures) are tagged **[C1]–[C7]** and correspond to `verify_conjectures.jl` (C1–C6) and `free_knot_vi.jl` (C7).

| File | Content | Dependencies |
|---|---|---|
| `relu_cgvi_equivalence.md` | Chinese version of this note | — |
| `relu_cgvi_equivalence_en.md` | This note | — |
| `theory_common.jl` | Reference implementation written from the definitions: discrete action, weak-form residual, Lagrange / truncated-power / B-spline / frozen-neuron bases, network residual and tangent vectors, Newton, LM, type-I solve | standard library only |
| `verify_proof_steps.jl` | Numerical check of every proof step V1.1–V5.6; prints PASS/FAIL per check and exits with code 1 on any failure | standard library only |
| `verify_conjectures.jl` | Part A: the actual `ShallowNet` and `CGVI` (C1–C3), reading $\Theta_n$ at every step and classifying it as in this note; Part B: C4–C6 on the reference implementation | `benchmark` project |
| `free_knot_vi.jl` | Free-knot spline variational integrator (outer root-finding on the knots, inner fixed-knot Galerkin, composite quadrature following the knots), compared with CGVI, CGVI with substeps and fixed-knot splines in parameters, FLOPs and error (C7) | standard library only |
| `relu_k_sweep.jl` | ShallowNet(ReLUᵏ / tanh, $S$) vs CGVI($\Pi^k$) over $k$, $S$ and $h$, with error in $q$ and relative Hamiltonian error | `benchmark` project |

```
julia --project=benchmark benchmark/theory/verify_proof_steps.jl
julia --project=benchmark benchmark/theory/verify_conjectures.jl                 # A + B
julia --project=benchmark benchmark/theory/verify_conjectures.jl --reference-only  # B only
julia --project=benchmark benchmark/theory/verify_conjectures.jl C1 C3a --final-time=40
julia benchmark/theory/free_knot_vi.jl                                         # C7
```

> **About the numbers.** All scripts have been run in Julia: `verify_proof_steps.jl` passes completely; Part B of `verify_conjectures.jl` (C4–C6) agrees digit by digit with the original Python prototype of the same algorithms, so the numbers labelled "prototype" in C4–C6 of §6 are also the Julia results; the C7 numbers are taken directly from `results/free_knot_vi.md`.

---

## 0. Notation

### 0.1 Correspondence with the dissertation and the code

| Object | Dissertation / this note | `theory_common.jl` | Package (`src/nvi/shallownet.jl`) |
|---|---|---|---|
| step size | $h$ | `h` | `timestep(int)` |
| local time in one step | $x = (t-t_n)/h \in [0,1]$ | `t` (called τ in comments) | — |
| Lagrangian | $L(q,\dot q)$ | $L = v^2/2 - V(q)$ (`ToyProblem`) | — |
| partial derivatives at the nodes | $\frac{\partial L}{\partial q}(q_j,v_j)$, $\frac{\partial L}{\partial \dot q}(q_j,v_j)$ | `f = -dV(q)`, `V = dq/h` (since $\partial L/\partial v = v$) | `F[j]`, `P[j]` |
| quadrature weights, nodes | $\{\alpha_j,\beta_j\}_{j=1}^R$ | `quad.b`, `quad.c` | `method(int).b`, `method(int).c` |
| network parameters of component $d$ | $\Theta = \{Q^i, W^i, b^i\}_{i=1}^S$ | `θ = (a, w, b)` | `ps[d]` |
| output weight / input weight / bias of neuron $i$ | $Q^i$, $W^i$, $b^i$ | `a[i]`, `w[i]`, `b[i]` | `ps[d][2].W[i]`, `ps[d][1].W[i]`, `ps[d][1].b[i]` |
| network basis function | $\psi^i(x) = \sigma(W^i x + b^i)$ | `neuron_basis` | first layer `Dense(1, S, σ)` |
| activation | $\sigma(s) = \mathrm{ReLU}^k(s) = \max(0,s)^k$ | `rp(s, k)` | activation of the first layer |
| kink of neuron $i$ | $z^i = -b^i/W^i$ | `-b ./ w` | — |
| polynomial space of degree $\le k$ | $\Pi^k$ | `monomial_basis(k)` | — |
| CGVI, Lagrange basis of degree $a$, $r$-point Gauss rule | $PaRrQu$ with $u = 2r$, e.g. $P3R4Q8$ | `cgvi_basis(a)`, `QRule01(r)` | `CGVI` with $a+1$ basis functions |
| NVI, $s$ neurons, activation $\sigma$, $r$-point Gauss rule | $SsRrQu\sigma$, e.g. $S4R4Q8\,\mathrm{ReLU}^3$ | `nn_residual(…, k, quad, …)` | `ShallowNet` |
| Jacobian regularization | $J_\lambda = J + \lambda I$ (dissertation) | not used; `lm_solve` uses Levenberg–Marquardt damping $(J^TJ+\mu I)\,\delta x=-J^Tr$, a different regularization | solver option `regularization_factor` |
| numerical flow map | $\Phi_h$ | `step_map_det` checks $\det\Phi_h'$ | — |

So $P3R4Q8$ is the `CGVI(S = 4, R = 4)` setting of the comparison script `compare_relu3_shallownet_vs_cgvi.jl` (4 Lagrange basis functions, i.e. degree 3), and $S4R4Q8\,\mathrm{ReLU}^3$ is `ShallowNet` with ReLU³, $S = 4$, $R = 4$. Since Gauss quadrature is used throughout, $u = 2R$ and we write CGVI($\Pi^k$, $R$) when the basis does not matter.

### 0.2 Conventions inside one step

Inside a step we drop the step index $n$ from the parameters: $\Theta = \Theta_n$, $Q^i = Q_n^i$, etc. As in the dissertation, `ShallowNet` uses an independent network for every component $q^d$, $d = 1,\dots,D$ (the parameter set `cache.ps[d]`); the trial set is the product of these networks and the one-step equations decouple into $D$ copies of the same structure, coupled only through $L$. All statements below are made for one component ($D=1$) and hold component by component.

The trajectory on $[t_n,t_{n+1}]$ is (dissertation, eq. `eqn:one_layer_coordinates`)

$$q_h(x;\Theta)=\sum_{i=1}^S Q^i\,\psi^i(x)=\sum_{i=1}^S Q^i\,\sigma(W^i x+b^i),\qquad \dot q_h=\frac1h\,q_h',\quad q_h' := \frac{d q_h}{dx}.$$

There is no output bias, identical to `ShallowNetBasis`: `Dense(1,S,σ) → Dense(S,1)`. We abbreviate $y^i(x) := W^i x + b^i$, and at the quadrature nodes

$$q_j := q_h(\beta_j),\qquad v_j := q_h'(\beta_j)/h .$$

Throughout, $k\ge 2$ (the dissertation uses $\mathrm{ReLU}^2$ and $\mathrm{ReLU}^3$; the checks cover $k = 2,\dots,6$). Note that $k$ here is the power of the activation, which equals the polynomial **degree** of the pieces; the dissertation's list of notations uses $k$ for the B-spline **order**, which is $k+1$ in the present notation. The symbol $J$ always denotes the Jacobian of the one-step equations; the canonical symplectic matrix is written $\mathbb J$. In $SsRrQu\sigma$ the letter $s$ is the number of neurons, while in $PsRrQu$ it is the polynomial degree (the dissertation's convention); in the text the width is always $S$.

### 0.3 Trial space and test space

Two basic notions of Galerkin methods are used:

- **Trial space** (or trial set): the set of functions in which the approximate trajectory $q_h$ is sought, i.e. all functions the ansatz can represent. For $P3R4Q8$ it is $\Pi^3$; for $SsRrQu\,\mathrm{ReLU}^k$ it is the set $\mathcal M := \{q_h(\cdot;\Theta)\}$ of all functions $\sum_i Q^i\sigma(W^i x+b^i)$. $\mathcal M$ is not a linear space.
- **Test space** $\mathcal T$: the set of variations $\delta q$ against which the discrete Euler–Lagrange equations are "tested". The exact solution makes the action stationary for **all** variations $\delta q$, which is the continuous Euler–Lagrange equation. A discrete method only requires stationarity for $\delta q \in \mathcal T$; each linearly independent direction in $\mathcal T$ yields one equation.

The trial space determines "what the solution looks like"; the test space determines "which equations it must satisfy". When both coincide the method is a (Bubnov–)Galerkin method. The central observation of this note (Observation 1.2) is that two methods can be compared by comparing these two objects alone.

---

## Summary of results

Let $\sigma(s) = \max(0,s)^k$ and let $z^i = -b^i/W^i$ be the kink of neuron $i$.

1. **Theorem 1 (no interior kink ⇒ NVI = CGVI($\Pi^k$)).** If no neuron has its kink in $(0,1)$, and the tangent vectors of the network span $\Pi^k$, then the DEL equations of NVI ($SsRrQu\,\mathrm{ReLU}^k$, eq. `eqn:DEL_NVI`) and of CGVI($\Pi^k$, $R$) (eq. `DEL_2`) are **equivalent**. The numbers of equations differ ($3S+1$ for NVI, $k+2$ for CGVI), but each set of equations is a linear combination of the other and both determine the same $(q_h, p_{n+1})$. The two methods therefore have the same one-step map $\Phi_h$, the same trajectory and the same error, for every $S$. As long as all kinks stay outside the interval, adding neurons does not reduce the error.
2. **Corollary 3.1 (Jacobian).** For each component the DEL equations have $3S+1$ unknowns ($Q^i, W^i, b^i$ for the $S$ neurons, plus $p_{n+1}$), but at a solution their Jacobian $J$ has rank only $k+2$. The remaining $3S-k-1$ directions form the **null space** $\ker J$, along which Newton's method cannot determine a step. They are exactly the **gauge freedom** (parameter symmetry): directions that change $\Theta$ but not $q_h$ (and hence not $p_{n+1}$). Examples: rescaling one neuron $(Q^i,W^i,b^i)\mapsto(Q^i/\gamma^k,\gamma W^i,\gamma b^i)$, or moving an exterior kink while readjusting the output weights. Because $J$ is singular, the plain Newton system has no unique solution; a regularization such as $J_\lambda = J+\lambda I$ (`regularization_factor`) makes the linear solves uniquely solvable (for $\lambda$ with $-\lambda$ not an eigenvalue of $J$, i.e. generically). If the regularized iteration converges to a root of the DEL equations, $\lambda$ only selects which of the equivalent parameter sets is reached; it changes neither $q_h$ nor $p_{n+1}$. $\mathrm{ReLU}^2$ is equivalent to CGVI($\Pi^2$), not CGVI($\Pi^3$). The step is symplectic.
3. **Propositions 4.1 / 4.2.** The condition "the kinks are distinct" cannot be dropped: there are spurious solutions with coalesced kinks that are not CGVI solutions. On the other hand, neurons with zero output weight and an interior kink are harmless: $q_h$ is still the CGVI solution.
4. **Theorem 2 (fixed interior kinks).** With $(W^i,b^i)$ frozen, NVI is the Galerkin variational integrator on the spline space $\mathcal S_k(Z)$, and the result does not depend on the basis (truncated powers, B-splines, or the frozen neurons).
5. **Theorem 3 (free kinks).** A full NVI step equals "the spline Galerkin step on the knots $Z^*$", where $Z^*$ is a stationary point of the discrete Lagrangian $L_d^{Z}(q_n,q_{n+1})$ with respect to $Z$. It is therefore still generated by a discrete Lagrangian and is symplectic.
6. **The right direction for generalization (C4).** Add $m$ fixed knots inside the interval. Here $m$ concerns both the function space and the quadrature. On the one hand, $m$ is the number of neurons whose kinks lie in $(0,1)$ (on top of $k+1$ neurons without interior kinks); they enlarge the trial space from $\Pi^k$ to $\mathcal S_k(Z)$ and raise its dimension from $k+1$ to $k+1+m$. On the other hand, the quadrature must be split at these knots (composite Gauss quadrature). With both in place the order stays $2k$ and the error is roughly that of CGVI($\Pi^k$) with step $h/(m+1)$. Adding neurons with a single global Gauss rule lets the quadrature error wipe out the benefit of the spline; splitting the quadrature without interior-kink neurons leaves the trial space at $\Pi^k$ and the error unchanged.

---

## 1. The one-step equations

### 1.1 From the Lagrange-multiplier action to a weak form

The dissertation builds both CGVI (eq. `action_2`) and NVI (§`sec:nvi_action`) on the discrete action with Lagrange multipliers. Restricted to one step $[t_n,t_{n+1}]$ it reads

$$\mathcal A_d^n(\Theta, q_n, q_{n+1}, p_n, \bar p_{n+1})=\underbrace{h\sum_{j=1}^R\alpha_j\,L\big(q_h(\beta_j;\Theta),\,q_h'(\beta_j;\Theta)/h\big)}_{=:~\mathbb L(q_h)}+p_n\big[q_h(0)-q_n\big]+\bar p_{n+1}\big[q_{n+1}-q_h(1)\big].$$

$\mathbb L(q_h)$ is the quadrature approximation of $\int_{t_n}^{t_{n+1}}L\,dt$ along the trial function. For CGVI, $\Theta$ is replaced by the coefficients $Q_n$ of a basis of $\Pi^k$. The variation of $\mathbb L$ in a direction $\delta q$ is

$$\delta\mathbb L(q_h)[\delta q]=\sum_{j=1}^R\alpha_j\Big(h\,\frac{\partial L}{\partial q}(q_j,v_j)\,\delta q(\beta_j)+\frac{\partial L}{\partial \dot q}(q_j,v_j)\,\delta q'(\beta_j)\Big).$$

Stationarity of $\sum_n \mathcal A_d^n$ with respect to $p_n$, $\bar p_{n+1}$ and $q_n$ gives (dissertation, eq. `DEL_2`) $q_h(0)=q_n$, $q_{n+1}=q_h(1)$ and $\bar p_n = p_n$, so the multipliers are the discrete momenta and $\bar p_{n+1}$ can be written $p_{n+1}$. Stationarity with respect to the trial parameters gives the remaining equations. Define the linear functional on variations

$$\rho(\delta q)\;=\;\rho_{q_h,p_{n+1}}(\delta q)\;:=\;p_{n+1}\,\delta q(1)-p_n\,\delta q(0)-\delta\mathbb L(q_h)[\delta q].$$

Given $(q_n,p_n)$, the unknowns are the trial function (through its parameters) and $p_{n+1}$, and the one-step equations are

$$\text{(E1)}\quad q_h(0)=q_n,\qquad\qquad \text{(E2)}\quad \rho_{q_h,p_{n+1}}(\delta q)=0\quad\forall\,\delta q\in\mathcal T .$$

After solving, $q_{n+1} := q_h(1)$.

- **CGVI($\Pi^k$, $R$).** The trial space is $\Pi^k$ and $\mathcal T=\Pi^k$. With any basis $\{\phi^i\}$ of $\Pi^k$, the first line of `DEL_2` is $D_iL_d+p_n\phi^i(0)-p_{n+1}\phi^i(1) = -\rho(\phi^i)=0$ and the second line is (E1). This is a system of $k+2$ equations for the $k+1$ coefficients and $p_{n+1}$. The documentation of `CGVI` in GeometricIntegrators states the same: continuity $q_h(t_n)=q_n$ is imposed weakly through a multiplier and the unknowns are the $S$ coefficients plus $p_{n+1}$.
- **NVI ($SsRrQu\,\mathrm{ReLU}^k$).** The DEL equations `eqn:DEL_NVI` have four groups of rows. Each row of `eqn:DEL_NVI` equals $-\rho(\cdot)$ of a tangent vector; `residual!` in `src/nvi/shallownet.jl` stores every row, including the constraint row `q̄ − r₀·X` $= q_n-q_h(0)$, with the opposite sign. (In the code quotes below, `b[j]` is a quadrature weight, `a` holds $\psi^{i\prime}$ at the nodes, and `k` is the component index; these are code names, not the symbols of this note.)
  - $Q$ rows: $h\sum_j\alpha_j\frac{\partial L}{\partial q}\psi^i(\beta_j)+\sum_j\alpha_j\frac{\partial L}{\partial\dot q}\psi^{i\prime}(\beta_j)+p_n\psi^i(0)-p_{n+1}\psi^i(1) = -\rho(\psi^i)$ (code: `(r₁[i,k] p̃[k] − r₀[i,k] p̄[k]) − Σⱼ b[j] (h m F + a P)`);
  - constraint row: $\sum_i Q^i\sigma(b^i)-q_n = 0$, i.e. (E1);
  - $W$ rows: $-\rho(\partial q_h/\partial W^i)$. They contain no $p_n$ term because $\partial q_h(0)/\partial W^i=Q^i\sigma'(b^i)\cdot 0=0$, as in `eqn:DEL_NVI` (code: `dqdWr₁[i,k] p̃[k] − Σ …`);
  - $b$ rows: $-\rho(\partial q_h/\partial b^i)$ (code: `(dqdbr₁ p̃ − dqdbr₀ p̄) − Σ …`).

  So NVI solves exactly (E1)(E2), with the test space equal to the tangent space of the trial set at $\Theta$:

  $$\mathcal T=T_\Theta\mathcal M:=\operatorname{span}\Big\{\frac{\partial q_h}{\partial Q^i},\frac{\partial q_h}{\partial W^i},\frac{\partial q_h}{\partial b^i}\Big\}_{i=1}^S .$$

  In total there are $3S+1$ equations for the $3S+1$ unknowns $(\Theta,p_{n+1})$, i.e. a $D(3S+1)\times D(3S+1)$ system as stated in the dissertation.

**Sign convention (residual rows).** From here on, the **residual row** of a parameter $\theta_\ell\in\Theta$ is

$$r_\ell:=\rho\big(t_\ell\big),\qquad t_\ell:=\partial q_h/\partial\theta_\ell,$$

and the residual row of the constraint is $r_0:=q_n-q_h(0)$. This is the convention of `residual!` and of `nn_residual` in `theory_common.jl`. Every row of `eqn:DEL_NVI` (and of `DEL_2`) is $-r_\ell$; the sign does not affect any solution set. "The $Q$ row", "the $b$ row", etc. below always mean residual rows.

**Lemma 1.1 (variational formulation).** Let $\mathcal G(\Theta,p_{n+1}) := \mathbb L(q_h(\cdot;\Theta))-p_{n+1}q_h(1;\Theta)+p_n q_h(0;\Theta)$. Then for every parameter $\theta_\ell\in\Theta$,

$$r_\ell=-\frac{\partial\mathcal G}{\partial\theta_\ell}.$$

*Proof.* By the chain rule, $\partial\mathbb L(q_h)/\partial\theta_\ell=\delta\mathbb L(q_h)[\partial_{\theta_\ell}q_h]$. The derivatives of the boundary terms are $-p_{n+1}\,\partial_{\theta_\ell}q_h(1)$ and $p_n\,\partial_{\theta_\ell}q_h(0)$. Collecting terms gives $\partial\mathcal G/\partial\theta_\ell=-\rho(\partial_{\theta_\ell}q_h)$. ∎ **[V1.1a]** checks $r = -\nabla_\Theta\mathcal G$ for the network and **[V1.1b]** for CGVI.

**Key observation 1.2.** The residual row of $\theta_\ell$ is $\rho_{q_h,p_{n+1}}(t_\ell)$. Hence $\Theta$ enters the equations only through two objects: the trial function $q_h(\cdot;\Theta)$ and the test space $T_\Theta\mathcal M$. **Whenever two parameterizations produce the same trial function and the same test space, their equations are equivalent.** All proofs below reduce to computing these two objects.

### 1.2 Discrete Legendre transform and symplecticity

**Lemma 1.3 (type I ⇔ type II, discrete Legendre transform).** Let $V$ be a linear space of functions on $[0,1]$ such that $\delta q\mapsto(\delta q(0),\delta q(1))$ is surjective on $V$. Define the discrete Lagrangian

$$L_d^V(q_a,q_b):=\mathbb L(q^*),$$

where $q^*$ is the stationary point of $\mathbb L$ on $\{q\in V:\ q(0)=q_a,\ q(1)=q_b\}$ (assumed locally unique and depending smoothly on $(q_a,q_b)$). Then the type-II step $(q_n,p_n)\mapsto(q_{n+1},p_{n+1})$ defined by (E1)(E2) with trial space $V$ and $\mathcal T=V$ satisfies the discrete Legendre transforms of the dissertation (eq. `position-momentum-form`):

$$p_n=-D_1L_d^V(q_n,q_{n+1}),\qquad p_{n+1}=D_2L_d^V(q_n,q_{n+1}).$$

*Proof.* The constrained stationarity condition is: there exist multipliers $\mu_0,\mu_1$ such that $\delta\mathbb L(q^*)[\delta q]=\mu_0\,\delta q(0)+\mu_1\,\delta q(1)$ for all $\delta q\in V$. Comparing with (E2), $\delta\mathbb L(q_h)[\delta q]=p_{n+1}\delta q(1)-p_n\delta q(0)$, and using surjectivity of the boundary evaluation gives $\mu_1=p_{n+1}$ and $\mu_0=-p_n$. By the envelope theorem, $D_1L_d^V=\mu_0$ and $D_2L_d^V=\mu_1$. ∎ **[V1.2a–d]**: the type-I solve with $q_{n+1}$ from the type-II step returns $p_n$ (V1.2a) and $p_{n+1}$ (V1.2b), and finite differences of $L_d$ give $p_n=-D_1L_d$ (V1.2c), $p_{n+1}=D_2L_d$ (V1.2d).

Hence CGVI($\Pi^k$, $R$) is the variational integrator generated by $L_d^{\Pi^k}$ and is symplectic. This is the classical statement of the dissertation's §`CGVI`; Lemma 1.3 is the form in which it is reused for the spline spaces of §5.

---

## 2. Lemmas on ReLUᵏ neurons

**Lemma 2.1 (restriction of a neuron with an exterior kink).** If $W\neq0$ and $z=-b/W\notin(0,1)$, then on $[0,1]$:

- if $Wx+b>0$ on $(0,1)$ (the neuron is **active**), $\sigma(Wx+b)=(Wx+b)^k = W^k(x-z)^k$;
- otherwise (the neuron is **inactive**), $\sigma(Wx+b)\equiv0$.

*Proof.* $Wx+b$ is affine and vanishes only at $x=z\notin(0,1)$, so it does not change sign on $(0,1)$. ∎ **[V2.1]**

**Lemma 2.2 (tangent vectors).** With $y^i=W^ix+b^i$,

$$\frac{\partial q_h}{\partial Q^i}=\sigma(y^i),\qquad \frac{\partial q_h}{\partial b^i}=Q^i\sigma'(y^i),\qquad \frac{\partial q_h}{\partial W^i}=Q^i\,x\,\sigma'(y^i),$$

and for the derivative $q_h'$,

$$\frac{\partial q_h'}{\partial Q^i}=W^i\sigma'(y^i),\qquad \frac{\partial q_h'}{\partial W^i}=Q^i\big(\sigma'(y^i)+xW^i\sigma''(y^i)\big),\qquad \frac{\partial q_h'}{\partial b^i}=Q^iW^i\sigma''(y^i).$$

These are the factors $\partial q_h^n/\partial W_n^i$, $\partial\dot q_h^n/\partial W_n^i$, … appearing in `eqn:DEL_NVI`. **[V2.2]** compares the closed forms with finite differences.

**Lemma 2.3 (apolar pairing).** On $\Pi^k$ define the bilinear form

$$\mathcal B_k(f,g)=\sum_{m=0}^k(-1)^m f^{(m)}(0)\,g^{(k-m)}(0).$$

Then $\mathcal B_k$ is nondegenerate, and for every $f\in\Pi^k$ and $z\in\mathbb R$:

$$\mathcal B_k\big(f,(x-z)^k\big)=k!\,f(z),\qquad \mathcal B_k\big(f,(x-z)^{k-1}\big)=-(k-1)!\,f'(z).$$

*Proof.*

1. Let $g=(x-z)^k$. Then $g^{(k-m)}(0)=\frac{k!}{m!}(-z)^m$, so

   $$\mathcal B_k(f,g)=\sum_{m=0}^k(-1)^m f^{(m)}(0)\,\frac{k!}{m!}(-z)^m=k!\sum_{m=0}^k\frac{f^{(m)}(0)}{m!}z^m=k!\,f(z),$$

   where the last step is Taylor's formula, exact because $f\in\Pi^k$.
2. Let $g=(x-z)^{k-1}$. The $m=0$ term vanishes since $g^{(k)}\equiv0$. For $m\ge1$, $g^{(k-m)}(0)=\frac{(k-1)!}{(m-1)!}(-z)^{m-1}$, and $(-1)^m(-1)^{m-1}=-1$, so

   $$\mathcal B_k(f,g)=-(k-1)!\sum_{m=1}^k\frac{f^{(m)}(0)}{(m-1)!}z^{m-1}=-(k-1)!\,f'(z).$$
3. Nondegeneracy: in the monomial basis, $\mathcal B_k(x^i,x^j)$ is nonzero only for $i+j=k$, with value $(-1)^i\,i!\,(k-i)!$. This is an anti-diagonal matrix with nonzero entries, hence of full rank $k+1$. ∎

**[V2.3a, V2.3b]** check the two identities, **[V2.3c]** checks nondegeneracy.

**Lemma 2.4 (spanning lemma).** Let $z_1,\dots,z_{d_z}$ be distinct. Then

$$\text{(a)}\quad \dim\operatorname{span}\{(x-z_j)^k\}_{j=1}^{d_z}=\min(d_z,k+1),$$

$$\text{(b)}\quad \dim\operatorname{span}\{(x-z_j)^k,(x-z_j)^{k-1}\}_{j=1}^{d_z}=\min(2d_z,k+1).$$

*Proof.* Let $U\subset\Pi^k$ be one of these spans and $U^\perp=\{f\in\Pi^k:\mathcal B_k(f,g)=0\ \forall g\in U\}$. Since $\mathcal B_k$ is nondegenerate, $\dim U=k+1-\dim U^\perp$.

- (a) By Lemma 2.3, $U^\perp=\{f\in\Pi^k:f(z_j)=0,\ j=1,\dots,d_z\}$, the multiples of $\prod_j(x-z_j)$ in $\Pi^k$, of dimension $\max(0,k+1-d_z)$.
- (b) $U^\perp=\{f\in\Pi^k:f(z_j)=f'(z_j)=0\}$, the multiples of $\prod_j(x-z_j)^2$, of dimension $\max(0,k+1-2d_z)$. ∎

For $d_z=k+1$ the determinant can be written explicitly. Arrange the monomial coefficients of $(x-z_i)^k$ in a matrix $C$ with entries $C_{ji}=\binom kj(-z_i)^{k-j}$, $j=0,\dots,k$. Factoring $\binom kj$ out of row $j$ leaves a Vandermonde matrix in $-z_i$, so

$$|\det C|=\prod_{j=0}^k\binom kj\prod_{i<l}|z_i-z_l|.$$

**[V2.4a]** checks (a) for $k=2,\dots,6$, $d_z=1,\dots,k+2$; **[V2.4b]** checks (b); **[V2.4c]** checks the determinant formula; **[V2.4d]** checks that the rank drops when two kinks coalesce.

**Lemma 2.5 (Euler homogeneity).** Since $\sigma$ is positively homogeneous of degree $k$, $s\,\sigma'(s)=k\,\sigma(s)$. Hence, for $W^i\ne0$,

$$\frac{\partial q_h}{\partial W^i}=\frac{k\,Q^i\,\partial_{Q^i}q_h-b^i\,\partial_{b^i}q_h}{W^i},$$

and the same identity holds for the rows of the system: $r_{W^i}=(k\,Q^i\,r_{Q^i}-b^i\,r_{b^i})/W^i$.

*Proof.*

$$x\,\sigma'(Wx+b)=\frac{(Wx+b)\sigma'(Wx+b)-b\,\sigma'(Wx+b)}{W}=\frac{k\,\sigma(Wx+b)-b\,\sigma'(Wx+b)}{W}.$$

Multiply by $Q$ and use Lemma 2.2. The identity for the rows follows from the linearity of $\rho$. ∎ **[V2.5a]** checks the tangent-vector identity, **[V2.5b]** the row identity.

So of the three rows of each neuron only two are independent: the $Q$ row and the $b$ row.

**Lemma 2.6 (gauge symmetry).** For $\gamma_i>0$, $(Q^i,W^i,b^i)\mapsto(Q^i/\gamma_i^k,\gamma_iW^i,\gamma_ib^i)$ leaves $q_h(\cdot;\Theta)$ unchanged; so does any permutation of the neurons. **[V2.6a, V2.6b]**

These are instances of the "different sets of neural network parameters that represent the same solution" mentioned in the dissertation's discussion of Newton solvers for NVI. The continuous rescalings give $S$ directions in the null space of the Jacobian at a solution (permutations are discrete and give none). Corollary 3.1 shows that in the kink-free regime the null space is exactly the set of directions with $\delta q_h=0$ and $\delta p_{n+1}=0$, of dimension $3S-k-1$; besides the rescalings it contains, e.g., moving an exterior kink while readjusting the output weights.

---

## 3. Theorem 1: NVI without interior kinks is CGVI($\Pi^k$)

**Theorem 1.** Let $\Theta$ satisfy

- **(H1)** No neuron has its kink in $(0,1)$, and $W^i\ne0$ for all $i$. By Lemma 2.1 each neuron is then either active or inactive on $[0,1]$.
- **(H2)** $T_\Theta\mathcal M=\Pi^k$. By Lemma 2.4(a), (H2) holds as soon as there are $k+1$ active neurons with distinct kinks. More generally, by Lemma 2.4(b), it holds whenever the active neurons have $d_z$ distinct kinks with $2d_z\ge k+1$ and every group of active neurons sharing a kink contains a neuron with $Q^iW^i\ne0$.

Then $(\Theta,p_{n+1})$ satisfies the NVI equations ($SsRrQu\,\mathrm{ReLU}^k$) **if and only if** $(q_h(\cdot;\Theta),p_{n+1})$ satisfies the CGVI($\Pi^k$, $R$) equations with the same $R$-point quadrature.

*Proof.*

- **Step 1: $q_h\in\Pi^k$.** By Lemma 2.1 each neuron restricted to $[0,1]$ is either $(W^i)^k(x-z^i)^k\in\Pi^k$ or identically zero. **[V3.1a]**
- **Step 2: $T_\Theta\mathcal M\subset\Pi^k$.** By Lemma 2.2, for an active neuron $\partial_{Q^i}q_h=(y^i)^k$, $\partial_{b^i}q_h=kQ^i(y^i)^{k-1}$ and $\partial_{W^i}q_h=kQ^i\,x\,(y^i)^{k-1}$, all in $\Pi^k$; for an inactive neuron all three vanish identically. **[V3.1a]**
- **Step 3: $T_\Theta\mathcal M=\Pi^k$.** This is (H2). For the active neuron $i$ with kink $z^i$, $\partial_{Q^i}q_h=(W^i)^k(x-z^i)^k$ and $\partial_{b^i}q_h=kQ^i(W^i)^{k-1}(x-z^i)^{k-1}$, so the tangent space contains the span of Lemma 2.4.
  - For $S=k+1$, $6$, $8$ with distinct kinks, **[V3.1b]** checks $\operatorname{rank}T_\Theta\mathcal M=k+1$;
  - with kinks coalesced in $d_z$ groups, **[V3.1c]** checks $\operatorname{rank}T_\Theta\mathcal M=\min(2d_z,k+1)$.
- **Step 4: the two systems are equivalent.** By Observation 1.2, the NVI equations are (E1) together with (E2) for $\delta q\in T_\Theta\mathcal M=\Pi^k$. The CGVI($\Pi^k$) equations are the same (E1) together with (E2) for $\delta q\in\Pi^k$. The trial function lies in $\Pi^k$ (Step 1), both test spaces equal $\Pi^k$, and the quadrature is the same; hence the two systems have the same solution set in terms of $(q_h,p_{n+1})$.
  - The numbers of equations and unknowns differ: NVI has $3S+1$ equations in $3S+1$ unknowns, CGVI has $k+2$ equations in $k+2$ unknowns. Equivalence means the following. Let $\{\phi^l\}_{l=0}^k$ be the CGVI basis (e.g. the Lagrange basis on the Gauss nodes for $P3R4Q8$). Every tangent vector $t_\ell=\partial q_h/\partial\theta_\ell\in\Pi^k$ can be expanded as $t_\ell=\sum_l M_{\ell l}\phi^l$. By linearity of $\rho$, $\rho(t_\ell)=\sum_l M_{\ell l}\rho(\phi^l)$, so every NVI row is a linear combination of CGVI rows. Conversely, by (H2) the $t_\ell$ span $\Pi^k$, so the matrix $M\in\mathbb R^{3S\times(k+1)}$ has rank $k+1$ and every CGVI row is a linear combination of NVI rows. The constraint row (E1) is common to both.
  - The $3S-(k+1)$ extra unknowns of NVI are exactly the gauge freedom (Corollary 3.1): they do not change $q_h$ and are not constrained by the equations, so the NVI solutions form a whole family of parameter sets $\Theta$ representing the same $q_h$. ∎

**Converse construction (existence).** Let $(q^*,p^*)$ be a CGVI($\Pi^k$, $R$) solution. Choose $k+1$ distinct kinks $z^i\notin[0,1]$ and signs of $W^i$ such that every neuron is active ($W^i>0$ for $z^i<0$, $W^i<0$ for $z^i>1$). By Lemma 2.4(a) the functions $\sigma(W^ix+b^i)|_{[0,1]}=|W^i|^k|x-z^i|^k$ form a basis of $\Pi^k$, so there is a unique set of output weights $\{Q^i\}$ with $q_h=q^*$. These neurons satisfy (H1) and (H2), so by Theorem 1 this $\Theta$ solves the NVI equations.

- **[V3.2a]** checks the fitting error;
- **[V3.2b]** checks that the NVI residual is $\le10^{-10}$ for the harmonic oscillator and the pendulum at $h\in\{0.1,1,5\}$.

**Uniqueness direction.** If the CGVI one-step solution is locally unique (its Jacobian is nonsingular, which the implicit function theorem guarantees for sufficiently small $h$), then every solution of the NVI equations in the (H1)(H2) region whose trial function lies in a neighbourhood of the CGVI solution gives the same $(q_{n+1},p_{n+1})$. **[V3.3a/b]** solve with LM from perturbed initial guesses and check that the kinks stay outside $[0,1]$ and that the result agrees with CGVI to $10^{-10}$.

By induction over the steps: as long as every NVI step lands in the (H1)(H2) region, the whole trajectory coincides with CGVI. **[V3.5]** checks a 20-step trajectory. The initial guess uses only a polynomial extrapolation of the previous NVI step, never the answer of the current step.

**Corollary 3.1 (structure of the Jacobian).** Let $(\Theta,p_{n+1})$ be a solution in the (H1)(H2) region with all kinks in $\mathbb R\setminus[0,1]$ (so that they stay outside under small perturbations), and assume that the CGVI($\Pi^k$, $R$) Jacobian at $(q_h,p_{n+1})$ is nonsingular (e.g. $h$ small). Then the $(3S+1)\times(3S+1)$ Jacobian $J$ of the NVI equations has rank $k+2$, and its null space consists exactly of the gauge directions, along which $q_h$ and $p_{n+1}$ do not change.

*Proof.* Write the residual of row $\ell$ as $r_\ell(\Theta,p_{n+1})=\rho_{q_h,p_{n+1}}(t_\ell(\Theta))$. Its differential in the direction $(\delta\Theta,\delta p)$ is

$$dr_\ell=(d\rho)[\delta q,\delta p](t_\ell)+\rho(dt_\ell),\qquad \delta q:=\sum_{\ell'}t_{\ell'}\,\delta\theta_{\ell'} .$$

1. For neurons with exterior kinks the second parameter derivatives $dt_\ell$ are again polynomials of degree $\le k$. At the solution $\rho$ vanishes on $\Pi^k$, so the second term drops out.
2. Hence $J=\Lambda\circ\iota$, where $\iota:(\delta\Theta,\delta p)\mapsto(\delta q,\delta p)\in\Pi^k\times\mathbb R$ is the linearized parameter-to-trajectory map, and $\Lambda$ maps $(\delta q,\delta p)$ to the values of $(d\rho)[\delta q,\delta p]$ on the $t_\ell$ together with $\delta q(0)$ (the linearized constraint row). By (H2), $\iota$ is surjective, with rank $k+2$.
3. $\Lambda$ is injective. If $\Lambda(\delta q,\delta p)=0$, then $(d\rho)[\delta q,\delta p]$ vanishes on all $t_\ell$, hence on their span $\Pi^k$, and $\delta q(0)=0$. These are exactly the linearized CGVI equations, whose Jacobian is nonsingular; hence $(\delta q,\delta p)=0$.
4. Therefore $\ker J=\ker\iota=\{(\delta\Theta,\delta p):\ \delta q=0,\ \delta p=0\}$ and $\operatorname{rank}J=\operatorname{rank}\iota=k+2$. ∎

**[V3.4a]** checks the rank, **[V3.4b]** checks that the null directions change neither $q_h$ nor $p_{n+1}$.

Practical meaning: of the $3S+1$ unknowns only $k+2$ are determined by the equations. The Jacobian of plain Newton is singular, so a regularization $J_\lambda=J+\lambda I$ (`regularization_factor`) or a Levenberg–Marquardt method is required. The null space has dimension $3S-k-1$ and grows with $S$.

**Corollary 3.2 (width and power).**

- $\mathrm{ReLU}^k$ NVI with any width $S\ge k+1$ is equivalent to CGVI($\Pi^k$), not CGVI($\Pi^{S-1}$), as long as all kinks are outside the interval. **[V3.7aa/ab]** check $S=6$ and **[V3.7ba/bb]** check $S=8$ (the script forms the tag as the id `V3.7a`/`V3.7b` followed by `a`/`b`).
- In particular, $S4R4Q8\,\mathrm{ReLU}^2$ is equivalent to $P2R4Q8$ and differs from $P3R4Q8$. **[V3.6a/b/c]**

**Corollary 3.3 (symplecticity).** In the (H1)(H2) region the NVI one-step map $\Phi_h$ is the CGVI one-step map, hence generated by $L_d^{\Pi^k}$ (Lemma 1.3) and therefore symplectic. **[V3.8]** checks $|\det\partial(q_{n+1},p_{n+1})/\partial(q_n,p_n)-1|\le10^{-7}$; for $D=1$ this determinant condition is equivalent to $\Phi_h'^{\,T}\mathbb J\,\Phi_h'=\mathbb J$, because $A^T\mathbb JA=(\det A)\,\mathbb J$ for every $2\times2$ matrix $A$. This is symplecticity of the one-step map in the sense of the dissertation's definition of a symplectic integrator.

The package currently has `issymplectic(::NetworkIntegratorMethod) = missing`. This corollary provides a proof under explicit conditions.

**Corollary 3.4 (order of convergence).** The NVI error equals the CGVI($\Pi^k$, $R$) error step by step. Ober-Blöbaum & Saake (*Construction and analysis of higher order Galerkin variational integrators*) report numerically that $PsRrQu$ has order $\min(2s,u)$ for $s\le r$; with Gauss quadrature ($u=2r$) and $r\ge s$ this is $2s$. Our prototype confirms this for the cases in **[C5]**. Hence $S4R4Q8\,\mathrm{ReLU}^3$, being equal to $P3R4Q8$, has order 6, consistent with the 6.00 measured for CGVI(4) by `run_convergence.jl`.

---

## 4. Other stationary points of the NVI equations: why Theorem 1 needs (H2)

**Proposition 4.1 (spurious solutions with coalesced kinks).** If all active neurons share the same kink $z$ ($d_z=1$), then $q_h=c_0(z-x)^k$ (or $c_0(x-z)^k$) and $T_\Theta\mathcal M=\operatorname{span}\{(x-z)^k,(x-z)^{k-1}\}$, of dimension $2<k+1$. The NVI equations then reduce to 3 independent conditions: $\rho$ vanishes on this two-dimensional space, plus (E1). The effective unknowns are also 3: $c_0$, $z$, $p_{n+1}$. So there are generically isolated solutions, and they are **not** CGVI solutions.

**[V4.1a/b/c]**:

1. solve the reduced equations by scanning and bisection over $W=-1$, $z>1$;
2. embed the solution into 4 neurons with different weights but the same kink and check that the full NVI residual is $\le10^{-9}$ (V4.1a);
3. check that the tangent-space rank equals 2 (V4.1b);
4. check $|q_{n+1}-q_{n+1}^{\mathrm{CGVI}}|\ge10^{-5}$ (V4.1c).

In the prototype, for the harmonic oscillator at $h=1$, $z^*\approx6.66$ (about 6.8–7 for the pendulum), and $q_{n+1}$ differs from CGVI by about $4\times10^{-2}$. The prototype's LM indeed converged on its own to such a solution from an ill-conditioned initial guess: all four kinks moved to 6.66. Diagnostics must therefore count the number of *distinct* kinks.

**Proposition 4.2 (zero-amplitude interior neurons are harmless).** Let $(q^*,p^*)$ be a CGVI($\Pi^k$) solution represented by some $\Theta$ without interior kinks, and let $\rho^*:=\rho_{q^*,p^*}$. If $z\in(0,1)$ satisfies

$$\rho^*\big((x-z)_+^k\big)=0,$$

then adding to $\Theta$ a neuron with $Q=0$, $W=1$, $b=-z$ gives a $\Theta'$ that still solves the NVI equations, and $q_h(\cdot;\Theta')=q^*$.

*Proof.* The trial function is unchanged because $Q=0$. The $Q$ row of the new neuron is $\rho^*((x-z)_+^k)$, which vanishes by the choice of $z$; its $W$ and $b$ rows contain the factor $Q=0$ (Lemma 2.2) and vanish; all other rows are unchanged. ∎

Conversely, if a converged NVI step contains an interior neuron with $Q^i=0$ and $W^i>0$, its $Q$ row enforces $\rho((x-z^i)_+^k)=0$. For $W^i<0$ the $Q$ row is $\rho(|W^i|^k(z^i-x)_+^k)$; by Lemma 5.1 this enforces the same condition if, in addition, the remaining neurons span $\Pi^k$ (so that $\rho$ vanishes on $\Pi^k$).

**[V4.2a]** checks that the residual is $\le10^{-9}$ at a root $z^*$ of $\rho^*$ (in the prototype the first root is $z^*\approx0.069$); **[V4.2b]** checks that the residual is $\ge10^{-6}$ at $z^*+0.05$, showing that the condition is not trivial.

**Step classification.** From these results, each step (for each component) falls into one of three classes:

- **$\Pi^k$-equiv**: only neurons without interior kinks and zero-amplitude neurons, and the tangent vectors of the former span $\Pi^k$. The step is exactly CGVI($\Pi^k$).
- **degenerate**: no interior kinks, but the tangent space is smaller than $\Pi^k$. Possibly a spurious solution (Proposition 4.1).
- **spline**: there is an interior kink with nonzero amplitude. See §5.

Part A of `verify_conjectures.jl` performs this classification step by step for the actual `ShallowNet` (`classify_step`; the class $\Pi^k$-equiv is printed as `P_k-equiv`).

---

## 5. Generalization: kinks inside the interval

**Lemma 5.1.** For $s\in\mathbb R$, $s_+^k+(-1)^k(-s)_+^k=s^k$, equivalently $(-s)_+^k=(-1)^k\big(s^k-s_+^k\big)$. **[V5.1]**

*Proof.* For $s\ge0$ the second term is 0; for $s<0$ the first term is 0 and the second is $(-1)^k(-s)^k=s^k$. ∎

So a neuron with $W<0$, $|W|^k(z-x)_+^k$, lies in $\Pi^k\oplus\operatorname{span}\{(x-z)_+^k\}$.

**Proposition 5.2 (trial space and tangent space).** Let $Z=\{z_1,\dots,z_m\}\subset(0,1)$ be the distinct interior kinks. Define

$$\mathcal S_k(Z):=\Pi^k\oplus\operatorname{span}\{(x-z)_+^k\}_{z\in Z},$$

the space of splines of degree $k$ on $[0,1]$ with simple knots $Z$, i.e. $C^{k-1}$ continuity at the knots. In the B-spline convention of the dissertation (order $= k+1$, knot vector $\Xi$), $\mathcal S_k(Z)$ is spanned by the B-splines of order $k+1$ on $\Xi=(0,\dots,0,z_1,\dots,z_m,1,\dots,1)$ with $0$ and $1$ repeated $k+1$ times (Curry–Schoenberg theorem, see de Boor, *A Practical Guide to Splines*); $\dim\mathcal S_k(Z)=k+1+m$. Then $q_h\in\mathcal S_k(Z)$, and

$$T_\Theta\mathcal M\subset\mathcal S_k(Z;2):=\mathcal S_k(Z)\oplus\operatorname{span}\{(x-z)_+^{k-1}\}_{z\in Z},$$

i.e. in the tangent space every knot becomes a double knot. If there are $k+1$ neurons with distinct exterior kinks and $m$ interior neurons with $Q^i\neq0$ and distinct kinks, then $\dim T_\Theta\mathcal M=k+1+2m$.

*Proof.* For an interior neuron with $W>0$ the three tangent vectors are $\partial_Qq_h=W^k(x-z)_+^k$, $\partial_bq_h=kQW^{k-1}(x-z)_+^{k-1}$ and $\partial_Wq_h=x\,\partial_bq_h$. Since $x(x-z)_+^{k-1}=(x-z)_+^k+z(x-z)_+^{k-1}$, $\partial_Wq_h$ also lies in $\mathcal S_k(Z;2)$. The case $W<0$ reduces to this via Lemma 5.1. The dimension count follows from the linear independence of $\{x^l\}_{l=0}^k\cup\{(x-z)_+^k,(x-z)_+^{k-1}\}_{z\in Z}$. ∎

**[V5.2a]** checks $q_h\in\mathcal S_k(Z)$ and $T_\Theta\mathcal M\subset\mathcal S_k(Z;2)$ via projection residuals; **[V5.2b]** checks the dimension for $k=2,\dots,4$, $m=1,\dots,3$.

This is the variational-integrator version of the statement "a one-dimensional shallow $\mathrm{ReLU}^k$ network is a free-knot spline" (cf. DeVore–Hanin–Petrova, *Acta Numerica* 2021): the trial set consists of free-knot splines, and the test space doubles every knot.

**Theorem 2 (fixed knots).** Freeze $\{W^i,b^i\}$ and require stationarity only with respect to the output weights $\{Q^i\}$ (keep only the $Q$ rows and (E1)). The resulting equations are the CGVI equations on the linear space $V=\operatorname{span}\{\sigma(W^ix+b^i)\}_{i=1}^S$. This is the case in the dissertation where "the parameters $\Theta_n$ are partitioned into degrees of freedom optimized by the Newton solver and fixed parameters". If $V=\mathcal S_k(Z)$, this is the Galerkin variational integrator on the spline space. Since (E1)(E2) depend only on $V$ (Observation 1.2), the result is independent of the basis: truncated powers, B-splines and frozen neurons give the same solution. It is symplectic by Lemma 1.3.

**[V5.3a]** checks the B-spline partition of unity; **[V5.3b]** checks that the three bases give the same solution ($\le10^{-11}$), both with composite quadrature and with a single global $R=8$ rule.

**Theorem 3 (free knots = discrete Lagrangian stationary in the knots).** Suppose every neuron has $Q^i\ne0$ and $W^i\ne0$, the neurons without interior kinks span $\Pi^k$, and the interior kinks are distinct and form the set $Z$. Suppose also that $\mathcal S_k(Z)$ satisfies the hypotheses of Lemma 1.3, and that the quadrature rule $\{\alpha_j,\beta_j\}$ does not depend on $\Theta$ (as in `ShallowNet`; V5.4 uses a global $R=8$ rule). Then $(\Theta,p_{n+1})$ satisfies the NVI equations **if and only if** both

- **(i)** $q_h$ is the fixed-knot Galerkin step on $\mathcal S_k(Z)$ (Theorem 2), and
- **(ii)** for every interior knot,

$$\frac{\partial}{\partial z_i}L_d^{\mathcal S_k(Z)}(q_n,q_{n+1})=0 .$$

*Proof.*

1. By Lemma 2.5, each neuron's $W$ row is a linear combination of its $Q$ row and $b$ row (coefficients $kQ/W$ and $-b/W$). So the system is equivalent to "all $Q$ rows + all $b$ rows + (E1)".
2. The $Q$ rows plus (E1) say that $\rho$ vanishes on $\operatorname{span}\{\psi^i\}=\mathcal S_k(Z)$, which is (i).
3. Write $q_h=\sum_lc_l\varphi_l(\cdot;Z)$ in the truncated-power basis of $\mathcal S_k(Z)$ and let $\partial_zq_h$ be the derivative with respect to a knot $z$ at fixed coefficients $c$. For an interior neuron with $W>0$, $Q\psi=QW^k(x-z)_+^k$ with coefficient $c=QW^k$, so $\partial_zq_h=-ck(x-z)_+^{k-1}=-kQW^k(x-z)_+^{k-1}$. By Lemma 2.2, $\partial_bq_h=kQ(Wx+b)_+^{k-1}=kQW^{k-1}(x-z)_+^{k-1}$, hence $\partial_zq_h=-W\,\partial_bq_h$, i.e. $\partial_bq_h=-\tfrac1W\partial_zq_h$, and the $b$ row equals $-\tfrac1W\rho(\partial_zq_h)$. For an interior neuron with $W<0$, Lemma 5.1 gives $Q\psi=QW^k(x-z)^k+c\,(x-z)_+^k$ with $c=-QW^k$, and then $\partial_bq_h=-\tfrac1W\partial_zq_h+kQW^{k-1}(x-z)^{k-1}$. The extra term lies in $\Pi^{k-1}\subset\mathcal S_k(Z)$, where $\rho$ vanishes by (i), so again the $b$ row equals $-\tfrac1W\rho(\partial_zq_h)$. (V5.4 tests only $W=+1$; the $W<0$ formulas were checked separately by finite differences for $k=2,\dots,5$.) For neurons without interior kinks, $\partial_bq_h\in\Pi^{k-1}\subset\mathcal S_k(Z)$ and the row vanishes automatically by (i).
4. **Envelope formula.** View $q_Z=\sum_lc_l\varphi_l(\cdot;Z)$ as the stationary point of the type-I problem on $\{q\in\mathcal S_k(Z):q(0)=q_a,\ q(1)=q_b\}$ with multipliers $\mu_0,\mu_1$. By the envelope theorem,

   $$\frac{dL_d}{dz}=\frac{\partial\mathbb L}{\partial z}\Big|_c-\mu_1\frac{\partial q(1)}{\partial z}\Big|_c-\mu_0\frac{\partial q(0)}{\partial z}\Big|_c .$$

   Substituting $\mu_1=p_{n+1}$, $\mu_0=-p_n$ (Lemma 1.3) gives

   $$\frac{dL_d}{dz}=\delta\mathbb L[\partial_zq]-p_{n+1}\,\partial_zq(1)+p_n\,\partial_zq(0)=-\rho(\partial_zq).$$

5. Hence $b$ row $=\tfrac1W\,dL_d/dz$, and the $b$ row vanishes if and only if (ii) holds. ∎

**[V5.4a]** checks the envelope formula (finite differences vs. closed form); **[V5.4b]** checks that the type-I solution satisfies all $Q$ rows; **[V5.4c]** checks that the $b$ row equals $\tfrac1W\,dL_d/dz$; **[V5.4d]** finds the stationary knot $Z^*$ ($Z^*\approx0.43849$ in the prototype) and checks that the full NVI residual is $\le10^{-8}$.

**Corollary 5.3 (symplecticity).** Near a nondegenerate stationary point $Z^*(q_a,q_b)$ (a smooth branch by the implicit function theorem), the NVI step is generated by

$$L_d^{\mathrm{NN}}(q_a,q_b)=L_d^{\mathcal S_k(Z^*(q_a,q_b))}(q_a,q_b).$$

Applying the envelope theorem once more, (ii) gives $D_iL_d^{\mathrm{NN}}=D_iL_d^{Z}|_{Z=Z^*}$, $i=1,2$. So the free-knot NVI step satisfies the discrete Legendre transforms of $L_d^{\mathrm{NN}}$; it is a variational integrator and hence symplectic. **[V5.6]** checks $|\det\Phi_h'-1|\le10^{-7}$ ($6\times10^{-11}$ in the prototype).

**Remark 5.4 (quadrature).** $q_h\in\mathcal S_k(Z)$ is $C^{k-1}$ at the kinks, so $q_h'$ is only $C^{k-2}$ there and so is the integrand $L(q_h,q_h'/h)$. A single global $R$-point Gauss rule integrates polynomials of degree $\le2R-1$ exactly, but not piecewise polynomials with a break inside $(0,1)$. A composite Gauss rule with $R$ points on each sub-interval between the kinks integrates every piecewise polynomial of degree $\le2R-1$ exactly. **[V5.5a]** checks exactness of the composite 4-point rule on $(x-z)_+^3x^j$, $j=0,\dots,3$ (degree $\le6$); **[V5.5b]** checks that the global 4-point rule is not exact on them (error $\ge10^{-7}$).

Consequences:

- With a global rule the quadrature error of the discrete action does not decrease like the error of the spline space; it wipes out the benefit of the spline (order 2 instead of 6 in C4).
- When $k+1+m>R$ there are fewer quadrature points than basis functions and the equations may even become singular (the case $m=3$ with a global $R=4$ rule in C4).
- The composite quadrature nodes move with $\Theta$, so in the free-knot problem $\mathbb L$ is only piecewise smooth in $\Theta$.

---

## 6. Conjectures and numerical tests

### C1–C3 (Part A, the actual package)

**C1.** Every step of $S4R4Q8\,\mathrm{ReLU}^3$ should be **$\Pi^k$-equiv** and equal to $P3R4Q8$: diff $\le\max(10^{-9},\,0.01\cdot\mathrm{err}_{\mathrm{CGVI}})$. If a step is classified as spline or degenerate, the theory makes no prediction for it and the script reports it as is; the "almost identical errors" seen by the original comparison script should fall in the $\Pi^k$-equiv class.

**C2.** For $S=6,8$, the theory predicts that every run whose steps are all $\Pi^k$-equiv equals $P3R4Q8$ and is no better. The OGA dictionary of the dissertation (§`sec:nvi_shallow_plain`) has $W=\pm1$ and biases on a uniform grid in an interval; its kinks are $z=-b/W$.

- `bias = [1.1, π]` puts the kinks of all dictionary atoms at $z\in[-\pi,-1.1]$ (for $W=1$) or $z\in[1.1,\pi]$ (for $W=-1$), outside the interval and active, so the initial guess satisfies (H1);
- `bias = [−π, π]` (the default of the dissertation) gives $z\in[-\pi,\pi]$, so OGA may pick interior kinks.

**C3.** $S4R4Q8\,\mathrm{ReLU}^2$ should equal $P2R4Q8$, and its difference from $P3R4Q8$ should be at least 10 times larger.

Decision rules:

- Only when all steps are $\Pi^k$-equiv and no solve hits `maxiter` does the theory make the hard prediction "must be equal". If they are then not equal the run is marked **VIOLATED** (check fails); if equal, **CONFIRMED**.
- In all other cases the result is only recorded ("equal anyway" or "differs") and not counted as a failure.

### C4 (Part B): can fixed interior knots reach a smaller error with fewer unknowns?

Prototype results: harmonic oscillator, $T=10$, maximum relative error in $q$.

| Method | h=0.1 | 0.2 | 0.5 | 1 | 2 | 5 |
|---|---|---|---|---|---|---|
| CGVI $P3R4Q8$ | 3.93e-11 | 2.51e-09 | 6.08e-07 | 3.79e-05 | 2.18e-03 | 1.43e-01 |
| spline m=1, global R4 | 5.54e-06 | 2.18e-05 | 1.23e-04 | 2.92e-04 | 2.14e-03 | 1.01e-01 |
| spline m=1, global R8 | 2.72e-08 | 8.34e-08 | 6.00e-07 | 1.93e-05 | 4.13e-04 | 2.72e-02 |
| **spline m=1, composite R4** | **1.09e-12** | 7.00e-11 | 1.74e-08 | 1.17e-06 | 8.85e-05 | 1.98e-02 |
| CGVI $P3R4Q8$, 2 substeps | 6.12e-13 | 3.93e-11 | 9.56e-09 | 6.08e-07 | 3.79e-05 | 5.20e-03 |
| spline m=3, global R4 | singular | | | | | |
| spline m=3, global R16 | 2.83e-05 | 2.84e-05 | 2.92e-05 | 3.14e-05 | 3.05e-05 | 7.20e-04 |
| **spline m=3, composite R4** | **2.58e-14** | 1.64e-12 | 4.02e-10 | 2.63e-08 | 1.84e-06 | 5.05e-04 |
| CGVI $P3R4Q8$, 4 substeps | 2.10e-14 | 6.12e-13 | 1.50e-10 | 9.56e-09 | 6.08e-07 | 9.74e-05 |

The script performs four automatic checks:

- **[C4a]** the frozen-neuron basis and the truncated-power basis give the same error (Theorem 2);
- **[C4b]** with composite quadrature the order stays 6 ($|p-6|\le0.4$);
- **[C4c]** the spline error is of the same magnitude as "CGVI on $m+1$ substeps" ($|\log_{10}\text{ratio}|\le0.7$; the prototype ratios are 1.8 and 2.7);
- **[C4d]** with a global R4 rule the order drops to $\le3$ (2.0 in the prototype).

Conclusion: with $m$ fixed interior knots and composite quadrature, one solve of dimension $k+2+m$ (coefficients plus $p_{n+1}$) reaches nearly the error of CGVI with step $h/(m+1)$, which needs $m+1$ sequential solves of dimension $k+2$. The order is unchanged ($2k$); what decreases is the error constant, by roughly a factor $(m+1)^{2k}$. C7, which counts FLOPs, further shows that fewer parameters do not mean less computation: at equal FLOPs, fixed-knot splines and CGVI substeps are roughly on par, with substeps slightly ahead.

### C5: the order of $PsRrQ(2r)$

Observed orders in the prototype, $h=0.2\to0.5$:

- $(s,R)=(2,3)$: 3.99
- $(2,4)$: 3.99
- $(3,3)$: 5.99
- $(3,4)$: 5.99
- $(4,4)$: 7.99
- $(4,5)$: 7.98

Conjecture: **the order is $2s$ when $R\ge s$.** This agrees with the numerical order $\min(2s,u)$, $u = 2R$, reported by Ober-Blöbaum & Saake for $s\le R$ (their Table 1 for Gauss quadrature); a proof for general $(s,R)$ is stated there as open.

Outside this hypothesis ($R<s$), $(4,3)$ agrees digit by digit with CGVI($\Pi^2$, 3) (order 4), and $(3,2)$ has order about 2: with too few quadrature points the higher-degree terms become "invisible". These cases are only printed, not checked; they lie outside the range $s\le r$ studied by Ober-Blöbaum & Saake.

### C6: what does LM converge to with free knots?

Setting: $k=3$, $S=5$ (4 neurons without interior kinks plus 1 neuron whose initial kink $z_0$ is interior), global $R=8$, pendulum, $h=1$.

- If the solve converges and is **$\Pi^k$-equiv**, check that it equals CGVI($\Pi^3$).
- If it is **spline**, check Theorem 3: the type-I solve reproduces the step and $\nabla_ZL_d=0$.

The 8 prototype runs:

- $(z_0,Q^5)=(0.8,10^{-3})$: converges to a zero-amplitude neuron (Proposition 4.2), $|\Delta q|=0$;
- $(0.2,0.1)$: converges to a genuine free-knot spline solution, $z=0.435$, $|\Delta q_1|=1.8\times10^{-5}$, Theorem 3 check value $3.5\times10^{-11}$;
- the other 6 runs do not converge within 500 LM iterations: the kinks drift towards 0 and 0.97.

This is consistent with Remark 5.4 and with "the stationary point is not a minimum": the free-knot problem is ill-conditioned.

### C7: free-knot spline VI vs fixed knots vs CGVI: parameters, FLOPs, error

**Method (`free_knot_vi.jl`).** The solution of Theorem 3 is computed in two levels:

- **Inner level**: for given knots $Z$, take one step of the Galerkin variational integrator on $\mathcal S_k(Z)$, with composite Gauss quadrature split at $Z$; the unknowns are the spline coefficients and $p_{n+1}$, solved by Newton with an analytic Jacobian. There is no gauge freedom at this level and no regularization is needed.
- **Outer level**: root-finding in the knots, $g(Z)=\partial L_d^{Z}(q_n,q_{n+1}(Z))/\partial Z=0$. $g$ is computed analytically and, besides the envelope formula, includes a term for the motion of the composite quadrature nodes with $Z$; its Jacobian uses forward differences ($m$ extra inner solves per outer iteration). The knots are kept in $(0.02,0.98)$ with spacing at least 0.02, each move is at most 0.1, and each step starts from the previous step's knots. If the outer level does not converge, the knots with the smallest $|g|$ are used and the step is counted as a fallback; such steps are no longer strictly variational.

Self-checks: the analytic $g$ agrees with finite differences of $L_d$; the free-knot one-step map satisfies $|\det\Phi_h'-1|\le10^{-7}$ (symplectic); CGVI in the monomial basis gives the same result as CGVI in the Lagrange basis.

**Accounting.** Parameters per step: CGVI $k+1$; $m+1$ substeps $(m+1)(k+1)$; fixed-knot spline $k+1+m$; free-knot spline $k+1+2m$ (the extra $m$ are the knot positions). A ShallowNet of the same function class has $3(k+1+m)$ parameters, of which only $k+1+2m$ are effective. FLOPs are counted by the code with an explicit model (residual, analytic Jacobian, LU, basis evaluation, knot gradient; sin/cos count as 10 operations); they represent the operation count of an efficient implementation, not measured wall time.

**Results** ($k=3$, $R=4$, $t\in(0,20)$, $(q_0,p_0)=(0.5,0)$; $q$ error is the maximum relative error on the grid; full tables in `results/free_knot_vi.md`):

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

1. **Fixed-knot splines improve a lot on CGVI at the same step size, at small cost.** For the harmonic oscillator at $h=1$, $m=1$ reduces the error 32-fold at 1.7× the FLOPs; $m=3$ reduces it about 1400-fold at 3.8× the FLOPs. The energy error is also small, e.g. harmonic oscillator, $h=5$, $m=3$: $q$ error $1.7\times10^{-3}$, $H$ error only $9.2\times10^{-6}$.
2. **But fixed-knot splines are not more efficient than CGVI substeps.** For the same $m$ their error is about 2–3 times that of $m+1$ substeps; at equal FLOPs the two are roughly on par, with substeps slightly ahead (harmonic oscillator, $h=0.5$: spline $m=3$ at about $1.0\times10^4$ FLOP/step with error $9.3\times10^{-10}$; 4 substeps at about $1.05\times10^4$ FLOP/step with error $3.5\times10^{-10}$). Their only advantage is the parameter count (7 vs 16 for $m=3$). The larger $m$, the worse for the spline: its Jacobian grows like $(k+1+m)^2$, while substeps are $m+1$ small problems of size $(k+1)^2$.
3. **Free knots bring no gain on these two problems.** They cost 5–110× the FLOPs of fixed knots (about two orders of magnitude for $m=3$), and for $m\ge2$ the outer level often fails to converge (up to 22 fallback steps); the error is mostly on par with or worse than fixed knots (7× worse for the pendulum at $h=1$, $m=2$), and the energy error is usually larger (harmonic oscillator, $h=5$, $m=1$: 0.146 vs 0.030). The only clear win is $h=5$, $m=3$ (4× better for the pendulum, 1.7× for the harmonic oscillator), but there CGVI with 4 substeps is even more accurate at about 1/34 of the FLOPs.
4. **Reason.** The variational principle selects knots that make the discrete action stationary, not knots that minimize the error. This guarantees symplecticity (Corollary 5.3) but not that the knots sit in "good" places. The free knots can be clearly better on the error of a single step (first step at $h=5$: $8.7\times10^{-4}$ with free knots vs $9.0\times10^{-3}$ with fixed $z=0.5$), but this advantage largely disappears over many steps.

**Conclusion (C7).** For smooth problems, what is really useful in the $\mathrm{ReLU}^k$ network function class is "a larger trial space + compatible quadrature". Its effect is essentially equivalent to subdividing the step and is not cheaper than simply using CGVI substeps; letting the knots move (the genuinely nonlinear part) brings no gain here. Free knots may still help for problems with fast local variation within one step (e.g. the pericenter passage of a highly eccentric Kepler orbit); this is an untested conjecture.

---

## 7. Relation to the observations in the dissertation

The dissertation reports several phenomena of NVI with shallow networks without a complete explanation. The results above give an explanation for each, under the stated hypotheses. The dissertation's statements are quoted as observations; the explanations are what this note proves or checks.

| Observation in the dissertation | Explanation in this note | Evidence |
|---|---|---|
| The Jacobian of the DEL equations is severely ill-conditioned (condition numbers up to about $10^{20}$), "likely due to the existence of null space in the parameter space", because the $\psi^i$ are linearly dependent (§`sec:nvi_action`, *Initial Guesses Optimization*). | In the kink-free regime the Jacobian is exactly singular: rank $k+2$ out of $3S+1$, and the null space is the gauge freedom (Corollary 3.1). In floating point an exactly singular matrix shows up as a finite but huge condition number. The null space has two sources. (a) Linear dependence of the $\psi^i$: in the kink-free regime all $\psi^i|_{[0,1]}\in\Pi^k$ (Step 1 of Theorem 1), so for $S>k+1$ they are linearly dependent. (b) Independently of the kinks, the $W$ rows are combinations of the $Q$ and $b$ rows (Lemma 2.5); differentiating this identity at a root shows that the $W$ rows of $J$ are combinations of the $Q$ and $b$ rows of $J$, so $J$ is singular at every solution with all $W^i\ne0$. Point (b) is a consequence of [V2.5b]; the rank of $J$ itself is checked only in the kink-free regime. | [V3.4a/b], [V2.5a/b] |
| Adding $J_\lambda = J+\lambda I$ with $\lambda\in[10^{-7},10^{-3}]$ stabilizes Newton and improves the energy behaviour (§`sec:nvi_shallow_plain`, *Regularization of Jacobian Matrix*). | Because $J$ is singular, a regularization is needed for the linear solves to be uniquely solvable. If the iteration converges to a root of the unregularized DEL equations, the regularization only selects a point on the gauge orbit and does not change $(q_{n+1},p_{n+1})$ (Theorem 1, Corollary 3.1). The reference checks use Levenberg–Marquardt, not $J_\lambda$; the package runs of C1–C3 use `regularization_factor`. | [V3.3a/b] (LM), [V3.4b], C1–C3 |
| "Increasing the number of neurons and the order of the ReLU activation function does not necessarily lead to smaller relative Hamiltonian error" (Figure `fig:NVI_HO_hamiltonian_error`). | As long as all kinks stay outside $[0,1]$, $S$ has no effect at all: every $S$ gives CGVI($\Pi^k$) (Theorem 1, Corollary 3.2). Increasing $k$ changes the method to CGVI($\Pi^k$), whose order is $2k$ only if $R\ge k$ (C5), and only if the step converges to the CGVI solution and not to a spurious one (Proposition 4.1). | [V3.6a/b/c], [V3.7aa/ab/ba/bb], C1–C3 |
| Non-unique parameterizations make NVI sensitive to the initial guess (§`sec:nvi_action`). | Besides the gauge orbit (harmless), the DEL equations have genuinely different solutions: coalesced-kink solutions (Proposition 4.1), zero-amplitude interior neurons (Proposition 4.2, harmless) and free-knot spline solutions (Theorem 3). The OGA dictionary with biases in $[-\pi,\pi]$ allows interior kinks; the choice $[1.1,\pi]$ excludes them from the initial guess. | [V4.1], [V4.2], [V5.4d], C2, C6 |

---

## 8. Answer to "generalize to many neurons and reach a similar error"

1. **The equivalence that can be proven rigorously is "same trial function + same test space".** For $\mathrm{ReLU}^k$ the test space of NVI is the tangent space $T_\Theta\mathcal M$ of the network with respect to its parameters. When all kinks are outside the interval it is exactly $\Pi^k$ (Theorem 1), so NVI of any width is **exactly equal** to CGVI($\Pi^k$): same error, and no better. Extra neurons only enlarge the gauge null space (Corollary 3.1) and add spurious solutions (Proposition 4.1).
2. **For more neurons to reduce the error, kinks must enter the interval.** The method then becomes a spline Galerkin variational integrator: Theorem 2 for fixed knots, Theorem 3 for free knots. In both cases it is provably a variational integrator generated by a discrete Lagrangian, hence symplectic (Corollary 5.3). But **being provably symplectic does not mean a smaller error**: in the measurements (C7), free knots are not more accurate than fixed knots on the harmonic oscillator and the pendulum, while costing one to two orders of magnitude more computation.
3. **Practical recommendations (revised according to the C4 and C7 measurements).**
   - If the goal is a variational integrator built from a $\mathrm{ReLU}^k$ network: pick the kinks with OGA (or simply a uniform grid), freeze $(W^i,b^i)$, solve only for the output weights $Q^i$ (Theorem 2), and use composite quadrature split at the kinks (Remark 5.4). This has a rigorous equivalence proof and symplecticity, no gauge null space and no spurious solutions; the order stays $2k$, the error is about that of CGVI($\Pi^k$) with step $h/(m+1)$, and the energy error is small.
   - But be clear about its place: measured in FLOPs it is roughly on par with, or slightly worse than, plain CGVI substeps; its only advantage is fewer parameters per step. For smooth problems this route is "another way of subdividing the step" and does not surpass linear methods.
   - **Free knots (Theorem 3) are not recommended for smooth problems.** They preserve symplecticity, but in C7 their error is mostly no better than fixed knots, their energy error is usually larger, they cost 5–110× the FLOPs, and for $m\ge2$ the outer root-finding often fails. Moving the knots according to the variational principle does not place them where the error is small. The earlier claim that "free knots are theoretically better" holds only for the single-step error, not over many steps.
   - Free knots may be worth trying for problems with fast local variation within a step (highly eccentric Kepler orbits, near-collisions, multiscale problems), where uniform knots are wasted on the smooth parts; this remains to be tested.
4. **Diagnostics.** At every step read out $\Theta_n$ and count: the number and amplitudes of interior kinks, the number of distinct kinks among the neurons without interior kinks, and the tangent-space rank. This determines whether the step is $\Pi^k$-equiv, spline or degenerate (`classify_step` in `verify_conjectures.jl`). We suggest adding this diagnostic to the package's `record_finer_solution!` or to the benchmarks.
