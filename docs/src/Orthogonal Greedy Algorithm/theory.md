# Theory

This page sets out what the greedy algorithm is approximating, why the classical
dictionary has the form it does, what distinguishes *orthogonal* greedy from plain
matching pursuit, and where the numerical difficulty comes from. The
[Algorithms](@ref) page then treats each implemented component in turn.

## The approximation problem

Over one time step the integrator represents each component of the trajectory by a
shallow network with `S` hidden units,

```math
u(t) \;=\; \sum_{k=1}^{S} c_k\, \sigma\!\left(w_k t + b_k\right) ,
\qquad t \in [0, 1] ,
```

with the step mapped affinely onto the unit interval. The unknowns are the output
weights ``c \in \mathbb{R}^S`` and the hidden parameters ``(w_k, b_k)``. The variational
integrator determines them by a nonlinear system; the OGA's job is only to produce a
starting point for that system, by fitting `u` to a target ``y`` sampled at the network's
input nodes.

The fit is measured in a **quadrature-weighted ``L^2`` norm**. With nodes ``t_j`` and
positive quadrature weights ``w_j`` (Simpson's rule on the network's input grid),

```math
\langle f, g \rangle_w \;=\; \sum_j w_j\, f(t_j)\, g(t_j) ,
\qquad \lVert f \rVert_w^2 = \langle f, f \rangle_w ,
```

which is the discrete stand-in for ``\int_0^1 f g \,\mathrm{d}t``. Writing
``\Phi_{ij} = \sigma(w_i t_j + b_i)`` for the value of dictionary atom `i` at node `j`,
the fit for a fixed set of hidden parameters is the linear least-squares problem

```math
\min_{c} \; \sum_j w_j \left( \sum_i c_i \Phi_{ij} - y_j \right)^{2} .
\tag{LS}
```

## Greedy approximation

The hidden parameters enter nonlinearly, so the fit over all of them at once is a
non-convex problem — which is precisely the problem the integrator's Newton solve is
already struggling with. Greedy approximation
[Temlyakov:2008](@cite) sidesteps it: fix a countable **dictionary**
``\mathcal{D} = \{g_a\}`` of candidate atoms, and build the approximation one atom at a
time, each time choosing the atom that best reduces the current residual. For shallow
networks this is a form of greedy training [Siegel:2023](@cite).

Two variants differ in what happens to the coefficients already chosen:

- **Matching pursuit** fits only the newly added atom, leaving earlier coefficients
  untouched.
- **Orthogonal** greedy re-solves (LS) over *all* selected atoms after every addition.

This package implements the orthogonal variant — hence the name — and the refit is where
all the numerical difficulty lives.

The algorithm, with ``\Lambda_k`` the index set selected after `k` steps:

```math
\begin{aligned}
&r_0 = y, \quad \Lambda_0 = \emptyset \\
&\textbf{for } k = 1, \dots, S: \\
&\qquad a_k = \arg\max_{a}\; \mathrm{score}(g_a; r_{k-1}, \Lambda_{k-1})
   &&\text{(selection)} \\
&\qquad \Lambda_k = \Lambda_{k-1} \cup \{a_k\} \\
&\qquad c^{(k)} = \arg\min_c \big\lVert \textstyle\sum_{i \in \Lambda_k} c_i g_i - y \big\rVert_w^2
   &&\text{(refit)} \\
&\qquad r_k = y - \textstyle\sum_{i \in \Lambda_k} c^{(k)}_i g_i
   &&\text{(residual)}
\end{aligned}
```

The two lines marked *selection* and *refit* are exactly the two axes the implementation
exposes as configurable ([`OGASelection`](@ref) and [`OGAFit`](@ref)); the dictionary
``\mathcal{D}`` is the third ([`OGADictionary`](@ref)).

## Why `±1` weights are the classical dictionary

The dictionary both original variants used is `{±1}` crossed with a uniform grid of
biases. That is not an arbitrary discretisation — for the activation the method was
derived for, it is *complete*.

`ReLUᵏ`, ``\sigma(x) = \max(0, x)^k``, is **positively homogeneous of degree ``k``**:
``\sigma(\lambda x) = \lambda^k \sigma(x)`` for ``\lambda > 0``. Hence for ``w \neq 0``

```math
\sigma(w t + b)
  \;=\; \sigma\!\left(\lvert w\rvert \left[\operatorname{sign}(w)\, t + \tfrac{b}{\lvert w\rvert}\right]\right)
  \;=\; \lvert w\rvert^{k}\, \sigma\!\left(\operatorname{sign}(w)\, t + \tfrac{b}{\lvert w\rvert}\right) .
\tag{H}
```

Read (H) as a statement about what `w` can and cannot change. Its **magnitude** produces
only two effects: a scalar factor ``\lvert w \rvert^k``, which the output weight ``c_k``
absorbs, and a rescaling of the bias, which the bias grid already covers. Neither changes
the atom's *shape*. What survives is the **sign** of `w`, which genuinely does — for
``k \geq 1``, ``\sigma(t + b)`` and ``\sigma(-t + b)`` are different functions (one
increasing, one decreasing on their active region), and no rescaling relates them.

So for `ReLUᵏ` the set

```math
\mathcal{D}_{1} = \{\pm 1\} \times \{b_0, \dots, b_n\}
```

covers every ridge direction the greedy step could want, and enlarging it with more
weight magnitudes adds nothing but duplicates. This is the shallow-`ReLU`
approximation-theory setting, and it is why [`BiasGrid1d`](@ref) is the default.

### Where the argument fails

Neither ELU nor GELU is positively homogeneous, so (H) does not hold for them and the
conclusion collapses. For a smooth, saturating activation, `w` controls the **length
scale** of the transition — how sharply the unit switches — and that is a genuine shape
parameter, independent of both `b` (which sets *where* the transition happens) and `c`
(which sets its amplitude). Consider ``\tanh(wt + b)``: increasing ``\lvert w\rvert``
sharpens the step, and no combination of a bias shift and an output weight reproduces it.

With ``\lvert w \rvert`` pinned to 1, the dictionary can therefore only offer transitions
of a *single* steepness, positioned along the bias axis. If the target needs a sharper or
gentler one, the greedy step cannot supply it; it compensates by stacking several atoms
that partially cancel, which is both a poor fit and a badly conditioned one. This is the
mechanism behind the measured regression of GELU under the `ReLU`-theory seed, and the
motivation for [`WeightBiasGrid2d`](@ref) and [`AngularGrid`](@ref).

### The atom sphere

There is a cleaner way to see both cases at once. Write the atom's parameters as a vector
``(w, b) \in \mathbb{R}^2`` and split it into a radius and a direction. By (H), for a
homogeneous activation the radius only rescales the atom, so the *shape* depends on the
direction alone — a point on the unit circle. The natural dictionary is therefore a grid
on that circle, which is how greedy approximation theory states it for
``\mathbb{R}^{d+1}``: a grid on the unit sphere.

That is what [`AngularGrid`](@ref) builds. It also explains a weakness of the bias grid:
uniform spacing in `b` at fixed ``\lvert w \rvert = 1`` is *not* uniform on the circle. It
concentrates resolution at large ``\lvert b \rvert``, where the atom is nearly constant on
``[0,1]`` and carries little information, and spreads it thinly near the origin, where the
interesting transitions live. For a non-homogeneous activation the radius stops being
redundant, and log-spaced radii recover the length-scale freedom without having to choose
a weight interval.

## The selection criterion

What should `score` be? Take the residual ``r`` orthogonal to the span of the atoms
already selected — which it is, by construction, since the refit is a projection. Adding a
candidate `g` and refitting reduces the squared residual by exactly

```math
\lVert r \rVert_w^2 - \lVert r_{\text{new}} \rVert_w^2
  \;=\; \frac{\langle r, g^{\perp} \rangle_w^{2}}{\lVert g^{\perp} \rVert_w^{2}}
  \;=\; \frac{\langle r, g \rangle_w^{2}}{\lVert g^{\perp} \rVert_w^{2}} ,
\tag{G}
```

where ``g^{\perp}`` is the component of `g` orthogonal to the selected span. The second
equality uses ``r \perp \operatorname{span}``, which kills the parallel part of `g` in the
inner product but *not* in the norm.

So the greedy step that maximises the one-step residual reduction ranks candidates by

```math
\mathrm{score}(g) = \frac{\lvert\langle r, g\rangle_w\rvert}{\lVert g^{\perp}\rVert_w} ,
```

which is [`OrthogonalProjection`](@ref). Two simplifications of it are also implemented:

- replacing ``\lVert g^{\perp} \rVert_w`` by ``\lVert g \rVert_w``, which ignores overlap
  with the selected set but keeps scale invariance — [`NormalizedProjection`](@ref). It is
  exact at the first step, where the selected set is empty and ``g^{\perp} = g``.
- dropping the denominator altogether — [`RawProjection`](@ref), the default rule.
  This is *not* scale invariant: a large-norm atom outranks a better-aligned small one.
  Harmless when all atoms have comparable norms, as on the `±1` grid; wrong on a 2-D grid,
  where norms differ by orders of magnitude.

Two caveats worth stating plainly. First, (G) is a **one-step** result: maximising it at
every step does not maximise the reduction after `S` steps. Greedy myopia is real here and
shows up in the measurements — on the harmonic-oscillator label sets,
`RawProjection` selects a worse first atom than the optimum but a better *quartet*.
Second, (G) is about the fit residual, whereas what actually matters is how well the seed
starts the Newton solve. The two are correlated but not identical, which is why the
[Studies](@ref) are split into a seed tier and an end-to-end tier.

### The rank-gain floor

(G) also gives a rank criterion for free. ``\lVert g^{\perp}\rVert_w`` is precisely the
amount of genuinely *new* direction the atom contributes: it is the diagonal entry the
atom would produce in a QR factorisation of the selected design matrix. If it collapses,
the atom is (numerically) in the span of its predecessors, and admitting it makes the
selected set rank-deficient.

[`OrthogonalProjection`](@ref) therefore refuses any atom with

```math
\lVert g^{\perp} \rVert_w \;<\; \texttt{min\_gain} \cdot \lVert g \rVert_w ,
\qquad \texttt{min\_gain} = \sqrt{\varepsilon(T)} \ \text{by default},
```

a *relative* test, so it is scale invariant. This is the direct fix for the
reduced-precision failure: a `SingularException: zero pivot found at index 3` — out of only
four neurons — is exactly this quantity going to zero at the third selected atom, and the
guard makes selecting such an atom impossible rather than merely detecting it afterwards.

## Conditioning: why the fit needed care

Fold the positive quadrature weights into a row scaling, ``\hat{A} =
\operatorname{diag}(\sqrt{w})\,\Phi^{\top}`` and ``\hat{y} =
\operatorname{diag}(\sqrt{w})\,y``. Then (LS) becomes an ordinary least-squares problem
``\min_c \lVert \hat{A} c - \hat{y}\rVert_2``, and the weighting disappears from every
subsequent formula. The implementation works entirely in this scaled space.

The original code solved (LS) through the **normal equations**

```math
G\,c = \Phi \operatorname{diag}(w)\, y , \qquad
G = \Phi \operatorname{diag}(w) \Phi^{\top} = \hat{A}^{\top}\hat{A} .
```

Forming `G` squares the condition number, ``\kappa(G) = \kappa(\hat{A})^{2}``
[GolubVanLoan:2013](@cite), [Higham:2002](@cite). Since a linear system is solvable in
floating point only while ``\kappa \lesssim 1/\varepsilon(T)``, squaring ``\kappa``
**halves the number of usable digits**. That is the whole story of the `Float64` island:
the dictionary is coherent enough that ``\kappa(\hat{A})`` is already large — near-duplicate
biases give near-parallel columns [HighamMary:2022](@cite) — so ``\kappa(\hat A)^2``
exceeded ``1/\varepsilon`` at `Float32` while still fitting under it at `Float64`.

Solving via a QR factorisation of ``\hat{A}`` instead [Bjorck:1996](@cite) never forms `G`,
so the accuracy is governed by ``\kappa(\hat{A})`` — recovering roughly the factor of two
in digits, which is exactly the `Float64`→`Float32` gap. This is
[`NonlinearIntegrators.weighted_lstsq`](@ref) and the [`WeightedQR`](@ref) fit.

Note what the island does *not* buy. The OGA result is a seed: it is rounded back to `T`
the moment it is stored, and the final accuracy is set by the working-precision Newton
solve. Double precision buys only robustness of an ill-conditioned solve, which a
well-conditioned formulation does not need.

## References

```@bibliography
Pages = ["theory.md"]
```
