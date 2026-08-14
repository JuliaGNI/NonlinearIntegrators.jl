# Algorithms

Every implemented component, with what it computes, how it is implemented, what it costs,
and when to reach for it. The mathematical background is on the [Theory](@ref) page; how to
select and combine these is on the [Usage](@ref) page.

Throughout, `M` is the number of quadrature nodes (11 by default), `N` the dictionary size,
`S` the number of hidden neurons, and `k ≤ S` the number of atoms selected so far. All
formulas are in the ``\sqrt{w}``-scaled space, where the Euclidean inner product *is* the
quadrature-weighted one.

## The greedy loop

[`oga_fit`](@ref) is the single implementation shared by all four network integrators. It
is deliberately integrator-agnostic: it takes a dictionary specification, an activation,
nodes, weights and a target, and returns neuron parameters. It knows nothing about
`GeometricIntegrator`, the parameter cache, or the variational equations, which is what
makes it directly testable at any precision.

```julia
oga_fit(oga, σ, nodes, w, y, nneurons;
        bias_interval, dict_amount, modulation = nothing, symmetry = NoSymmetry())
    -> OGAResult
```

Per step it:

1. scores every dictionary atom against the current residual
   ([`oga_scores!`](@ref NonlinearIntegrators.oga_scores!), one `mul!` against the
   dictionary — the dominant cost, ``O(NM)``);
2. masks atoms blocked by the coherence guard and takes an `argmax`;
3. optionally polishes the winner off-grid ([`Refined`](@ref));
4. appends the atom's column(s) to the maintained QR, rejecting the atom and continuing to
   the next-best candidate if it adds no new direction;
5. refits all output weights ([`oga_solve`](@ref NonlinearIntegrators.oga_solve));
6. updates the residual and blocks atoms too coherent with the winner.

[`OGAResult`](@ref) carries the neuron parameters together with the diagnostics that make
rank behaviour visible: `atoms` (the selected indices, in order), `neurons` (how many were
actually placed), `gains` (the ``\lVert g^{\perp}\rVert_w`` of each accepted atom — a
sequence collapsing towards zero is the fingerprint of the reduced-precision failure),
`rejected`, and the final weighted residual norm.

### Atoms, neurons and columns

One atom does not always mean one neuron. [`OGASymmetry`](@ref) declares the mapping,
which is what lets the two time-reversible integrators share this loop:

| Symmetry | Neurons per atom | Design columns | Output weights | Used by |
|---|---|---|---|---|
| [`NoSymmetry`](@ref) | 1 | 1 | independent | `ShallowNet`, `ShallowNetAutodiff` |
| [`MirrorPairs`](@ref) | 2 | 2 | independent | `ShallowNetReversible` |
| [`SharedMirrorPairs`](@ref) | 2 | 1 (their sum) | shared | `ShallowNetAutodiffReversible` |

The mirror map is ``(w, b) \mapsto (-w,\, w + b)``, which sends ``\sigma(wt + b)`` to
``\sigma(w(1-t) + b)`` — a reflection about the midpoint of the step. For
`SharedMirrorPairs` both members carry the *same* output weight, and that sharing is what
actually enforces time-reversal symmetry of the ansatz; with independent weights the pair
can drift apart.

### The boundary ansatz

`ShallowNetAutodiff` and `ShallowNetAutodiffReversible` represent the step as

```math
q(t) = (1-t)\,\bar{q} + t\,\tilde{q} + t(1-t)\, u(t) ,
```

so the network only has to fit what is left after the linear part, and every dictionary
atom carries the ``t(1-t)`` factor. Both enter through `oga_fit`'s `modulation` argument
(the ``t(1-t)`` vector) and a target with the straight line subtracted — no separate code
path.

## Dictionaries

### [`BiasGrid1d`](@ref)

`{±1}` crossed with a uniform grid of `dict_amount + 1` biases over the method's
`bias_interval`, giving ``2(\texttt{dict\_amount}+1)`` atoms.

*Theory.* Complete for `ReLUᵏ`: by positive homogeneity the magnitude of `w` carries no
shape information the bias grid and output weight do not already absorb, so only the sign
remains. See [Theory](@ref).

*Implementation.* The grid comes from
[`NonlinearIntegrators.bias_grid`](@ref), which generates an integer-indexed range in
`Float64` and casts once to `T`. Computing `lo:(hi-lo)/n:hi` at `T` instead carries a
half-precision trap: `Float16(70000)` overflows to `Inf`, the step evaluates to zero, and
the range constructor throws `ArgumentError: range step cannot be zero`. Atom order — the
`w = -1` block first, then `w = +1` — is load-bearing, since `argmax` breaks ties by first
index.

*Cost.* ``N = 2(\texttt{dict\_amount}+1)``.

*When.* The default, and the right choice for `ReLUᵏ`.

### [`WeightBiasGrid2d`](@ref)

A genuine 2-D grid: `weight_amount + 1` magnitudes spaced logarithmically over
``2^{\texttt{octaves}[1]} \ldots 2^{\texttt{octaves}[2]}``, optionally sign-symmetric,
crossed with the bias grid.

*Theory.* For an activation that is *not* positively homogeneous, ``\lvert w \rvert`` sets
the transition's length scale, an independent shape parameter. The weight axis is spaced
logarithmically because length scales compare by ratio, not by difference; the default
`octaves = (-3, 3)` spans a factor of 64.

*Implementation.* [`NonlinearIntegrators.weight_grid`](@ref) exponentiates an
integer-indexed `Float64` range and casts once, so an octave outside the range of `T`
saturates predictably rather than overflowing mid-computation. `bias_amount` overrides
`dict_amount` on the bias axis, which is how the total atom count is held roughly constant
while the weight axis is added.

*Cost.* ``N = n_w \cdot (n_b + 1)``, with ``n_w = 2(\texttt{weight\_amount}+1)`` when
signed. Since the greedy step is linear in `N`, trim `bias_amount` to compensate.

*When.* ELU, GELU, tanh — any non-homogeneous activation. Setting
`octaves = (0, 0), weight_amount = 0` recovers [`BiasGrid1d`](@ref) *exactly* (asserted in
`test/unit/oga_kernels.jl`), so this is a strict generalisation: neutral for the homogeneous
activations and enabling for the smooth ones. Pair with
[`NormalizedProjection`](@ref) — with raw projection the large-``\lvert w\rvert`` atoms
would be ranked by amplitude rather than by fit.

### [`AngularGrid`](@ref)

Atoms on rays through the origin of `(w, b)` space: ``(w, b) = r\,(\cos\theta,
\sin\theta)`` for a uniform grid of angles over ``[0, 2\pi)`` and each radius in `radii`.

*Theory.* This is the dictionary greedy approximation theory is stated for — a grid on the
unit sphere of ``\mathbb{R}^{d+1}``. It unifies the two cases: for a homogeneous activation
one radius suffices and the set covers the same ridge directions as `{±1} × grid`, but
sampled *uniformly in atom space*. The bias grid is not: uniform spacing in `b` at
``\lvert w \rvert = 1`` concentrates resolution where ``\lvert b \rvert`` is large and the
atom is nearly constant on ``[0,1]``, i.e. where it matters least. For a non-homogeneous
activation the radius *is* the length scale.

*Implementation.* The full circle, not a half circle: ``\sigma(t)`` and ``\sigma(-t)`` are
different functions, so the sign of `w` is real shape information even though its *scale*
is redundant. The endpoint is excluded because ``2\pi`` wraps onto 0.

*Cost.* ``N = \lvert\texttt{radii}\rvert \cdot (\texttt{amount}+1)``.

*When.* As an alternative to `WeightBiasGrid2d` for smooth activations, and worth trying
for `ReLUᵏ` too — in the seed study it wins more often than the bias grid does, which the
non-uniform-coverage argument above predicts.

### [`Refined`](@ref)

A decorator: after the greedy `argmax` picks a grid atom, its `(w, b)` are polished *off*
the grid by locally maximising the selection score.

*Theory.* The grid then only has to identify the right neighbourhood. This decouples
accuracy from dictionary size — a few dozen atoms plus refinement can match hundreds of
thousands — and since the greedy step is linear in `N`, that is a large cost saving. It is
the standard "OGA with inner optimisation".

*Implementation.* A derivative-free compass search: evaluate the score at ``(w \pm h, b)``
and ``(w, b \pm h)``, step to any improvement, shrink `h` when none improves, repeat
`iterations` times. Derivative-free on purpose — the score is only piecewise smooth for
`ReLUᵏ` (the kink crosses a quadrature node), and it keeps the activation off the
ForwardDiff path entirely, so no `Dual` tag can leak into the working precision.

The polished objective is always the **normalised** score, even under
[`RawProjection`](@ref). The raw inner product is a ranking heuristic among atoms of
comparable norm, not an objective: maximised continuously over `(w, b)` it rewards growing
the atom rather than fitting the residual, and the search would drift to large
``\lvert w\rvert`` while the fit got worse.

*Cost.* ``4 \cdot \texttt{iterations}`` extra score evaluations per step, each ``O(M)`` —
negligible against the ``O(NM)`` scan.

*When.* Whenever the dictionary is coarse, and as a cheap accuracy win at any size. Note
the guarantee is one-step: refinement provably improves the *first* atom's fit but, by
greedy myopia, not necessarily the final quartet.

## Selection rules

All three fill a score vector by one `mul!` of the dictionary against the residual and mark
unusable atoms with `-1`, so `argmax` skips them. An atom is unusable if its projection or
its norm is non-finite, or its norm is below the floor.

### [`RawProjection`](@ref)

``\mathrm{score}(g) = \lvert\langle r, g\rangle_w\rvert``.

The default, and what the regression tests pin:
normalising before selection changes which neurons are picked and steers the Newton solve
into a different — empirically worse — basin. Not scale invariant, which is harmless on the
`±1` grid (comparable norms) and wrong on a 2-D grid.

*Cost.* ``O(NM)``.

### [`NormalizedProjection`](@ref)

``\mathrm{score}(g) = \lvert\langle r, g\rangle_w\rvert / \lVert g\rVert_w``.

The textbook criterion: it measures how much of the residual the atom *explains*,
independently of amplitude, which the output weight absorbs anyway. Exact for the first
atom. Mandatory for [`WeightBiasGrid2d`](@ref) and [`AngularGrid`](@ref). Also
`ShallowNetAutodiff`'s rule, hence the [`OGA1dNormalized`](@ref) preset.

*Cost.* ``O(NM)``; the norms are precomputed once per fit.

### [`OrthogonalProjection`](@ref)

``\mathrm{score}(g) = \lvert\langle r, g\rangle_w\rvert / \lVert g^{\perp}\rVert_w``, with
any atom whose orthogonal part has collapsed refused outright.

The actual orthogonal-greedy criterion — it maximises the one-step residual reduction
exactly (equation (G) on the [Theory](@ref) page) — and the direct fix for the
reduced-precision failure.

*Implementation.* ``\lVert g^{\perp}\rVert^2 = \lVert g \rVert^2 - \lVert Q^{\top}g
\rVert^2``, so all `N` deflated norms come from one product of the dictionary against the
maintained `Q`. Because the residual is already orthogonal to the selected span, the
numerator is unchanged from `NormalizedProjection`; only the denominator differs.
Subtracting in `T` can go slightly negative for an atom already fully explained — exactly
the atom to reject — so the result is clamped rather than erroring. The `min_gain` floor
defaults to ``\sqrt{\varepsilon(T)}``.

*Cost.* one extra ``O(NMk)`` product per step, the same order as the score itself.

*When.* At reduced precision, where it guarantees a full-rank selected set. Measured, it is
the strongest selection rule in the seed study. Be aware of the trade-off: at `Float64` its
one-step optimality does not survive greedy myopia, and the rank-gain floor can refuse
atoms that would have been usable — end-to-end it helps at 16 bits and can hurt above.

## Fits

!!! tip "The fit is free, so choose it for robustness alone"
    The selected system is *tiny*: ``\hat{A}`` is ``M \times k`` with ``M \approx 11``
    quadrature nodes and ``k \le S \le 8`` neurons, and the Gram matrix it replaces would be
    ``k \times k``. The only operation that scales with anything large is the greedy
    selection scan over the dictionary, which is a matrix–vector product plus an `argmax`
    and is already precision-robust.

    So the fit contributes essentially nothing to the runtime, at any of the five
    factorisations, and there is no performance argument for preferring a cheaper one. Pick
    on numerical behaviour: the measurements in [Studies](@ref) put the rank-revealing fits
    ahead at `Float64`, which is not what a cost-first reading would have suggested.

All five receive the ``\sqrt{w}``-scaled design matrix ``\hat{A}`` (``M \times k``) and
target ``\hat{y}``, so they minimise the same objective and differ only in the
factorisation. [`oga_solve`](@ref NonlinearIntegrators.oga_solve) wraps every one of them
with a single guarantee: **the result has one entry per column and every entry is finite**.
Where a factorisation throws on a rank-deficient design or divides by a pivot that survived
truncation, it falls back to the ridged solve. Enforcing that once, rather than per fit,
means it holds for a new fit by construction.

### [`WeightedQR`](@ref)

QR of ``\hat{A}``, re-solved from scratch each step
([`NonlinearIntegrators.weighted_lstsq`](@ref)). Conditioned on ``\kappa(\hat A)`` rather
than ``\kappa(\hat A)^2``; no Gram matrix, and no ridge unless the plain solve returns
non-finite.

The default, and the fit whose arithmetic the `Float64`/`Float32` regression tests pin.

*Cost.* ``O(Mk^2)`` per step.

!!! note "`\` can throw, not just mislead"
    For a non-BLAS element type — i.e. at `Float16` — Julia's `\` **throws**
    `SingularException` on a rank-deficient matrix rather than returning garbage. Letting
    that escape would put a `SingularException` back on the seed path at exactly the
    precision this fit exists to rescue, so it is caught and routed to the ridged solve.

### [`IncrementalQR`](@ref)

Reuses the QR maintained across greedy steps
([`NonlinearIntegrators.IncrementalQRState`](@ref)): one triangular solve instead of a fresh
factorisation.

*Implementation.* Columns are appended by modified Gram–Schmidt with **one
reorthogonalisation pass**. That second pass is not bookkeeping: plain MGS loses
orthogonality in proportion to the condition number — which at `Float16` is the whole
problem — whereas reorthogonalising once restores it to ``O(\varepsilon(T))`` for any
conditioning that has not already collapsed.

*Cost.* ``O(Mk)`` per step, against ``O(Mk^2)`` for a fresh factorisation. It also produces
two quantities the loop wants anyway: the appended column's deflated norm (the rank gain)
and `Q` itself, which is what makes [`OrthogonalProjection`](@ref) one matrix product.

*When.* Pair it with `OrthogonalProjection` — together they are the "textbook" efficient
*and* stable OGA, and it is numerically equivalent to `WeightedQR` up to rounding.

### [`PivotedQR`](@ref)

Householder QR with column pivoting, truncated where the pivot norm drops below
``\texttt{rtol}`` times the first pivot
([`NonlinearIntegrators.pivoted_qr_lstsq`](@ref)).

*Theory.* Pivoting is what makes it rank-revealing: at each step the column with the largest
remaining norm is brought forward, so a numerically dependent column is pushed to the end
and detected by its collapsed pivot instead of being solved through. Columns past the
detected rank receive a zero coefficient.

*Implementation.* Hand-rolled, because `qr(Â, ColumnNorm())` is LAPACK-only and therefore
does not exist at `Float16` — the precision the remedy is *for*. Widening to `Float32` to
borrow LAPACK would reintroduce the `Float64` island in miniature. Column norms are
*recomputed* after each elimination rather than downdated — downdating loses accuracy
precisely where the pivot norms collapse, which is the regime the factorisation exists to
detect, and at ``k \le 8`` columns the recomputation is free; the reflector follows the
LAPACK convention
(``v_1 = 1``, unnormalised) to avoid a division that can underflow at half precision.

*Cost.* ``O(Mk^2)`` per step, plus the pivot search.

### [`TruncatedSVD`](@ref)

Minimum-norm solve through a truncated pseudo-inverse, dropping singular directions with
``\sigma < \texttt{rtol}\,\sigma_{\max}``
([`NonlinearIntegrators.truncated_svd_lstsq`](@ref)).

*Theory.* The most robust of the five — a rank-deficient selected set yields a bounded
solution rather than amplified rounding noise — and the one to reach for if a single
configuration must work unchanged at every precision. Where `PivotedQR` zeroes a dependent
*column*, this returns the minimum-norm solution, spreading weight across a duplicated pair;
the residual is the same and ``\lVert c \rVert`` is smaller.

*Implementation.* One-sided Jacobi ([`NonlinearIntegrators.jacobi_svd`](@ref)): orthogonalise
the columns pairwise by plane rotations until mutually orthogonal, at which point the column
norms are the singular values. Chosen over bidiagonalisation because it is short, generic in
the element type (`svd` is likewise LAPACK-only), and has *high relative accuracy on the
small singular values* — exactly the ones that decide whether the selected atoms are still
independent.

!!! warning "A `Float16` trap in the convergence test"
    The rotation is skipped when ``\lvert\beta\rvert \le \varepsilon(T)\sqrt{\alpha}
    \sqrt{\gamma}`` — deliberately *not* ``\sqrt{\alpha\gamma}``. The product of two squared
    column norms overflows `Float16` once the norms exceed about 16, at which point the
    threshold becomes `Inf`, every column pair tests as "already orthogonal", and the routine
    silently returns the **unrotated** matrix: wrong singular values, no error raised. This
    was a real bug, caught by comparing against LAPACK at `Float16`.

*Cost.* ``O(Mk^2)`` per sweep, a handful of sweeps at ``k \le 8``.

### [`NormalEquationsFit`](@ref)

``G c = \Phi\operatorname{diag}(w) y`` with ``G = \hat{A}^{\top}\hat{A}``, with the
precision-scaled Tikhonov ridge of [`NonlinearIntegrators.oga_tikhonov`](@ref) and the
`Float64` island as **independent switches**.

This is the *baseline*, not a recommendation: forming `G` squares the condition number. It
exists so that "island vs working precision" and "ridge vs no ridge" are two knobs on one
code path that can be ablated, rather than differences buried in forked implementations.
`NormalEquationsFit(ridge = false, island = true)` reproduces the arithmetic of
[`OGA1dNormalEquations`](@ref).

*Cost.* ``O(Mk^2 + k^3)``.

!!! note
    `island = true` deliberately violates the package's precision discipline. It is the
    thing being measured against — see [Precision](@ref).

## Guard rails

These are fields on [`OGA`](@ref), orthogonal to the three axes.

**Norm floor** (`norm_guard`, default `true`). Atoms whose weighted norm falls below
[`NonlinearIntegrators.oga_norm_floor`](@ref) — ``\sqrt{\varepsilon(T)}`` times the largest
atom norm — are unusable, rather than normalised by noise. It replaced a hard-coded absolute
`1e-12`, which sat *below* ``\varepsilon(\texttt{Float32})`` and so never fired in reduced
precision. Atoms with a **non-finite** norm are always excluded regardless of the flag; that
is correctness, not policy — at `Float16` a high `ReLUᵏ` power over a wide bias interval
overflows ``\sigma(b)^k``, and an `Inf` norm would sail past a bare `n > floor` test and then
divide to `NaN`.

**Coherence guard** (`coherence`, default `true`). After an atom is selected, atoms whose
weighted-``L^2`` coherence with it exceeds ``1 - \sqrt{\varepsilon(T)}`` are blocked from
future selection, keeping the selected set independent. Inert at `Float64`/`Float32`, where
distinct atoms are well separated; it only bites at `Float16`, where many grid biases collapse
onto the same value. Computed on the fly as ``\langle g_i, g_{\text{best}}\rangle /
(\lVert g_i\rVert \lVert g_{\text{best}}\rVert)``, which avoids materialising a second
dictionary-sized array.

**Unused-neuron fill** (`fill_unused`, default `true`). When the loop runs out of usable atoms
before all `S` neurons are placed, the remainder get *distinct, well-separated* `(w, b)` with
zero output weight. Without this they would all keep `(0, 0)`, become identical rows of the
Newton Jacobian, and trade a rank-deficient seed for a rank-deficient solve — the failure
would move rather than go away.

**Activation check.** [`oga_fit`](@ref) calls
[`NonlinearIntegrators.oga_check_precision`](@ref) once per fit and throws if `σ(::T)` is not
a `T`. See [Precision](@ref).

**Neuron-count check.** [`oga_fit`](@ref) calls
[`NonlinearIntegrators.oga_check_neuron_count`](@ref) and throws if `nneurons` is not a
multiple of `neurons_per_atom(symmetry)` — i.e. if an odd count is requested under
[`MirrorPairs`](@ref) or [`SharedMirrorPairs`](@ref). The greedy loop places whole atoms, so
an odd count would run one step fewer and leave the last neuron at `(0, 0)`, which the
unused-neuron fill cannot repair either since it fills a pair at a time. That is exactly the
duplicated-row state the fill exists to prevent, so the count is rejected rather than
half-honoured. The two time-reversible integrators enforce the same condition on `S` at
construction.
