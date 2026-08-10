# Proposal: a 2-D OGA dictionary for smooth activations

> **Status: implemented.** This note is kept as the design record. The proposal landed as
> `WeightBiasGrid2d` in `src/oga/dictionaries.jl`, reachable via the `OGA2d()` preset, with
> `AngularGrid` and `Refined` as further generalisations along the same lines. See the
> *Orthogonal Greedy Algorithm* page of the documentation for the implemented design, and
> `benchmark/oga_sweep.jl`'s `smooth` stage for the measurements. Note the name
> `OGA1d_Legacy` used below is now `OGA1dNormalEquations`.

*Status when written: recommendation, not implemented in this repository.* This is a proposed change to
[NonlinearIntegrators.jl](https://github.com/JuliaGNI/NonlinearIntegrators.jl) — specifically the
OGA seeds in `src/network_integrators/NonLinear_OneLayer_GML.jl` — motivated by the activation &
seed study run here (`scripts/nonlinear_activation_study.jl`).

## Background

The `NonLinear_OneLayer_GML` integrator seeds its per-step nonlinear solve with an Orthogonal
Greedy Algorithm (OGA): it greedily builds a one-layer network `u(x) = Σₖ cₖ σ(wₖ x + bₖ)` by
repeatedly picking, from a fixed **dictionary** of candidate neurons `(w, b)`, the atom most
correlated with the current fit residual, then refitting the output weights `c` by a
quadrature-weighted least-squares solve.

Both OGA variants build the candidate dictionary the same way — **fixed weights `w = ±1`** paired
with a **1-D uniform grid of biases `b`**:

- `OGA1d` (working-precision QR): `A = hcat(vcat(-ones(T,…), ones(T,…)), vcat(B, B))`, with
  `B = bias_grid(bias_interval..., dict_amount, T)` — `NonLinear_OneLayer_GML.jl:350-351`.
- `OGA1d_Legacy` (Float64-island normal equations): the same `±1 × grid` construction at
  `:451-453`.

So the dictionary has `2 · (dict_amount + 1)` atoms, all with `|w| = 1`, differing only in the
sign of `w` and the value of `b`.

## Why ±1 weights suit ReLU but under-serve ELU/GELU

For a **positively homogeneous** activation — `ReLU` and, up to the power, `ReLU^k`
(`x ↦ max(0, x)^k`) — the magnitude of `w` is *not* an independent degree of freedom of the atom
*shape*. Because `σ(w x + b) = |w|ᵏ · σ(sign(w) x + b/|w|)`, scaling `w` only rescales the atom and
shifts the bias — effects the output weight `c` and the bias grid already absorb. The only shape
information left in `w` is its **sign**. This is exactly the shallow-ReLU approximation-theory
setting the ±1 dictionary is derived from, and there it is complete: `{±1} × (bias grid)` spans the
ridge directions the greedy step needs.

Smooth activations break this. **ELU** (`x` for `x > 0`, `exp(x) − 1` otherwise) and **GELU**
(`≈ 0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))`) are **not** positively homogeneous: `w` sets an
intrinsic **length scale** (how sharply the unit transitions), which is a genuine shape parameter
independent of `b` and `c`. With `|w|` pinned to `1`, the dictionary can only place transitions of a
single steepness along the bias axis; it cannot represent sharper or gentler transitions, so the
greedy seed is a poor fit and the resulting Newton system is ill-conditioned. This is the mechanism
behind the prior observation that GELU + the ReLU-theory OGA seed regressed even the Float64 solve.

## Proposal: a 2-D `(w, b)` dictionary

Replace the `{±1} × (bias grid)` set with a genuine **2-D grid over `(w, b)`**, keeping everything
downstream unchanged:

1. **Dictionary construction.** Build a grid of weights `w ∈ [w_lo, w_hi]` (spanning several
   octaves of length scale, e.g. logarithmically spaced and sign-symmetric,
   `±{2⁻ᵐ … 2⁺ᵐ}`) crossed with the existing bias grid `b`. Assemble the atom matrix as
   `A = [vec(w_grid) vec(b_grid)]` (all `(w, b)` pairs), reusing `bias_grid` for the bias axis and
   an analogous helper for the weight axis so the reduced-precision coordinate safeguards carry
   over. Add a `weight_interval` (and its resolution) alongside the existing `bias_interval` /
   `dict_amount` constructor keywords.
2. **Greedy selection and refit unchanged.** The design matrix `gx_quad = activation.(A * nodes)`,
   the correlation-maximising `argmax` step, the coherence/dedup guard, and the weighted-QR refit
   (`weighted_lstsq`) all operate on the atom matrix generically and need no change — only `A` grows
   a meaningful weight column.
3. **Reduced-precision safeguards preserved.** `oga_norm_floor` (atom-norm floor), the coherence
   cap, and `weighted_lstsq`'s Tikhonov fallback are activation- and dictionary-agnostic and apply
   as-is. Unit-norm normalization becomes *more* important with a 2-D grid, since atoms of very
   different `|w|` have very different raw norms.

### Cost

The dictionary size grows from `2·(dict_amount+1)` to `n_w · (n_b+1)`. The greedy step is linear in
the dictionary size (one matrix–vector product against the residual per selected neuron), so a
modest weight grid (a handful of octaves) is affordable, and `dict_amount` can be reduced on the
bias axis to compensate. As with the 1-D dictionary, the grid coordinates may be generated in
`Float64` and cast to the working precision `T`.

### Backward compatibility

For ReLU/ReLU^k the extra weight degrees of freedom are redundant (see above), so a 2-D dictionary
should reproduce the 1-D result up to the greedy tie-breaking — restricting the weight grid to
`{±1}` recovers the current behaviour exactly. The proposal is therefore a strict generalization:
neutral for the homogeneous activations the benchmark ships with, and enabling for the smooth
activations this study evaluates.

## Scope

This document is a recommendation for the pinned NonlinearIntegrators git revision. It is **out of
scope** for the SolverBenchmark repository, which only constructs and drives the integrator; the
activation & seed study here uses the existing `OGA1d` seed and measures how far smooth activations
get with it.
