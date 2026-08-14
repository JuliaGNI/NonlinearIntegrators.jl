# ---- The OGA initial-guess method -------------------------------------------
#
# One composable seed type rather than one singleton per variant. The three axes —
# which atoms are on offer, how the greedy step ranks them, and how the output weights
# are refit — are independent, so crossing them as fields keeps a dozen-odd variants
# behind a single `initial_params!` implementation and makes every combination
# benchmarkable. Named presets recover the useful corners.

"""
    OGA(dictionary = BiasGrid1d(), selection = RawProjection(), fit = WeightedQR();
        coherence = true, norm_guard = true, fill_unused = true)

Orthogonal Greedy Algorithm initial guess for the network integrators.

At every time step the integrator must solve a nonlinear system for the parameters of a
shallow network `u(x) = Σₖ cₖ σ(wₖ x + bₖ)`. The OGA produces the starting point: it
repeatedly picks, from a fixed dictionary of candidate neurons `(w, b)`, the atom most
correlated with the current fit residual, then refits *all* output weights `c` by a
quadrature-weighted least-squares solve — the "orthogonal" part, as opposed to plain
matching pursuit, which would only fit the new atom.

The three axes:

* `dictionary::`[`OGADictionary`](@ref) — the candidate neuron set: the original
  `{±1} × (bias grid)` ([`BiasGrid1d`](@ref)), a genuine 2-D `(w, b)` grid
  ([`WeightBiasGrid2d`](@ref)), an angular grid on the atom sphere
  ([`AngularGrid`](@ref)), or any of them with off-grid polish ([`Refined`](@ref)).
* `selection::`[`OGASelection`](@ref) — how candidates are ranked:
  [`RawProjection`](@ref), [`NormalizedProjection`](@ref) or
  [`OrthogonalProjection`](@ref).
* `fit::`[`OGAFit`](@ref) — how the output weights are refit: [`WeightedQR`](@ref),
  [`IncrementalQR`](@ref), [`PivotedQR`](@ref), [`TruncatedSVD`](@ref) or
  [`NormalEquationsFit`](@ref).

Guard rails, all scaled to `eps(T)` rather than to absolute constants:

* `coherence` — after an atom is selected, block dictionary atoms whose weighted L²
  coherence with it exceeds `1 - sqrt(eps(T))`. Inert at `Float64`/`Float32`; it only
  bites where distinct atoms have rounded together.
* `norm_guard` — treat atoms whose weighted norm falls below
  [`oga_norm_floor`](@ref) as unusable rather than normalising by noise. Atoms with a
  non-finite norm (reachable at `Float16` with a high `ReLUᵏ` power over a wide bias
  interval, where `σ(b)ᵏ` overflows) are always excluded, guard or not.
* `fill_unused` — when the greedy loop runs out of usable atoms before all `S` neurons
  are placed, give the remaining neurons *distinct, well-separated* `(w, b)` with zero
  output weight. Without this they would all keep `(0, 0)` and become identical rows of
  the Newton Jacobian — trading a rank-deficient seed for a rank-deficient solve.

Everything runs at the solver's working precision `T`; see the precision note in
`src/oga/numerics.jl`.

# Examples

```julia
ShallowNet(basis, quadrature)                                # OGA1d(), the default
ShallowNet(basis, quadrature; initial_guess_method = OGA2d())
ShallowNet(basis, quadrature;
    initial_guess_method = OGA(BiasGrid1d(), OrthogonalProjection(), TruncatedSVD()))
```
"""
struct OGA{D<:OGADictionary,S<:OGASelection,F<:OGAFit} <: InitialParametersMethod
    dictionary::D
    selection::S
    fit::F
    coherence::Bool
    norm_guard::Bool
    fill_unused::Bool

    function OGA(dictionary::D = BiasGrid1d(), selection::S = RawProjection(),
                 fit::F = WeightedQR();
                 coherence::Bool = true, norm_guard::Bool = true,
                 fill_unused::Bool = true) where {D<:OGADictionary,S<:OGASelection,F<:OGAFit}
        new{D,S,F}(dictionary, selection, fit, coherence, norm_guard, fill_unused)
    end
end

"""
    OGA1d(; kwargs...)

The default seed: the original `{±1} × (bias grid)` dictionary, raw-projection selection,
and a QR fit of the `√w`-scaled design matrix.

Its atom choices are pinned by the regression tests: normalising before selection steers
the Newton solve into a different and empirically worse basin.
"""
OGA1d(; kwargs...) = OGA(BiasGrid1d(), RawProjection(), WeightedQR(); kwargs...)

"""
    OGA1dNormalized(; kwargs...)

`OGA1d`, but selecting on the *normalized* inner product.

`ShallowNetAutodiff`'s constructor default: alone among the four integrators it ranks candidates
by `|⟨r, g⟩_w| / ‖g‖_w` rather than by the raw projection. Which of the two an integrator
uses changes which neurons get picked and therefore which basin the Newton solve lands in,
so each keeps the rule it was tuned with rather than inheriting a single default.
"""
OGA1dNormalized(; kwargs...) = OGA(BiasGrid1d(), NormalizedProjection(), WeightedQR(); kwargs...)

"""
    OGA1dStable(; kwargs...)

The same 1-D dictionary made robust at reduced precision: orthogonal-greedy selection
with a rank-gain floor, on top of the incrementally maintained QR.

The combination aimed squarely at the 16-bit failure mode: an atom that adds no new
direction is never selected, so the selected design matrix cannot go rank-deficient
regardless of precision.
"""
OGA1dStable(; kwargs...) = OGA(BiasGrid1d(), OrthogonalProjection(), IncrementalQR(); kwargs...)

"""
    OGA2d(; dictionary = WeightBiasGrid2d(), kwargs...)

A 2-D `(w, b)` dictionary with normalised selection and the incremental QR fit: the
variant for activations that are *not* positively homogeneous (ELU, GELU, tanh), where
`|w|` is a genuine length-scale degree of freedom rather than redundant with `b` and `c`.
"""
OGA2d(; dictionary = WeightBiasGrid2d(), kwargs...) =
    OGA(dictionary, NormalizedProjection(), IncrementalQR(); kwargs...)

"""
    OGASphere(; dictionary = AngularGrid(), kwargs...)

Atoms sampled uniformly on the sphere of `(w, b)` space rather than uniformly in bias —
the dictionary the underlying approximation theory is stated for. See
[`AngularGrid`](@ref).
"""
OGASphere(; dictionary = AngularGrid(), kwargs...) =
    OGA(dictionary, NormalizedProjection(), IncrementalQR(); kwargs...)

"""
    OGA1dNormalEquations()

The reference implementation from the original paper, kept as a selectable baseline for
comparison.

The dictionary and the greedy least-squares fit are assembled in `Float64` — a
"double-precision island" — the output weights come from the normal equations
`Gₖ xₖ = bₖ`, and the result is rounded into the working-precision cache. It carries none
of the precision-scaled guard rails: no norm floor, no coherence guard, no ridge, and no
rank detection.

That combination is why it is the baseline rather than the default. Forming `Gₖ` squares
the condition number, so the fit needs roughly twice the digits the problem does, which is
what the `Float64` island supplies; and at 16 bits the atom *selection* degrades until the
third or fourth selected neuron is linearly dependent on its predecessors, at which point
the Gram solve raises `SingularException` before the Newton iteration has begun. See the
"Orthogonal Greedy Algorithm" section of the documentation for the full analysis, and
[`NormalEquationsFit`](@ref) for the same arithmetic available as a fit strategy inside the
modern composable [`OGA`](@ref), where the island and the ridge can be toggled
independently.

Select it with `ShallowNet(...; initial_guess_method = OGA1dNormalEquations())`.
"""
struct OGA1dNormalEquations <: InitialParametersMethod end

oga_label(oga::OGA) = string(oga_label(oga.dictionary), "/", oga_label(oga.selection),
                             "/", oga_label(oga.fit))
oga_label(::OGA1dNormalEquations) = "reference/raw/normaleq+f64"
