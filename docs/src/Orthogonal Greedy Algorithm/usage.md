# Usage

## Presets

Five named configurations cover the useful corners. All are keyword functions returning an
[`OGA`](@ref), so they compose with the guard-rail keywords.

| Preset | Dictionary | Selection | Fit |
|---|---|---|---|
| [`OGA1d`](@ref) | [`BiasGrid1d`](@ref) | [`RawProjection`](@ref) | [`WeightedQR`](@ref) |
| [`OGA1dNormalized`](@ref) | [`BiasGrid1d`](@ref) | [`NormalizedProjection`](@ref) | [`WeightedQR`](@ref) |
| [`OGA1dStable`](@ref) | [`BiasGrid1d`](@ref) | [`OrthogonalProjection`](@ref) | [`IncrementalQR`](@ref) |
| [`OGA2d`](@ref) | [`WeightBiasGrid2d`](@ref) | [`NormalizedProjection`](@ref) | [`IncrementalQR`](@ref) |
| [`OGASphere`](@ref) | [`AngularGrid`](@ref) | [`NormalizedProjection`](@ref) | [`IncrementalQR`](@ref) |

[`OGA1dNormalEquations`](@ref) is the exception: a separate implementation rather than a
preset, kept unchanged so that claims about the modern variants are measured against the
published algorithm rather than a paraphrase of it. It is available for
`NonLinear_OneLayer_GML` only, and it carries none of the guard rails.

```julia
using NonlinearIntegrators, QuadratureRules

relu3(x) = max(zero(x), x)^3                       # float-generic — see Precision
basis = OneLayerNetwork_GML{Float64}(relu3, 4)     # S = 4 hidden neurons
quad  = QuadratureRules.GaussLegendreQuadrature(Float64, 8)

# The default: OGA1d()
method = NonLinear_OneLayer_GML(basis, quad; bias_interval = [-π, π], dict_amount = 400)

# A different preset
method = NonLinear_OneLayer_GML(basis, quad; bias_interval = [-π, π], dict_amount = 400,
                                initial_guess_method = OGA1dStable())

# The original-paper reference, for comparison
method = NonLinear_OneLayer_GML(basis, quad; bias_interval = [-π, π], dict_amount = 400,
                                initial_guess_method = OGA1dNormalEquations())
```

## Composing your own

The presets are corners of one type; any combination of the three axes is equally valid.

```julia
# Rank-revealing fit on the classical dictionary
OGA(BiasGrid1d(), OrthogonalProjection(), TruncatedSVD())

# 2-D dictionary with a wider weight range and a trimmed bias axis, so the total atom
# count stays comparable to the 1-D grid (the greedy step is linear in it)
OGA(WeightBiasGrid2d(octaves = (-4, 4), weight_amount = 8, bias_amount = 44),
    NormalizedProjection(), IncrementalQR())

# Off-grid refinement on a deliberately coarse grid
OGA(Refined(BiasGrid1d(); iterations = 5), NormalizedProjection(), IncrementalQR())

# Ablations: turn the guard rails off one at a time
OGA(BiasGrid1d(), RawProjection(), WeightedQR(); coherence = false)
OGA(BiasGrid1d(), RawProjection(), NormalEquationsFit(ridge = false, island = true))
```

The dictionary's bias axis is configured on the *method* (`bias_interval`, `dict_amount`),
since those predate this subsystem and every integrator already carries them. Axes that only
some dictionaries have — the weight range, the radii, the angular resolution — live on the
dictionary object, so no integrator struct had to change.

Because `OGA{D,S,F}` holds only singletons and immutables it stays `isbits`, so the method
struct's type parameter remains concrete and there is no dispatch cost.

## Choosing a variant

A short decision guide; the numbers behind it are in [Studies](@ref).

- **`ReLUᵏ` at `Float64`** — [`OGA1d`](@ref). The classical dictionary is complete, and the
  pinned raw-projection selection is what the accuracy guards were tuned against.
- **`ReLUᵏ` at 16 bits** — [`OGA1dStable`](@ref), or `Refined(BiasGrid1d())`. The rank-gain
  floor guarantees a full-rank selected set. Note the trade-off: it helps at `Float16` and
  can *hurt* at `Float64`, where one-step optimality does not survive greedy myopia.
- **ELU, GELU, tanh** — [`OGA2d`](@ref) or [`OGASphere`](@ref). The 1-D dictionary cannot
  represent more than one transition steepness, which is a structural limitation and not a
  numerical one; no fit or selection rule repairs it.
- **A coarse dictionary, or a cost budget** — wrap it in [`Refined`](@ref). It lets a few
  dozen atoms stand in for hundreds of thousands.
- **"Must never fail at any precision"** — [`TruncatedSVD`](@ref) with
  [`OrthogonalProjection`](@ref).

## Per-integrator behaviour

All four network integrators share [`oga_fit`](@ref). What differs is declared, not
reimplemented:

| Integrator | Ansatz | Symmetry | Constructor default |
|---|---|---|---|
| `NonLinear_OneLayer_GML` | — | [`NoSymmetry`](@ref) | [`OGA1d`](@ref) |
| `Hardcode_int` | `t(1-t)`, target minus the linear part | [`NoSymmetry`](@ref) | [`OGA1dNormalized`](@ref) |
| `Time_reversible_OneLayer` | — | [`MirrorPairs`](@ref) | [`OGA1d`](@ref) |
| `Time_Reversible_Hardcode` | `t(1-t)`, target minus the linear part | [`SharedMirrorPairs`](@ref) | [`OGA1d`](@ref) |

`Hardcode_int` is the odd one out, and deliberately so: its pre-refactor greedy step ranked
candidates by the normalised inner product where the other three used the raw one. Since that
choice decides which neurons are picked and hence which basin the Newton solve lands in, each
integrator keeps the rule it was tuned with rather than inheriting a single shared default.

Two further notes. The two boundary-ansatz integrators differ in where the endpoint of the
linear part comes from — `Hardcode_int` uses the last label (i.e. the initial-trajectory
integrator's estimate), `Time_Reversible_Hardcode` uses the cache's endpoint estimate `q̃` —
and both write that endpoint into the nonlinear solution vector, which the `OneLayer` variants
do not (there the corresponding slot is the momentum, set by `initial_trajectory!`).

`NonLinear_DenseNet_GML` has no OGA seed; it uses `TrainingMethod` or `LSGD`.

## Reading the result

[`oga_fit`](@ref) can also be called directly, which is how the seed study measures variants
without an integrator in the way:

```julia
using NonlinearIntegrators
const NI = NonlinearIntegrators

T = Float32
nodes = T.((0:10) ./ 10)
w     = NI.simpson_quadrature(10, T)
y     = T.(cos.(3 .* Float64.(nodes)))

r = oga_fit(OGA1dStable(), x -> max(zero(x), x)^3, nodes, w, y, 4;
            bias_interval = [-T(π), T(π)], dict_amount = 400)

r.W, r.b, r.c    # hidden weights, hidden biases, output weights (one entry per neuron)
r.atoms          # dictionary indices selected, in order
r.neurons        # how many of the 4 were actually placed
r.gains          # ‖g⊥‖ of each accepted atom — the rank gain it contributed
r.rejected       # candidates refused for adding no new direction
r.residual       # final weighted L² residual norm
```

`gains` and `rejected` are the diagnostics worth watching. A `gains` sequence collapsing
towards zero across the four atoms is the fingerprint of the failure this subsystem exists to
remove; `neurons < S` means the loop ran out of atoms that add a new direction and the
remainder were filled with zero-weight placeholders.

## Extending

Adding a component means one type and one method.

```julia
# A new dictionary
struct MyGrid <: OGADictionary
    n::Int
end
function NonlinearIntegrators.oga_atoms(d::MyGrid, bias_interval, dict_amount, ::Type{T}) where {T}
    # return an (natoms × 2) Matrix{T}: column 1 is w, column 2 is b
end
NonlinearIntegrators.oga_label(::MyGrid) = "mygrid"      # for the study reports

# A new fit
struct MyFit <: OGAFit end
function NonlinearIntegrators._oga_solve(::MyFit, Â::AbstractMatrix{T}, ŷ, qr) where {T}
    # return a length-size(Â,2) Vector{T}
end
```

Define `_oga_solve`, not `oga_solve`: the latter is the guarded wrapper that guarantees a
finite, correctly sized result for every fit, and defining it directly would opt out of that
guarantee. A new dictionary inherits the no-op [`NonlinearIntegrators.oga_refine`](@ref)
unless wrapped in [`Refined`](@ref).

Whatever you add must respect the precision discipline — see [Precision](@ref). The
`eltype === T` and `@inferred` assertions in `test/unit/oga_kernels.jl` sweep every
dictionary × selection × fit combination at three precisions, so a new component is covered
by adding it to the lists there.
