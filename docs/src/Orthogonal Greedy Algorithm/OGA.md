# Orthogonal Greedy Algorithm Initial Guess

The network integrators in this package (`NonLinear_OneLayer_GML`, `Hardcode_int`,
`Time_reversible_OneLayer`, `Time_reversible_Hardcode_int`) solve, at every time step, a
nonlinear system for the parameters of a shallow neural network that represents the
trajectory between two discrete nodes. That nonlinear (Newton) solve needs a good initial
guess. This guess is produced by an **Orthogonal Greedy Algorithm (OGA)**
[Temlyakov:2008](@cite), a form of greedy training for shallow networks
[Siegel:2023](@cite): candidate neurons are drawn from a fixed dictionary, the neuron most
correlated with the current residual is added one at a time, and the output weights are
refit by (quadrature-weighted) least squares after each addition.

All of it lives in `src/oga/`, behind a single composable type.

## The pages

| Page | Contents |
|---|---|
| [Theory](@ref) | what is being approximated, why the classical dictionary has `±1` weights, the greedy selection criterion and where it comes from, the conditioning analysis |
| [Algorithms](@ref) | every dictionary, selection rule and fit in turn — mechanism, implementation, cost, when to use |
| [Usage](@ref) | presets, composing your own, per-integrator behaviour, reading the result, extending |
| [Precision](@ref) | the no-implicit-conversion invariant, how it is enforced, and the `Float16` traps |
| [Studies](@ref) | the two-tier measurement setup and what it found |

## The shape of the problem

Per time step and per solution component, the algorithm greedily builds

```math
u(t) = \sum_{k=1}^{S} c_k\, \sigma(w_k t + b_k)
```

by repeatedly picking, from a fixed dictionary of candidate neurons `(w, b)`, the atom most
correlated with the current fit residual, then refitting **all** output weights `c` by a
quadrature-weighted least-squares solve. That refit — the *orthogonal* part, as opposed to
plain matching pursuit, which would fit only the new atom — is where all the numerical
difficulty lives.

The seed is configured along three independent axes, crossed as fields of one [`OGA`](@ref)
type rather than enumerated as a type per combination:

- **dictionary** ([`OGADictionary`](@ref)) — which candidate neurons are on offer;
- **selection** ([`OGASelection`](@ref)) — how candidates are ranked against the residual;
- **fit** ([`OGAFit`](@ref)) — how the output weights are refit.

Named presets recover the useful corners: [`OGA1d`](@ref) (the default, and the
pre-refactor behaviour exactly), [`OGA1dNormalized`](@ref), [`OGA1dStable`](@ref),
[`OGA2d`](@ref), [`OGASphere`](@ref). The original-paper reference implementation is kept
separately as [`OGA1dNormalEquations`](@ref).

## Why this needed rework

Two problems motivated the current design.

### 1. The seed collapsed at reduced precision

The original fit solved the least-squares problem through the **normal equations**, forming
the Gram matrix ``G = \Phi\operatorname{diag}(w)\Phi^{\top}``. That squares the condition
number, ``\kappa(G) = \kappa(\Phi)^2`` [GolubVanLoan:2013](@cite), [Higham:2002](@cite),
which halves the number of usable digits — and the dictionary is coherent enough that the
squared condition number exceeded ``1/\varepsilon`` at `Float32` while still fitting under
it at `Float64`. Hence the `Float64` island: the seed was assembled in double precision
regardless of the solver's working type.

Crucially the island bought **no accuracy**. The OGA result is a seed, rounded back to `T`
the moment it is stored, and the final accuracy is set by the working-precision Newton
solve. It bought only robustness of an ill-conditioned solve — which is unnecessary once
the solve is no longer ill-conditioned.

Three hard-coded guard-rail constants encoded the same `Float64` assumption and were
silently ineffective in reduced precision:

- a dictionary-norm floor `dict_norms < 1e-12`, which sits *below*
  ``\varepsilon(\texttt{Float32}) \approx 1.2\times10^{-7}`` and so never fired;
- Tikhonov ridges `G + 1e-12·I` and `G + 1e-14·I`, which round away entirely below
  ``\varepsilon(\texttt{Float32})``;
- a bias grid `lo:(hi-lo)/dict_amount:hi` that threw `ArgumentError: range step cannot be
  zero` at `Float16`, because a large `dict_amount` overflows `Float16(dict_amount)` to
  `Inf`.

The fit is now a **QR factorisation of the ``\sqrt{w}``-scaled design matrix**
([`NonlinearIntegrators.weighted_lstsq`](@ref)), conditioned on ``\kappa(\Phi)`` rather than
``\kappa(\Phi)^2``, and every guard rail scales with ``\varepsilon(T)``. See
[Theory](@ref) for the analysis and [Algorithms](@ref) for the alternatives —
rank-revealing and incremental factorisations, and a selection rule that makes a
rank-deficient selected set impossible rather than merely detectable.

### 2. The dictionary was `ReLU`-only

Both original variants used `{±1}` crossed with a bias grid. That set is *complete* for a
positively homogeneous activation — for `ReLUᵏ`, the magnitude of `w` carries no shape
information that the bias grid and the output weight do not already absorb, so only its sign
remains. It starves ELU, GELU and tanh, for which `w` sets a genuine transition length
scale. [`WeightBiasGrid2d`](@ref) and [`AngularGrid`](@ref) supply that missing degree of
freedom; restricting the former to `{±1}` recovers the original dictionary exactly, so it is
a strict generalisation. See [Theory](@ref).

## A didactic `Float16` example

The following self-contained example reproduces, in miniature, the failure that used to
force the `Float64` island. We build four `ReLU³` neurons — two of them with almost
identical biases (`0.300` and `0.305`), the situation that makes the Gram matrix
ill-conditioned — and fit a target that is a known combination of them. We solve the
least-squares problem both the old way (normal equations / Gram matrix) and the new way (QR
on the ``\sqrt{w}``-scaled design matrix), in `Float64` and in `Float16`.

```@example oga
using LinearAlgebra

σ(x) = max(zero(x), x)^3          # ReLU³ activation

function setup(::Type{T}) where {T}
    t = T.(range(0, 1; length = 9))                 # quadrature nodes on [0,1]
    w = fill(one(T) / 9, 9)                         # (toy) positive weights
    biases = T[0.300, 0.305, -0.20, 1.50]           # atoms 1 and 2 nearly identical
    Φ = reduce(vcat, (permutedims(σ.(t .+ b)) for b in biases))   # (natoms × nnodes)
    y = T(0.7) .* σ.(t .+ T(0.30)) .- T(0.4) .* σ.(t .- T(0.20))  # target
    return t, w, Φ, y
end

# OLD: normal-equations (Gram) solve
function gram_solve(Φ, w, y)
    G   = Φ * (w .* Φ')
    rhs = Φ * (w .* y)
    return G, G \ rhs
end

# NEW: QR on the √w-scaled design matrix
function qr_solve(Φ, w, y)
    sw = sqrt.(w)
    Â  = sw .* Φ'
    return Â, Â \ (sw .* y)
end
nothing # hide
```

In `Float64` both methods recover the true weights `[0.7, 0, -0.4, 0]`, and we can see the
condition number relationship ``\kappa(G) = \kappa(\hat{A})^2`` directly:

```@example oga
t, w, Φ, y = setup(Float64)
G, xg = gram_solve(Φ, w, y)
Â, xq = qr_solve(Φ, w, y)

fit_err(x) = sqrt(sum(w .* (Φ' * x .- y) .^ 2))

println("cond(G)  = ", cond(G))
println("cond(Â)² = ", cond(Â)^2)
println("Gram x   = ", xg, "   fit-err = ", fit_err(xg))
println("QR   x   = ", xq, "   fit-err = ", fit_err(xq))
```

In `Float16`, the achievable condition number is only about
``1/\texttt{eps(Float16)} \approx 10^3``. The Gram condition number is far beyond that, so
the Gram solve returns *finite garbage* — the two near-duplicate atoms receive huge weights
of opposite sign — while the QR solve stays bounded and close to the truth:

```@example oga
t, w, Φ, y = setup(Float16)
G, xg = gram_solve(Φ, w, y)
Â, xq = qr_solve(Φ, w, y)

fit_err(x) = sqrt(sum(w .* (Φ' * x .- y) .^ 2))

println("1/eps(Float16) = ", 1 / eps(Float16))
println("cond(G)        = ", Float64(cond(Float64.(G))))
println("cond(Â)        = ", Float64(cond(Float64.(Â))))
println("Gram x = ", Float64.(xg), "   fit-err = ", Float64(fit_err(xg)))
println("QR   x = ", Float64.(xq), "   fit-err = ", Float64(fit_err(xq)))
```

The QR result matches what the shipped fit produces (`weighted_lstsq` implements exactly
this QR solve, plus the ridged fallback for the rank-deficient case):

```@example oga
using NonlinearIntegrators
t, w, Φ, y = setup(Float16)
println("weighted_lstsq x = ", Float64.(NonlinearIntegrators.weighted_lstsq(Φ, w, y)))
```

In the full integrator the `Float16` Gram garbage does not merely produce a poor seed: fed
into the Newton solve it makes the parameter Jacobian singular (two nearly identical neurons
are nearly linearly dependent), which previously surfaced as a `SingularException` or a
`NaN`. The QR reformulation, the coherence guard and the ridged fallback together keep the
seed finite and well-behaved, so the run proceeds at the working precision — and the
[Studies](@ref) measure how far that gets each variant.

## API

The helpers — `weighted_lstsq`, `scaled_lstsq`, `ridged_lstsq`, `oga_norm_floor`,
`oga_tikhonov`, `bias_grid`, `weight_grid`, `pivoted_qr_lstsq`, `jacobi_svd`,
`truncated_svd_lstsq`, `oga_check_precision` and the `IncrementalQRState` factorisation —
are documented with their full docstrings in the API listing on the home page.

## References

```@bibliography
Pages = ["OGA.md"]
```
