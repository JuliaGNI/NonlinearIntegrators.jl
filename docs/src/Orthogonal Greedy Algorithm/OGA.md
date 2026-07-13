# Orthogonal Greedy Algorithm Initial Guess

The network integrators in this package (`NonLinear_OneLayer_GML`, `Hardcode_int`,
`Time_reversible_OneLayer`, `Time_reversible_Hardcode_int`) solve, at every time
step, a nonlinear system for the parameters of a shallow neural network that
represents the trajectory between two discrete nodes. That nonlinear (Newton)
solve needs a good initial guess. This guess is produced by an **Orthogonal Greedy
Algorithm (OGA)** [Temlyakov:2008](@cite), a form of greedy training for shallow
networks [Siegel:2023](@cite): candidate neurons are drawn from a fixed dictionary,
the neuron most correlated with the current residual is added one at a time, and the
output weights are refit by (quadrature-weighted) least squares after each addition.

This page documents why the OGA fit used to be pinned to `Float64`, why that pin was
unnecessary for accuracy but *was* necessary given the original formulation, and how
the fit was reformulated so that the whole seed runs at the solver's working
precision `T` — including `Float32` (the GPU-native type) and `Float16`.

## The problem with the original formulation

Per spatial dimension, the OGA fit repeatedly solves a linear least-squares problem
for the output weights ``x`` of the currently selected neurons,

```math
\min_{x} \; \sum_j w_j \Bigl( \sum_i x_i\, \Phi_{ij} - y_j \Bigr)^2 ,
```

where each row of ``\Phi`` is a dictionary atom sampled at the quadrature nodes,
``w_j > 0`` are the quadrature weights and ``y_j`` is the fit target at node ``j``.
The original code solved this through the **normal equations**

```math
G\,x = \Phi\,\operatorname{diag}(w)\,\Phi^\top x = \Phi\,\operatorname{diag}(w)\,y .
```

Forming the Gram matrix ``G = \Phi\,\operatorname{diag}(w)\,\Phi^\top`` **squares the
condition number**, ``\kappa(G) = \kappa(\Phi)^2`` [GolubVanLoan:2013](@cite), [Higham:2002](@cite).
A matrix can only be solved reliably while its condition number stays below
``1/\varepsilon``, where ``\varepsilon`` is the unit roundoff of the working type
(`eps(T)`). Squaring the condition number therefore halves the number of usable
digits, and in reduced precision the Gram matrix becomes numerically singular: two
dictionary neurons whose biases round to nearly the same low-precision value produce
nearly identical columns of ``\Phi``, so ``G`` is (near-)rank-deficient
[HighamMary:2022](@cite). This is why the seed used to be assembled in `Float64`.

Crucially, the extra `Float64` precision brought **no accuracy benefit**: the OGA
result is only a *seed*, immediately rounded back to `T` when stored in the cache,
and the final accuracy is set by the working-precision Newton solve. The `Float64`
island only ever bought *robustness* of the ill-conditioned Gram solve — a robustness
that is unnecessary once the solve itself is well-conditioned.

Three hard-coded guard-rail constants encoded the `Float64` assumption and were
silently ineffective in reduced precision:

- a dictionary-norm floor `dict_norms < 1e-12`, which sits *below* `eps(Float32)`
  ``\approx 1.2\times 10^{-7}`` and so never fires;
- Tikhonov ridges `G + 1e-12·I` and `G + 1e-14·I`, which round away entirely below
  `eps(Float32)`;
- a bias grid `lo:(hi-lo)/dict_amount:hi` that threw `ArgumentError: range step
  cannot be zero` in `Float16` when a large `dict_amount` overflowed
  `Float16(dict_amount)` to `Inf`.

## The reformulated algorithm

The fit is now solved by a **QR factorization of the ``\sqrt{w}``-scaled design
matrix** rather than the normal equations. Folding the (positive) weights into a row
scaling turns the weighted problem into an ordinary least-squares problem,

```math
\min_{x} \; \bigl\lVert \hat{A}\,x - \hat{y} \bigr\rVert_2^2 , \qquad
\hat{A} = \operatorname{diag}(\sqrt{w})\,\Phi^\top , \qquad
\hat{y} = \operatorname{diag}(\sqrt{w})\,y ,
```

solved by QR [Bjorck:1996](@cite), [GolubVanLoan:2013](@cite). Because QR works directly on
``\hat{A}``, the accuracy is governed by ``\kappa(\Phi)`` instead of
``\kappa(\Phi)^2`` — roughly *doubling* the resolvable conditioning at every
precision, which is exactly the `Float64`→`Float32` gap. The Gram matrix is never
formed. This is implemented by [`NonlinearIntegrators.weighted_lstsq`](@ref).

Three further ingredients make the seed robust and precision-generic:

1. **Precision-scaled guard rails.** The hard-coded constants are replaced by
   formulas that track the working precision:
   - the norm floor becomes ``\sqrt{\varepsilon}\,\lVert\cdot\rVert_{\max}``
     ([`NonlinearIntegrators.oga_norm_floor`](@ref));
   - the Tikhonov ridge becomes ``\lambda = C\,\varepsilon\,\operatorname{tr}(G)/n``
     with a modest safety factor ``C`` ([`NonlinearIntegrators.oga_tikhonov`](@ref)).
   The ridge is applied only as a **fallback**: `weighted_lstsq` uses the plain QR
   solve whenever it is finite (so the well-conditioned `Float64`/`Float32` atom
   choice is unchanged and the greedy residual is unperturbed) and augments the
   design matrix with a ``\sqrt{\lambda}\,I`` block only when the plain solve returns
   a non-finite result — the genuinely rank-deficient `Float16` case.

2. **A coherence guard.** After a neuron is selected, dictionary atoms whose
   quadrature-weighted L² coherence with it exceeds ``1 - \sqrt{\varepsilon}`` are
   blocked from future selection. This keeps the selected neurons linearly
   independent. It is inert at `Float64`/`Float32` (distinct atoms have coherence
   well below the threshold) and only bites at `Float16`, where many grid biases
   collapse onto the same value.

3. **A safe bias grid.** [`NonlinearIntegrators.bias_grid`](@ref) builds the grid
   from an integer-indexed range and casts to `T`, so a large `dict_amount` cannot
   overflow the step to zero.

The greedy *selection* itself is deliberately left unchanged (it still maximizes the
raw inner product between the dictionary and the residual). Normalizing the
dictionary before selection would change which neurons are picked and steer the
Newton solve into a different — empirically worse — solution basin; normalization is
therefore used only to *measure* coherence for the guard above.

## A didactic `Float16` example

The following self-contained example reproduces, in miniature, the failure that used
to force the `Float64` island. We build four `ReLU³` neurons — two of them with
almost identical biases (`0.300` and `0.305`), the situation that makes the Gram
matrix ill-conditioned — and fit a target that is a known combination of them. We
solve the least-squares problem both the old way (normal equations / Gram matrix) and
the new way (QR on the ``\sqrt{w}``-scaled design matrix), in `Float64` and in
`Float16`.

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

In `Float64` both methods recover the true weights `[0.7, 0, -0.4, 0]`, and we can
see the condition number relationship ``\kappa(G) = \kappa(\hat{A})^2`` directly:

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
``1/\texttt{eps(Float16)} \approx 10^3``. The Gram condition number is far beyond
that, so the Gram solve returns *finite garbage* — the two near-duplicate atoms
receive huge weights of opposite sign — while the QR solve stays bounded and close to
the truth:

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

The QR result matches what the shipped fit produces (`weighted_lstsq` implements
exactly this QR solve, plus the ridged fallback for the rank-deficient case):

```@example oga
using NonlinearIntegrators
t, w, Φ, y = setup(Float16)
println("weighted_lstsq x = ", Float64.(NonlinearIntegrators.weighted_lstsq(Φ, w, y)))
```

In the full integrator the `Float16` Gram garbage does not merely produce a poor
seed: fed into the Newton solve it makes the parameter Jacobian singular (two nearly
identical neurons are nearly linearly dependent), which previously surfaced as a
`SingularException` or a `NaN`. The QR reformulation, the coherence guard and the
ridged fallback together keep the seed finite and well-behaved, so the run proceeds
at the working precision.

The helper functions `weighted_lstsq`, `oga_norm_floor`, `oga_tikhonov` and
`bias_grid` are documented (with their full docstrings) in the API listing on the
home page.

## References

```@bibliography
Pages = ["OGA.md"]
```
