# OGA 1d instability

> **Status: implemented.** This note is kept as the design record. Remedy 1 (weighted QR)
> was already in place; remedies 3 (`PivotedQR`, `TruncatedSVD` — hand-rolled, since
> `qr(…, ColumnNorm())` and `svd` are LAPACK-only and so absent at `Float16`), 4
> (`IncrementalQR`) and 5 (`NormalEquationsFit`'s scale-aware ridge) landed in
> `src/oga/fits.jl`, and remedy 2's normalisation-and-coherence recommendation became the
> `NormalizedProjection` / `OrthogonalProjection` selection rules in
> `src/oga/selection.jl`.
>
> Two corrections to the closing paragraph. The `network_inputs` range is *not* a trap:
> `1/nstages` is `Float64` (`nstages::Int`), so the range is built and counted in double
> precision and only cast on assignment to the `Matrix{T}` field. And the GPU cleanups are
> only **partly** done — the dominant operations are vectorised (`mul!` for both the
> selection scan and the coherence guard, so the per-step cost is a mat-vec plus an
> `argmax`), but building the dictionary's design matrix and placing the selected neurons
> are still scalar loops. `src/oga/` is therefore precision-generic but not yet
> device-ready; making it so needs the dictionary build expressed as a broadcast or a
> kernel.
>
> See the *Orthogonal Greedy Algorithm* documentation page for the implemented design and
> `benchmark/oga_fit_study.jl` for the measurements.

The actual problem

Both variants form the **normal equations** and solve them:

```julia
Gk  = selected_g * (selected_g .* quad_weights')'   # Gram matrix Φ Wᵈ Φᵀ  (k×k)
rhs = selected_g * (labels .* quad_weights)
xk  = Gk \ rhs                                       # or (Gk + λI) \ rhs
```

Forming `Gk = ΦᵀWΦ` **squares the condition number**: `κ(Gk) = κ(Φ)²`. At `Float64` you have ~16 digits to burn, so even a dictionary that's coherent to 8 digits survives. At `Float32` (~7 digits) that same dictionary produces a numerically singular `Gk` and `\` returns garbage. This is the real reason for the `Float64` island — not the seed accuracy, but the κ² blow-up in the solve. So the fix is: **stop forming the Gram matrix.**

The good news for GPU: the selected system is *tiny* — `Gk` is `k×k` with `k ≤ S` neurons, `Φ` is `k × nstages` (~11 nodes). The only large, GPU-parallel operation is the greedy selection scan over the dictionary (`gx_quad * (...)` → `argmax`), which is just a mat-vec + reduction and is already precision-robust. That means you can afford the *most* robust dense solver for the fit without any performance concern.

## Remedies, most-impactful first

### 1. Solve the least-squares problem directly (QR on the weighted design matrix) — the main fix

Fold the (positive) Simpson weights into a `√w` row scaling so it becomes an ordinary LSQ, then use QR instead of normal equations:

```julia
sw  = sqrt.(quad_weights)                 # weights are positive ⇒ real sqrt
Â   = (selected_g .* sw')'                 # (nstages × k) weighted design matrix
ŷ   = network_labels[d, :] .* sw
xk  = Â \ ŷ                                # Julia dispatches to QR least-squares
```

This works on `κ(Φ)`, not `κ(Φ)²` — you recover roughly *half the lost digits*, which is exactly the `Float64`→`Float32` gap. In most cases this alone lets `Float32` match what `Float64`-normal-equations did. `qr` is available on GPU arrays via CUDA.jl / cuSOLVER (`geqrf`), and for a tiny tall-skinny matrix it's essentially free.

### 2. Kill rank deficiency at the source: dictionary normalization + coherence pruning

The rank deficiency comes from *duplicate/near-duplicate atoms* (neurons whose biases round together). Attack that directly rather than patching the solve:

- **Normalize** every dictionary atom to unit weighted-L² norm. The `Hardcode` variant already does this (lines 327–329); `NonLinear_OneLayer_GML` does **not** — add it there. It makes the greedy inner-product selection scale-invariant and improves conditioning for free.
- **Coherence guard in the greedy step:** when picking the next atom, skip candidates whose absolute correlation with the already-selected set exceeds `1 − ε`. This guarantees the selected columns stay linearly independent, so `Φ` can't become rank-deficient regardless of precision. Cheap (you already compute the projections for selection).

### 3. If you keep a direct solver, make it rank-revealing

For maximum robustness against a near-singular selected set, use a pivoted/​truncated factorization instead of plain `\`:

- **Column-pivoted QR:** `qr(Â, ColumnNorm())` — reveals and handles rank drop. Note: cuSOLVER's pivoted QR support is thinner than unpivoted, so on GPU prefer option 4 if you need this.
- **Truncated SVD / pseudo-inverse:** `F = svd(Â); xk = F.V * (F.U' * ŷ ./ F.S with tiny σ dropped)`. Most robust of all; `svd` (Jacobi `gesvdj`) is available on GPU and, at `k×nstages`, trivially cheap. This is what I'd reach for if you want a single method that never fails across `Float16/32/64`.

### 4. Incremental (rank-1-updated) QR — the "textbook" efficient+stable OGA

Because OGA adds exactly one atom per iteration, the canonical formulation maintains a QR factorization of the selected design matrix and updates it by one column per greedy step (one Householder/Givens update + a triangular solve). This is both *more stable* (never forms a Gram matrix) and *cheaper* (`O(k·nstages)` per step vs. re-solving `k×k` from scratch each iteration). It's more code, but it's the principled version of what the loop is hand-rolling today, and it batches cleanly over the `d` dimensions on GPU.

### 5. Tikhonov as a floor, done scale-aware

The `Hardcode` variants' `(Gk + 1e-12*I) \ rhs` is a band-aid on the normal equations and the *absolute* `1e-12`/`1e-14` are meaningless once you're in `Float32` (below `eps(Float32)≈1.2e-7`). If you keep any ridge term, scale it to the problem and precision:

```julia
λ = eltype(Â) === Float32 ? 1f-4 * tr(Gk)/k : 1e-10 * tr(Gk)/k
```

But regularization on `ΦᵀWΦ` is strictly inferior to options 1/3 — treat it as a last resort, not the primary mechanism.

## My recommendation for GPU

Do **(1) + (2)**: switch the fit to a QR least-squares solve on the `√w`-weighted design matrix, and add dictionary normalization + a coherence guard to the greedy selection. That combination removes the κ² blow-up *and* the rank-deficiency root cause, so the whole OGA path becomes precision-generic and runs natively in `Float32` on the device — no `Float64` island, no crashes. Add **(3, SVD)** only if you find a residual case that still degrades. Reserve **(4)** for when the per-step re-solve shows up in profiling.

Two GPU-portability cleanups to fold in while you're there:
- The bias grid `lo:(hi-lo)/dict_amount:hi` should be built index-based to avoid the `Float16` `Inf`/zero-step trap and to be GPU-friendly: `B = lo .+ (hi - lo) .* (0:dict_amount) ./ dict_amount`.
- Replace the scalar-indexed writes (`W[k] = ...`, per-`i` loops at the end) and `argmin`/`argmax` on CPU arrays with vectorized ops so the kernel doesn't force scalar indexing on the device.

Want me to prototype the QR + normalization + coherence-guard version of `initial_params!` for `NonLinear_OneLayer_GML` and verify it reproduces the `Float64` seed quality while running in `Float32`?
