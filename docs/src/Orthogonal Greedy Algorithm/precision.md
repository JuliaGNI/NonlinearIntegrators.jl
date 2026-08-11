# Precision

Everything on the OGA path runs at the solver's working type `T`, with no implicit
conversion.

This is not tidiness. A reduced-precision run that computes its seed in `Float64`
internally measures nothing about reduced precision — and
[`OGA1dNormalEquations`](@ref) is exactly that failure mode, preserved deliberately as the
baseline to measure against. Once the seed is a `Float64` island, every statement of the
form "`Float16` converges here" is a statement about the island, not about `Float16`.

## The rules

- **Every array on the path is an `Array{T}`** — the atom matrix, the design matrix, the
  weights, the target, the residual, the coefficients, and every fit's internal factors.
  Built with `zeros(T, …)` / `ones(T, …)`, never from untyped literals.
- **No bare float literals.** Constants materialise through the argument or the type:
  `one(T)`, `zero(T)`, `T(C) * eps(T)`, `oftype(x, 0.044715)`, `sqrt(eps(T))`.
- **Reductions are spelled `sqrt(sum(abs2, ·))`, not `norm`** — the generic `norm` fallback
  rescales through `float`, which is a promotion waiting to happen.
- **`simpson_quadrature(nstages, T)` is always passed `T`.** Its signature defaults to
  `Float64`, and that default is precisely how the legacy path acquired its island.
- **The rank-revealing fits stay generic.** [`PivotedQR`](@ref) and
  [`TruncatedSVD`](@ref) are hand-rolled so that they run *at* `T`; widening to `Float32` to
  borrow LAPACK would reintroduce the island in miniature and is explicitly rejected.
- **Diagnostics are quarantined.** Condition numbers, smallest singular values and the
  study's comparable fit errors are computed in `Float64` for *reporting only*, after the
  fit returns. Nothing derived from them may re-enter the fit.

### Three documented exceptions

Each computes a *scalar or a grid coordinate* in `Float64` and converts once — never a
matrix, and never a solve.

1. [`NonlinearIntegrators.bias_grid`](@ref) and
   [`NonlinearIntegrators.weight_grid`](@ref) generate coordinates from an integer-indexed
   `Float64` range and cast to `T`, so a large `dict_amount` cannot overflow the step to
   zero. (Computed at `Float16`, `lo:(hi-lo)/n:hi` throws `ArgumentError: range step cannot
   be zero` for `n = 70000`, since `Float16(70000) == Inf`.)
2. The regularization ladder is formed as `T(2.0^k * sqrt(Float64(eps(T))))`, because
   `T(2)^k * sqrt(eps(T))` overflows to `Inf` for `Float16` well inside the ladder.
3. [`OGA1dNormalEquations`](@ref), the baseline being measured against.

## Enforcement

Two mechanisms, because prose in a docstring does not catch a regression.

**A runtime check on the activation.** [`oga_fit`](@ref) calls
[`NonlinearIntegrators.oga_check_precision`](@ref) once per fit and throws if `σ(::T)` is
not a `T`:

```julia
julia> oga_fit(OGA1d(), x -> max(0.0, x)^3, nodes, w, y, 4; bias_interval = [-π, π], dict_amount = 400)
ERROR: ArgumentError: activation returned Float64 for a Float16 argument, so the OGA seed
would not run at the working precision. Write the activation float-generically —
`max(zero(x), x)^k`, `oftype(x, c)` — rather than with bare Float64 literals.
```

`max(0.0, x)^k` instead of `max(zero(x), x)^k` promotes every evaluation to `Float64`. It
costs one scalar call per fit to rule out, and the failure it catches is otherwise visible
only as suspiciously *good* half-precision accuracy — the worst kind of bug, because it
looks like a result.

Note the activation field of `OneLayerNetwork_GML` is untyped, so the activation is boxed as
`Any` and a `Float64`-returning one would not be caught by inference. Hence a value-level
check.

**Value-level assertions in the test suite.** `test/unit/oga_kernels.jl` asserts
`eltype === T` on every array [`oga_fit`](@ref) returns, for every dictionary × selection ×
fit combination, at `Float16`, `Float32` and `Float64` — with `@inferred` on top to catch the
type instability that would let a promotion through. The existing `assert_no_upcast` helper
covers the end-to-end path (`eltype(q[end]) == T` on the final state), and the studies record
a run whose final state has left the working precision as its own `upcast` status rather than
folding it into "converged".

## `Float16`: squares overflow long before values do

The half-precision ceiling is 65504, so a quantity of magnitude above about 256 cannot be
squared, and two such quantities cannot be multiplied after squaring. That is easy to reach
here: a `ReLU³` atom over `bias_interval = [-π, π]` has norm ≈ 43, which squares to 1874, and
two of those multiply to 3.5 million.

Two consequences shaped the implementation.

**It broke the Jacobi SVD silently.** The convergence test compared
``\lvert\beta\rvert`` against ``\varepsilon(T)\sqrt{\alpha\gamma}``, a product of two squared
column norms. Once it overflowed, the threshold became `Inf`, every column pair tested as
"already orthogonal", and [`NonlinearIntegrators.jacobi_svd`](@ref) returned the **unrotated**
matrix — wrong singular values, `‖UᵀU − I‖ = 1.27`, and no error raised. The fix is to compare
against ``\sqrt{\alpha}\sqrt{\gamma}``, which never forms the product.

**The dictionary is rescaled.** [`oga_fit`](@ref) multiplies the whole dictionary by a single
**power of two** so the largest atom has norm ≈ 1, which keeps squared quantities near 1
instead of near ``\lVert\cdot\rVert_{\max}^2`` throughout the factorisations. Three properties
make this safe rather than a perturbation:

- a power of two is **exact** in binary floating point — a pure exponent shift, no rounding —
  so `Float64` and `Float32` atom selection is bit-for-bit unchanged;
- *every* row is scaled by the same factor, so even the non-scale-invariant
  [`RawProjection`](@ref) ranks candidates identically;
- the residual needs no correction, since ``(s\Psi)^{\top}(c/s) = \Psi^{\top}c``; only the
  final coefficients are unscaled, by one multiplication.

The reciprocal itself must be representable: for a largest norm down near the subnormal range
`ldexp(one(Float16), 20)` already overflows, so the scaling falls back to 1 rather than
multiplying the dictionary by `Inf`.

## The tolerances have to scale too

The same argument that motivates the ``\sqrt{\varepsilon(T)}`` regularization ladder applies
to the solver's residual tolerance, and getting it wrong invalidates a whole precision rather
than degrading it.

The integrator default is
``f_{\text{abstol}} = \max(8, \texttt{solversize})\,\varepsilon(\texttt{datatype(problem)})``
— scaled to the working precision, and merged with any options the caller passes. Both
properties are required of the dependency (`[compat]` pins GeometricIntegratorsBase 0.5),
because an absolute tolerance that does not scale with `eps(T)` is simply unreachable in
reduced precision: at `Float32` (``\varepsilon \approx 1.2\times10^{-7}``) or `Float16`
(``\approx 9.8\times10^{-4}``) the run then sits at its residual floor and burns the entire
iteration budget while parked on the right answer. Measured that way, `ReLU³` at `Float32`
reports 1000 iterations at *every* regularization factor with an accuracy of
``1.8\times10^{-7}``; read as non-convergence — which is what a naive status check does —
that makes a whole precision column an artefact of the tolerance rather than a fact about
the seed.

The studies pin an explicit ``f_{\text{abstol}} = 256\,\varepsilon(T)`` (`oga_f_abstol` in
`scripts/oga_activations.jl`) on top of that, because the default scales with `solversize`,
which varies with `S` across the sweep: pinning keeps cases of different network width
comparable.

## Not yet device-ready

Precision genericity was the prerequisite for GPU portability — `Float32` is the device-native
type, and a `Float64` island in the seed would have forced a host round-trip every time step —
but the subsystem is not portable yet.

What is already in the right shape: the dominant per-step operations are the selection scan and
the coherence guard, each one `mul!` against the dictionary plus an `argmax`, i.e. a matrix–vector
product and a reduction. Those map to a device directly. [`IncrementalQR`](@ref) also batches
cleanly over the solution components, since each component's factorisation is independent.

What blocks it: building the dictionary's design matrix and placing the selected neurons are
still scalar loops, which would force scalar indexing on a device array. Getting there needs the
dictionary build expressed as a broadcast or a kernel.

One trade-off to expect when it happens. The hand-rolled
[`NonlinearIntegrators.pivoted_qr_lstsq`](@ref) and
[`NonlinearIntegrators.jacobi_svd`](@ref) exist because LAPACK has no `Float16` path; on a
device at `Float32` the vendor libraries are available instead and would be preferable —
unpivoted QR through cuSOLVER's `geqrf`, and a Jacobi SVD through `gesvdj`. Note that
cuSOLVER's *pivoted* QR support is thinner than its unpivoted support, so on a device
[`IncrementalQR`](@ref) or [`TruncatedSVD`](@ref) is the more natural choice than
[`PivotedQR`](@ref). None of that changes the fit's cost, which is negligible either way (see
the note at the top of the [Algorithms](@ref) page).
