# Benchmarks

## OGA initial-guess comparison

`oga_comparison.jl` compares the two Orthogonal Greedy Algorithm initial-guess
variants of `NonLinear_OneLayer_GML`:

- **`OGA1d`** (default) — the seed is assembled at the working precision `T` and the
  output weights come from a QR fit of the `√w`-scaled design matrix;
- **`OGA1d_Legacy`** — the previous algorithm: the seed is assembled in `Float64`
  (a "double-precision island") and the output weights come from the normal
  equations `Gk \ rhs`, then rounded into the working type.

Both are run end-to-end (OGA seed + Newton solve) on the harmonic oscillator over
`t ∈ [0, 1]`, sweeping problem complexity along two axes — the time-step length `dt`
(hence the number of steps `1/dt`) and the number of neurons `S` (hence the number
of network parameters) — at three working precisions (`Float64`, `Float32`,
`Float16`). The script reports, per case, the solver status, the error of the final
state against the analytic solution, and the wall-clock time (after a warm-up run).

The background: the legacy Gram/normal-equations solve has condition number
`κ(Φ)²`, which forces the `Float64` island and becomes rank-deficient in reduced
precision; the QR reformulation is conditioned on `κ(Φ)`. See the "Orthogonal Greedy
Algorithm" section of the package documentation for the full analysis.

### Running

```
julia --project=benchmark -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=benchmark benchmark/oga_comparison.jl
```

### What to expect

- **`Float64` / `Float32`** — where both algorithms converge they reach essentially
  the same accuracy (the seed is only a warm start; the Newton solve sets the final
  error). This confirms the reformulation is a no-regression change.
- **Higher parameter counts** (`S = 8`) — the legacy seed already tends to go
  `singular` (near-duplicate neurons make the Gram matrix / Newton Jacobian
  rank-deficient), while the QR seed still solves.
- **`Float16`** — the legacy variant fails (`singular` / `diverged`) across the
  board, whereas the QR seed stays finite and lets the solve proceed for the smaller
  networks.
