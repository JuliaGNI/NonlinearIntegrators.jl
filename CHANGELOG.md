# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **The Orthogonal Greedy Algorithm (OGA) initial guess is now precision-generic.**
  Previously the OGA seed that warm-starts the Newton solve in the network
  integrators was assembled in `Float64` regardless of the solver's working type,
  because the least-squares step used the normal equations `Φ diag(w) Φᵀ`, whose
  condition number is `κ(Φ)²` and which becomes rank-deficient in reduced
  precision. The seed is now built entirely at the working type
  `T = eltype(nlsolution(int))`, so the whole path (dictionary construction,
  greedy selection, least-squares fit) is GPU-portable and consistent with the
  rest of the solver. This affects `NonLinear_OneLayer_GML`, `Hardcode_int`,
  `Time_reversible_OneLayer`, and `Time_reversible_Hardcode_int`. See the
  "Orthogonal Greedy Algorithm initial guess" section of the documentation for
  the full analysis.
- The greedy least-squares fit now uses a **QR factorization of the `√w`-scaled
  design matrix** (conditioned on `κ(Φ)` instead of `κ(Φ)²`) instead of forming
  and solving the Gram matrix. This removes the need for the `Float64` island and
  lets the fit run at `Float32`/`Float16`.

### Added

- `OGA1d_Legacy` initial-guess method: the previous Float64 / normal-equations OGA
  algorithm, kept as a selectable alternative to the default `OGA1d` for
  `NonLinear_OneLayer_GML`. Select it with
  `NonLinear_OneLayer_GML(...; initial_guess_method = OGA1d_Legacy())`.
- `benchmark/oga_comparison.jl` (with its own `Project.toml` and `README.md`): an
  end-to-end comparison of `OGA1d` vs `OGA1d_Legacy` across problems of increasing
  complexity (time-step length and number of network neurons) at Float64/Float32/
  Float16.
- Shared OGA numerical helpers in `src/network_integrators/utilities.jl`:
  - `weighted_lstsq(Φ, w, y)` — quadrature-weighted least squares via QR on the
    `√w`-scaled design matrix, with a Tikhonov-ridged fallback that only engages
    when the plain solve returns a non-finite result (the genuinely
    rank-deficient `Float16` case).
  - `oga_norm_floor(T, ref) = sqrt(eps(T)) · ref` — precision-scaled floor for
    the dictionary-normalization guard.
  - `oga_tikhonov(G; C = 100) = C · eps(T) · tr(G) / n` — precision-scaled
    Tikhonov floor (used as the ridge in the `weighted_lstsq` fallback).
  - `bias_grid(lo, hi, n, T)` — index-based construction of the bias grid.
- A coherence guard in the greedy selection that blocks atoms whose
  quadrature-weighted L² coherence with an already-selected atom exceeds
  `1 - sqrt(eps(T))`. It is inert at `Float64`/`Float32` and only bites at
  `Float16`, where it keeps the selected neurons linearly independent.
- Documentation section describing the findings, the reformulated algorithm and
  its references, with a self-contained didactic `Float16` example.

### Fixed

- Replaced the hard-coded regularization/guard constants that were silently
  ineffective in reduced precision:
  - the `1e-12` dictionary-norm guard (which sat below `eps(Float32)` and so
    never fired) is now `oga_norm_floor(T, …)`;
  - the `Gk + 1e-12·I` and `Gk + 1e-14·I` Tikhonov ridges (which round away
    entirely below `eps(Float32)`) are replaced by the precision-scaled ridge in
    `weighted_lstsq`.
- The bias grid is built from an integer-indexed range cast to `T`, avoiding the
  `Float16` "`range step cannot be zero`" trap that occurred when a large
  `dict_amount` overflowed `T(dict_amount)` to `Inf`.
- Removed a stray `global xk_low` from the OGA fit in
  `Time_reversible_Hardcode_int.jl`.
