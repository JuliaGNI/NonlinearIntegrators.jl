# ---- OGA fit strategies -----------------------------------------------------
#
# The greedy loop refits the output weights of all selected atoms after every
# selection. How that linear least-squares problem is solved is the single biggest
# lever on reduced-precision robustness, because the original formulation solved it
# on the normal equations `G = Φ diag(w) Φᵀ`, and forming `G` squares the condition
# number: `κ(G) = κ(Φ)²`. A matrix is solvable while `κ ≲ 1/eps(T)`, so squaring κ
# halves the usable digits — which is exactly the `Float64`→`Float32` gap that made
# the seed's `Float64` island look necessary.
#
# Each fit below receives the *already* `√w`-scaled design matrix `Â` (nnodes × k) and
# target `ŷ`, so all of them minimise the same quadrature-weighted objective and differ
# only in the factorisation. Every one returns a finite result: the shared last resort
# is `ridged_lstsq`.

"""
    OGAFit

How the OGA refits the output weights of the selected atoms. One of
[`WeightedQR`](@ref), [`IncrementalQR`](@ref), [`PivotedQR`](@ref),
[`TruncatedSVD`](@ref) or [`NormalEquationsFit`](@ref); see each for the trade-off.
"""
abstract type OGAFit end

"""
    WeightedQR()

QR least squares on the `√w`-scaled design matrix (see [`weighted_lstsq`](@ref)):
conditioned on `κ(Φ)` rather than `κ(Φ)²`, with no Gram matrix and no ridge unless the
plain solve comes back non-finite.

The default, and the fit whose behaviour the `Float64`/`Float32` regression tests pin —
it re-solves from scratch at each greedy step, matching the pre-refactor arithmetic
exactly.
"""
struct WeightedQR <: OGAFit end

"""
    IncrementalQR()

Reuse the incrementally maintained QR factorisation (see [`IncrementalQRState`](@ref)):
one triangular solve per greedy step instead of a fresh `k × k` factorisation.

Numerically equivalent to [`WeightedQR`](@ref) up to rounding, but `O(k · nnodes)`
rather than `O(k² · nnodes)` per step, and it is the fit that shares its `Q` with
[`OrthogonalProjection`](@ref) — pairing the two is the "textbook" efficient *and*
stable OGA.
"""
struct IncrementalQR <: OGAFit end

"""
    PivotedQR(; rtol = nothing)

Rank-revealing Householder QR with column pivoting, truncated below
`rtol · (largest pivot)`; `rtol = nothing` uses `eps(T) · max(4, k)` with
`k = min(nnodes, ncols)` — see the note above `_rtol` for why not `sqrt(eps(T))`.

Unlike [`WeightedQR`](@ref), a numerically dependent selected atom is *detected* and
given a zero coefficient rather than solved through. Hand-rolled because
`qr(Â, ColumnNorm())` is LAPACK-only and so does not exist at `Float16` — the precision
that needs it.
"""
struct PivotedQR{R} <: OGAFit
    rtol::R
    PivotedQR(; rtol = nothing) = new{typeof(rtol)}(rtol)
end

"""
    TruncatedSVD(; rtol = nothing)

Minimum-norm solve through a truncated pseudo-inverse, dropping singular directions
with `σ < rtol · σ_max`; `rtol = nothing` uses `eps(T) · max(4, k)` with
`k = min(nnodes, ncols)` — see the note above `_rtol` for why not `sqrt(eps(T))`.

The most robust of the fits — a rank-deficient selected set gives a bounded solution
instead of amplified rounding noise — and the one to reach for if a single variant must
work unchanged across every precision. Uses the generic one-sided Jacobi
[`jacobi_svd`](@ref), since `svd` is likewise LAPACK-only.
"""
struct TruncatedSVD{R} <: OGAFit
    rtol::R
    TruncatedSVD(; rtol = nothing) = new{typeof(rtol)}(rtol)
end

"""
    NormalEquationsFit(; ridge = true, island = false)

Solve the normal equations `G x = Φ diag(w) y` with `G = Φ diag(w) Φᵀ`, optionally with
the precision-scaled Tikhonov ridge of [`oga_tikhonov`](@ref) (`ridge = true`) and
optionally in `Float64` regardless of the working precision (`island = true`).

This is the *baseline*, not a recommendation: forming `G` squares the condition number.
It exists so that "island vs. working precision" and "ridge vs. no ridge" are two knobs
on one code path that can be ablated in a benchmark, rather than differences buried in
forked implementations. `NormalEquationsFit(ridge = false, island = true)` reproduces
the arithmetic of [`OGA1dNormalEquations`](@ref), the original-paper reference.

Note `island = true` deliberately violates the package's precision discipline; it is
the thing being measured against.
"""
struct NormalEquationsFit <: OGAFit
    ridge::Bool
    island::Bool
    NormalEquationsFit(; ridge = true, island = false) = new(ridge, island)
end

# Resolve a `nothing` tolerance to the precision-scaled default, following `pinv`'s
# convention of `eps(T)` times the dimension — the level at which a singular value is
# indistinguishable from accumulated rounding.
#
# Deliberately *not* `sqrt(eps(T))`: that is the scale of `OrthogonalProjection`'s
# rank-gain floor, which admits a column whose orthogonal part is exactly that fraction of
# its norm, so a `sqrt(eps)` truncation would discard the very directions the selection
# rule just decided were usable. Measured at `Float16`, the two thresholds differ by a
# factor of 20 in the resulting fit residual.
_rtol(::Nothing, ::Type{T}, k::Int) where {T} = eps(T) * max(4, k)
_rtol(r, ::Type{T}, ::Int) where {T} = T(r)

"""
    oga_solve(fit, Â, ŷ, qr) -> Vector

Refit the output weights of the currently selected atoms. `Â` (nnodes × k) is the
`√w`-scaled design matrix, `ŷ` the `√w`-scaled target, and `qr` the incrementally
maintained factorisation of `Â` (used only by [`IncrementalQR`](@ref), but kept in the
signature so every fit is interchangeable).

**Guarantees, for every fit, at every precision:** the result has one entry per column of
`Â` and every entry is finite. That is enforced here rather than in each fit, so it holds
for a new fit by construction. Where a factorisation throws on a rank-deficient design, or
returns `Inf`/`NaN` from a division by a pivot that survived truncation, the fit falls back
to the ridged solve of [`ridged_lstsq`](@ref). A seed the Newton solve can start from is
worth more than a faithful report that the fit was impossible — and the rank-deficiency
itself is already reported, through `OGAResult`'s `gains` and `rejected`.
"""
function oga_solve(fit::OGAFit, Â::AbstractMatrix{T}, ŷ::AbstractVector{T},
                   qr::IncrementalQRState{T}) where {T}
    x = try
        _oga_solve(fit, Â, ŷ, qr)
    catch e
        _is_rank_failure(e) || rethrow()
        return ridged_lstsq(Â, ŷ)
    end
    (length(x) == size(Â, 2) && all(isfinite, x)) && return x
    return ridged_lstsq(Â, ŷ)
end

_oga_solve(::WeightedQR, Â::AbstractMatrix, ŷ::AbstractVector, ::IncrementalQRState) =
    scaled_lstsq(Â, ŷ)

_oga_solve(::IncrementalQR, ::AbstractMatrix, ŷ::AbstractVector, qr::IncrementalQRState) =
    oga_qr_solve(qr, ŷ)

_oga_solve(fit::PivotedQR, Â::AbstractMatrix{T}, ŷ::AbstractVector{T}, ::IncrementalQRState) where {T} =
    pivoted_qr_lstsq(Â, ŷ, _rtol(fit.rtol, T, minimum(size(Â))))

_oga_solve(fit::TruncatedSVD, Â::AbstractMatrix{T}, ŷ::AbstractVector{T}, ::IncrementalQRState) where {T} =
    truncated_svd_lstsq(Â, ŷ, _rtol(fit.rtol, T, minimum(size(Â))))

function _oga_solve(fit::NormalEquationsFit, Â::AbstractMatrix{T}, ŷ::AbstractVector{T}, ::IncrementalQRState) where {T}
    if fit.island
        # The `Float64` island: the whole solve is widened, then rounded back into `T`.
        x = _normal_equations(Float64.(Â), Float64.(ŷ), fit.ridge)
        return T.(x)
    end
    return _normal_equations(Â, ŷ, fit.ridge)
end

function _normal_equations(Â::AbstractMatrix{T}, ŷ::AbstractVector{T}, ridge::Bool) where {T}
    G   = Â' * Â
    rhs = Â' * ŷ
    if ridge
        λ = oga_tikhonov(G)
        for i in axes(G, 1)
            G[i, i] += λ
        end
    end
    # A singular Gram matrix is the documented failure mode of this fit — and the reason the
    # greedy seed used to abort the whole time step. Let it throw: `oga_solve` catches rank
    # failures uniformly and falls back to the ridged solve.
    return G \ rhs
end
