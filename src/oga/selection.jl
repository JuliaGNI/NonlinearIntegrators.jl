# ---- OGA selection rules ----------------------------------------------------
#
# The greedy step picks the dictionary atom most correlated with the current residual.
# *How* correlation is measured decides which atoms get chosen and therefore, through
# the seed, which basin the Newton solve lands in — so the rules below are genuinely
# different algorithms, not cosmetic variations.
#
# All scores are computed in the `√w`-scaled space, where the Euclidean inner product
# is the quadrature-weighted one. A score of `-1` marks an atom as unusable (blocked by
# the coherence guard, degenerate, or contributing no new direction); scores are
# otherwise non-negative, so `argmax` skips them.

"""
    OGASelection

How the greedy step scores candidate atoms against the current residual. One of
[`RawProjection`](@ref), [`NormalizedProjection`](@ref) or
[`OrthogonalProjection`](@ref).
"""
abstract type OGASelection end

"""
    RawProjection()

Score by the bare weighted inner product `|⟨r, g⟩_w|`.

The default, and what the `Float64`/`Float32` regression tests pin. Note it is *not*
scale-invariant — an atom with a large norm outranks a
better-aligned small one — which is harmless for the `{±1} × (bias grid)` dictionary,
whose atoms all have comparable norms, and wrong for a 2-D `(w, b)` dictionary, whose
atoms do not.
"""
struct RawProjection <: OGASelection end

"""
    NormalizedProjection()

Score by the cosine-like ratio `|⟨r, g⟩_w| / ‖g‖_w`.

This is the textbook greedy criterion: it measures how much of the residual the atom
*explains*, independently of the atom's amplitude, which the output weight absorbs
anyway. Scale invariance makes it mandatory for [`WeightBiasGrid2d`](@ref) and
[`AngularGrid`](@ref), where atoms differ in norm by orders of magnitude.
"""
struct NormalizedProjection <: OGASelection end

"""
    OrthogonalProjection(; min_gain = nothing)

Score by the projection onto the part of the atom *orthogonal to the already selected
ones*, `|⟨r, g⟩_w| / ‖g⊥‖_w`, and refuse any atom whose orthogonal part has collapsed
(`‖g⊥‖ < min_gain · ‖g‖`); `min_gain = nothing` uses `sqrt(eps(T))`.

This is what makes the algorithm *orthogonal* greedy rather than matching pursuit, and
it is the direct fix for the observed reduced-precision failure. The residual is already
orthogonal to the selected span, so the numerator is unchanged from
[`NormalizedProjection`](@ref) — but the denominator penalises an atom that mostly
duplicates what is already there, and the `min_gain` floor rules it out entirely. An atom
that adds no new direction is therefore never selected, which is the condition that
otherwise surfaces downstream as `SingularException: zero pivot found at index 3` out of
four neurons.

Costs one dictionary-sized matrix product against the maintained `Q` per step, the same
order as the score itself.
"""
struct OrthogonalProjection{G} <: OGASelection
    min_gain::G
    OrthogonalProjection(; min_gain = nothing) = new{typeof(min_gain)}(min_gain)
end

_min_gain(::Nothing, ::Type{T}) where {T} = sqrt(eps(T))
_min_gain(g, ::Type{T}) where {T} = T(g)

# An atom is usable if its projection is finite and its norm is finite and above the
# floor. The `isfinite(n)` half is correctness, not policy: at `Float16` a high `ReLUᵏ`
# power over a wide bias interval overflows `σ(b)ᵏ`, giving an `Inf` norm that would sail
# past a bare `n > nfloor` test and then divide to `NaN`.
_usable(s::T, n::T, nfloor::T) where {T} = isfinite(s) && isfinite(n) && n > nfloor

"""
    oga_scores!(score, rule, Ψ, rownorms, nfloor, r̂, qr, proj)

Fill `score` with the selection score of every dictionary atom.

* `Ψ` — (natoms × nnodes) `√w`-scaled dictionary, one atom per row.
* `rownorms` — the `√w`-weighted L² norm of each atom.
* `nfloor` — norm floor below which an atom counts as numerically zero.
* `r̂` — the `√w`-scaled current residual.
* `qr` — the factorisation of the selected columns (used by
  [`OrthogonalProjection`](@ref)).
* `proj` — (natoms × maxcols) scratch for `Ψ Q`.

Returns `score`. Unusable atoms are set to `-one(T)`.
"""
function oga_scores!(score::AbstractVector{T}, ::RawProjection, Ψ::AbstractMatrix{T},
                     rownorms::AbstractVector{T}, nfloor::T, r̂::AbstractVector{T},
                     ::IncrementalQRState{T}, ::AbstractMatrix{T}) where {T}
    mul!(score, Ψ, r̂)
    @inbounds for i in eachindex(score)
        s = abs(score[i])
        score[i] = _usable(s, rownorms[i], nfloor) ? s : -one(T)
    end
    return score
end

function oga_scores!(score::AbstractVector{T}, ::NormalizedProjection, Ψ::AbstractMatrix{T},
                     rownorms::AbstractVector{T}, nfloor::T, r̂::AbstractVector{T},
                     ::IncrementalQRState{T}, ::AbstractMatrix{T}) where {T}
    mul!(score, Ψ, r̂)
    @inbounds for i in eachindex(score)
        n = rownorms[i]
        s = abs(score[i])
        score[i] = _usable(s, n, nfloor) ? s / n : -one(T)
    end
    return score
end

function oga_scores!(score::AbstractVector{T}, rule::OrthogonalProjection, Ψ::AbstractMatrix{T},
                     rownorms::AbstractVector{T}, nfloor::T, r̂::AbstractVector{T},
                     qr::IncrementalQRState{T}, proj::AbstractMatrix{T}) where {T}
    mul!(score, Ψ, r̂)
    k = qr.k
    gainfloor = _min_gain(rule.min_gain, T)

    if k > 0
        # `proj[i, j] = ⟨gᵢ, qⱼ⟩`, so `‖gᵢ⊥‖² = ‖gᵢ‖² − Σⱼ proj[i, j]²`.
        P = view(proj, :, 1:k)
        mul!(P, Ψ, view(qr.Q, :, 1:k))
    end

    @inbounds for i in eachindex(score)
        n = rownorms[i]
        s = abs(score[i])
        if !_usable(s, n, nfloor)
            score[i] = -one(T)
            continue
        end
        explained = zero(T)
        for j in 1:k
            explained += proj[i, j] * proj[i, j]
        end
        residual² = n * n - explained
        # Subtracting in `T` can go slightly negative for an atom already fully
        # explained; that atom is exactly the one to reject, so clamp rather than error.
        gain = residual² > zero(T) ? sqrt(residual²) : zero(T)
        score[i] = gain > gainfloor * n ? s / gain : -one(T)
    end
    return score
end
