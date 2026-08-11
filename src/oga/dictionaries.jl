# ---- OGA dictionaries -------------------------------------------------------
#
# The candidate set of neurons `(w, b)` the greedy step chooses from. Both original
# variants used the same one: fixed weights `w = ±1` crossed with a uniform grid of
# biases.
#
# That set is *complete* for a positively homogeneous activation. Since
# `σ(w x + b) = |w|ᵏ σ(sign(w) x + b/|w|)` for `ReLUᵏ`, scaling `w` only rescales the
# atom and shifts the bias — effects the output weight `c` and the bias grid already
# absorb — so the only shape information left in `w` is its sign. This is exactly the
# shallow-ReLU approximation-theory setting the ±1 dictionary is derived from.
#
# Smooth activations break that identity. ELU and GELU are not positively homogeneous:
# there `w` sets an intrinsic *length scale* (how sharply the unit transitions), a
# genuine shape parameter independent of `b` and `c`. With `|w|` pinned to 1 the
# dictionary can only place transitions of a single steepness along the bias axis, so
# the greedy seed is a poor fit and the Newton system it feeds is ill-conditioned — the
# mechanism behind the measured regression of GELU with the ReLU-theory seed.
#
# The dictionaries below therefore range from the original 1-D set to genuine 2-D
# `(w, b)` sets, plus an off-grid refinement that composes with any of them.

"""
    OGADictionary

The candidate neuron set the greedy step selects from. One of [`BiasGrid1d`](@ref),
[`WeightBiasGrid2d`](@ref), [`AngularGrid`](@ref), or any of those wrapped in
[`Refined`](@ref).
"""
abstract type OGADictionary end

"""
    BiasGrid1d()

The original dictionary: weights `w = ±1` crossed with a uniform grid of
`dict_amount + 1` biases over the method's `bias_interval`, for `2·(dict_amount + 1)`
atoms.

Complete for `ReLUᵏ` (see the discussion at the top of this file) and the default. Atom
order — the `w = -1` block first, then `w = +1` — is load-bearing: `argmax` breaks ties by
first index, so reordering changes which atoms are selected.
"""
struct BiasGrid1d <: OGADictionary end

"""
    WeightBiasGrid2d(; octaves = (-3.0, 3.0), weight_amount = 6, signed = true,
                       bias_amount = nothing)

A genuine 2-D grid over `(w, b)`: `weight_amount + 1` weight magnitudes spaced
logarithmically over `2^octaves[1] … 2^octaves[2]`, optionally sign-symmetric, crossed
with the bias grid.

The weight axis spans length scales by *ratio*, so the default covers a factor of 64
between the sharpest and gentlest transition. `bias_amount` overrides the method's
`dict_amount` on the bias axis (`nothing` keeps it), which is how the total dictionary
size is held roughly constant while the weight axis is added — the greedy step is linear
in the dictionary size.

Set `octaves = (0.0, 0.0), weight_amount = 0` to recover [`BiasGrid1d`](@ref) exactly:
the extra weight degrees of freedom are redundant for `ReLUᵏ`, so this is a strict
generalisation — neutral for the homogeneous activations and enabling for the smooth
ones. Pair it with [`NormalizedProjection`](@ref): atoms of very different `|w|` have
very different raw norms, so [`RawProjection`](@ref) would rank them by amplitude
rather than by fit.
"""
struct WeightBiasGrid2d{B} <: OGADictionary
    octaves::Tuple{Float64,Float64}
    weight_amount::Int
    signed::Bool
    bias_amount::B

    function WeightBiasGrid2d(; octaves = (-3.0, 3.0), weight_amount::Int = 6,
                                signed::Bool = true, bias_amount = nothing)
        oct = (Float64(octaves[1]), Float64(octaves[2]))
        new{typeof(bias_amount)}(oct, weight_amount, signed, bias_amount)
    end
end

"""
    AngularGrid(; radii = (1.0,), amount = nothing)

Atoms placed on rays through the origin of `(w, b)` space:
`(w, b) = r · (cos θ, sin θ)` for a uniform grid of `amount + 1` angles over `[0, 2π)`
and each radius in `radii`. `amount = nothing` uses the method's `dict_amount`.

This is the dictionary the underlying approximation theory is stated for — a grid on
the unit sphere of `ℝ^{d+1}` — and it unifies the 1-D and 2-D cases. For a homogeneous
activation the radius is redundant (it only rescales the atom), so a single radius
suffices and the set covers the same ridge directions as `{±1} × (bias grid)`, but
sampled *uniformly in atom space* rather than uniformly in bias: the ±1 grid over
`[-π, π]` clusters its resolution where `|b|` is large and the atom is nearly constant
on `t ∈ [0, 1]`, which is where it matters least. For a smooth activation the radius
*is* the length scale, and log-spaced `radii` give the second degree of freedom without
having to pick a weight interval.

The full circle is used rather than a half circle because `ReLUᵏ(x)` and `ReLUᵏ(-x)`
are different functions — the sign of `w` is real shape information (only the *scale*
is redundant).
"""
struct AngularGrid{R,A} <: OGADictionary
    radii::R
    amount::A

    AngularGrid(; radii = (1.0,), amount = nothing) =
        new{typeof(radii),typeof(amount)}(radii, amount)
end

"""
    Refined(inner; iterations = 3, shrink = 0.5)

Wrap any dictionary so that, after the greedy `argmax` picks a grid atom, the atom's
`(w, b)` are polished *off* the grid by locally maximising the same selection score.

The grid then only has to get the neighbourhood right, which decouples accuracy from
dictionary size: a few dozen atoms plus refinement can match a dictionary of hundreds of
thousands, and the greedy step is linear in the dictionary size. This is the standard
"OGA with inner optimisation".

The local search is a derivative-free compass search — evaluate the score at `(w ± h, b)`
and `(w, b ± h)`, step to the best improvement, shrink `h` by `shrink` when none
improves — repeated `iterations` times. Derivative-free on purpose: the score is only
piecewise smooth for `ReLUᵏ` (the kink crosses a quadrature node), and it keeps the
activation off the ForwardDiff path entirely, so no `Dual` tag can leak into the
working precision.

The polished objective is always the *normalised* score, even when the selection rule is
[`RawProjection`](@ref) — see the note in `_candidate_score`. Maximising the raw inner
product continuously over `(w, b)` would reward growing the atom rather than fitting the
residual.
"""
struct Refined{D<:OGADictionary} <: OGADictionary
    inner::D
    iterations::Int
    shrink::Float64

    Refined(inner::D; iterations::Int = 3, shrink = 0.5) where {D<:OGADictionary} =
        new{D}(inner, iterations, Float64(shrink))
end

"""
    oga_atoms(dict, bias_interval, dict_amount, ::Type{T}) -> Matrix{T}

Build the candidate atom matrix: one row per atom, column 1 the hidden weight `w` and
column 2 the bias `b`, in precision `T`.
"""
function oga_atoms(::BiasGrid1d, bias_interval, dict_amount::Integer, ::Type{T}) where {T}
    B = bias_grid(bias_interval[1], bias_interval[2], dict_amount, T)
    n = length(B)
    return hcat(vcat(-ones(T, n), ones(T, n)), vcat(B, B))
end

function oga_atoms(dict::WeightBiasGrid2d, bias_interval, dict_amount::Integer, ::Type{T}) where {T}
    nb = dict.bias_amount === nothing ? dict_amount : dict.bias_amount
    B = bias_grid(bias_interval[1], bias_interval[2], nb, T)
    mags = weight_grid(dict.octaves[1], dict.octaves[2], dict.weight_amount, T)
    W = dict.signed ? vcat(-mags, mags) : mags

    A = Matrix{T}(undef, length(W) * length(B), 2)
    i = 0
    for w in W, b in B
        i += 1
        A[i, 1] = w
        A[i, 2] = b
    end
    return A
end

function oga_atoms(dict::AngularGrid, bias_interval, dict_amount::Integer, ::Type{T}) where {T}
    n = dict.amount === nothing ? dict_amount : dict.amount
    # Angles from an integer-indexed `Float64` range, cast once — same safeguard as
    # `bias_grid`. The endpoint is excluded because 2π wraps onto 0.
    θ = T.(2π .* (0:n) ./ (n + 1))
    radii = T.(collect(dict.radii))

    A = Matrix{T}(undef, length(radii) * length(θ), 2)
    i = 0
    for r in radii, t in θ
        i += 1
        A[i, 1] = r * cos(t)
        A[i, 2] = r * sin(t)
    end
    return A
end

oga_atoms(dict::Refined, bias_interval, dict_amount::Integer, ::Type{T}) where {T} =
    oga_atoms(dict.inner, bias_interval, dict_amount, T)

"""
    oga_refine(dict, score, w, b) -> (w, b)

Polish a selected atom off the grid. `score(w, b)` returns the selection score of the
candidate atom (larger is better; `-Inf`/`NaN` for an invalid one). The default is a
no-op, so only [`Refined`](@ref) does any work.
"""
oga_refine(::OGADictionary, score, w::T, b::T) where {T} = (w, b)

function oga_refine(dict::Refined, score, w::T, b::T) where {T}
    best = score(w, b)
    isfinite(best) || return (w, b)
    # Start from a step comparable to the atom's own scale, so the search is
    # scale-invariant and cannot be swamped at reduced precision.
    h = (sqrt(eps(T)) + abs(w) + abs(b)) / T(8)

    for _ in 1:dict.iterations
        improved = false
        for (dw, db) in ((h, zero(T)), (-h, zero(T)), (zero(T), h), (zero(T), -h))
            s = score(w + dw, b + db)
            if isfinite(s) && s > best
                best = s
                w += dw
                b += db
                improved = true
            end
        end
        improved || (h *= T(dict.shrink))
        h > eps(T) || break
    end
    return (w, b)
end

# Labels used by the benchmark reports; kept next to the definitions so a new
# dictionary cannot be added without one.
oga_label(::BiasGrid1d) = "grid1d"
oga_label(d::WeightBiasGrid2d) = "grid2d($(d.weight_amount + 1)w)"
oga_label(d::AngularGrid) = "angular($(length(d.radii))r)"
oga_label(d::Refined) = "refined[" * oga_label(d.inner) * "]"
oga_label(::RawProjection) = "raw"
oga_label(::NormalizedProjection) = "normalized"
oga_label(::OrthogonalProjection) = "orthogonal"
oga_label(::WeightedQR) = "qr"
oga_label(::IncrementalQR) = "incqr"
oga_label(::PivotedQR) = "pivqr"
oga_label(::TruncatedSVD) = "tsvd"
oga_label(f::NormalEquationsFit) =
    "normaleq" * (f.ridge ? "+ridge" : "") * (f.island ? "+f64" : "")
