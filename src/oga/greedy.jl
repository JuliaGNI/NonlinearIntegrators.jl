# ---- The greedy loop --------------------------------------------------------
#
# One integrator-agnostic implementation. It knows nothing about
# `GeometricIntegrator`, the parameter cache, or the variational equations — it takes a
# dictionary spec, an activation, quadrature nodes and weights, and a target, and returns
# neuron parameters. That makes it directly unit-testable at any precision, and it
# replaces four hand-rolled copies of this loop that had drifted apart into four
# different guard-rail policies.
#
# Everything happens in the `√w`-scaled space: with `Â = √w ⊙ Φᵀ` and `ŷ = √w ⊙ y`, the
# plain Euclidean inner product *is* the quadrature-weighted one, so the greedy scores,
# the residual and the least-squares fit all read as ordinary linear algebra and no
# re-scaling happens inside the loop. It is also why the fit is conditioned on `κ(Φ)`
# rather than `κ(Φ)²`: there is never a Gram matrix to form.
#
# On portability: the per-step cost is dominated by two `mul!`s against the dictionary —
# the selection scan and the coherence guard — plus an `argmax`, all of which are already
# vectorised. Building the dictionary's design matrix and placing the selected neurons are
# still scalar loops, so this is precision-generic but not yet device-ready; running it on
# a GPU array would force scalar indexing there.

"""
    OGASymmetry

How one selected dictionary atom turns into network neurons. [`NoSymmetry`](@ref) is the
plain case; the two mirror variants exist for the time-reversible integrators, whose
ansatz requires neurons to come in pairs related by `t ↦ 1 - t`.
"""
abstract type OGASymmetry end

"""
    NoSymmetry()

One atom → one neuron → one design-matrix column.
"""
struct NoSymmetry <: OGASymmetry end

"""
    MirrorPairs()

One atom `(w, b)` → two neurons `(w, b)` and `(-w, w + b)` → two *independent* design
columns, so each member of the pair gets its own output weight.

The map `(w, b) ↦ (-w, w + b)` sends `σ(w t + b)` to `σ(w (1 - t) + b)`, i.e. reflects the
neuron about the midpoint of the unit time interval. Used by `Time_reversible_OneLayer`.
"""
struct MirrorPairs <: OGASymmetry end

"""
    SharedMirrorPairs()

One atom → two mirrored neurons → *one* design column, their sum, so the pair shares a
single output weight.

Sharing the weight is what actually enforces time-reversal symmetry of the ansatz (with
independent weights the pair can drift apart). Used by `Time_Reversible_Hardcode`.
"""
struct SharedMirrorPairs <: OGASymmetry end

neurons_per_atom(::NoSymmetry) = 1
neurons_per_atom(::MirrorPairs) = 2
neurons_per_atom(::SharedMirrorPairs) = 2

columns_per_atom(::NoSymmetry) = 1
columns_per_atom(::MirrorPairs) = 2
columns_per_atom(::SharedMirrorPairs) = 1

# The mirror image of an atom about t = 1/2.
_mirror(w::T, b::T) where {T} = (-w, w + b)

"""
    OGAResult{T}

What [`oga_fit`](@ref) returns.

* `W`, `b`, `c` — hidden weights, hidden biases and output weights, one entry per neuron.
* `atoms` — the dictionary indices selected, in order.
* `neurons` — how many neurons the greedy loop actually placed; a smaller number than
  requested means it ran out of atoms that add a new direction, and the remainder were
  filled with zero-weight placeholders (see `fill_unused` in [`OGA`](@ref)).
* `residual` — the weighted L² norm of the final fit residual.
* `gains` — the rank gain `‖g⊥‖` of each accepted atom: how much genuinely new direction
  it contributed. A gain collapsing towards zero across the sequence is the fingerprint of
  the reduced-precision failure this subsystem exists to remove.
* `rejected` — how many candidate atoms were skipped for adding no new direction.
"""
struct OGAResult{T}
    W::Vector{T}
    b::Vector{T}
    c::Vector{T}
    atoms::Vector{Int}
    neurons::Int
    residual::T
    gains::Vector{T}
    rejected::Int
end

"""
    oga_check_precision(σ, ::Type{T})

Assert that the activation evaluates at the working precision, i.e. that `σ(::T)` is a
`T`.

This is the one trap that silently invalidates a reduced-precision run: an activation
written `max(0.0, x)^k` instead of `max(zero(x), x)^k` promotes every evaluation to
`Float64`, so the seed is computed in double precision and the measurement says nothing
about `T`. It costs one scalar call per fit to rule out, and the failure it catches is
otherwise visible only as suspiciously good half-precision accuracy.
"""
function oga_check_precision(σ, ::Type{T}) where {T}
    v = σ(one(T) / T(3))
    v isa T && return nothing
    throw(ArgumentError(
        "activation returned $(typeof(v)) for a $T argument, so the OGA seed would not " *
        "run at the working precision. Write the activation float-generically — " *
        "`max(zero(x), x)^k`, `oftype(x, c)` — rather than with bare Float64 literals."))
end

"""
    oga_fit(oga, σ, nodes, w, y, nneurons; bias_interval, dict_amount,
            modulation = nothing, symmetry = NoSymmetry()) -> OGAResult

Greedily fit `nneurons` neurons of a one-layer network to the target `y` sampled at
`nodes`, under the quadrature weights `w`.

* `oga::`[`OGA`](@ref) — the dictionary, selection rule, fit and guard rails.
* `σ` — the activation; must evaluate at `eltype(nodes)` (checked, see
  [`oga_check_precision`](@ref)).
* `modulation` — optional per-node factor multiplying every atom, for the boundary
  ansatz `q(t) = (1-t) q̄ + t q + t(1-t) u(t)`, where the dictionary is
  `t(1-t) σ(w t + b)`. Pass the `t(1-t)` vector; `nothing` means no modulation.
* `symmetry::`[`OGASymmetry`](@ref) — how atoms map to neurons.

Runs entirely at `T = eltype(nodes)`.
"""
function oga_fit(oga::OGA, σ, nodes::AbstractVector{T}, w::AbstractVector{T},
                 y::AbstractVector{T}, nneurons::Int;
                 bias_interval, dict_amount::Integer,
                 modulation::Union{Nothing,AbstractVector{T}} = nothing,
                 symmetry::OGASymmetry = NoSymmetry()) where {T}
    oga_check_precision(σ, T)
    M = length(nodes)
    @assert length(w) == M && length(y) == M

    mod = modulation === nothing ? ones(T, M) : modulation
    sw  = sqrt.(w)                      # quadrature weights are positive ⇒ real sqrt
    ŷ   = sw .* y

    A = oga_atoms(oga.dictionary, bias_interval, dict_amount, T)
    natoms = size(A, 1)

    # Scoring rows of the dictionary, `√w`-scaled. One row per atom; for
    # `SharedMirrorPairs` the row is already the summed pair, since that is the column the
    # fit will use.
    Ψ = Matrix{T}(undef, natoms, M)
    g = Vector{T}(undef, M)
    @inbounds for i in 1:natoms
        _score_column!(g, σ, A[i, 1], A[i, 2], nodes, mod, sw, symmetry)
        for j in 1:M
            Ψ[i, j] = g[j]
        end
    end

    rownorms = T[sqrt(sum(abs2, view(Ψ, i, :))) for i in 1:natoms]
    finite_max = zero(T)
    @inbounds for n in rownorms
        isfinite(n) && n > finite_max && (finite_max = n)
    end

    # Rescale the whole dictionary by a single *power of two* so the largest atom has norm
    # ≈ 1. Squared norms then stay near 1 instead of near `finite_max²`, which at `Float16`
    # is the difference between arithmetic that works and arithmetic that overflows — a
    # `ReLU³` atom of norm 43 already squares to 1874, and two of those multiply past the
    # 65504 ceiling inside the factorisations.
    #
    # A power of two is exact in binary floating point — a pure exponent shift, no rounding
    # — so this leaves the `Float64`/`Float32` atom selection bit-for-bit unchanged, and
    # because *every* row is scaled by the same factor, even the non-scale-invariant
    # `RawProjection` ranks candidates identically. The coefficients are unscaled at the
    # end; the residual needs no correction, since `(sΨ)ᵀ(x/s) = Ψᵀx`.
    #
    # The reciprocal itself has to be representable: for a `finite_max` down near the
    # subnormal range, `ldexp(one(Float16), 20)` already overflows, and scaling by `Inf`
    # would destroy the dictionary rather than condition it. Fall back to no scaling.
    scale = (finite_max > zero(T) && isfinite(finite_max)) ?
        ldexp(one(T), -exponent(finite_max)) : one(T)
    (isfinite(scale) && scale > zero(T)) || (scale = one(T))
    if scale != one(T)
        Ψ .*= scale
        rownorms .*= scale
        finite_max *= scale
        # Fold the factor into `sw` so every column built later in the loop — the accepted
        # atoms' fit columns and the off-grid refinement's candidates — is scaled the same
        # way as `Ψ`. `ŷ` keeps the *unscaled* weights, which is what makes the unscaling of
        # the coefficients a single multiplication at the end.
        sw = sw .* scale
    end

    nfloor = oga.norm_guard ? oga_norm_floor(T, finite_max) : zero(T)
    coherence_cap = one(T) - sqrt(eps(T))

    nsteps  = nneurons ÷ neurons_per_atom(symmetry)
    percol  = columns_per_atom(symmetry)
    maxcols = nsteps * percol

    qr   = IncrementalQRState{T}(M, maxcols)
    Âsel = zeros(T, M, maxcols)
    r̂    = copy(ŷ)
    score   = Vector{T}(undef, natoms)
    blocked = falses(natoms)
    # `Ψ Q` scratch, only needed by `OrthogonalProjection`; the other rules ignore it, so
    # they get a zero-column matrix instead of a dictionary-sized allocation.
    proj = oga.selection isa OrthogonalProjection ? Matrix{T}(undef, natoms, maxcols) :
                                                    Matrix{T}(undef, natoms, 0)

    W = zeros(T, nneurons)
    B = zeros(T, nneurons)
    c = zeros(T, nneurons)
    atoms = Int[]
    gains = T[]
    gcand = Vector{T}(undef, M)      # off-grid refinement scratch, reused

    # Only `OrthogonalProjection` imposes a *relative* rank-gain floor. The other rules
    # reject a candidate only when it contributes exactly nothing, which is what the
    # pre-refactor implementations would have solved through into garbage — so the
    # well-conditioned atom sequence they produce is unchanged.
    gainfloor = oga.selection isa OrthogonalProjection ?
        _min_gain(oga.selection.min_gain, T) : zero(T)

    step = 0
    rejected = 0
    ncols = 0
    # Each rejected candidate is blocked, so the loop cannot revisit it; the cap is a
    # backstop against a dictionary in which nearly everything is dependent.
    maxattempts = nsteps + 8 * nsteps + 8

    for _ in 1:maxattempts
        step < nsteps || break

        oga_scores!(score, oga.selection, Ψ, rownorms, nfloor, r̂, qr, proj)
        @inbounds for i in eachindex(score)
            blocked[i] && (score[i] = -one(T))
        end
        best = argmax(score)
        score[best] < zero(T) && break          # no usable atom left at this precision

        wa, ba = A[best, 1], A[best, 2]
        wa, ba = oga_refine(oga.dictionary,
                            (ww, bb) -> _candidate_score(oga.selection, σ, ww, bb, nodes,
                                                         mod, sw, symmetry, r̂, qr, gcand,
                                                         nfloor, gainfloor),
                            wa, ba)

        # Tentatively append this atom's columns; roll the factorisation back if any of
        # them adds no new direction.
        k0 = qr.k
        ok = true
        gain = zero(T)
        for col in 1:percol
            _fit_column!(g, σ, wa, ba, nodes, mod, sw, symmetry, col)
            ρ = oga_qr_append!(qr, g; min_gain = gainfloor)
            if qr.k == k0 + col
                Âsel[:, ncols+col] .= g
                gain = col == 1 ? ρ : min(gain, ρ)
            else
                ok = false
                break
            end
        end

        if !ok
            qr.k = k0
            blocked[best] = true
            rejected += 1
            continue
        end

        step += 1
        ncols += percol
        push!(atoms, best)
        push!(gains, gain)
        _place_neurons!(W, B, step, wa, ba, symmetry)

        x = oga_solve(oga.fit, view(Âsel, :, 1:ncols), ŷ, qr)
        _place_coefficients!(c, x, step, symmetry)

        r̂ .= ŷ
        mul!(r̂, view(Âsel, :, 1:ncols), x, -one(T), one(T))

        blocked[best] = true
        if oga.coherence
            _block_coherent!(blocked, Ψ, rownorms, best, coherence_cap, score)
        end
    end

    # Undo the dictionary scaling: the coefficients were fit against `scale · Ψ`.
    scale != one(T) && (c .*= scale)

    nplaced = step * neurons_per_atom(symmetry)
    oga.fill_unused && nplaced < nneurons &&
        _fill_unused!(W, B, A, blocked, nplaced, nneurons, symmetry)

    return OGAResult{T}(W, B, c, atoms, nplaced, sqrt(sum(abs2, r̂)), gains, rejected)
end

# ---- atom → columns ---------------------------------------------------------

# The `√w`-scaled row used for *scoring* a candidate atom. For the shared-pair symmetry
# this is the summed pair, because that is also the column the fit sees.
function _score_column!(g::AbstractVector{T}, σ, w::T, b::T, nodes::AbstractVector{T},
                        mod::AbstractVector{T}, sw::AbstractVector{T},
                        sym::OGASymmetry) where {T}
    if sym isa SharedMirrorPairs
        wm, bm = _mirror(w, b)
        @inbounds for j in eachindex(nodes)
            t = nodes[j]
            g[j] = (σ(w * t + b) + σ(wm * t + bm)) * mod[j] * sw[j]
        end
    else
        @inbounds for j in eachindex(nodes)
            g[j] = σ(w * nodes[j] + b) * mod[j] * sw[j]
        end
    end
    return g
end

# The `√w`-scaled column number `col` contributed by an accepted atom. Only
# `MirrorPairs` has a second column, the mirrored neuron with its own output weight.
function _fit_column!(g::AbstractVector{T}, σ, w::T, b::T, nodes::AbstractVector{T},
                      mod::AbstractVector{T}, sw::AbstractVector{T},
                      sym::OGASymmetry, col::Int) where {T}
    if sym isa MirrorPairs && col == 2
        wm, bm = _mirror(w, b)
        @inbounds for j in eachindex(nodes)
            g[j] = σ(wm * nodes[j] + bm) * mod[j] * sw[j]
        end
        return g
    end
    return _score_column!(g, σ, w, b, nodes, mod, sw, sym)
end

function _place_neurons!(W::AbstractVector{T}, B::AbstractVector{T}, step::Int,
                         w::T, b::T, sym::OGASymmetry) where {T}
    if sym isa NoSymmetry
        W[step] = w
        B[step] = b
    else
        wm, bm = _mirror(w, b)
        W[2step-1] = w
        B[2step-1] = b
        W[2step]   = wm
        B[2step]   = bm
    end
    return nothing
end

function _place_coefficients!(c::AbstractVector{T}, x::AbstractVector{T}, step::Int,
                              sym::OGASymmetry) where {T}
    if sym isa SharedMirrorPairs
        # One coefficient per atom, copied to both members of the pair — the constraint
        # that makes the ansatz time-reversible.
        for j in 1:step
            c[2j-1] = x[j]
            c[2j]   = x[j]
        end
    else
        for i in eachindex(x)
            c[i] = x[i]
        end
    end
    return nothing
end

# ---- guard rails ------------------------------------------------------------

# Block atoms whose weighted-L² coherence with the just-selected atom exceeds the cap.
# The coherence is computed on the fly as `⟨gᵢ, g_best⟩ / (‖gᵢ‖ ‖g_best‖)` rather than
# from a normalised copy of the dictionary, which saves a second dictionary-sized array.
# `score` is reused as scratch — it is recomputed from scratch next iteration.
function _block_coherent!(blocked::BitVector, Ψ::AbstractMatrix{T},
                          rownorms::AbstractVector{T}, best::Int, cap::T,
                          scratch::AbstractVector{T}) where {T}
    nb = rownorms[best]
    (nb == zero(T) || !isfinite(nb)) && return nothing
    mul!(scratch, Ψ, view(Ψ, best, :))
    @inbounds for i in eachindex(scratch)
        n = rownorms[i]
        (n == zero(T) || !isfinite(n)) && continue
        abs(scratch[i]) > cap * n * nb && (blocked[i] = true)
    end
    return nothing
end

# Give the neurons the greedy loop could not place distinct, well-separated `(w, b)` and a
# zero output weight. Leaving them at `(0, 0)` instead would make them identical, turning
# a rank-deficient seed into a rank-deficient Newton Jacobian — the failure moves rather
# than goes away.
function _fill_unused!(W::AbstractVector{T}, B::AbstractVector{T}, A::AbstractMatrix{T},
                       blocked::BitVector, nplaced::Int, nneurons::Int,
                       sym::OGASymmetry) where {T}
    natoms = size(A, 1)
    candidates = findall(!, blocked)
    isempty(candidates) && (candidates = collect(1:natoms))

    missing_neurons = nneurons - nplaced
    nfill = sym isa NoSymmetry ? missing_neurons : missing_neurons ÷ 2
    # Spread the placeholders across the dictionary rather than taking neighbours, so they
    # stay mutually distinct.
    stride = max(1, length(candidates) ÷ max(1, nfill))

    for f in 1:nfill
        idx = candidates[min(length(candidates), 1 + (f - 1) * stride)]
        _place_neurons!(W, B, nplaced ÷ neurons_per_atom(sym) + f, A[idx, 1], A[idx, 2], sym)
    end
    return nothing
end

# ---- off-grid refinement ----------------------------------------------------

# Score a single candidate atom `(w, b)` that is not in the dictionary, using the same
# criterion as the selection rule so the polish optimises what the greedy step ranks.
function _candidate_score(rule::OGASelection, σ, w::T, b::T, nodes::AbstractVector{T},
                          mod::AbstractVector{T}, sw::AbstractVector{T},
                          sym::OGASymmetry, r̂::AbstractVector{T},
                          qr::IncrementalQRState{T}, g::AbstractVector{T},
                          nfloor::T, gainfloor::T) where {T}
    _score_column!(g, σ, w, b, nodes, mod, sw, sym)
    n = sqrt(sum(abs2, g))
    p = abs(dot(g, r̂))
    _usable(p, n, nfloor) || return T(-Inf)

    # Note the deliberate asymmetry: refinement normalises even under `RawProjection`.
    # The raw inner product is a *ranking heuristic* among atoms of comparable norm, not an
    # objective — maximised continuously over `(w, b)` it rewards growing the atom rather
    # than fitting the residual, and the search would drift towards large `|w|` while the
    # fit got worse. The normalised score is the actual residual-reduction objective, so
    # that is what gets polished.
    denom = n
    if rule isa OrthogonalProjection && qr.k > 0
        explained = sum(abs2, view(qr.Q, :, 1:qr.k)' * g)
        residual² = n * n - explained
        denom = residual² > zero(T) ? sqrt(residual²) : zero(T)
        denom > gainfloor * n || return T(-Inf)
    end
    return p / denom
end
