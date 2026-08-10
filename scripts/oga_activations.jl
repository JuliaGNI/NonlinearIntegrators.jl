# Activation functions for the OGA benchmarks.
#
# Every one is float-generic: constants are materialised through the argument
# (`zero`/`one`/`oftype`), never as bare `Float64` literals, so a `Float16` sweep evaluates
# at `Float16` instead of silently promoting the whole seed to double precision. `oga_fit`
# checks this at runtime (`oga_check_precision`), but getting it right here is what makes
# the reduced-precision numbers mean anything.
#
# They must also trace through two non-numeric paths: `ForwardDiff.Dual` (the Newton
# Jacobian) and `Symbolics.Num` (the `SymbolicNeuralNetwork` built when the basis is
# constructed). That is why the branches are written with `max`/`min` rather than `?:` or
# `ifelse` — a comparison on a symbolic value cannot produce a `Bool`.

# ReLUᵏ. Positively homogeneous, so `σ(w x + b) = |w|ᵏ σ(sign(w) x + b/|w|)`: the magnitude
# of `w` carries no shape information the bias grid and output weight do not already absorb.
# This is the activation the classical ±1-weight OGA dictionary is derived for.
relu_k(k::Int) = x -> max(zero(x), x)^k

# ELU (α = 1) and the tanh approximation of GELU. Neither is positively homogeneous, so
# `|w|` is a genuine length-scale parameter for them — the motivation for a 2-D dictionary.
elu(x)  = max(zero(x), x) + min(zero(x), exp(x) - one(x))
gelu(x) = x / 2 * (one(x) + tanh(sqrt(oftype(x, 2 / pi)) *
                                (x + oftype(x, 0.044715) * x^3)))

# ReLU¹ is not C¹, so the Newton solve on it is expected to struggle; it is included as the
# k = 1 end of the ReLUᵏ axis rather than as a candidate for use.
const OGA_ACTIVATIONS_RELU = [("relu1", relu_k(1)), ("relu2", relu_k(2)),
                              ("relu3", relu_k(3)), ("relu4", relu_k(4))]
const OGA_ACTIVATIONS_SMOOTH = [("elu", elu), ("gelu", gelu), ("tanh", tanh)]
const OGA_ACTIVATIONS_ALL = vcat(OGA_ACTIVATIONS_RELU, OGA_ACTIVATIONS_SMOOTH)

# ---- the regularization ladder ----------------------------------------------
#
# `regularization_factor` is swept as multiples of `√eps(T)` rather than as absolute
# values, so that the Jacobian-diagonal shift is scaled to the precision it protects: an
# absolute `1e-5` is far below `√eps` at anything but `Float64` and so cannot lift a
# near-singular Jacobian in reduced precision.
#
# Each value is quoted as its **multiple of `√eps(T)`**, which is the readable identifier:
# `λ = 16√eps(T)` says what the shift is, whereas an index into a list does not. The set of
# multiples differs by precision so that the ladders span a comparable dynamic range —
# `Float16`/`Float32` step by powers of two from 2 to 64, `Float64` from 4 to 4096. The value
# `16√eps(T)` appears in both, and is the one the package documents as its recommended default.
oga_reg_multiples(::Type{Float64}) = (4, 16, 64, 256, 1024, 4096)   # 2^2 … 2^12
oga_reg_multiples(::Type{T}) where {T} = (2, 4, 8, 16, 32, 64)      # 2^1 … 2^6

# Formed in `Float64` and converted once: `T(multiple) * sqrt(eps(T))` overflows to `Inf` for
# `Float16` well inside the ladder.
oga_reg_factor(::Type{T}, multiple::Integer) where {T} =
    T(multiple * sqrt(Float64(eps(T))))

# ---- the residual tolerance --------------------------------------------------
#
# The solver's default `f_abstol` is `1.78e-15` — an *absolute* value scaled to `Float64`.
# It is unreachable at `Float32` (`eps ≈ 1.2e-7`) and `Float16` (`eps ≈ 9.8e-4`), so a
# reduced-precision run sits at its residual floor and burns the entire iteration budget
# while parked on the right answer: measured, `ReLU³` at `Float32` reports 1000 iterations
# at every regularization factor with an accuracy of `1.8e-7`. Reading that as non-convergence
# would make the whole `Float32` column an artefact of the tolerance rather than a fact about
# the seed.
#
# Same defect class as the absolute λ values the ladder above replaces, and the same fix:
# scale it to the precision. The factor 256 matches what SolverBenchmark's nonlinear
# specifications use for these problems.
oga_f_abstol(::Type{T}; factor = 256) where {T} = T(factor) * eps(T)

"""
    oga_reg_ladder(T) -> Vector{@NamedTuple{multiple::Int, factor::T}}

The regularization factors to sweep for working type `T`: the `λ = 0` control followed by six
values of the form `multiple · √eps(T)`.

`multiple` is `λ / √eps(T)` — the readable label for a factor, and `0` for the control.
`factor` is the `λ` actually passed to the solver.
"""
function oga_reg_ladder(::Type{T}) where {T}
    ladder = [(multiple = 0, factor = zero(T))]
    for m in oga_reg_multiples(T)
        push!(ladder, (multiple = m, factor = oga_reg_factor(T, m)))
    end
    return ladder
end
