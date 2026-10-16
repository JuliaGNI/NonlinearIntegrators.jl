# Can the rank deficiency of issue #98 be designed out of the ansatz? A standalone prototype.
#
#     julia --startup-file=no --project=scripts scripts/free_knot_rank.jl
#
# `scripts/newton_jacobian_rank.jl` measures the ShallowNet Newton Jacobian at rank 5 of 13 at
# `S = 4`: the converged ansatz is a cubic polynomial on the element (4 coefficients + p̃), with
# the scaling gauge of `relu_k` and dead neurons (`cᵢ ≈ 0`, so `wᵢ`, `bᵢ` are free) on top. This
# script tests the design that would remove all three — fix the gauge, carry the cubic
# explicitly, and keep the kinks inside the element:
#
#     q(t) = Σⱼ aⱼ tʲ  (j = 0…3)  +  Σᵢ cᵢ (t − τᵢ)₊³,     t ∈ [0, 1]
#
# `w = +1` for every neuron is the gauge fix; with the cubic explicit, the sign of `w` carries no
# information either, because `(τ − t)₊³ = (t − τ)₊³ − (t − τ)³`. Unknowns `(a, c, τ, p̃)`,
# `4 + 2S + 1`. The residual is the one `residual!` in `src/nvi/shallownet.jl` assembles by hand —
# the gradient of `p̃ q(1) − p̄ q(0) − h Σⱼ βⱼ L(q(tⱼ), q'(tⱼ)/h)` and the constraint `q(0) = q̄` —
# here taken by ForwardDiff of the discrete action, so it is exact in every parametrisation.
#
# Three variants, each on the harmonic oscillator (m = 1, k = 0.5, h = 0.5), two steps from
# `(q̄, p̄) = (0.5, 0)`, `S = 4, 5, 6`, and four knot seeds: uniform `τᵢ = i/(S+1)`, and the
# raw, normalised and orthogonal greedy rules of the OGA seeds, run on an interior knot grid of
# 401 points against the 11 labels a ShallowNet step fits.
#
#   fixed  — Gauss 8 on [0, 1], the quadrature ShallowNet uses.
#   split  — Gauss 4 on every piece between consecutive interior kinks: exact for a piecewise
#            cubic on this problem, and smooth in τ.
#   gaps   — split, with τᵢ = Σ_{j≤i} e^{φⱼ} / Σ_{j≤S+1} e^{φⱼ} (φ_{S+1} = 0): knots ordered,
#            distinct and inside (0, 1) for every finite φ.
#
# The cubic step's endpoint error is 5.9e-10 after one step and 2.2e-9 after two. Measured, on
# the row and column equilibrated Jacobian at sqrt(eps):
#
#   fixed — 16 of 24 rows reach full rank, σ_n/σ_1 ≈ 2e-3, but every one of them has
#           `max|c|` ≤ 1.2e-12, and the cubic's error or the one its first step left: the knots
#           are dead. A dead knot is still
#           pinned — at `c = 0` the knot row vanishes, and the `c` row selects the `τ` where the
#           cubic's residual is orthogonal to the atom — so the rank is full while the ansatz is
#           the cubic. Where `c` is not dead, the rank is lost, and in three of those rows the
#           step is *worse* than the cubic (≈ 1e-6). The `τ` scan at the end shows why: at
#           `S = 1` the stationary `c(τ)` changes sign every ≈ 0.04 in `τ` and reaches 1e6 … 1e9
#           near the ends — the knot fits the quadrature error of Gauss 8 on a piecewise cubic,
#           not the dynamics.
#   split — the one accurate stationary point: `S = 4` from the uniform seed, knots spread
#           evenly with all `cᵢ ≈ 2e-4`, error 1e-13 after one step and 4e-13 after two, and a
#           spectrum that decays smoothly to ≈ 7e-7 with no gap. Everywhere else the knots merge,
#           sit at 0 or 1, or leave the element, and the error is the cubic's.
#   gaps  — the same accurate point at `S = 4` from the uniform seed; elsewhere the knots gather
#           at 0 and 1, and the rows that keep full rank there do so with knots a hair apart.
#
# So the free-knot ansatz has an accurate, nondegenerate stationary point, but its basin under
# Newton is small: the cubic collapse attracts every OGA-like seed, and with Gauss 8 the full
# rank that remains is the dead-knot kind. Column equilibration matters for the reading: the τ
# columns scale with `c ≈ 2e-4`, and row equilibration alone (what `newton_jacobian_rank.jl`
# does) reports `n − (S − 1)` at the accurate point.

using LinearAlgebra
using Printf
using ForwardDiff
using QuadratureRules

const m, k = 1.0, 0.5
const h = 0.5
const ω = sqrt(k / m)
const RTOL = sqrt(eps())

L(q, v) = m * v^2 / 2 - k * q^2 / 2
relu3(z) = max(zero(z), z)^3
drelu3(z) = 3 * max(zero(z), z)^2

const FIXED = QuadratureRules.GaussLegendreQuadrature(Float64, 8)
const PIECE = QuadratureRules.GaussLegendreQuadrature(Float64, 4)

# θ = (a₀…a₃, c₁…c_S, τ₁…τ_S)
function ansatz(θ, S, t)
    sum(θ[j + 1] * t^j for j in 0:3) + sum(θ[4 + i] * relu3(t - θ[4 + S + i]) for i in 1:S)
end
function dansatz(θ, S, t)
    sum(j * θ[j + 1] * t^(j - 1) for j in 1:3) +
    sum(θ[4 + i] * drelu3(t - θ[4 + S + i]) for i in 1:S)
end

"The quadrature nodes and weights on [0, 1]: one rule, or one rule per piece between kinks."
function rule(θ, S, quadrature)
    quadrature === :fixed &&
        return QuadratureRules.nodes(FIXED), QuadratureRules.weights(FIXED)
    T = eltype(θ)
    edges = vcat(zero(T), sort([τ for τ in θ[(5 + S):(4 + 2S)] if 0 < τ < 1]), one(T))
    ξ, β = QuadratureRules.nodes(PIECE), QuadratureRules.weights(PIECE)
    t = [l + (r - l) * x for (l, r) in zip(edges[1:(end - 1)], edges[2:end]) for x in ξ]
    w = [(r - l) * b for (l, r) in zip(edges[1:(end - 1)], edges[2:end]) for b in β]
    t, w
end

function action(θ, p̃, S, p̄, quadrature)
    t, w = rule(θ, S, quadrature)
    s = p̃ * ansatz(θ, S, 1.0) - p̄ * ansatz(θ, S, 0.0)
    for j in eachindex(t)
        s -= h * w[j] * L(ansatz(θ, S, t[j]), dansatz(θ, S, t[j]) / h)
    end
    s
end

# The knot map of the `gaps` variant, and its inverse for a sorted τ inside (0, 1).
knots(φ) = (e = exp.(vcat(φ, zero(eltype(φ)))); cumsum(e)[1:(end - 1)] ./ sum(e))
gaps(τ) = (g = diff(vcat(0.0, τ, 1.0)); log.(g[1:(end - 1)] ./ g[end]))

"y holds τ or φ in its knot slots; the rest is (a, c, …, p̃) in both."
function to_τ(y, S, variant)
    variant === :gaps || return y
    vcat(y[1:(4 + S)], knots(y[(5 + S):(4 + 2S)]), y[end])
end

"The τ-form equations, at the τ that `y` maps to."
function residual(y, S, q̄, p̄, variant)
    quadrature = variant === :fixed ? :fixed : :split
    n = 4 + 2S
    x = to_τ(y, S, variant)
    g = ForwardDiff.gradient(θ -> action(θ, x[end], S, p̄, quadrature), x[1:n])
    vcat(g, q̄ - ansatz(x[1:n], S, 0.0))
end

function newton(y, S, q̄, p̄, variant; iterations = 200)
    for it in 1:iterations
        r = residual(y, S, q̄, p̄, variant)
        norm(r) < 1e-13 && return y, norm(r)
        J = ForwardDiff.jacobian(y -> residual(y, S, q̄, p̄, variant), y)
        y = y - qr(J, ColumnNorm()) \ r
    end
    y, norm(residual(y, S, q̄, p̄, variant))
end

function equilibrate(J)
    E = copy(J)
    for i in axes(E, 1)
        s = maximum(abs, view(E, i, :))
        s > 0 && (view(E, i, :) ./= s)
    end
    for j in axes(E, 2)
        s = maximum(abs, view(E, :, j))
        s > 0 && (view(E, :, j) ./= s)
    end
    E
end

# ---- seeds ------------------------------------------------------------------

const LABEL_T = collect(range(0, 1, 11))
const GRID = collect(range(0.01, 0.99, 401))

exact(q̄, p̄, t) = q̄ * cos(ω * h * t) + p̄ / (m * ω) * sin(ω * h * t)
design(τs) = hcat([LABEL_T .^ j for j in 0:3]..., [relu3.(LABEL_T .- τ) for τ in τs]...)

"Greedy knots against the cubic least-squares residual of the labels, by one OGA selection rule."
function greedy_knots(S, y, selection)
    chosen = Float64[]
    for _ in 1:S
        A = design(chosen)
        r = y - A * (A \ y)
        Q = Matrix(qr(A).Q)[:, 1:size(A, 2)]
        score(g) = selection === :raw ? abs(dot(r, g)) :
                   selection === :normalized ? abs(dot(r, g)) / norm(g) :
                   (gp = g - Q * (Q' * g);
            norm(gp) < 1e-10 * norm(g) ? -Inf : abs(dot(r, gp)) / norm(gp))
        candidates = [τ for τ in GRID if !(τ in chosen)]
        push!(chosen, candidates[argmax([score(relu3.(LABEL_T .- τ)) for τ in candidates])])
    end
    chosen
end

function seed(S, q̄, p̄, knotseed, variant)
    y = exact.(q̄, p̄, LABEL_T)
    τ = knotseed === :uniform ? collect((1:S) ./ (S + 1)) : greedy_knots(S, y, knotseed)
    p̃ = -m * ω * q̄ * sin(ω * h) + p̄ * cos(ω * h)
    if variant === :gaps          # sorted, pulled apart, φ = gaps(τ)
        τ = sort(τ)
        for i in 2:S
            τ[i] = max(τ[i], τ[i - 1] + 1e-3)
        end
        τ = clamp.(τ, 1e-3, 1 - 1e-3)
        return vcat(design(τ) \ y, gaps(τ), p̃)
    end
    vcat(design(τ) \ y, τ, p̃)
end

# ---- the table ----------------------------------------------------------------

println("Julia $(VERSION); harmonic oscillator m = 1, k = 0.5, h = 0.5; rank at sqrt(eps) of the")
println("row and column equilibrated Jacobian; `error` is |q(t_n) − q_exact(t_n)| after `step`.")
println()
@printf("%-6s %2s %-10s %4s %3s %5s %9s %9s %9s %9s  %s\n",
    "", "S", "seed", "step", "n", "rank", "residual", "error", "σ_n/σ_1", "max|c|", "knots")

for variant in (:fixed, :split, :gaps), S in (4, 5, 6),
    knotseed in (:uniform, :raw, :normalized, :orthogonal)
    q̄, p̄ = 0.5, 0.0
    for step in 1:2
        y, res = newton(seed(S, q̄, p̄, knotseed, variant), S, q̄, p̄, variant)
        J = ForwardDiff.jacobian(y -> residual(y, S, q̄, p̄, variant), y)
        σ = svdvals(equilibrate(J))
        x = to_τ(y, S, variant)
        qend = ansatz(x[1:(4 + 2S)], S, 1.0)
        @printf("%-6s %2d %-10s %4d %3d %5d %9.1e %9.1e %9.1e %9.1e  %s\n",
            variant, S, knotseed, step, length(y), count(>(RTOL * σ[1]), σ), res,
            abs(qend - exact(0.5, 0.0, step)), σ[end] / σ[1],
            maximum(abs, x[5:(4 + S)]),
            join(round.(x[(5 + S):(4 + 2S)]; sigdigits = 3), " "))
        q̄, p̄ = qend, y[end]
    end
end

# ---- why the fixed quadrature kills the knots: S = 1, τ held fixed -------------------

println()
println("fixed quadrature, S = 1, τ held fixed: the stationary c(τ) and the knot row R_τ")
@printf("%6s %12s %12s\n", "τ", "c", "R_τ")
for τ in 0.02:0.04:0.98
    full(z) = vcat(z[1:5], τ, z[6])
    reduced(z) = residual(full(z), 1, 0.5, 0.0, :fixed)[[1, 2, 3, 4, 5, 7]]
    z = zeros(6)
    for _ in 1:20
        r = reduced(z)
        norm(r) < 1e-15 && break
        z -= ForwardDiff.jacobian(reduced, z) \ r
    end
    @printf("%6.2f %12.3e %12.3e\n", τ, z[5], residual(full(z), 1, 0.5, 0.0, :fixed)[6])
end
