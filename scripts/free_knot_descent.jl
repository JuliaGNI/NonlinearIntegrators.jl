# Can a solve reach the accurate free-knot stationary point of `free_knot_rank.jl` from the greedy
# (OGA-like) knot seeds? The standalone prototype of the solve in `FreeKnotSpline`.
#
# **A frozen prototype.** It keeps its own copy of the method and does not follow
# `src/free_knot/free_knot_spline.jl`, which differs from it: here the descent accepts a step on
# Armijo alone, the inner solve runs a fixed number of iterations, and p̃ is seeded from RK4. Its
# figures are the prototype's. `free_knot_spline_study.jl` measures `FreeKnotSpline` itself.
#
#     julia --startup-file=no --project=scripts scripts/free_knot_descent.jl
#
# The ansatz and the quadrature are `free_knot_rank.jl`'s `split` variant:
#
#     q(t) = Σⱼ aⱼ tʲ  (j = 0…3)  +  Σᵢ cᵢ (t − τᵢ)₊³,     t ∈ [0, 1],
#
# Gauss 4 on every piece between consecutive knots. There, plain Newton on the full system reaches
# the accurate point (S = 4, error 1e-13 against the cubic's 6e-10) only from uniform knots.
#
# The solve here is a descent on the discrete action with fixed endpoints, by variable projection:
#
#   1. Q = q(1) from the cubic step (S = 0), whose endpoint is already within 1e-9.
#   2. For knots τ, eliminate a₀ = q̄ and a₃ (from q(1) = Q) and minimise the action over
#      z = (a₁, a₂, c) by Newton — one step for the harmonic oscillator, where it is quadratic.
#      That is f(τ) = min_z A(z, τ): for a quadratic action with exact quadrature,
#      f(τ) − A(q_exact) is half the squared energy-norm error of the best spline with knots τ.
#   3. Minimise f over φ, τ = knots(φ) (the ordered gap map of `free_knot_rank.jl`, so knots stay
#      ordered and inside (0, 1); a merge is φ → −∞), by Newton with the absolute-eigenvalue
#      Hessian, a step cap and Armijo backtracking.
#   4. Polish with plain Newton on the full τ-form system (a, c, τ, p̃), which also frees Q.
#
# For each problem, S = 4, 5, 6, and the uniform, raw, normalised and orthogonal greedy seeds,
# over two steps, the table reports plain Newton from the seed (`newton`) beside descent + polish
# (`descent`), with the rank of the row and column equilibrated Jacobian at sqrt(eps).
# `live` is yes when every knot sits at least 1e-3 inside (0, 1) and from its neighbour, and
# every |cᵢ| ≥ 1e-3 max|c|.
#
# Measured (h = 0.5, k = 0.5; oscillator q̄ = 0.5, pendulum L = v²/2 + k cos q, q̄ = 1.0):
#
#   descent — all 48 rows (2 problems × S = 4, 5, 6 × 4 seeds × 2 steps) reach full rank, live
#             knots and one and the same point per (problem, S, step), whatever the seed, in 4–19
#             descent iterations. The knots spread near-uniformly (S = 4: 0.21 0.40 0.59 0.79),
#             all cᵢ ≈ 2e-4, σ_n/σ_1 ≈ 1e-7 … 9e-7. Endpoint error after one / two steps:
#             oscillator 1e-13 / 4e-13 (S = 4), 3e-14 / 1.2e-13 (S = 5), 1.5e-14 / 5e-14 (S = 6);
#             pendulum 1e-13 / 4.7e-13, 4e-14 / 1.7e-13, 2e-14 / 6e-14 to 9e-14. The cubic:
#             5.9e-10 / 2.2e-9 (oscillator), 5.8e-10 / 2.9e-9 (pendulum).
#   newton  — from the same seeds, full rank with live knots in 8 of 48 rows, all from the
#             uniform seed; every greedy row collapses (knots merged, at 0 or 1, or outside),
#             with the rank lost and an error of 1.7e-12 at best, the cubic's in most rows.
#             (The plain-Newton seed here takes (a, c) from the reduced solve, not the label fit
#             of `free_knot_rank.jl`, so its uniform rows differ from that script's.)
#
# So the basin problem is the solver's, not the seed's: the reduced action f(τ) has, at
# S = 4, 5, 6, a single minimiser reachable by descent from every greedy seed tried, and it is
# the accurate nondegenerate stationary point.
#
# The descent here accepts a step on Armijo alone. That is enough for these two steps from p̄ = 0,
# and not in general: near a zero of q the gain of the knots falls below the round-off of f, the
# descent stops at an arbitrary point, and the polish can reach the dead-knot point. `FreeKnotSpline`
# also accepts a step that halves the reduced gradient; `free_knot_spline_study.jl` measures it.

using LinearAlgebra
using Printf
using ForwardDiff
using QuadratureRules

const h = 0.5
const k = 0.5
const RTOL = sqrt(eps())
const PIECE = QuadratureRules.GaussLegendreQuadrature(Float64, 4)

relu3(z) = max(zero(z), z)^3
drelu3(z) = 3 * max(zero(z), z)^2

const OSCILLATOR = (name = "oscillator", q₀ = 0.5,
    L = (q, v) -> v^2 / 2 - k * q^2 / 2, force = q -> -k * q)
const PENDULUM = (name = "pendulum", q₀ = 1.0,
    L = (q, v) -> v^2 / 2 + k * cos(q), force = q -> -k * sin(q))

# ---- the exact flow: RK4 at h/40000, error ≈ 4e-15, from round-off ------------------------

function flow(P, q, p, T; n = ceil(Int, 40000 * abs(T)))
    dt = T * h / n
    for _ in 1:n
        k1q, k1p = p, P.force(q)
        k2q, k2p = p + dt / 2 * k1p, P.force(q + dt / 2 * k1q)
        k3q, k3p = p + dt / 2 * k2p, P.force(q + dt / 2 * k2q)
        k4q, k4p = p + dt * k3p, P.force(q + dt * k3q)
        q += dt / 6 * (k1q + 2k2q + 2k3q + k4q)
        p += dt / 6 * (k1p + 2k2p + 2k3p + k4p)
    end
    q, p
end

# ---- ansatz, quadrature, action ---------------------------------------------------------

# θ = (a₀…a₃, c₁…c_S, τ₁…τ_S)
function ansatz(θ, S, t)
    q = θ[1] + θ[2] * t + θ[3] * t^2 + θ[4] * t^3
    for i in 1:S
        q += θ[4 + i] * relu3(t - θ[4 + S + i])
    end
    q
end
function dansatz(θ, S, t)
    v = θ[2] + 2θ[3] * t + 3θ[4] * t^2
    for i in 1:S
        v += θ[4 + i] * drelu3(t - θ[4 + S + i])
    end
    v
end

"Gauss 4 on every piece between consecutive interior knots."
function rule(θ, S)
    T = eltype(θ)
    edges = vcat(zero(T), sort([τ for τ in θ[(5 + S):(4 + 2S)] if 0 < τ < 1]), one(T))
    ξ, β = QuadratureRules.nodes(PIECE), QuadratureRules.weights(PIECE)
    t = [l + (r - l) * x for (l, r) in zip(edges[1:(end - 1)], edges[2:end]) for x in ξ]
    w = [(r - l) * b for (l, r) in zip(edges[1:(end - 1)], edges[2:end]) for b in β]
    t, w
end

"h ∫ L dt by the piecewise rule: the action without its boundary terms."
function interior(P, θ, S)
    t, w = rule(θ, S)
    s = zero(eltype(θ))
    for j in eachindex(t)
        s += h * w[j] * P.L(ansatz(θ, S, t[j]), dansatz(θ, S, t[j]) / h)
    end
    s
end

"The full τ-form system of `free_knot_rank.jl`: ∇θ[p̃ q(1) − p̄ q(0) − A] and q(0) = q̄."
function residual(P, x, S, q̄, p̄)
    n = 4 + 2S
    action(θ) = x[end] * ansatz(θ, S, 1.0) - p̄ * ansatz(θ, S, 0.0) - interior(P, θ, S)
    vcat(ForwardDiff.gradient(action, x[1:n]), q̄ - ansatz(x[1:n], S, 0.0))
end

function newton(P, x, S, q̄, p̄; iterations = 200)
    for _ in 1:iterations
        r = residual(P, x, S, q̄, p̄)
        norm(r) < 1e-13 && return x, norm(r)
        J = ForwardDiff.jacobian(x -> residual(P, x, S, q̄, p̄), x)
        x = x - qr(J, ColumnNorm()) \ r
    end
    x, norm(residual(P, x, S, q̄, p̄))
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

# ---- the reduced action f(φ) = min_z A(z, knots(φ)) with q(0) = q̄, q(1) = Q ----------------

knots(φ) = (e = exp.(vcat(φ, zero(eltype(φ)))); cumsum(e)[1:(end - 1)] ./ sum(e))
gaps(τ) = (g = diff(vcat(0.0, τ, 1.0)); log.(g[1:(end - 1)] ./ g[end]))

"θ from z = (a₁, a₂, c) and τ, with a₀ = q̄ and a₃ such that q(1) = Q."
function theta(z, τ, S, q̄, Q)
    a₁, a₂, c = z[1], z[2], z[3:end]
    a₃ = Q - q̄ - a₁ - a₂ - sum(c[i] * (1 - τ[i])^3 for i in 1:S)
    vcat(q̄ + zero(a₃), a₁, a₂, a₃, c, τ)
end

function inner(P, τ, S, q̄, Q; iterations = P === OSCILLATOR ? 1 : 6)
    F(z) = interior(P, theta(z, τ, S, q̄, Q), S)
    z = zeros(eltype(τ), 2 + S)
    for _ in 1:iterations
        z = z - ForwardDiff.hessian(F, z) \ ForwardDiff.gradient(F, z)
    end
    z
end

function reduced(P, φ, S, q̄, Q)
    (τ = knots(φ); interior(P, theta(inner(P, τ, S, q̄, Q), τ, S, q̄, Q), S))
end

function descend(P, φ, S, q̄, Q; iterations = 300, cap = 1.0)
    f(φ) = reduced(P, φ, S, q̄, Q)
    fφ = f(φ)
    for it in 1:iterations
        g = ForwardDiff.gradient(f, φ)
        λ, V = eigen(Symmetric(ForwardDiff.hessian(f, φ)))
        d = -V * ((V' * g) ./ max.(abs.(λ), 1e-10 * maximum(abs, λ)))
        s = min(1.0, cap / norm(d, Inf))
        accepted = false
        for _ in 1:40
            fnew = f(φ + s * d)
            if fnew < fφ + 1e-4 * s * dot(g, d)
                φ, fφ, accepted = φ + s * d, fnew, true
                break
            end
            s /= 2
        end
        (!accepted || norm(s * d, Inf) < 1e-12) && return φ, it
    end
    φ, iterations
end

# ---- seeds --------------------------------------------------------------------------------

const LABEL_T = collect(range(0, 1, 11))
const GRID = collect(range(0.01, 0.99, 401))

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

"Sorted seed knots, pulled 1e-3 apart and inside, so that `gaps` is finite."
function seed_knots(P, S, q̄, p̄, knotseed)
    knotseed === :uniform && return collect((1:S) ./ (S + 1))
    y = [flow(P, q̄, p̄, t)[1] for t in LABEL_T]
    τ = sort(greedy_knots(S, y, knotseed))
    for i in 2:S
        τ[i] = max(τ[i], τ[i - 1] + 1e-3)
    end
    clamp.(τ, 1e-3, 1 - 1e-3)
end

# ---- one step by each method ----------------------------------------------------------------

function cubic(P, q̄, p̄)
    p̃ = flow(P, q̄, p̄, 1)[2]
    x, _ = newton(P, [q̄, 0.0, 0.0, 0.0, p̃], 0, q̄, p̄)
    x
end

function step(P, S, q̄, p̄, knotseed, method)
    xc = cubic(P, q̄, p̄)
    τ = seed_knots(P, S, q̄, p̄, knotseed)
    iterations = 0
    if method === :descent
        φ, iterations = descend(P, gaps(τ), S, q̄, ansatz(xc, 0, 1.0))
        τ = knots(φ)
    end
    Q = ansatz(xc, 0, 1.0)
    θ = theta(inner(P, τ, S, q̄, Q), τ, S, q̄, Q)
    x, res = newton(P, vcat(θ, xc[end]), S, q̄, p̄)
    x, res, iterations
end

function report(P, method, S, knotseed)
    q̄, p̄ = P.q₀, 0.0
    qc, pc = q̄, p̄
    for n in 1:2
        x, res, its = step(P, S, q̄, p̄, knotseed, method)
        xc = cubic(P, qc, pc)
        σ = svdvals(equilibrate(ForwardDiff.jacobian(x -> residual(P, x, S, q̄, p̄), x)))
        c, τ = x[5:(4 + S)], x[(5 + S):(4 + 2S)]
        qend, qexact = ansatz(x, S, 1.0), flow(P, P.q₀, 0.0, n)[1]
        gap = minimum(diff(vcat(0.0, sort(τ), 1.0)))
        live = gap ≥ 1e-3 && minimum(abs, c) ≥ 1e-3 * maximum(abs, c)
        @printf("%-8s %2d %-10s %4d %4d %3d %5d %9.1e %9.1e %9.1e %9.1e %9.1e %9.1e %-4s %s\n",
            method, S, knotseed, n, its, length(x), count(>(RTOL * σ[1]), σ), res,
            abs(qend - qexact), abs(ansatz(xc, 0, 1.0) - qexact), σ[end] / σ[1],
            maximum(abs, c), gap, live ? "yes" : "no",
            join(round.(τ; sigdigits = 3), " "))
        q̄, p̄ = qend, x[end]
        qc, pc = ansatz(xc, 0, 1.0), xc[end]
    end
end

println("Julia $(VERSION); h = $h, k = $k; rank at sqrt(eps) of the row and column equilibrated")
println("Jacobian; `error` is |q(t_n) − q_exact(t_n)|, `cubic` the same for the cubic (S = 0) step;")
println("`its` counts descent iterations; `gap` is the smallest distance between knots or to 0, 1.")
for P in (OSCILLATOR, PENDULUM)
    println()
    println("$(P.name), q̄ = $(P.q₀), p̄ = 0")
    @printf("%-8s %2s %-10s %4s %4s %3s %5s %9s %9s %9s %9s %9s %9s %-4s %s\n",
        "method", "S", "seed", "step", "its", "n", "rank", "residual", "error", "cubic",
        "σ_n/σ_1", "max|c|", "gap", "live", "knots")
    for S in (4, 5, 6), knotseed in (:uniform, :raw, :normalized, :orthogonal),
        method in (:newton, :descent)
        report(P, method, S, knotseed)
    end
end
