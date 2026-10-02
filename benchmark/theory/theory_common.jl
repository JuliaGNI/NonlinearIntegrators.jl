# Self-contained reference implementation for the ReLUᵏ-network ⇔ CGVI proofs.
#
# Uses only the Julia standard library (LinearAlgebra, Printf, Random), so the proof checks run
# independently of NonlinearIntegrators / GeometricIntegrators: every equation in
# `relu_cgvi_equivalence.md` is re-implemented here from its definition, not taken from the
# package. `verify_conjectures.jl` then compares the package against the same statements.
#
# Conventions (one time step, one degree of freedom, D = 1):
#   t = tₙ + h τ,  τ ∈ [0, 1],   L(q, v) = v²/2 − V(q),   ϑ = ∂L/∂v = v,   f = ∂L/∂q = −V′(q)
#   discrete action  A_h(q) = h Σⱼ bⱼ L(q(cⱼ), q′(cⱼ)/h)      (q′ = dq/dτ)
#   one-step weak form (type II, the form both CGVI and ShallowNet solve):
#       for every test function δq:  δq(1) pₙ₊₁ − δq(0) pₙ − δA_h(q)[δq] = 0,   q(0) = qₙ
#   δA_h(q)[δq] = Σⱼ bⱼ ( h f(Qⱼ) δq(cⱼ) + ϑ(Vⱼ) δq′(cⱼ) ),  Qⱼ = q(cⱼ), Vⱼ = q′(cⱼ)/h.

using LinearAlgebra
using Printf
using Random

# ---------------------------------------------------------------------------
# Quadrature on [0, 1]
# ---------------------------------------------------------------------------

"""Gauss–Legendre nodes/weights on [0,1] (Golub–Welsch)."""
function gauss_legendre01(R::Int)
    R == 1 && return [0.5], [1.0]
    β = [i / sqrt(4i^2 - 1) for i in 1:(R - 1)]
    E = eigen(SymTridiagonal(zeros(R), β))
    x = E.values
    w = 2 .* E.vectors[1, :] .^ 2
    p = sortperm(x)
    return (x[p] .+ 1) ./ 2, w[p] ./ 2
end

"""A quadrature rule on [0,1]."""
struct QRule01
    c::Vector{Float64}
    b::Vector{Float64}
end
QRule01(R::Int) = QRule01(gauss_legendre01(R)...)

"""Composite Gauss rule: R points on every sub-interval cut by the interior `breaks`."""
function composite_rule(R::Int, breaks)
    br = vcat(0.0, sort(filter(z -> 0 < z < 1, collect(Float64, breaks))), 1.0)
    c0, b0 = gauss_legendre01(R)
    c = Float64[]
    b = Float64[]
    for i in 1:(length(br) - 1)
        l, r = br[i], br[i + 1]
        append!(c, l .+ (r - l) .* c0)
        append!(b, (r - l) .* b0)
    end
    QRule01(c, b)
end

# ---------------------------------------------------------------------------
# Toy problems  L = v²/2 − V(q)
# ---------------------------------------------------------------------------

struct ToyProblem
    name::String
    V::Function
    dV::Function
end
const TOY_HO = ToyProblem("harmonic_oscillator", q -> q^2 / 2, q -> q)
const TOY_PEND = ToyProblem("pendulum", q -> 1 - cos(q), q -> sin(q))
const TOY_PROBLEMS = (TOY_HO, TOY_PEND)

"""Exact harmonic-oscillator solution with ω = 1."""
ho_exact(q0, p0, t) = q0 * cos(t) + p0 * sin(t)

discrete_action(q, dq, h, quad::QRule01, prob::ToyProblem) =
    h * sum(quad.b .* (0.5 .* (dq ./ h) .^ 2 .- prob.V.(q)))

# ---------------------------------------------------------------------------
# ReLUᵏ and its derivatives (Heaviside-correct for exponent 0)
# ---------------------------------------------------------------------------

rp(x, m::Int) = m == 0 ? (x > 0 ? one(x) : zero(x)) : max(zero(x), x)^m
σk(x, k) = rp(x, k)
dσk(x, k) = k * rp(x, k - 1)
d2σk(x, k) = k >= 2 ? k * (k - 1) * rp(x, k - 2) : zero(x)

# ---------------------------------------------------------------------------
# Linear bases: every basis is a closure  phi(t::Vector) -> (P, D),
# P[r, i] = φᵢ(t_r), D[r, i] = φᵢ′(t_r).
# ---------------------------------------------------------------------------

function lagrange_basis(nodes::Vector{Float64})
    n = length(nodes)
    function phi(t::AbstractVector)
        P = ones(length(t), n)
        D = zeros(length(t), n)
        for (r, τ) in enumerate(t), i in 1:n
            for j in 1:n
                j == i && continue
                P[r, i] *= (τ - nodes[j]) / (nodes[i] - nodes[j])
            end
            for m in 1:n
                m == i && continue
                term = 1 / (nodes[i] - nodes[m])
                for j in 1:n
                    (j == i || j == m) && continue
                    term *= (τ - nodes[j]) / (nodes[i] - nodes[j])
                end
                D[r, i] += term
            end
        end
        P, D
    end
    phi
end

"""CGVI basis of degree s: Lagrange on the s+1 Gauss nodes (as `galerkin_method`)."""
cgvi_basis(s::Int) = lagrange_basis(gauss_legendre01(s + 1)[1])

function monomial_basis(k::Int)
    phi(t::AbstractVector) = ([τ^j for τ in t, j in 0:k],
        [j == 0 ? 0.0 : j * τ^(j - 1) for τ in t, j in 0:k])
    phi
end

"""Truncated-power basis of S_k(Z): 1, τ, …, τᵏ, (τ − zᵢ)₊ᵏ."""
function truncated_power_basis(k::Int, Z)
    Z = collect(Float64, Z)
    function phi(t::AbstractVector)
        P = hcat([τ^j for τ in t, j in 0:k], [rp(τ - z, k) for τ in t, z in Z])
        D = hcat([j == 0 ? 0.0 : j * τ^(j - 1) for τ in t, j in 0:k],
            [dσk(τ - z, k) for τ in t, z in Z])
        P, D
    end
    phi
end

"""Double-knot space: truncated powers of order k and k−1 at every knot."""
function double_knot_basis(k::Int, Z)
    Z = collect(Float64, Z)
    function phi(t::AbstractVector)
        P = hcat([τ^j for τ in t, j in 0:k], [rp(τ - z, k) for τ in t, z in Z],
            [rp(τ - z, k - 1) for τ in t, z in Z])
        P, zeros(size(P))
    end
    phi
end

"""Frozen neurons σ(wᵢτ + bᵢ) as a linear basis."""
function neuron_basis(k::Int, w::Vector{Float64}, b::Vector{Float64})
    function phi(t::AbstractVector)
        Zm = t .* w' .+ b'
        σk.(Zm, k), dσk.(Zm, k) .* w'
    end
    phi
end

"""Clamped B-spline basis of degree k with interior knots Z (Cox–de Boor)."""
function bspline_basis(k::Int, Z)
    kn = vcat(zeros(k + 1), sort(collect(Float64, Z)), ones(k + 1))
    n = length(kn) - k - 1
    function N(i, p, t)
        if p == 0
            kn[i] <= t < kn[i + 1] && return 1.0
            # right end point: belongs to the last non-empty interval
            (t == kn[end] && kn[i] < kn[i + 1] && kn[i + 1] == kn[end]) && return 1.0
            return 0.0
        end
        s = 0.0
        d1 = kn[i + p] - kn[i]
        d1 > 0 && (s += (t - kn[i]) / d1 * N(i, p - 1, t))
        d2 = kn[i + p + 1] - kn[i + 1]
        d2 > 0 && (s += (kn[i + p + 1] - t) / d2 * N(i + 1, p - 1, t))
        s
    end
    function dN(i, p, t)
        s = 0.0
        d1 = kn[i + p] - kn[i]
        d1 > 0 && (s += p / d1 * N(i, p - 1, t))
        d2 = kn[i + p + 1] - kn[i + 1]
        d2 > 0 && (s -= p / d2 * N(i + 1, p - 1, t))
        s
    end
    phi(t::AbstractVector) = ([N(i, k, τ) for τ in t, i in 1:n],
        [dN(i, k, τ) for τ in t, i in 1:n])
    phi
end

nbasis_of(phi) = size(phi([0.5])[1], 2)

# ---------------------------------------------------------------------------
# Shallow ReLUᵏ network  q_θ(τ) = Σᵢ aᵢ σ(wᵢ τ + bᵢ),  θ = [a; w; b]
# (same ansatz as ShallowNetBasis: Dense(1,S,σ) → Dense(S,1, no bias))
# ---------------------------------------------------------------------------

function unpack(θ)
    S = length(θ) ÷ 3
    θ[1:S], θ[(S + 1):(2S)], θ[(2S + 1):(3S)]
end
pack(a, w, b) = vcat(a, w, b)
kinks(θ) = (x = unpack(θ); -x[3] ./ x[2])

function nn_eval(θ, t::AbstractVector, k)
    a, w, b = unpack(θ)
    Zm = t .* w' .+ b'
    σk.(Zm, k) * a, (dσk.(Zm, k) .* w') * a
end

"""Tangent vectors ∂q/∂θ and ∂q′/∂θ at the points t (columns ordered a, w, b)."""
function nn_tangent(θ, t::AbstractVector, k)
    a, w, b = unpack(θ)
    Zm = t .* w' .+ b'
    s = σk.(Zm, k)
    ds = dσk.(Zm, k)
    dds = d2σk.(Zm, k)
    TQ = hcat(s, a' .* t .* ds, a' .* ds)
    TV = hcat(w' .* ds, a' .* (ds .+ t .* w' .* dds), a' .* w' .* dds)
    TQ, TV
end

# ---------------------------------------------------------------------------
# One-step residuals (type II).  Unknowns: coefficients (or θ) and pₙ₊₁.
# ---------------------------------------------------------------------------

"""Weak-form residual for a linear space with basis phi. x = [X; p₁]."""
function linear_residual(x, q0, p0, h, phi, quad::QRule01, prob::ToyProblem)
    n = length(x) - 1
    X = x[1:n]
    p1 = x[n + 1]
    P, D = phi(quad.c)
    Q = P * X
    V = (D * X) ./ h
    f = -prob.dV.(Q)
    r0 = vec(phi([0.0])[1])
    r1 = vec(phi([1.0])[1])
    res = r1 .* p1 .- r0 .* p0 .- (P' * (quad.b .* h .* f) .+ D' * (quad.b .* V))
    vcat(res, q0 - dot(r0, X))
end

"""Weak-form residual of the network; rows ordered [a (S); w (S); b (S); q(0) = qₙ]."""
function nn_residual(x, q0, p0, h, k, quad::QRule01, prob::ToyProblem)
    θ = x[1:(end - 1)]
    p1 = x[end]
    q, dq = nn_eval(θ, quad.c, k)
    V = dq ./ h
    f = -prob.dV.(q)
    TQ, TV = nn_tangent(θ, quad.c, k)
    T0 = vec(nn_tangent(θ, [0.0], k)[1])
    T1 = vec(nn_tangent(θ, [1.0], k)[1])
    res = T1 .* p1 .- T0 .* p0 .- (TQ' * (quad.b .* h .* f) .+ TV' * (quad.b .* V))
    vcat(res, q0 - nn_eval(θ, [0.0], k)[1][1])
end

# ---------------------------------------------------------------------------
# Solvers (finite-difference Jacobian; the unknown counts are tiny)
# ---------------------------------------------------------------------------

function fd_jacobian(F, x; ε = 1e-7)
    f0 = F(x)
    J = zeros(length(f0), length(x))
    for i in eachindex(x)
        δ = ε * max(1.0, abs(x[i]))
        e = zeros(length(x))
        e[i] = δ
        J[:, i] = (F(x .+ e) .- F(x .- e)) ./ (2δ)
    end
    J
end

"""Plain Newton (for the linear-space methods, whose Jacobian is regular)."""
function newton_solve(F, x0; tol = 1e-13, maxit = 60)
    x = copy(x0)
    for it in 0:maxit
        r = F(x)
        norm(r, Inf) < tol && return (x = x, iters = it, converged = true)
        it == maxit && break
        x = x .- fd_jacobian(F, x) \ r
    end
    (x = x, iters = maxit, converged = norm(F(x), Inf) < 1e-11)
end

"""Levenberg–Marquardt (for the network, whose Jacobian is singular by the gauge)."""
function lm_solve(F, x0; tol = 1e-13, maxit = 500)
    x = copy(x0)
    r = F(x)
    nr = norm(r)
    μ = 1e-6
    for it in 0:maxit
        norm(r, Inf) < tol && return (x = x, iters = it, converged = true)
        J = fd_jacobian(F, x)
        A = J' * J
        g = J' * r
        accepted = false
        while μ < 1e12
            dx = -((A + μ * I) \ g)
            rn = F(x .+ dx)
            nrn = norm(rn)
            if isfinite(nrn) && nrn < nr
                x = x .+ dx
                r = rn
                nr = nrn
                μ = max(μ / 10, 1e-15)
                accepted = true
                break
            end
            μ *= 10
        end
        accepted || break
    end
    (x = x, iters = maxit, converged = norm(r, Inf) < 1e-11)
end

# ---------------------------------------------------------------------------
# Steps and trajectories
# ---------------------------------------------------------------------------

"""One type-II step in a linear space (CGVI when phi = cgvi_basis(s))."""
function linear_step(q0, p0, h, phi, quad, prob)
    tt = collect(range(0, 1; length = 41))
    X0 = phi(tt)[1] \ fill(q0, length(tt))
    sol = newton_solve(x -> linear_residual(x, q0, p0, h, phi, quad, prob), vcat(X0, p0))
    n = length(X0)
    X = sol.x[1:n]
    (q1 = dot(vec(phi([1.0])[1]), X), p1 = sol.x[end], X = X, converged = sol.converged)
end

"""One network step by LM from the initial guess x0 = [θ; p₁]."""
function nn_step(q0, p0, h, k, quad, prob, x0; kw...)
    sol = lm_solve(x -> nn_residual(x, q0, p0, h, k, quad, prob), x0; kw...)
    θ = sol.x[1:(end - 1)]
    (q1 = nn_eval(θ, [1.0], k)[1][1], p1 = sol.x[end], θ = θ, converged = sol.converged,
        x = sol.x)
end

"""Trajectory of a linear method; returns (q, p) vectors of length N+1."""
function linear_trajectory(q0, p0, h, N, phi, quad, prob)
    q = [q0]
    p = [p0]
    for _ in 1:N
        s = linear_step(q[end], p[end], h, phi, quad, prob)
        s.converged || error("linear step did not converge")
        push!(q, s.q1)
        push!(p, s.p1)
    end
    q, p
end

const TT_FIT = collect(range(0, 1; length = 41))

"""
Network trajectory. Each step is started from the previous step's network: if that network is
kink-free (a polynomial on [0,1]) it is continued polynomially to τ ∈ [1,2] and refitted with
the same (w, b) — an extrapolated start value, like the package's warm start, which never uses
the answer of the current step. Otherwise the previous θ itself is used.
"""
function nn_trajectory(q0, p0, h, N, k, quad, prob, θ0)
    q = [q0]
    p = [p0]
    θ = copy(θ0)
    thetas = [copy(θ)]
    Mk = monomial_basis(k)
    for _ in 1:N
        guess = θ
        if is_kinkfree(θ, k)
            cpoly = Mk(TT_FIT)[1] \ nn_eval(θ, TT_FIT, k)[1]
            qext = Mk(1 .+ TT_FIT)[1] * cpoly
            _, w, b = unpack(θ)
            guess = fit_output_weights(TT_FIT, qext, w, b, k)
        end
        s = nn_step(q[end], p[end], h, k, quad, prob, vcat(guess, p[end]))
        s.converged || error("network step did not converge")
        push!(q, s.q1)
        push!(p, s.p1)
        θ = s.θ
        push!(thetas, copy(θ))
    end
    q, p, thetas
end

"""
det of the Jacobian of the network one-step map (qₙ, pₙ) ↦ (qₙ₊₁, pₙ₊₁), by central
differences, every perturbed solve warm-started from the converged x (same solution branch).
det = 1 ⇔ area preserving ⇔ symplectic (one degree of freedom).
"""
function step_map_det(q0, p0, h, k, quad, prob, xw; e = 1e-5)
    M = zeros(2, 2)
    for (j, (dq, dp)) in enumerate(((e, 0.0), (0.0, e)))
        sp = nn_step(q0 + dq, p0 + dp, h, k, quad, prob, xw)
        sm = nn_step(q0 - dq, p0 - dp, h, k, quad, prob, xw)
        (sp.converged && sm.converged) || return NaN
        M[:, j] = [(sp.q1 - sm.q1) / (2 * e), (sp.p1 - sm.p1) / (2 * e)]
    end
    det(M)
end

# ---------------------------------------------------------------------------
# Type-I (both end points fixed) and the discrete Lagrangian
# ---------------------------------------------------------------------------

"""
Stationary point of A_h on {q ∈ span(phi): q(0) = qa, q(1) = qb}.
Returns coefficients, L_d(qa, qb) and the two discrete Legendre momenta
p_a = −∂L_d/∂qa (= −λ₀), p_b = ∂L_d/∂qb (= λ₁).
"""
function typeI_step(qa, qb, h, phi, quad, prob)
    n = nbasis_of(phi)
    P, D = phi(quad.c)
    r0 = vec(phi([0.0])[1])
    r1 = vec(phi([1.0])[1])
    function F(x)
        c = x[1:n]
        λ0 = x[n + 1]
        λ1 = x[n + 2]
        Q = P * c
        V = (D * c) ./ h
        gA = P' * (quad.b .* h .* (-prob.dV.(Q))) .+ D' * (quad.b .* V)
        vcat(gA .- λ0 .* r0 .- λ1 .* r1, dot(r0, c) - qa, dot(r1, c) - qb)
    end
    tt = collect(range(0, 1; length = 41))
    c0 = phi(tt)[1] \ (qa .+ (qb - qa) .* tt)
    sol = newton_solve(F, vcat(c0, 0.0, 0.0))
    c = sol.x[1:n]
    (c = c, Ld = discrete_action(P * c, D * c, h, quad, prob), p_a = -sol.x[n + 1],
        p_b = sol.x[n + 2], converged = sol.converged)
end

# ---------------------------------------------------------------------------
# Linear-algebra helpers
# ---------------------------------------------------------------------------

"""Numerical rank relative to the largest singular value."""
numrank(M; rtol = 1e-10) = (s = svdvals(M); isempty(s) ? 0 : count(>(rtol * s[1]), s))

"""Relative residual of projecting the columns of M onto span(B) (least squares)."""
proj_residual(M, B) = norm(M - B * (B \ M)) / max(norm(M), eps())

"""Monomial coefficients (τ⁰ … τᵏ) of (τ − z)ʲ padded to length k+1."""
shift_power_coeffs(z, j, k) = [i <= j ? binomial(j, i) * (-z)^(j - i) : 0.0 for i in 0:k]

"""Apolar pairing  B(f, g) = Σₘ (−1)ᵐ f⁽ᵐ⁾(0) g⁽ᵏ⁻ᵐ⁾(0) on coefficient vectors of P_k."""
function apolar_pairing(f::AbstractVector, g::AbstractVector, k::Int)
    sum((-1)^m * factorial(m) * f[m + 1] * factorial(k - m) * g[k - m + 1] for m in 0:k)
end
polyval(c, x) = sum(c[i] * x^(i - 1) for i in eachindex(c))
polyder_val(c, x) = sum((i - 1) * c[i] * x^(i - 2) for i in 2:length(c); init = 0.0)

"""Neurons with distinct kinks outside [0,1] that are active on the whole interval."""
function kinkfree_neurons(S::Int)
    w = [isodd(i) ? 1.0 : -1.0 for i in 1:S]
    z = [isodd(i) ? -0.3 - 0.35 * (i ÷ 2) : 1.3 + 0.35 * (i ÷ 2 - 1) for i in 1:S]
    w, -z .* w
end

"""Represent a function given by samples (tt, q) with fixed neurons (w, b): returns θ."""
function fit_output_weights(tt, qvals, w, b, k)
    a = σk.(tt .* w' .+ b', k) \ qvals
    pack(a, w, b)
end

"""Classify neurons of θ on [0,1]."""
function neuron_classes(θ, k; amp_rtol = 1e-8)
    a, w, b = unpack(θ)
    amp = abs.(a) .* abs.(w) .^ k
    amax = maximum(amp; init = 0.0)
    cls = String[]
    for i in eachindex(a)
        if abs(w[i]) < 1e-14
            push!(cls, b[i] > 0 ? "outside" : "inactive")
            continue
        end
        z = -b[i] / w[i]
        if 0 < z < 1
            push!(cls, amp[i] <= amp_rtol * max(amax, eps()) ? "interior0" : "interior")
        else
            push!(cls, w[i] * 0.5 + b[i] > 0 ? "outside" : "inactive")
        end
    end
    cls
end

"""No neuron has its kink inside (0,1) (active kink-free or inactive neurons only)."""
is_kinkfree(θ, k) = !any(c -> startswith(c, "interior"), neuron_classes(θ, k))

"""Number of distinct kinks (clustered with a relative tolerance)."""
function n_distinct(zs; rtol = 1e-6)
    isempty(zs) && return 0
    s = sort(zs)
    n = 1
    for i in 2:length(s)
        abs(s[i] - s[i - 1]) > rtol * max(1.0, abs(s[i])) && (n += 1)
    end
    n
end

# ---------------------------------------------------------------------------
# Check bookkeeping
# ---------------------------------------------------------------------------

const CHECKS = NamedTuple[]

"""Record a check `value ≤ tol` and print one line."""
function vcheck(id, desc, value, tol)
    v = Float64(value)
    pass = isfinite(v) && v <= tol
    push!(CHECKS, (id = id, desc = desc, value = v, tol = Float64(tol), pass = pass))
    @printf("  [%s] %-7s %-72s %10.3e  (≤ %.1e)\n", pass ? "PASS" : "FAIL", id, desc, v, tol)
    pass
end

"""Record a check `value ≥ lower` (used for 'must differ' statements)."""
function vcheck_ge(id, desc, value, lower)
    v = Float64(value)
    pass = isfinite(v) && v >= lower
    push!(CHECKS, (id = id, desc = desc, value = v, tol = Float64(lower), pass = pass))
    @printf("  [%s] %-7s %-72s %10.3e  (≥ %.1e)\n", pass ? "PASS" : "FAIL", id, desc, v, lower)
    pass
end

vcheck_eq(id, desc, got::Integer, expected::Integer) = begin
    pass = got == expected
    push!(CHECKS, (id = id, desc = desc, value = Float64(got), tol = Float64(expected), pass = pass))
    @printf("  [%s] %-7s %-72s %10d  (= %d)\n", pass ? "PASS" : "FAIL", id, desc, got, expected)
    pass
end

section(title) = (println(); println("── ", title, " ", "─"^max(0, 90 - length(title))))

function summarize_checks()
    n = length(CHECKS)
    nf = count(c -> !c.pass, CHECKS)
    println()
    println("="^100)
    @printf("%d checks, %d passed, %d failed\n", n, n - nf, nf)
    for c in CHECKS
        c.pass || @printf("  FAILED %-7s %s  (value %.3e, bound %.1e)\n", c.id, c.desc, c.value, c.tol)
    end
    println("="^100)
    nf == 0
end
