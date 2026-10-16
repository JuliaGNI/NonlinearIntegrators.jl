@doc raw"""
    FreeKnotSpline(S; quadrature = QuadratureRules.GaussLegendreQuadrature(4),
        descent_iterations = 100, newton_iterations = 50)

Variational integrator on a cubic spline with `S` free interior knots per step.

On a step ``[t_n, t_n + h]``, with ``t \in [0, 1]`` the scaled time, every degree of freedom
``d`` is

```math
q_d(t) = \sum_{j=0}^{3} a_{jd}\, t^j + \sum_{i=1}^{S} c_{id}\, (t - \tau_i)_+^3 ,
```

an explicit cubic plus `S` truncated cubics. The knots ``\tau_i`` are shared by all degrees of
freedom. The unknowns are ``(a, c, \tau, \tilde p)``, ``4D + SD + S + D`` of them, and the step
solves the discrete Euler–Lagrange system

```math
\nabla_\theta \bigl[ \tilde p \cdot q(1) - \bar p \cdot q(0) - A(\theta) \bigr] = 0 , \qquad
q(0) = \bar q ,
```

with ``\theta = (a, c, \tau)`` and the action ``A = h \int_0^1 L(t_n + h t, q, \dot q / h)\, dt``.
`quadrature` is applied to **each piece between consecutive knots**, so its nodes move with the
knots. On a piecewise cubic, that rule is smooth in ``\tau``; a rule on the whole step is not, and
its free knots fit the quadrature error instead of the dynamics. The residual is the exact
gradient of the discrete action, taken by ForwardDiff, so the motion of the nodes is included.

# The solve

Plain Newton on this system reaches the accurate stationary point from uniform knots only and
collapses from other seeds: the knots merge, move to the ends, or leave the step, all of which are
stationary too. Each step therefore runs four stages:

1. The cubic step (``S = 0``), by Newton from ``a_1 = h \bar v``. Its endpoint ``Q = q(1)`` is
   fixed for stages 2 and 3.
2. For given knots, the action with fixed endpoints ``q(0) = \bar q``, ``q(1) = Q`` is minimised
   over ``(a_1, a_2, c)`` by Newton. This is the reduced action ``f(\tau)``.
3. ``f`` is minimised over the knots, from uniform knots, by Newton with the absolute-eigenvalue
   Hessian and a step cap. A step is accepted on Armijo, or where the Hessian is positive
   definite, when it halves the gradient of ``f``: near the minimum the decrease of ``f`` is below
   its round-off. The knots are parametrised by their gaps,
   ``\tau_i = \sum_{j \le i} e^{\varphi_j} / \sum_{j \le S+1} e^{\varphi_j}``, so they stay ordered
   and inside ``(0, 1)``.
4. Newton on the full system from that point. This also frees ``q(1)``.

For a short step the action with fixed endpoints is a minimum in the trajectory, so stage 3 is
a descent towards the best spline and cannot settle on a collapsed point, which lies level with
the cubic. The step raises an error if stage 4 does not converge.

# Requirements

The problem must be a Lagrangian problem whose Lagrangian `l(t, q, v, params)` accepts
`ForwardDiff.Dual` arguments. Of the problem's functions, the step evaluates `l` and the initial
guess `v̄` only. It does not evaluate `ϑ`, `f` or `ω`, so nothing checks them against `l`.

# Arguments

  - `S`: the number of free interior knots, at least 1.

# Keywords

  - `quadrature`: the rule applied on each piece. Gauss 4 integrates the action of a
    quadratic Lagrangian exactly on a piecewise cubic.
  - `descent_iterations`: the cap on stage 3.
  - `newton_iterations`: the cap on each Newton solve of stages 1, 2 and 4.

# Measured

Harmonic oscillator (``m = 1``, ``k = 0.5``, ``\bar q = 0.5``) and pendulum
(``L = \dot q^2/2 + 0.5 \cos q``, ``\bar q = 1``), ``h = 0.5``, endpoint error after one step:

| ``S`` | oscillator | pendulum |
|:--|--:|--:|
| 0 (the cubic) | ``5.9 \cdot 10^{-10}`` | ``5.8 \cdot 10^{-10}`` |
| 4 | ``9.8 \cdot 10^{-14}`` | ``9.6 \cdot 10^{-14}`` |
| 5 | ``3.3 \cdot 10^{-14}`` | ``3.8 \cdot 10^{-14}`` |
| 6 | ``1.3 \cdot 10^{-14}`` | ``2.1 \cdot 10^{-14}`` |

At these points the Jacobian of the full system has full rank, with row and column equilibration
at ``\sqrt{\varepsilon}``. The scripts are `scripts/free_knot_rank.jl` and
`scripts/free_knot_descent.jl`.

# Note

This is a research method. Each step allocates and runs nested ForwardDiff, so it is much slower
than the network integrators. The integrator does not use the framework's nonlinear solver, so a
`solver` keyword passed to `GeometricIntegrator` has no effect.
"""
struct FreeKnotSpline{QT} <: LODEMethod
    S::Int
    quadrature::QT
    descent_iterations::Int
    newton_iterations::Int

    function FreeKnotSpline(S::Int;
            quadrature = QuadratureRules.GaussLegendreQuadrature(4),
            descent_iterations::Int = 100, newton_iterations::Int = 50)
        S ≥ 1 || throw(ArgumentError("FreeKnotSpline needs at least one knot, got S = $S."))
        new{typeof(quadrature)}(S, quadrature, descent_iterations, newton_iterations)
    end
end

nknots(method::FreeKnotSpline) = method.S

GeometricIntegratorsBase.isexplicit(::Union{FreeKnotSpline, Type{<:FreeKnotSpline}}) = false
GeometricIntegratorsBase.isimplicit(::Union{FreeKnotSpline, Type{<:FreeKnotSpline}}) = true

"""
    FreeKnotSplineCache{ST}

The unknowns of the last step, `x = (vec(a), vec(c), τ, p̃)`, with `a` of size `4 × D` and `c`
of size `S × D`.
"""
struct FreeKnotSplineCache{ST} <: IODEIntegratorCache{ST}
    x::Vector{ST}

    function FreeKnotSplineCache{ST}(D::Int, S::Int) where {ST}
        new{ST}(zeros(ST, 4D + S * D + S + D))
    end
end

GeometricIntegratorsBase.nlsolution(cache::FreeKnotSplineCache) = cache.x

function GeometricIntegratorsBase.Cache{ST}(
        problem::AbstractProblemIODE, method::FreeKnotSpline; kwargs...) where {ST}
    FreeKnotSplineCache{ST}(length(vec(initial_conditions(problem).q)), nknots(method))
end

@inline function GeometricIntegratorsBase.CacheType(
        ST, ::AbstractProblemIODE, ::FreeKnotSpline)
    FreeKnotSplineCache{ST}
end

# ---- the ansatz -------------------------------------------------------------------------------

relu3(z) = max(zero(z), z)^3
drelu3(z) = 3 * max(zero(z), z)^2

# θ = (vec(a), vec(c), τ): a is 4 × D, c is S × D.
spline_a(θ, D) = reshape(view(θ, 1:(4D)), 4, D)
spline_c(θ, S, D) = reshape(view(θ, (4D + 1):(4D + S * D)), S, D)
spline_τ(θ, S, D) = view(θ, (4D + S * D + 1):(4D + S * D + S))

"q(t) of every degree of freedom, at the scaled time `t`."
function spline_q(θ, S, D, t)
    a, c, τ = spline_a(θ, D), spline_c(θ, S, D), spline_τ(θ, S, D)
    [a[1, d] + a[2, d] * t + a[3, d] * t^2 + a[4, d] * t^3 +
     sum(c[i, d] * relu3(t - τ[i]) for i in 1:S; init = zero(eltype(θ))) for d in 1:D]
end

"dq/dt of every degree of freedom, at the scaled time `t`."
function spline_dq(θ, S, D, t)
    a, c, τ = spline_a(θ, D), spline_c(θ, S, D), spline_τ(θ, S, D)
    [a[2, d] + 2a[3, d] * t + 3a[4, d] * t^2 +
     sum(c[i, d] * drelu3(t - τ[i]) for i in 1:S; init = zero(eltype(θ))) for d in 1:D]
end

"""
    spline_action(θ, S, D, ξ, β, l, t̄, h, params)

`h ∫₀¹ L(t̄ + h t, q, q̇/h) dt`, with the rule `(ξ, β)` on `[0, 1]` applied on every piece
between consecutive knots inside `(0, 1)`.
"""
function spline_action(θ, S, D, ξ, β, l, t̄, h, params)
    T = eltype(θ)
    edges = vcat(zero(T), sort!([τ for τ in spline_τ(θ, S, D) if 0 < τ < 1]), one(T))
    s = zero(T)
    for k in 1:(length(edges) - 1)
        lo, w = edges[k], edges[k + 1] - edges[k]
        for j in eachindex(ξ, β)
            t = lo + w * ξ[j]
            s += h * w * β[j] *
                 l(t̄ + h * t, spline_q(θ, S, D, t), spline_dq(θ, S, D, t) / h,
                     params)
        end
    end
    s
end

"""
    spline_residual(x, S, D, ξ, β, l, t̄, h, params, q̄, p̄)

The discrete Euler–Lagrange system of [`FreeKnotSpline`](@ref) at `x = (θ, p̃)`: the gradient
of `p̃ ⋅ q(1) − p̄ ⋅ q(0) − A(θ)` with respect to `θ`, then `q̄ − q(0)`.
"""
function spline_residual(x, S, D, ξ, β, l, t̄, h, params, q̄, p̄)
    n = length(x) - D
    p̃ = x[(n + 1):end]
    G(θ) = sum(p̃ .* spline_q(θ, S, D, one(eltype(θ)))) -
           sum(p̄ .* spline_q(θ, S, D, zero(eltype(θ)))) -
           spline_action(θ, S, D, ξ, β, l, t̄, h, params)
    θ = x[1:n]
    vcat(ForwardDiff.gradient!(similar(θ), G, θ), q̄ .- spline_q(θ, S, D, zero(eltype(θ))))
end

# ---- the reduced action with fixed endpoints -------------------------------------------------

"Knots from gaps: `τᵢ = Σ_{j≤i} e^{φⱼ} / Σ_{j≤S+1} e^{φⱼ}` with `φ_{S+1} = 0`."
function knots_from_gaps(φ)
    e = exp.(vcat(φ, zero(eltype(φ))))
    cumsum(e)[1:(end - 1)] ./ sum(e)
end

"""
    spline_theta(z, φ, S, D, q̄, Q)

`θ` from `z = (a₁, a₂, vec(c))` and the gaps `φ`, with `a₀ = q̄` and `a₃` chosen so that
`q(1) = Q`.
"""
function spline_theta(z, φ, S, D, q̄, Q)
    τ = knots_from_gaps(φ)
    a₁, a₂ = z[1:D], z[(D + 1):(2D)]
    c = reshape(z[(2D + 1):end], S, D)
    a₃ = [Q[d] - q̄[d] - a₁[d] - a₂[d] - sum(c[i, d] * (1 - τ[i])^3 for i in 1:S)
          for d in 1:D]
    T = promote_type(eltype(z), eltype(φ))
    a = permutedims(hcat(T.(q̄), a₁, a₂, a₃))
    vcat(vec(a), vec(c), τ)
end

"""
    dense_newton(F, x, tol, iterations)

Newton on `F = 0` from `x`, with a ForwardDiff Jacobian and a column-pivoted QR. It stops when
`|F|∞ ≤ tol`, or at the round-off floor: once `|F|∞ ≤ 10⁴ tol`, a step that does not halve it
ends the solve. `nothing` if neither happens within `iterations`.
"""
function dense_newton(F, x, tol, iterations)
    r = F(x)
    ρ = maximum(abs, r)
    for _ in 1:iterations
        isfinite(ρ) || return nothing
        ρ ≤ tol && return x
        J = ForwardDiff.jacobian!(similar(r, length(r), length(x)), F, x)
        xnew = x - qr(J, ColumnNorm()) \ r
        rnew = F(xnew)
        ρnew = maximum(abs, rnew)
        ρ ≤ 10_000 * tol && !(ρnew < ρ / 2) && return ρnew < ρ ? xnew : x
        x, r, ρ = xnew, rnew, ρnew
    end
    ρ ≤ tol ? x : nothing
end

"""
    reduced_derivatives(A, z, φ)

The gradient and the Hessian of `f(φ) = min_z A(z, φ)` at `(z, φ)`, by the Schur complement of
the Hessian of `A` in `(z, φ)`. `z` need not be the exact minimiser: the gradient carries the
first-order correction for a residual `∇_z A`.
"""
function reduced_derivatives(A, z, φ)
    nz = length(z)
    y = vcat(z, φ)
    Ay(y) = A(y[1:nz], y[(nz + 1):end])
    g = ForwardDiff.gradient!(similar(y), Ay, y)
    H = ForwardDiff.hessian!(similar(y, length(y), length(y)), Ay, y)
    Hzz, Hzφ = H[1:nz, 1:nz], H[1:nz, (nz + 1):end]
    Hzz⁻¹Hzφ = Hzz \ Hzφ
    gr = g[(nz + 1):end] - Hzz⁻¹Hzφ' * g[1:nz]
    Hr = Symmetric(H[(nz + 1):end, (nz + 1):end] - Hzφ' * Hzz⁻¹Hzφ)
    gr, Hr
end

"""
    descend_knots(A, z, φ, iterations, newton_iterations, tol)

Minimise `f(φ) = min_z A(z, φ)` by Newton on `φ` with the reduced Hessian, made positive by the
absolute value of its eigenvalues, and a step cap of 1, for at most `iterations` steps. The inner
minimum is warm-started from the last `z`.

A trial step is accepted on either of two tests:

  - **Armijo on `f`**, while `f` resolves the decrease. It stops doing so near the minimum: `f`
    exceeds the action of the exact solution by about the squared error, so the gain of the
    knots over the cubic falls below the round-off of `f` once the step is accurate.
  - **The reduced gradient halves**, where the reduced Hessian is positive definite. The gradient
    comes from automatic differentiation and resolves the minimum where `f` does not.

The descent ends when neither test accepts a step of any length.
"""
function descend_knots(A, z, φ, iterations, newton_iterations, tol)
    inner(z, φ) = dense_newton(
        y -> ForwardDiff.gradient!(similar(y), w -> A(w, φ), y), z, tol, newton_iterations)
    z = inner(z, φ)
    z === nothing && return nothing
    fφ = A(z, φ)
    gr, Hr = reduced_derivatives(A, z, φ)
    for _ in 1:iterations
        λ, V = eigen(Hr)
        convex = minimum(λ) > 0
        d = -V * ((V' * gr) ./ max.(abs.(λ), sqrt(eps(eltype(λ))) * maximum(abs, λ)))
        s = min(one(eltype(d)), inv(maximum(abs, d)))
        accepted = false
        while s * maximum(abs, d) > eps(eltype(d))
            φtrial = φ + s * d
            ztrial = inner(z, φtrial)
            if ztrial !== nothing
                ftrial = A(ztrial, φtrial)
                grtrial, Hrtrial = reduced_derivatives(A, ztrial, φtrial)
                if ftrial < fφ + s * dot(gr, d) / 10_000 ||
                   (convex && norm(grtrial) < norm(gr) / 2)
                    z, φ, fφ, gr, Hr = ztrial, φtrial, ftrial, grtrial, Hrtrial
                    accepted = true
                    break
                end
            end
            s /= 2
        end
        accepted || break
    end
    z, φ
end

# ---- one step -------------------------------------------------------------------------------

function GeometricIntegratorsBase.integrate_step!(
        sol, history, params, int::GeometricIntegrator{
            <:FreeKnotSpline, <:AbstractProblemIODE})
    local m = method(int)
    local S = nknots(m)
    local h = timestep(int)
    local t̄ = sol.t - h
    local q̄, p̄ = copy(vec(sol.q)), copy(vec(sol.p))
    local D = length(q̄)
    local T = eltype(q̄)
    local ξ = T.(QuadratureRules.nodes(m.quadrature))
    local β = T.(QuadratureRules.weights(m.quadrature))
    local l = equations(int).l
    local tol = 16 * eps(T) * (1 + maximum(abs, q̄) + maximum(abs, p̄))

    # 1. the cubic step, from the velocity guess of the problem
    v̄ = zeros(T, D)
    GeometricBase.initialguess(problem(int)).v(v̄, t̄, sol.q, sol.p, params)
    cubic = dense_newton(x -> spline_residual(x, 0, D, ξ, β, l, t̄, h, params, q̄, p̄),
        vcat(vec(permutedims(hcat(q̄, h .* v̄, zeros(T, D, 2)))), p̄), tol,
        m.newton_iterations)
    cubic === nothing && error("FreeKnotSpline: the cubic step did not converge at t = $t̄.")
    Q = spline_q(cubic[1:(4D)], 0, D, one(T))
    p̃ = cubic[(4D + 1):end]

    # 2. and 3. descent on the reduced action with q(1) = Q, from uniform knots
    A(z, φ) = spline_action(spline_theta(z, φ, S, D, q̄, Q), S, D, ξ, β, l, t̄, h, params)
    z₀ = vcat(cubic[2:4:(4D)], cubic[3:4:(4D)], zeros(T, S * D))
    descent = descend_knots(
        A, z₀, zeros(T, S), m.descent_iterations, m.newton_iterations, tol)
    descent === nothing &&
        error("FreeKnotSpline: the reduced action has no minimum in z at t = $t̄.")
    z, φ = descent

    # 4. Newton on the full system
    x = dense_newton(x -> spline_residual(x, S, D, ξ, β, l, t̄, h, params, q̄, p̄),
        vcat(spline_theta(z, φ, S, D, q̄, Q), p̃), tol, m.newton_iterations)
    x === nothing &&
        error("FreeKnotSpline: the full Newton solve did not converge at t = $t̄.")

    copyto!(nlsolution(int), x)
    sol.q .= reshape(spline_q(x[1:(end - D)], S, D, one(T)), size(sol.q))
    sol.p .= reshape(x[(end - D + 1):end], size(sol.p))
end
