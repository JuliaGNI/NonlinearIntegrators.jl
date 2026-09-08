# The rank of the Newton Jacobian of a ShallowNet integrator, at a converged point.
#
#     julia --startup-file=no --project=scripts scripts/newton_jacobian_rank.jl
#
# This is the check behind issue #98 and behind the `default_options` docstring in
# `src/nvi/network_integrator_core.jl`. The claim it establishes is *not* "the Jacobian is
# ill-conditioned" but the stronger and quite different one that it is **exactly rank
# deficient**: the spectrum has a gap of many orders rather than a decay, so the deficiency is a
# property of the ansatz and no tolerance choice makes it go away.
#
# Two mechanisms stack, and the script reports the evidence for each.
#
#   1. **Positive homogeneity of the activation.** `relu_k(k)` satisfies `σ(λx) = λᵏσ(x)`, so
#      rescaling one neuron as `(wᵢ, bᵢ, cᵢ) ↦ (λwᵢ, λbᵢ, λ⁻ᵏcᵢ)` leaves the trajectory
#      pointwise unchanged. The residual depends on the parameters only through the trajectory,
#      so its Jacobian annihilates the `S` generators `(δwᵢ, δbᵢ, δcᵢ) = (wᵢ, bᵢ, −k cᵢ)`. This
#      half is intrinsic and permanent.
#   2. **Collapse onto a polynomial.** Where every pre-activation keeps one sign across the
#      element, `max(0, z)ᵏ = zᵏ` identically and the whole ansatz is a degree-`k` polynomial —
#      `3S + 1` parameters onto `k + 2` coefficients.
#
# What is measured here is their *combined* effect, the rank, and not the split between them:
# separating the two needs the pre-activation signs across the element, which this does not
# collect. What the `S`-dependence does show is an **offset**. Mechanism 1 alone predicts a
# deficiency of exactly `S` — one generator per neuron — i.e. 4, 5, 6. Measured on the converged
# `OGA1d` rows it is 8, 9, 10: the slope is 1, so mechanism 1 accounts for all of the growth, and
# a constant 4 sits on top of it. Four is what a cubic collapse would contribute, `k + 1`
# coefficients of the polynomial the ansatz degenerates to.
#
# **Why ForwardDiff and not finite differences.** The null directions here are annihilated to
# `σ_{r+1}/σ₁ ≈ 5e-18` — below `eps(Float64)` relative to the largest singular value. A
# central-difference Jacobian of a function whose argument has `‖x‖ ≈ 1.6e4` cannot resolve
# anything of that size: its own truncation and cancellation error puts a floor many orders
# above, which makes an exact null space look merely "suppressed" and turns the measurement into
# the ill-conditioning reading this script exists to refute. (The floor itself is not measured
# here — the point is only that it is far above 5e-18, which the column reports.) The Jacobian
# read back is the solver's own, and `NewtonSolver` builds it with `JacobianAutodiff`, i.e.
# ForwardDiff through the symbolic basis with no fallback.
#
# **Why a converged point.** The constraint row annihilates the scaling fibre unconditionally,
# but the second-derivative rows only do so at a critical point of the discrete action, so a rank
# measured where Newton did not converge means nothing. The table shows this directly: the rows
# marked `NOT CONVERGED` report ranks anywhere from 4 to full rank, against a converged 5 of 13 at
# the same `S`. Their gaps are not evidence either way — measured `Inf, 4.6, 1.5, 1.9, 2.6`, where
# the `Inf` is structural (a full-rank row has no dropped singular value to compare against) — but
# none of them is the twelve-to-fourteen the converged rows show. Every row therefore carries the
# residual it was measured at, and only the unmarked ones are evidence.

using LinearAlgebra
using Printf
using QuadratureRules
using SimpleSolvers
using SimpleSolvers: jacobianmatrix, cache as solvercache
using GeometricIntegratorsBase
using GeometricProblems.HarmonicOscillator
using NonlinearIntegrators

const T = Float64

# The tolerance the rank is reported at. Deliberately the same one
# `SimpleSolvers.rank_tolerance` defaults to, so that the number here is the number the
# integrator's own `PivotedQR` acts on.
const RTOL = sqrt(eps(T))

# Defined here rather than imported: `relu_k` lives in `test/testsetup.jl`, and a script in
# `scripts/` that reaches into the test suite would break the moment the suite is reorganised.
# This is the same one line, and it is the activation the whole of #98 is about.
relu_k(k::Int = 3) = x -> max(zero(x), x)^k

# Likewise a test-suite shorthand rather than a package export.
gauss(::Type{T}, R = 8) where {T} = QuadratureRules.GaussLegendreQuadrature(T, R)

"The harmonic-oscillator LODE the unit tests drive, at the element these methods integrate over."
function ho_problem(::Type{T}) where {T}
    HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)];
        timespan = (T(0.0), T(1.0)), timestep = T(0.5))
end

"""
Integrate two steps and hand back the solver, so that its last Jacobian and its residual can be
read off together. Two steps and not one: the first is driven from the initial condition, and it
is the second that runs from a state the method itself produced.
"""
function converged_solver(S, seed)
    basis = ShallowNetBasis{T}(relu_k(3), S)
    method = ShallowNet(basis, gauss(T, 8);
        show_status = false, bias_interval = [-T(pi), T(pi)], dict_amount = 400,
        initial_guess_method = seed)
    prob = ho_problem(T)
    int = GeometricIntegrator(prob, method)
    integrate(int)
    GeometricIntegratorsBase.solver(int), GeometricIntegratorsBase.solverstate(int)
end

"""
Row-equilibrate `J` — scale each row to unit ∞-norm — before taking its singular values.

Without this the spectrum reports the row scaling of the residual as much as the rank: the
constraint row and the second-derivative rows of this system differ by orders of magnitude in
norm, and that spread lands in the singular values on top of the deficiency being measured.
A zero row is left alone rather than divided by zero.
"""
function row_equilibrate(J)
    E = copy(J)
    for i in axes(E, 1)
        s = maximum(abs, view(E, i, :))
        s > 0 && (view(E, i, :) ./= s)
    end
    E
end

println("Julia $(VERSION), rank tolerance = sqrt(eps) = ", RTOL)
println("harmonic oscillator, D = 1, gauss(T, 8), relu_k(3), dict_amount = 400, ",
    "bias_interval = [-π, π]")
println()
@printf("%3s %22s %7s %6s %6s %11s %11s %11s\n",
    "S", "seed", "unknown", "rank", "gap", "residual", "σ_r/σ_1", "σ_r+1/σ_1")

# Named explicitly: all three seeds are `OGA{...}` aliases, so `nameof(typeof(seed))` prints
# `OGA` for every one of them and the table cannot be read.
const SEEDS = (("OGA1d", OGA1d()),
    ("OGA1dNormalized", OGA1dNormalized()),
    ("OGA1dStable", OGA1dStable()))

for S in (4, 5, 6), (seed_name, seed) in SEEDS

    s, st = converged_solver(S, seed)
    J = jacobianmatrix(solvercache(s))
    σ = svdvals(row_equilibrate(J))
    n = size(J, 2)
    r = count(>(RTOL * σ[1]), σ)
    resid = SimpleSolvers.status(s, st).rfₐ

    # the gap: how many orders separate the last kept singular value from the first dropped one
    gap = r < n ? log10(σ[r] / σ[r + 1]) : Inf

    @printf("%3d %22s %7d %6d %6.1f %11.2e %11.2e %11.2e%s\n",
        S, seed_name, n, r, gap, resid, σ[r] / σ[1],
        r < n ? σ[r + 1] / σ[1] : 0.0,
        resid < 1e-8 ? "" : "   NOT CONVERGED — not evidence")
end

println()
println("`unknown` is D*(3S+1); `rank` counts σ > sqrt(eps)*σ₁; `gap` is log₁₀(σ_r/σ_{r+1}).")
println("The done condition of Tasks/'Fix the singular Newton Jacobian…' is rank == unknown.")
