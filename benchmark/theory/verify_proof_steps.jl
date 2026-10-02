# Numerical verification of every step of the proofs in `relu_cgvi_equivalence.md`.
#
#   julia benchmark/theory/verify_proof_steps.jl
#   julia --project=benchmark benchmark/theory/verify_proof_steps.jl      (same thing)
#
# Needs only the standard library. Each check prints PASS/FAIL with the measured quantity and
# its bound; the IDs (V1.1, V2.3, …) are the ones quoted in the markdown next to the step they
# verify. The script exits with status 1 if any check fails.

include(joinpath(@__DIR__, "theory_common.jl"))

Random.seed!(20260922)

# ===========================================================================
section("§1  The one-step equations")
# ===========================================================================

# V1.1 — the residual rows are −∇θ of the augmented functional
#        𝒢(θ, pₙ₊₁) = A_h(q_θ) − pₙ₊₁ q_θ(1) + pₙ q_θ(0)
let k = 3, S = 5, h = 0.7, quad = QRule01(4), prob = TOY_PEND
    θ = pack(randn(S), [1.0, -1.3, 0.8, -0.6, 1.1], [0.2, 1.1, -0.35, 0.4, 0.9])
    p0, p1 = 0.3, -0.2
    𝒢(θ) = begin
        q, dq = nn_eval(θ, quad.c, k)
        discrete_action(q, dq, h, quad, prob) - p1 * nn_eval(θ, [1.0], k)[1][1] +
        p0 * nn_eval(θ, [0.0], k)[1][1]
    end
    g = vec(fd_jacobian(θ -> [𝒢(θ)], θ))
    r = nn_residual(vcat(θ, p1), 0.0, p0, h, k, quad, prob)[1:(3S)]
    vcheck("V1.1a", "network: residual rows = −∇θ𝒢 (rel.)", norm(r .+ g) / norm(g), 1e-6)

    phi = cgvi_basis(3)
    X = randn(4)
    𝒢lin(X) = begin
        P, D = phi(quad.c)
        discrete_action(P * X, D * X, h, quad, prob) - p1 * dot(vec(phi([1.0])[1]), X) +
        p0 * dot(vec(phi([0.0])[1]), X)
    end
    gl = vec(fd_jacobian(X -> [𝒢lin(X)], X))
    rl = linear_residual(vcat(X, p1), 0.0, p0, h, phi, quad, prob)[1:4]
    vcheck("V1.1b", "CGVI: residual rows = −∇X𝒢 (rel.)", norm(rl .+ gl) / norm(gl), 1e-6)
end

# V1.2 — type II (given qₙ, pₙ) and type I (given qₙ, qₙ₊₁) describe the same step, and the
#        momenta are the discrete Legendre transforms of L_d.
let h = 0.8, quad = QRule01(4), prob = TOY_PEND, phi = cgvi_basis(3), q0 = 0.5, p0 = 0.1
    s2 = linear_step(q0, p0, h, phi, quad, prob)
    s1 = typeI_step(q0, s2.q1, h, phi, quad, prob)
    vcheck("V1.2a", "type I with qₙ₊₁ from type II returns pₙ", abs(s1.p_a - p0), 1e-10)
    vcheck("V1.2b", "type I with qₙ₊₁ from type II returns pₙ₊₁", abs(s1.p_b - s2.p1), 1e-10)
    e = 1e-6
    dLa = (typeI_step(q0 + e, s2.q1, h, phi, quad, prob).Ld -
           typeI_step(q0 - e, s2.q1, h, phi, quad, prob).Ld) / (2 * e)
    dLb = (typeI_step(q0, s2.q1 + e, h, phi, quad, prob).Ld -
           typeI_step(q0, s2.q1 - e, h, phi, quad, prob).Ld) / (2 * e)
    vcheck("V1.2c", "pₙ = −∂L_d/∂qₙ", abs(p0 + dLa), 1e-7)
    vcheck("V1.2d", "pₙ₊₁ = ∂L_d/∂qₙ₊₁", abs(s2.p1 - dLb), 1e-7)
end

# ===========================================================================
section("§2  Lemmas on ReLUᵏ neurons")
# ===========================================================================

const TT = collect(range(0, 1; length = 201))
const TT_OFF = collect(range(0.0013, 0.9987; length = 157))   # avoids landing on a kink

# V2.1 — restriction of a neuron with its kink outside [0,1]
let err = 0.0
    for k in 2:5
        w, b = kinkfree_neurons(6)
        for i in eachindex(w)
            err = max(err, maximum(abs.(σk.(w[i] .* TT .+ b[i], k) .- (w[i] .* TT .+ b[i]) .^ k)))
        end
        # an inactive neuron (kink outside, sign negative on [0,1]) is identically 0
        err = max(err, maximum(abs.(σk.(-TT .- 0.2, k))))
    end
    vcheck("V2.1", "kink ∉ [0,1]: σ(wτ+b) = (wτ+b)ᵏ (active) or 0 (inactive)", err, 1e-14)
end

# V2.2 — closed-form tangent vectors (q and q′) against finite differences
let err = 0.0
    for k in 2:4
        θ = pack(randn(5), [1.0, -1.3, 0.8, -0.6, 1.1], [0.2, 1.1, -0.35, 0.4, 0.9])
        TQ, TV = nn_tangent(θ, TT_OFF, k)
        FQ = fd_jacobian(θ -> nn_eval(θ, TT_OFF, k)[1], θ)
        FV = fd_jacobian(θ -> nn_eval(θ, TT_OFF, k)[2], θ)
        err = max(err, norm(TQ - FQ) / norm(FQ), norm(TV - FV) / norm(FV))
    end
    vcheck("V2.2", "∂q/∂(a,w,b) and ∂q′/∂(a,w,b) closed forms (rel. to FD)", err, 1e-6)
end

# V2.3 — the apolar pairing:  B(f,(τ−z)ᵏ) = k! f(z),  B(f,(τ−z)ᵏ⁻¹) = −(k−1)! f′(z),
#        and B is non-degenerate on P_k
let e1 = 0.0, e2 = 0.0, rk_ok = true
    for k in 2:6, _ in 1:5
        f = randn(k + 1)
        z = 3 * randn()
        e1 = max(e1, abs(apolar_pairing(f, shift_power_coeffs(z, k, k), k) -
                         factorial(k) * polyval(f, z)) / (1 + abs(factorial(k) * polyval(f, z))))
        e2 = max(e2, abs(apolar_pairing(f, shift_power_coeffs(z, k - 1, k), k) +
                         factorial(k - 1) * polyder_val(f, z)) /
                     (1 + abs(factorial(k - 1) * polyder_val(f, z))))
        G = [apolar_pairing(Matrix(1.0I, k + 1, k + 1)[:, i], Matrix(1.0I, k + 1, k + 1)[:, j], k)
             for i in 1:(k + 1), j in 1:(k + 1)]
        rk_ok &= numrank(G) == k + 1
    end
    vcheck("V2.3a", "B(f,(τ−z)ᵏ) = k!·f(z)", e1, 1e-10)
    vcheck("V2.3b", "B(f,(τ−z)ᵏ⁻¹) = −(k−1)!·f′(z)", e2, 1e-10)
    vcheck("V2.3c", "Gram matrix of B on monomials has full rank k+1", rk_ok ? 0.0 : 1.0, 0.5)
end

# V2.4 — spanning lemma: rank span{(τ−zⱼ)ᵏ} = min(d, k+1),
#        rank span{(τ−zⱼ)ᵏ, (τ−zⱼ)ᵏ⁻¹} = min(2d, k+1); Vandermonde determinant formula.
let bad1 = 0, bad2 = 0, detrel = 0.0
    for k in 2:6, d in 1:(k + 2)
        z = sort(-0.2 .- 2 .* rand(d))
        C1 = hcat([shift_power_coeffs(zj, k, k) for zj in z]...)
        C2 = hcat(C1, hcat([shift_power_coeffs(zj, k - 1, k) for zj in z]...))
        numrank(C1; rtol = 1e-12) == min(d, k + 1) || (bad1 += 1)
        numrank(C2; rtol = 1e-12) == min(2d, k + 1) || (bad2 += 1)
        if d == k + 1
            pred = prod(binomial(k, j) for j in 0:k) *
                   prod(abs(z[i] - z[l]) for i in 1:d for l in (i + 1):d)
            detrel = max(detrel, abs(abs(det(C1)) - pred) / pred)
        end
    end
    vcheck_eq("V2.4a", "rank{(τ−zⱼ)ᵏ} = min(d,k+1)   (#violations over k=2..6, d=1..k+2)", bad1, 0)
    vcheck_eq("V2.4b", "rank{(τ−zⱼ)ᵏ,(τ−zⱼ)ᵏ⁻¹} = min(2d,k+1)   (#violations)", bad2, 0)
    vcheck("V2.4c", "|det C| = Πⱼ C(k,j) · Π_{i<l}|zᵢ−z_l|  (rel.)", detrel, 1e-8)
    k = 3
    Ccoal = hcat([shift_power_coeffs(z, k, k) for z in (-0.7, -0.7, -1.4, -2.0)]...)
    vcheck_eq("V2.4d", "two coincident kinks: rank drops to 3 < k+1", numrank(Ccoal; rtol = 1e-12), 3)
end

# V2.5 — Euler homogeneity: ∂q/∂wᵢ = (k aᵢ ∂q/∂aᵢ − bᵢ ∂q/∂bᵢ)/wᵢ, hence the same for residual rows
let k = 3, S = 5, h = 0.7, quad = QRule01(4), prob = TOY_PEND
    θ = pack(randn(S), [1.0, -1.3, 0.8, -0.6, 1.1], [0.2, 1.1, -0.35, 0.4, 0.9])
    a, w, b = unpack(θ)
    TQ, _ = nn_tangent(θ, TT, k)
    lhs = TQ[:, (S + 1):(2S)]
    rhs = (k .* a' .* TQ[:, 1:S] .- b' .* TQ[:, (2S + 1):(3S)]) ./ w'
    vcheck("V2.5a", "tangent identity ∂_w = (k a ∂_a − b ∂_b)/w", norm(lhs - rhs) / norm(lhs), 1e-12)
    r = nn_residual(vcat(θ, 0.2), 0.4, 0.1, h, k, quad, prob)
    ra, rw, rb = r[1:S], r[(S + 1):(2S)], r[(2S + 1):(3S)]
    vcheck("V2.5b", "residual identity r_w = (k a r_a − b r_b)/w",
        norm(rw .- (k .* a .* ra .- b .* rb) ./ w) / max(norm(rw), 1e-300), 1e-10)
end

# V2.6 — gauge invariances of q_θ: positive rescaling of a neuron and permutations
let k = 3, S = 4
    θ = pack(randn(S), [1.0, -1.3, 0.8, -0.6], [0.2, 1.1, -0.35, 0.4])
    a, w, b = unpack(θ)
    λ = [0.3, 2.0, 5.0, 0.7]
    θλ = pack(a ./ λ .^ k, λ .* w, λ .* b)
    vcheck("V2.6a", "(aᵢ,wᵢ,bᵢ) → (aᵢ/λᵢᵏ, λᵢwᵢ, λᵢbᵢ), λᵢ>0 leaves q_θ unchanged",
        maximum(abs.(nn_eval(θλ, TT, k)[1] .- nn_eval(θ, TT, k)[1])), 1e-13)
    p = [3, 1, 4, 2]
    vcheck("V2.6b", "permuting neurons leaves q_θ unchanged",
        maximum(abs.(nn_eval(pack(a[p], w[p], b[p]), TT, k)[1] .- nn_eval(θ, TT, k)[1])), 1e-14)
end

# ===========================================================================
section("§3  Theorem 1 — kink-free networks are CGVI(P_k)")
# ===========================================================================

# V3.1 — q_θ ∈ P_k, T_θM ⊂ P_k, and T_θM = P_k iff #distinct kinks d satisfies 2d ≥ k+1
let maxproj = 0.0, bad = 0
    for k in 2:4, S in (k + 1, 6, 8)
        w, b = kinkfree_neurons(S)
        θ = pack(randn(S), w, b)
        Mk = monomial_basis(k)(TT)[1]
        TQ, _ = nn_tangent(θ, TT, k)
        maxproj = max(maxproj, proj_residual(nn_eval(θ, TT, k)[1][:, :], Mk), proj_residual(TQ, Mk))
        numrank(TQ; rtol = 1e-9) == k + 1 || (bad += 1)
    end
    vcheck("V3.1a", "kink-free: q_θ ∈ P_k and ∂q/∂θ ∈ P_k (projection residual)", maxproj, 1e-11)
    vcheck_eq("V3.1b", "kink-free, S ≥ k+1 distinct kinks: rank T_θM = k+1 (#violations)", bad, 0)
    bad = 0
    for k in 2:5, d in 1:(k + 1)
        # 2 neurons per kink cluster with different |w| (same kink), all outside [0,1]
        z = -0.4 .- 0.5 .* (0:(d - 1))
        w = repeat([1.0, 2.0], d)
        zz = repeat(collect(z); inner = 2)
        θ = pack(randn(2d), w, -zz .* w)
        numrank(nn_tangent(θ, TT, k)[1]; rtol = 1e-9) == min(2d, k + 1) || (bad += 1)
    end
    vcheck_eq("V3.1c", "d distinct kinks (clustered neurons): rank T_θM = min(2d,k+1) (#viol.)", bad, 0)
end

"""CGVI(P_s, quad) step and a kink-free network θ representing its polynomial."""
function cgvi_and_network(q0, p0, h, s, k, S, quad, prob)
    phi = cgvi_basis(s)
    st = linear_step(q0, p0, h, phi, quad, prob)
    qpoly = phi(TT)[1] * st.X
    w, b = kinkfree_neurons(S)
    θ = fit_output_weights(TT, qpoly, w, b, k)
    st, θ, maximum(abs.(nn_eval(θ, TT, k)[1] .- qpoly))
end

# V3.2 / V3.3 / V3.4 — both directions of Theorem 1 and the Jacobian structure
let quad = QRule01(4), k = 3, S = 4
    rmax = 0.0
    fitmax = 0.0
    dmax = 0.0
    kinks_ok = true
    rank_bad = 0
    gauge_max = 0.0
    for prob in TOY_PROBLEMS, h in (0.1, 1.0, 5.0)
        q0, p0 = 0.5, 0.1
        st, θ, fiterr = cgvi_and_network(q0, p0, h, 3, k, S, quad, prob)
        fitmax = max(fitmax, fiterr)
        F = x -> nn_residual(x, q0, p0, h, k, quad, prob)
        xstar = vcat(θ, st.p1)
        rmax = max(rmax, norm(F(xstar), Inf))
        # ⇒ : solve the network equations from a perturbed start
        sol = nn_step(q0, p0, h, k, quad, prob, xstar .+ 1e-3 .* randn(length(xstar)))
        kinks_ok &= sol.converged && is_kinkfree(sol.θ, k)
        dmax = max(dmax, abs(sol.q1 - st.q1), abs(sol.p1 - st.p1))
        # Jacobian rank k+2 and the null space is pure gauge
        J = fd_jacobian(F, sol.x)
        numrank(J; rtol = 1e-7) == k + 2 || (rank_bad += 1)
        N = svd(J).V[:, (k + 3):end]                    # 3S+1 − (k+2) null directions
        TQ, _ = nn_tangent(sol.θ, TT, k)
        gauge_max = max(gauge_max, norm(TQ * N[1:(3S), :]) / norm(TQ), norm(N[end, :]))
    end
    vcheck("V3.2a", "CGVI polynomial is represented by 4 kink-free neurons (max fit error)", fitmax, 1e-12)
    vcheck("V3.2b", "⇐ : that θ solves the network equations (max |residual|)", rmax, 1e-10)
    vcheck("V3.3a", "⇒ : LM from a perturbed start converges with all kinks outside [0,1]",
        kinks_ok ? 0.0 : 1.0, 0.5)
    vcheck("V3.3b", "⇒ : and reproduces (qₙ₊₁, pₙ₊₁) of CGVI", dmax, 1e-10)
    vcheck_eq("V3.4a", "rank of the network Jacobian = k+2 at every solution (#violations)", rank_bad, 0)
    vcheck("V3.4b", "null directions of the Jacobian leave q_θ and pₙ₊₁ unchanged", gauge_max, 1e-4)
end

# V3.5 — whole trajectories coincide
let quad = QRule01(4), k = 3, S = 4, h = 0.5, N = 20
    dmax = 0.0
    for prob in TOY_PROBLEMS
        q0, p0 = 0.5, 0.0
        qc, pc = linear_trajectory(q0, p0, h, N, cgvi_basis(3), quad, prob)
        # start from the kink-free fit of the straight line q₀ + h p₀ τ (not from the answer)
        w0, b0 = kinkfree_neurons(S)
        θ0 = fit_output_weights(TT, q0 .+ h * p0 .* TT, w0, b0, k)
        qn, pn, thetas = nn_trajectory(q0, p0, h, N, k, quad, prob, θ0)
        all(θ -> is_kinkfree(θ, k), thetas) || (dmax = Inf)
        dmax = max(dmax, maximum(abs.(qn .- qc)), maximum(abs.(pn .- pc)))
    end
    vcheck("V3.5", "20-step trajectories: network (k=3,S=4) ≡ CGVI(P₃,R=4)", dmax, 1e-9)
end

# V3.6 / V3.7 — corollaries: ReLU² ≡ CGVI(P₂); more neurons do not change the method
let quad = QRule01(4)
    for (id, k, S, s) in (("V3.6", 2, 4, 2), ("V3.7a", 3, 6, 3), ("V3.7b", 3, 8, 3))
        dmax = 0.0
        rmax = 0.0
        for prob in TOY_PROBLEMS, h in (0.5, 2.0)
            q0, p0 = 0.5, 0.1
            st, θ, _ = cgvi_and_network(q0, p0, h, s, k, S, quad, prob)
            F = x -> nn_residual(x, q0, p0, h, k, quad, prob)
            rmax = max(rmax, norm(F(vcat(θ, st.p1)), Inf))
            sol = nn_step(q0, p0, h, k, quad, prob, vcat(θ, st.p1) .+ 1e-3 .* randn(3S + 1))
            is_kinkfree(sol.θ, k) || (dmax = Inf)
            dmax = max(dmax, abs(sol.q1 - st.q1), abs(sol.p1 - st.p1))
        end
        vcheck(id * "a", "ReLU^$k, S=$S: CGVI(P$s) solution solves the network equations", rmax, 1e-10)
        vcheck(id * "b", "ReLU^$k, S=$S: network solve ≡ CGVI(P$s)", dmax, 1e-10)
    end
    # and ReLU² is *not* CGVI(P₃)
    st2 = linear_step(0.5, 0.1, 1.0, cgvi_basis(2), quad, TOY_PEND)
    st3 = linear_step(0.5, 0.1, 1.0, cgvi_basis(3), quad, TOY_PEND)
    vcheck_ge("V3.6c", "sanity: CGVI(P₂) and CGVI(P₃) differ (so V3.6 is not vacuous)",
        abs(st2.q1 - st3.q1), 1e-6)
end

# V3.8 — corollary: the kink-free network step is symplectic (det of the step map = 1)
let quad = QRule01(4), k = 3, S = 4, h = 1.0, q0 = 0.5, p0 = 0.1
    worst = 0.0
    for prob in TOY_PROBLEMS
        st, θ, _ = cgvi_and_network(q0, p0, h, 3, k, S, quad, prob)
        worst = max(worst, abs(step_map_det(q0, p0, h, k, quad, prob, vcat(θ, st.p1)) - 1))
    end
    vcheck("V3.8", "kink-free network step map is area preserving: |det − 1|", worst, 1e-7)
end

# ===========================================================================
section("§4  Other stationary points of the network equations")
# ===========================================================================

"""Bisection for a sign change of g on [lo, hi]."""
function bisect(g, lo, hi; iters = 80)
    glo = g(lo)
    for _ in 1:iters
        mid = (lo + hi) / 2
        gm = g(mid)
        if sign(gm) == sign(glo)
            lo, glo = mid, gm
        else
            hi = mid
        end
    end
    (lo + hi) / 2
end

# V4.1 — coalesced kinks: a genuine solution of the network equations that is NOT CGVI
let quad = QRule01(4), k = 3, q0 = 0.5, p0 = 0.0
    for prob in TOY_PROBLEMS
        h = 1.0
        # one neuron, w = −1 fixed, kink z = b > 1; unknowns (a, p₁) from the a-row and q(0)=qₙ
        function solve_ap(z)
            F = x -> (r = nn_residual([x[1], -1.0, z, x[2]], q0, p0, h, k, quad, prob); [r[1], r[4]])
            s = newton_solve(F, [q0 / z^k, p0])
            s.x, nn_residual([s.x[1], -1.0, z, s.x[2]], q0, p0, h, k, quad, prob)[3]
        end
        g(z) = try
            solve_ap(z)[2]
        catch
            NaN
        end
        zs = 1 .+ 10 .^ range(-2, log10(50); length = 400)
        vals = [g(z) for z in zs]
        idx = findfirst(i -> isfinite(vals[i]) && isfinite(vals[i + 1]) &&
                                  sign(vals[i]) != sign(vals[i + 1]), 1:(length(zs) - 1))
        if idx === nothing
            vcheck("V4.1", "$(prob.name): no coalesced-kink solution found in scan", 1.0, 0.5)
            continue
        end
        zstar = bisect(g, zs[idx], zs[idx + 1])
        (a, p1), _ = solve_ap(zstar)
        # embed into S = 4 neurons with different weights but the same kink
        wv = [-1.0, -0.5, -2.0, -1.5]
        cfac = [0.4, 0.3, 0.2, 0.1]                       # split of the amplitude
        a4 = cfac .* a ./ (abs.(wv) .^ k)
        θ4 = pack(a4, wv, -zstar .* wv)
        r4 = norm(nn_residual(vcat(θ4, p1), q0, p0, h, k, quad, prob), Inf)
        st = linear_step(q0, p0, h, cgvi_basis(3), quad, prob)
        q1 = nn_eval(θ4, [1.0], k)[1][1]
        @printf("      %s: coalesced kink z* = %.6f, q₁ = %.8f vs CGVI %.8f\n", prob.name, zstar, q1, st.q1)
        vcheck("V4.1a", "$(prob.name): 4 neurons with one common kink solve the equations", r4, 1e-9)
        vcheck_eq("V4.1b", "$(prob.name): their tangent space has rank 2 (< k+1 = 4)",
            numrank(nn_tangent(θ4, TT, k)[1]; rtol = 1e-9), 2)
        vcheck_ge("V4.1c", "$(prob.name): and the step differs from CGVI (|Δqₙ₊₁|)", abs(q1 - st.q1), 1e-5)
    end
end

# V4.2 — a zero-amplitude neuron with an interior kink at a root of ρ(z) leaves q = CGVI
let quad = QRule01(4), k = 3, S = 4, q0 = 0.5, p0 = 0.0, h = 1.0
    for prob in TOY_PROBLEMS
        st, θ, _ = cgvi_and_network(q0, p0, h, 3, k, S, quad, prob)
        a, w, b = unpack(θ)
        # ρ(z) = residual functional of the CGVI solution in the direction (τ − z)₊ᵏ
        ρ(z) = nn_residual(vcat(pack(vcat(a, 0.0), vcat(w, 1.0), vcat(b, -z)), st.p1),
            q0, p0, h, k, quad, prob)[S + 1]
        zs = collect(range(0.02, 0.98; length = 481))
        vals = ρ.(zs)
        idx = findfirst(i -> isfinite(vals[i]) && isfinite(vals[i + 1]) &&
                                  sign(vals[i]) != sign(vals[i + 1]), 1:(length(zs) - 1))
        if idx === nothing
            vcheck("V4.2", "$(prob.name): ρ has no root in (0,1)", 1.0, 0.5)
            continue
        end
        zstar = bisect(ρ, zs[idx], zs[idx + 1])
        θ5 = pack(vcat(a, 0.0), vcat(w, 1.0), vcat(b, -zstar))
        F = x -> nn_residual(x, q0, p0, h, k, quad, prob)
        @printf("      %s: first root of ρ at z* = %.6f\n", prob.name, zstar)
        vcheck("V4.2a", "$(prob.name): CGVI + zero-amplitude neuron at z* solves the equations",
            norm(F(vcat(θ5, st.p1)), Inf), 1e-9)
        θbad = pack(vcat(a, 0.0), vcat(w, 1.0), vcat(b, -(zstar + 0.05)))
        vcheck_ge("V4.2b", "$(prob.name): … but not at z* + 0.05 (|residual|)",
            norm(F(vcat(θbad, st.p1)), Inf), 1e-6)
    end
end

# ===========================================================================
section("§5  Interior kinks: splines, fixed knots, free knots")
# ===========================================================================

# V5.1 — x₊ᵏ + (−1)ᵏ (−x)₊ᵏ = xᵏ
let err = 0.0
    xs = collect(range(-3, 3; length = 601))
    for k in 1:6
        err = max(err, maximum(abs.(rp.(xs, k) .+ (-1)^k .* rp.(-xs, k) .- xs .^ k)))
    end
    vcheck("V5.1", "identity x₊ᵏ + (−1)ᵏ(−x)₊ᵏ = xᵏ", err, 1e-12)
end

# V5.2 — q_θ ∈ S_k(Z), T_θM ⊂ S_k(Z, double knots), dim T_θM = k+1+2m
let maxproj = 0.0, bad = 0
    for k in 2:4, m in 1:3
        wk, bk = kinkfree_neurons(k + 1)
        Z = m == 1 ? [0.5] : collect(range(0.15, 0.85; length = m))
        wi = [isodd(i) ? 1.0 : -1.3 for i in 1:m]
        θ = pack(randn(k + 1 + m), vcat(wk, wi), vcat(bk, -Z .* wi))
        B1 = truncated_power_basis(k, Z)(TT_OFF)[1]
        B2 = double_knot_basis(k, Z)(TT_OFF)[1]
        TQ, _ = nn_tangent(θ, TT_OFF, k)
        maxproj = max(maxproj, proj_residual(nn_eval(θ, TT_OFF, k)[1][:, :], B1), proj_residual(TQ, B2))
        numrank(TQ; rtol = 1e-9) == k + 1 + 2m || (bad += 1)
    end
    vcheck("V5.2a", "q_θ ∈ S_k(Z) and ∂q/∂θ ∈ S_k(Z, double knots) (projection residual)", maxproj, 1e-10)
    vcheck_eq("V5.2b", "dim T_θM = k+1+2m for k+1 kink-free + m interior neurons (#viol.)", bad, 0)
end

# V5.3 — Theorem 2: with frozen (w,b) the network is the Galerkin VI on S_k(Z), in any basis
let k = 3, Z = [0.3, 0.6], q0 = 0.5, p0 = 0.1, h = 1.0, prob = TOY_PEND
    wk, bk = kinkfree_neurons(k + 1)
    w = vcat(wk, [1.0, 1.0])
    b = vcat(bk, -Z)
    ttb = collect(range(0, 1; length = 101))
    vcheck("V5.3a", "B-spline basis: partition of unity",
        maximum(abs.(sum(bspline_basis(k, Z)(ttb)[1]; dims = 2) .- 1)), 1e-13)
    for (tag, quad) in (("composite R=4", composite_rule(4, Z)), ("global R=8", QRule01(8)))
        sols = [linear_step(q0, p0, h, phi, quad, prob)
                for phi in (truncated_power_basis(k, Z), bspline_basis(k, Z), neuron_basis(k, w, b))]
        d = maximum(max(abs(s.q1 - sols[1].q1), abs(s.p1 - sols[1].p1)) for s in sols)
        vcheck("V5.3b", "$tag: truncated-power ≡ B-spline ≡ frozen-neuron solutions", d, 1e-11)
    end
end

# V5.4 — Theorem 3 (free knots): discrete Legendre, envelope formula, knot equation
let k = 3, quad = QRule01(8), prob = TOY_PEND, h = 1.0, qa = 0.5, qb = 0.3
    w = [1.0, 1.0, -1.0, -1.0, 1.0]
    zz = [-0.5, -1.0, 1.5, 2.0, 0.45]
    bvec(z5) = -vcat(zz[1:4], z5) .* w
    T1(z5) = typeI_step(qa, qb, h, neuron_basis(k, w, bvec(z5)), quad, prob)
    s = T1(zz[5])
    e = 1e-6
    fd = (T1(zz[5] + e).Ld - T1(zz[5] - e).Ld) / (2 * e)
    # formula: ∂A/∂z − p_b ∂q(1)/∂z + p_a ∂q(0)/∂z  at fixed coefficients
    qz(z5, t) = neuron_basis(k, w, bvec(z5))(t)
    Afix(z5) = (P = qz(z5, quad.c); discrete_action(P[1] * s.c, P[2] * s.c, h, quad, prob))
    dA = (Afix(zz[5] + e) - Afix(zz[5] - e)) / (2 * e)
    dq1 = (dot(vec(qz(zz[5] + e, [1.0])[1]), s.c) - dot(vec(qz(zz[5] - e, [1.0])[1]), s.c)) / (2 * e)
    dq0 = (dot(vec(qz(zz[5] + e, [0.0])[1]), s.c) - dot(vec(qz(zz[5] - e, [0.0])[1]), s.c)) / (2 * e)
    formula = dA - s.p_b * dq1 + s.p_a * dq0
    vcheck("V5.4a", "envelope: dL_d/dz = ∂A/∂z − pₙ₊₁∂q(1)/∂z + pₙ∂q(0)/∂z (rel.)",
        abs(fd - formula) / abs(fd), 1e-4)
    # the b-row of the network residual equals (1/w)·dL_d/dz, and the a-rows vanish
    x = vcat(pack(s.c, w, bvec(zz[5])), s.p_b)
    r = nn_residual(x, qa, s.p_a, h, k, quad, prob)
    vcheck("V5.4b", "type-I solution satisfies all a-rows of the network equations",
        norm(r[1:5], Inf), 1e-10)
    vcheck("V5.4c", "b-row of the interior neuron = (1/w)·dL_d/dz (rel.)",
        abs(r[2 * 5 + 5] - fd / w[5]) / abs(fd), 1e-4)
    # stationary knot Z*: dL_d/dz = 0 ⇒ the network equations hold completely
    g(z5) = (T1(z5 + e).Ld - T1(z5 - e).Ld) / (2 * e)
    zs = collect(range(0.1, 0.9; length = 81))
    vals = g.(zs)
    idx = findfirst(i -> isfinite(vals[i]) && isfinite(vals[i + 1]) &&
                                  sign(vals[i]) != sign(vals[i + 1]), 1:(length(zs) - 1))
    if idx === nothing
        vcheck("V5.4d", "no stationary knot found in (0.1, 0.9)", 1.0, 0.5)
    else
        zstar = bisect(g, zs[idx], zs[idx + 1]; iters = 50)
        sstar = T1(zstar)
        rstar = nn_residual(vcat(pack(sstar.c, w, bvec(zstar)), sstar.p_b), qa, sstar.p_a, h, k, quad, prob)
        @printf("      stationary knot z* = %.8f\n", zstar)
        vcheck("V5.4d", "at the stationary knot the full network residual vanishes", norm(rstar, Inf), 1e-8)
        # V5.6 — corollary: the free-knot step is generated by L_d^{NN}, hence symplectic
        xw = vcat(pack(sstar.c, w, bvec(zstar)), sstar.p_b)
        dt = step_map_det(qa, sstar.p_a, h, k, quad, prob, xw)
        vcheck("V5.6", "free-knot network step map is area preserving: |det − 1|", abs(dt - 1), 1e-7)
    end
end

# V5.5 — Gauss quadrature is not exact on kinked integrands; the composite rule is
let k = 3, z = 0.37
    worst_global = 0.0
    worst_comp = 0.0
    for j in 0:3
        f(τ) = rp(τ - z, k) * τ^j
        c8 = composite_rule(8, [z])
        exact = sum(c8.b .* f.(c8.c))
        c4 = composite_rule(4, [z])
        g4 = QRule01(4)
        worst_comp = max(worst_comp, abs(sum(c4.b .* f.(c4.c)) - exact))
        worst_global = max(worst_global, abs(sum(g4.b .* f.(g4.c)) - exact))
    end
    vcheck("V5.5a", "composite Gauss (split at the kink) integrates (τ−z)₊³τʲ exactly", worst_comp, 1e-14)
    vcheck_ge("V5.5b", "global 4-point Gauss does not", worst_global, 1e-7)
end

ok = summarize_checks()
ok || exit(1)
