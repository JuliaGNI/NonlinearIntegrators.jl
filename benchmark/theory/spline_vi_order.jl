# Order and linear stability of the B-spline Galerkin variational integrator
#
#     CGVI(BSplineBasis(UniformMesh(M, 0 .. 1), p), CompositeQuadrature(GaussLegendreQuadrature(r), M))
#
# p = spline degree, M = cells per time step, r = Gauss points per cell, Δt = h / M.
# Statements checked (benchmark/theory/spline_vi_order_en.md):
#   A1  p = 1: one step of size h with M cells = M steps of CGVI(P₁) of size h / M
#   A2  r ≥ p + 1 (exact for the harmonic oscillator): global error ∝ Δt^{2p}, order 2p in h
#   A3  reduced rule r = p: order still 2p
#   B2  harmonic oscillator: one-step map G(z), z = ωh, det G = 1, stable iff |tr G| < 2
#
#   julia --project=benchmark benchmark/theory/spline_vi_order.jl
#
# Writes benchmark/results/spline_vi_order_{local,ab,stability,identity,global,trace}.csv.

using GeometricIntegrators
using SimpleSplines
using QuadratureRules: GaussLegendreQuadrature, CompositeQuadrature
import GeometricProblems.HarmonicOscillator as ho
import GeometricProblems.HenonHeilesPotential as hh
using LinearAlgebra
using Printf

const RESULTS = joinpath(@__DIR__, "..", "results")

spline_cgvi(p, M, r, T = Float64) = CGVI(BSplineBasis(UniformMesh(M, T(0) .. T(1)), p),
    CompositeQuadrature(GaussLegendreQuadrature(T, r), M))

"""
One-step map (q₀, p₀) ↦ (q₁, p₁) of `method::CGVI` for q̈ = −q (ω = 1) and step z, assembled from the
coefficients of the method. The CGVI residual is linear here:
    (z Mq − Ka / z) X + r₁ p₁ = r₀ p₀,   r₀ᵀ X = q₀,   q₁ = r₁ᵀ X,
with Mq = mᵀ diag(b) m and Ka = aᵀ diag(b) a.
"""
function step_map(method::CGVI, z)
    m, a, b, r₀, r₁ = method.m, method.a, method.b, method.r₀, method.r₁
    S = length(r₀)
    A = [z * m' * Diagonal(b) * m - a' * Diagonal(b) * a / z r₁; r₀' 0]
    Y = A \ [zeros(S) r₀; 1 0]                 # columns: (q₀, p₀) = (1, 0) and (0, 1)
    [r₁' * Y[1:S, :]; Y[S+1:S+1, :]]
end

exact_map(z) = [cos(z) sin(z); -sin(z) cos(z)]

"Least-squares slope of log(err) against log(h) over the points with err above `floor`."
function slope(h, err; floor = 1e-12)
    i = findall(e -> e > floor, err)
    length(i) < 2 && return NaN
    x, y = log.(h[i]), log.(err[i])
    sum((x .- sum(x) / length(x)) .* (y .- sum(y) / length(y))) / sum(abs2, x .- sum(x) / length(x))
end

const CASES = [(p, M, r) for p in 1:4 for M in (1, 2, 4) for r in (p, p + 1)]

# ---------------------------------------------------------------------------------------------
# B2 + local error: one-step map in BigFloat, ω = 1
# ---------------------------------------------------------------------------------------------
setprecision(BigFloat, 256)
hs_local = [BigFloat(2)^(-j) for j in 1:8]
rows = ["p,M,r,h,local_err,det_minus_1"]
println("local error |G(h) − exp(hJ)|, slope expected 2p+1")
for (p, M, r) in CASES
    method = spline_cgvi(p, M, r, BigFloat)
    err = [maximum(abs, step_map(method, h) - exact_map(h)) for h in hs_local]
    dets = [det(step_map(method, h)) - 1 for h in hs_local]
    for (h, e, d) in zip(hs_local, err, dets)
        push!(rows, @sprintf("%d,%d,%d,%.6e,%.6e,%.3e", p, M, r, Float64(h), Float64(e), Float64(d)))
    end
    @printf("p = %d  M = %d  r = %d   slope = %6.3f   max |det G − 1| = %.1e\n", p, M, r,
        slope(Float64.(hs_local), Float64.(err); floor = 1e-60), Float64(maximum(abs, dets)))
end
write(joinpath(RESULTS, "spline_vi_order_local.csv"), join(rows, "\n") * "\n")

# ---------------------------------------------------------------------------------------------
# HO proof (spline_vi_order_en.md §3): with symmetric knots and rule, L_d = [a (q₀² + q₁²) − 2b q₀q₁] / 2h, so
#     G = [a/b  h/b; (a² − b²)/(bh)  a/b],   a = b G₁₂ ... read off as b = h/G₁₂, a = b G₁₁,
# exact: a_E = z cot z, b_E = z / sin z. The proof needs Δa − Δb = O(z^{2p+2}) and Δa + Δb = O(z^{2p}).
# ---------------------------------------------------------------------------------------------
rows = ["p,M,r,z,G11_minus_G22,da_minus_db,da_plus_db"]
println("\nHO: a, b of L_d; slopes of Δa − Δb (proof needs ≥ 2p+2) and Δa + Δb (needs ≥ 2p); max |G₁₁ − G₂₂|")
for (p, M, r) in CASES
    method = spline_cgvi(p, M, r, BigFloat)
    d = map(hs_local) do z
        G = step_map(method, z)
        b = z / G[1, 2]
        a = b * G[1, 1]
        (G[1, 1] - G[2, 2], (a - z * cot(z)) - (b - z / sin(z)), (a - z * cot(z)) + (b - z / sin(z)))
    end
    for (z, x) in zip(hs_local, d)
        push!(rows, @sprintf("%d,%d,%d,%.6e,%.3e,%.6e,%.6e", p, M, r, Float64(z), Float64.(x)...))
    end
    @printf("p = %d  M = %d  r = %d   Δa − Δb: %6.3f   Δa + Δb: %6.3f   max |G₁₁ − G₂₂| = %.1e\n", p, M, r,
        slope(Float64.(hs_local), Float64.(abs.(getindex.(d, 2))); floor = 1e-70),
        slope(Float64.(hs_local), Float64.(abs.(getindex.(d, 3))); floor = 1e-70),
        Float64(maximum(x -> abs(x[1]), d)))
end
write(joinpath(RESULTS, "spline_vi_order_ab.csv"), join(rows, "\n") * "\n")

# ---------------------------------------------------------------------------------------------
# B2: stability set Σ = {z : |tr G(z)| ≤ 2} on z ∈ (0, ZMAX]
#
# |tr G| = 2 exactly where G has the eigenvalue ±1, i.e. where a step maps some (q₀, p₀) to ±(q₀, p₀).
# With y = z p₀ and λ = z² the step equations then read
#     (Ka − λ Mq) X + (r₀ ∓ r₁) y = 0,   (r₁ ∓ r₀)ᵀ X = 0,
# a generalised eigenvalue problem in λ: its finite positive eigenvalues are the band edges. Between two
# edges |tr G| − 2 has one sign, read off at the midpoint.
# ---------------------------------------------------------------------------------------------
const ZMAX = 100.0

function band_edges(method::CGVI)
    m, a, b, r₀, r₁ = method.m, method.a, method.b, method.r₀, method.r₁
    Mq, Ka = m' * Diagonal(b) * m, a' * Diagonal(b) * a
    S = length(r₀)
    edges = Float64[]
    for σ in (1, -1)
        λ = eigvals([Ka r₀-σ*r₁; (r₁-σ*r₀)' 0], [Mq zeros(S); zeros(1, S + 1)])
        append!(edges, sqrt.(real(λ[@. isfinite(λ) & (abs(imag(λ)) < 1e-8) & (real(λ) > 0)])))
    end
    sort(filter(z -> 1e-6 < z < ZMAX, edges))         # z = 0: G = I
end

rows = ["p,M,r,z_lo,z_hi,max_half_trace_minus_1,half_trace_minus_1_midpoint_bigfloat"]
println("\nunstable z-intervals (|tr G| > 2) in (0, $ZMAX]")
for (p, M, r) in CASES
    method = spline_cgvi(p, M, r)
    method_big = spline_cgvi(p, M, r, BigFloat)
    pts = vcat(0.0, band_edges(method), ZMAX)
    intervals = Tuple{Float64,Float64,Float64}[]
    for (lo, hi) in zip(pts[1:end-1], pts[2:end])
        hi - lo < 1e-12 && continue
        # samples at a Ritz value give a singular system (NaN): skipped
        worst = maximum(filter(!isnan, [abs(tr(step_map(method, z))) / 2 for z in range(lo, hi, length = 41)[2:end-1]]))
        worst > 1 + 1e-10 && push!(intervals, (lo, hi, worst))
    end
    for (lo, hi, worst) in intervals
        mid = Float64(abs(tr(step_map(method_big, (BigFloat(lo) + BigFloat(hi)) / 2))) / 2 - 1)
        push!(rows, @sprintf("%d,%d,%d,%.6f,%.6f,%.3e,%.3e", p, M, r, lo, hi, worst - 1, mid))
    end
    # poles of a, b: Ritz values z of q̈ = −z² q, q(0) = q(1) = 0, on the interior functions 2:S−1
    Mq, Ka = method.m' * Diagonal(method.b) * method.m, method.a' * Diagonal(method.b) * method.a
    S = length(method.r₀)
    ritz = S > 2 ? sqrt.(filter(<(ZMAX^2), eigvals(Ka[2:S-1, 2:S-1], Mq[2:S-1, 2:S-1]))) : Float64[]
    @printf("p = %d  M = %d  r = %d   Ritz z: %s\n", p, M, r, join([@sprintf("%.4f", z) for z in ritz], " "))
    @printf("p = %d  M = %d  r = %d   %s\n", p, M, r, isempty(intervals) ? "none" :
        join([@sprintf("[%.4f, %.4f]", I[1], I[2]) for I in first(intervals, 4)], " ") *
        (length(intervals) > 4 ? " … ($(length(intervals)) intervals)" : ""))
end
write(joinpath(RESULTS, "spline_vi_order_stability.csv"), join(rows, "\n") * "\n")

# p = 1 closed forms (§1): tr G/2 = (1 − z²/3)/(1 + z²/6) for r = 2, (1 − z²/4)/(1 + z²/4) for r = 1
zz = [0.3, 1.0, 2.5, 5.0, 40.0]
dev = [maximum(abs(tr(step_map(spline_cgvi(1, 1, r), z)) / 2 - f(z)) for z in zz)
       for (r, f) in ((2, z -> (1 - z^2 / 3) / (1 + z^2 / 6)), (1, z -> (1 - z^2 / 4) / (1 + z^2 / 4)))]
@printf("\np = 1 trace formulas, max deviation on z ∈ %s: exact rule %.1e, midpoint %.1e\n", zz, dev...)
write(joinpath(RESULTS, "spline_vi_order_trace.csv"), @sprintf("rule,max_dev\nexact,%.3e\nmidpoint,%.3e\n", dev...))

# ---------------------------------------------------------------------------------------------
# A1: p = 1, M cells, step h  vs  CGVI(P₁) with step h / M
# ---------------------------------------------------------------------------------------------
rows = ["M,r,h,max_diff_q,max_diff_p"]
println("\nA1: S₁(M cells) at step h vs CGVI(P₁) at step h/M, harmonic oscillator, T = 16")
for M in (2, 4), r in (1, 2), h in (1.0, 0.25)
    sol_spline = integrate(ho.lodeproblem(timespan = (0, 16), timestep = h), spline_cgvi(1, M, r))
    sol_p1 = integrate(ho.lodeproblem(timespan = (0, 16), timestep = h / M), spline_cgvi(1, 1, r))
    dq = maximum(abs(sol_spline.q[n][1] - sol_p1.q[M*n][1]) for n in eachindex(sol_spline.q))
    dp = maximum(abs(sol_spline.p[n][1] - sol_p1.p[M*n][1]) for n in eachindex(sol_spline.p))
    push!(rows, @sprintf("%d,%d,%g,%.3e,%.3e", M, r, h, dq, dp))
    @printf("M = %d  r = %d  h = %5.3f   max |Δq| = %.2e   max |Δp| = %.2e\n", M, r, h, dq, dp)
end
write(joinpath(RESULTS, "spline_vi_order_identity.csv"), join(rows, "\n") * "\n")

# ---------------------------------------------------------------------------------------------
# A2/A3: global error, harmonic oscillator (exact solution) and Hénon–Heiles (Gauss(8) reference)
# ---------------------------------------------------------------------------------------------
T = 16.0
hs = [2.0^(-j) for j in -1:5]
"max over the time levels of |q − q_ref| and |p − p_ref|; `ref` holds the reference at the same times."
max_error(sol, ref) = max(maximum(maximum.(abs, sol.q[0:end] .- ref.q)),
    maximum(maximum.(abs, sol.p[0:end] .- ref.p)))

const HREF = 2.0^(-8)
rows = ["problem,p,M,r,h,err,seconds"]
println("\nglobal error max_n |(q, p)_n − (q, p)(t_n)| on [0, $T]; expected slope 2p (fit over h ≤ 1/2)")
for (name, lode, ref) in (
    ("harmonic_oscillator", h -> ho.lodeproblem(timespan = (0, T), timestep = h),
        ho.exact_solution(ho.podeproblem(timespan = (0, T), timestep = HREF))),
    ("henon_heiles", h -> hh.lodeproblem(timespan = (0, T), timestep = h),
        integrate(hh.lodeproblem(timespan = (0, T), timestep = HREF), Gauss(8))))
    for (p, M, r) in CASES
        err = Float64[]
        secs = @elapsed for h in hs
            sol = integrate(lode(h), spline_cgvi(p, M, r))
            stride = round(Int, h / HREF)
            push!(err, max_error(sol, (q = ref.q[0:stride:end], p = ref.p[0:stride:end])))
        end
        for (h, e) in zip(hs, err)
            push!(rows, @sprintf("%s,%d,%d,%d,%g,%.6e,%.2f", name, p, M, r, h, e, secs))
        end
        @printf("%-19s p = %d  M = %d  r = %d   slope = %6.3f   err(h = 1/8)·M^{2p} = %.3e   (%.1f s)\n",
            name, p, M, r, slope(hs[3:end], err[3:end]), err[5] * M^(2p), secs)
    end
end
write(joinpath(RESULTS, "spline_vi_order_global.csv"), join(rows, "\n") * "\n")
println("wrote ", RESULTS)
