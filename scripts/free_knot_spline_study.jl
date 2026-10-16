# How `FreeKnotSpline` behaves over many steps and as h shrinks.
#
#     julia --startup-file=no --project=scripts scripts/free_knot_spline_study.jl
#
# Harmonic oscillator (m = 1, k = 0.5) from (q, p) = (0.5, 0):
#
#   1. Convergence in h: the error at t = 4 for h = 1, 0.5, 0.25, S = 4, with the observed
#      order between consecutive rows.
#   2. Long run: h = 0.5 over 200 steps (t = 100), S = 4. The largest energy error and the
#      largest q error, and the smallest knot gap and the smallest |cᵢ|/max|c| of the last step.
#   3. Time per step, from the long run, after compilation.
#   4. Local error per step, h = 0.5, 16 steps: the step from the numerical state against the
#      exact flow from that state. Steps 5 and 14 lie next to a zero of q, where the cᵢ are about
#      1e-5 and change sign; a step that collapses onto the cubic shows the cubic's local error
#      there, 1e-11.

using NonlinearIntegrators
using GeometricIntegratorsBase
using GeometricIntegratorsBase: nlsolution
using GeometricProblems.HarmonicOscillator
using Printf

const PARAMS = HarmonicOscillator.default_parameters()
exact(t) = HarmonicOscillator.exact_solution_q(t, 0.5, 0.0, 0.0, PARAMS)
energy(q, p) = p^2 / 2 + PARAMS.k * q^2 / 2

function problem(h, T)
    HarmonicOscillator.lodeproblem([0.5], [0.0];
        timespan = (0.0, T), timestep = h, parameters = PARAMS)
end

println("Julia $(VERSION); harmonic oscillator m = 1, k = 0.5, (q, p) = (0.5, 0); S = 4")
println()
println("1. error at t = 4")
@printf("%6s %10s %6s\n", "h", "error", "order")
let previous = NaN
    for h in (1.0, 0.5, 0.25)
        sol = integrate(problem(h, 4.0), FreeKnotSpline(4))
        e = abs(collect(sol.q[:, 1])[end] - exact(4.0))
        @printf("%6.3f %10.2e %6s\n", h, e,
            isnan(previous) ? "" : @sprintf("%.2f", log2(previous / e)))
        previous = e
    end
end

println()
println("2. h = 0.5 over 200 steps")
int = GeometricIntegrator(problem(0.5, 100.0), FreeKnotSpline(4))
t = @elapsed sol = integrate(int)
q, p = collect(sol.q[:, 1]), collect(sol.p[:, 1])
ts = collect(range(0.0, 100.0, length(q)))
x = nlsolution(int)
τ, c = NonlinearIntegrators.spline_τ(x, 4, 1), NonlinearIntegrators.spline_c(x, 4, 1)
@printf("max |H − H₀| = %.2e, max |q − q_exact| = %.2e\n",
    maximum(abs, energy.(q, p) .- energy(q[1], p[1])), maximum(abs, q .- exact.(ts)))
@printf("last step: knots %s, smallest gap %.3f, min |c| / max |c| = %.3f\n",
    join(round.(τ; digits = 3), " "), minimum(diff(vcat(0.0, sort(τ), 1.0))),
    minimum(abs, c) / maximum(abs, c))
println()
@printf("3. %.0f ms per step (200 steps in %.1f s)\n", 1000t / 200, t)

println()
println("4. local error per step, h = 0.5")
@printf("%4s %8s %8s %9s %9s  %s\n", "step", "q", "p", "error q", "error p", "knots")
const ω = sqrt(PARAMS.k / PARAMS.m)
flow(q, p, t) = (q * cos(ω * t) + p / ω * sin(ω * t), -q * ω * sin(ω * t) + p * cos(ω * t))
let q = 0.5, p = 0.0
    for n in 1:16
        prob = HarmonicOscillator.lodeproblem([q], [p];
            timespan = (0.0, 0.5), timestep = 0.5, parameters = PARAMS)
        int = GeometricIntegrator(prob, FreeKnotSpline(4))
        sol = integrate(int)
        qn, pn = collect(sol.q[:, 1])[end], collect(sol.p[:, 1])[end]
        qe, pe = flow(q, p, 0.5)
        knots = NonlinearIntegrators.spline_τ(nlsolution(int), 4, 1)
        @printf("%4d %8.4f %8.4f %9.1e %9.1e  %s\n", n, q, p, abs(qn - qe), abs(pn - pe),
            join(round.(knots; digits = 3), " "))
        q, p = qn, pn
    end
end
