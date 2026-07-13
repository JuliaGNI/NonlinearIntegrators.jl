# Comparison of the two OGA initial-guess algorithms for `NonLinear_OneLayer_GML`:
#
#   * `OGA1d`        — the default: seed assembled at the working precision `T`,
#                      output weights from a QR fit of the √w-scaled design matrix.
#   * `OGA1d_Legacy` — the previous algorithm: seed assembled in `Float64` (a
#                      "double-precision island"), output weights from the normal
#                      equations `Gk \ rhs`, then rounded into the working type.
#
# Both are run end-to-end (OGA seed + Newton solve) on the harmonic oscillator over
# t ∈ [0, 1], across problems of increasing complexity along two axes:
#
#   * time-step length  dt  (⇒ number of steps = 1/dt), and
#   * number of neurons S   (⇒ number of network parameters, 3·S per dimension),
#
# and at three working precisions (Float64, Float32, Float16). For each case we
# report the solver status, the error of the final state against the analytic
# solution, and the wall-clock time (after a warm-up run to exclude compilation).
#
# Run with:
#   julia --project=benchmark benchmark/oga_comparison.jl

using NonlinearIntegrators
using GeometricProblems.HarmonicOscillator
using QuadratureRules
using GeometricIntegratorsBase
using LinearAlgebra: SingularException
using Printf

# Type-generic ReLU^k activation (never `max(0.0, x)`, which would upcast).
relu_k(k::Int = 3) = x -> max(zero(x), x)^k

# One end-to-end run. Returns (status, error-vs-exact, seconds). A warm-up run is
# discarded first so the timing excludes compilation (S enters the method type, so
# each S recompiles).
function run_case(::Type{T}, guess; S::Int, R::Int, timestep) where {T}
    params = HarmonicOscillator.default_parameters(T)
    prob = HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)];
        timespan = (T(0), T(1)), timestep = T(timestep), parameters = params)
    basis = OneLayerNetwork_GML{T}(relu_k(3), S)
    method = NonLinear_OneLayer_GML(basis, QuadratureRules.GaussLegendreQuadrature(T, R);
        bias_interval = [-T(pi), T(pi)], dict_amount = 400, initial_guess_method = guess)

    solve() = integrate(prob, method; regularization_factor = T(1e-5), max_iterations = 10000)
    ref = Float64(HarmonicOscillator.exact_solution_q(T(1), T(0.5), T(0.0), T(0.0), params))

    try
        solve()                                    # warm-up (compile) — discarded
        t = @elapsed res = solve()
        qend = Float64(collect(res.sol.q[:, 1])[end])
        err = abs(qend - ref)
        return (status = isfinite(err) ? "ok" : "nonfinite", err = err, secs = t)
    catch e
        e isa SingularException && return (status = "singular", err = NaN, secs = NaN)
        # A NonlinearSolverException means the Newton solve failed to converge.
        name = string(nameof(typeof(e)))
        short = occursin("NonlinearSolver", name) ? "diverged" : first(name, 9)
        return (status = short, err = NaN, secs = NaN)
    end
end

fmt_err(e) = isnan(e) ? "     —     " : @sprintf("%.3e", e)
fmt_ms(s)  = isnan(s) ? "   —   "     : @sprintf("%7.1f", 1000s)

# (S, dt) pairs of increasing complexity, plus the working precisions to sweep.
const CONFIGS = [(2, 0.5), (4, 0.1), (8, 0.05)]
const TYPES   = (Float64, Float32, Float16)
const R       = 8

println("OGA comparison — NonLinear_OneLayer_GML, harmonic oscillator on t ∈ [0, 1]")
println("  new = OGA1d (working-precision QR)   legacy = OGA1d_Legacy (Float64 normal equations)")
println()
@printf("%-8s %3s %6s %6s │ %-9s %-11s %-8s │ %-9s %-11s %-8s\n",
        "T", "S", "dt", "steps", "new", "new err", "new ms", "legacy", "leg err", "leg ms")
println("─"^86)

for T in TYPES
    for (S, dt) in CONFIGS
        steps = round(Int, 1 / dt)
        n = run_case(T, OGA1d();        S = S, R = R, timestep = dt)
        l = run_case(T, OGA1d_Legacy(); S = S, R = R, timestep = dt)
        @printf("%-8s %3d %6.3g %6d │ %-9s %-11s %-8s │ %-9s %-11s %-8s\n",
                string(T), S, dt, steps,
                n.status, fmt_err(n.err), fmt_ms(n.secs),
                l.status, fmt_err(l.err), fmt_ms(l.secs))
    end
end

println("─"^86)
println("""
Reading the table:
  * Float64 / Float32: both algorithms should reach the same accuracy — the seed is
    only a warm start and the Newton solve sets the final error. The legacy variant
    computes that seed in Float64 either way; the new one matches it at the working
    precision, so this confirms the reformulation is a no-regression change.
  * Float16: the legacy Gram solve is ill-conditioned (κ(Φ)²) and its rounded seed
    tends to drive the Newton Jacobian singular ("singular") or non-finite, whereas
    the new QR seed stays finite and lets the solve proceed.
""")
