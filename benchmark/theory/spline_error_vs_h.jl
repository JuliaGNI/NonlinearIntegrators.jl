# Error against the time step h of Lagrange-polynomial, fixed-knot and free-knot spline
# variational integrators on the harmonic oscillator, the pendulum and the double pendulum, for
# degrees k = 2, 3, 4:
#
#   Lagrange     CGVI(P_k), Lagrange basis on k + 1 Gauss points (and P6 for the overview)
#   FixedSpline  CGVI on S_k with n = 1, 2, 3 uniform interior knots, Gauss(k + 1) on every cell
#   FreeSpline   VISplineFree(k, n, Gauss(k + 1)), n = 1, 2, 3
#
# Harmonic oscillator: GeometricProblems' default (ω = 1/√2), exact reference. Pendulum:
# PerturbedPendulum with ϵ = 0, i.e. L = q̇²/2 + ω² cos q, ω = 1, Gauss(8) reference with dt = 2⁻¹⁰
# (checked against dt/2), (q₀, p₀) = (0.5, 0). Double pendulum: GeometricProblems' DoublePendulum
# with l = m = g = 1, (q₀, p₀) = ((0.75, 1.5), (0.25, 0.5)) as in relu_k_sweep.jl, Gauss(8) reference.
# t ∈ (0, 20), h = 2⁻⁴ … 4.
# Error: max over the grid n h of |q − q_ref|∞ / max |q_ref|∞, NaN for a run that failed. The figures
# leave out failed runs and runs with a step whose residual exceeds 1e-10 (`unconverged` > 0).
#
#   julia --project=benchmark benchmark/theory/spline_error_vs_h.jl         # ≈ 6 min
#   julia --project=benchmark benchmark/theory/spline_error_vs_h.jl plot    # figures from the csv only
#
# Writes benchmark/results/spline_error_vs_h.csv, spline_error_vs_h.png (overview: P3, P4, P6 and
# the degree-3 splines) and spline_error_vs_h_k<k>.png (one figure per degree k).

module HardProblems
include(joinpath(@__DIR__, "..", "hard_problems", "integration.jl"))
end
using .HardProblems: integrate_counting, WarnCounter, cgvi, qmatrix, relerr
using GeometricIntegrators
using NonlinearIntegrators: VISplineFree
using QuadratureRules: GaussLegendreQuadrature, CompositeQuadrature
using SimpleSplines: BSplineBasis, UniformMesh, (..)
import GeometricProblems.HarmonicOscillator as ho
import GeometricProblems.PerturbedPendulum as pp
import GeometricProblems.DoublePendulum as dp
using Logging: with_logger
using CairoMakie
using Printf

const T = 20.0
const HS = [2.0^j for j in -4:2]
const HREF = 2.0^(-10)
const DEGREES = (2, 3, 4)
const RESULTS = joinpath(@__DIR__, "..", "results")

spline_cgvi(k, n) = CGVI(BSplineBasis(UniformMesh(n + 1, 0 .. 1), k),
    CompositeQuadrature(GaussLegendreQuadrature(Float64, k + 1), n + 1))

const SUB = Dict(2 => "₂", 3 => "₃", 4 => "₄")

const METHODS = vcat(
    [(degree = 6, family = "Lagrange", label = "P6", method = cgvi(6))],
    [m for k in DEGREES for m in vcat(
        [(degree = k, family = "Lagrange", label = "P$(k)", method = cgvi(k))],
        [(degree = k, family = "FixedSpline", label = "fixed S$(SUB[k]) n=$(n)", method = spline_cgvi(k, n))
         for n in 1:3],
        [(degree = k, family = "FreeSpline", label = "free S$(SUB[k]) n=$(n)",
             method = VISplineFree(k, n, GaussLegendreQuadrature(Float64, k + 1))) for n in 1:3])])

const PENDULUM = (ω = 1.0, ϵ = 0.0, ϕ = π / 3)
# as in relu_k_sweep.jl: unit masses, lengths and gravity, a state that is not chaotic over T = 20
const DP_PARAMETERS = (l₁ = 1.0, l₂ = 1.0, m₁ = 1.0, m₂ = 1.0, g = 1.0)
const DP_Q₀, DP_P₀ = [0.75, 1.5], [0.25, 0.5]
const PROBLEMS = (
    (name = "harmonic oscillator",
        build = h -> ho.lodeproblem([0.5], [0.0]; timespan = (0, T), timestep = h),
        reference = dt -> qmatrix(ho.exact_solution(ho.podeproblem([0.5], [0.0]; timespan = (0, T), timestep = dt)))),
    (name = "pendulum",
        build = h -> pp.lodeproblem([0.5], [0.0]; timespan = (0, T), timestep = h, parameters = PENDULUM),
        reference = dt -> qmatrix(integrate(pp.lodeproblem([0.5], [0.0]; timespan = (0, T), timestep = dt,
            parameters = PENDULUM), Gauss(8)))),
    (name = "double pendulum",
        build = h -> dp.lodeproblem(DP_Q₀, DP_P₀; timespan = (0, T), timestep = h, parameters = DP_PARAMETERS),
        reference = dt -> qmatrix(integrate(dp.lodeproblem(DP_Q₀, DP_P₀; timespan = (0, T), timestep = dt,
            parameters = DP_PARAMETERS), Gauss(8)))))

const CSV = joinpath(RESULTS, "spline_error_vs_h.csv")

function run_all()
    rows = ["problem,degree,family,label,h,status,unconverged,fallbacks,q_err,secs"]
    for P in PROBLEMS
        Qref = P.reference(HREF)
        @printf("\n%s: reference |dt − dt/2| = %.2e\n", P.name, relerr(Qref, P.reference(HREF / 2)[:, 1:2:end]))
        problems = Dict(h => P.build(h) for h in HS)    # PerturbedPendulum generates its code symbolically
        for M in METHODS, h in HS
            secs = @elapsed r = with_logger(() -> integrate_counting(problems[h], M.method), WarnCounter())
            stride = round(Int, h / HREF)
            err = r.status == "ok" ? relerr(qmatrix(r.sol), Qref[:, 1:stride:end]) : NaN
            C = HardProblems.GIB.cache(r.int)
            fallbacks = hasproperty(C, :fallbacks) ? C.fallbacks[1] : 0
            push!(rows, @sprintf("%s,%d,%s,%s,%g,%s,%d,%d,%.6e,%.3e", P.name, M.degree, M.family, M.label, h,
                r.status, r.unconverged, fallbacks, err, secs))
            @printf("  %-18s h = %-7g %-22s fallbacks = %-4d q_err = %.2e  %.2f s\n", M.label, h, r.status,
                fallbacks, err, secs)
        end
    end
    mkpath(RESULTS)
    write(CSV, join(rows, "\n") * "\n")
end

"plot" in ARGS || run_all()      # `plot`: only redraw the figures from the csv

# ---- figures: one panel per problem, colour = family, marker and line style = order within the family

const COLORS = Dict("Lagrange" => colorant"#2a78d6", "FixedSpline" => colorant"#eb6834",
    "FreeSpline" => colorant"#1baf7a")
const MARKERS = (:circle, :rect, :utriangle)
const STYLES = (:solid, :dash, :dot)

data = [split(l, ",") for l in readlines(CSV)[2:end]]

function error_figure(file, methods, title)
    fig = Figure()
    Label(fig[0, 1:length(PROBLEMS)], title; fontsize = 20, font = :bold)
    for (j, P) in enumerate(PROBLEMS)
        ax = Axis(fig[1, j]; width = 360, height = 280, xscale = log10, yscale = log10, xlabel = "h",
            xticks = (HS, ["1/16", "1/8", "1/4", "1/2", "1", "2", "4"]), yticks = 10.0 .^ (-14:2:2),
            ytickformat = vs -> [rich("10", superscript(string(round(Int, log10(v))))) for v in vs],
            ylabel = j == 1 ? "max relative error in q" : "", title = P.name, titlesize = 18)
        for (i, M) in enumerate(methods)
            v = count(m -> m.family == M.family, methods[1:i])
            d = filter(r -> r[1] == P.name && r[4] == M.label, data)
            h, e = parse.(Float64, getindex.(d, 5)), parse.(Float64, getindex.(d, 9))
            ok = isfinite.(e) .& (e .> 0) .& (getindex.(d, 7) .== "0")     # no step with residual > 1e-10
            scatterlines!(ax, h[ok], e[ok]; color = COLORS[M.family], linewidth = 2, markersize = 10,
                marker = MARKERS[v], linestyle = STYLES[v], label = M.label)
        end
    end
    Legend(fig[1, length(PROBLEMS) + 1], fig.content[2]; framevisible = false, labelsize = 14)
    resize_to_layout!(fig)
    save(joinpath(RESULTS, file), fig; px_per_unit = 2)
    println("wrote ", joinpath(RESULTS, file))
end

error_figure("spline_error_vs_h.png",
    vcat([only(filter(M -> M.label == l, METHODS)) for l in ("P3", "P4", "P6")],
        filter(M -> M.degree == 3 && M.family != "Lagrange", METHODS)),
    "Lagrange P3, P4, P6 and degree-3 splines")
for k in DEGREES
    error_figure("spline_error_vs_h_k$(k).png", filter(M -> M.degree == k, METHODS), "degree k = $(k)")
end
