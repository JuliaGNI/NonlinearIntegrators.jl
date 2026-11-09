# Free-knot vs fixed-knot spline variational integrators vs Lagrange polynomials (CGVI) on the hard
# problems (PLAN.md §2: N4 against L1), all of degree 3 with 4 Gauss points per cell:
#
#   Lagrange     CGVI(P_s), Lagrange basis on R = s + 1 Gauss points, s = 3..6, one step per h;
#                and CGVI(P₃) with m = 2..4 substeps per h
#   FixedSpline  CGVI on S₃ with n = 1..3 uniform interior knots, composite Gauss(4) on every cell
#   FreeSpline   VISplineFree(3, n, Gauss(4)), n = 1..3 interior knots found by the variational principle
#
#   julia --project=benchmark benchmark/hard_problems/run_splines.jl                      # quick cases
#   julia --project=benchmark benchmark/hard_problems/run_splines.jl full --cases=KeplerProblem_e0.99
#   julia --project=benchmark benchmark/hard_problems/run_splines.jl quick --c=1,3,10      # some h Ω only
#   julia --project=benchmark benchmark/hard_problems/run_splines.jl report               # md from the csv
#
# Same cases, steps and metrics as run_baselines.jl (see its header); additionally `params`, the
# parameters per step h (s + 1, 3m + 1, 4 + n and 4 + 2n), and `fallbacks`, the steps of
# FreeSpline whose knot equation did not converge. For FreeSpline, `iterations` and the residual
# behind `status` / `max_res` are those of the last inner solve of a step, which for a fallback
# step is not the accepted one; `secs` is its cost measure.
#
# Writes results/<case>_splines.csv; `report` writes results/splines.md.

include(joinpath(@__DIR__, "integration.jl"))

using NonlinearIntegrators: VISplineFree
using SimpleSplines: BSplineBasis, UniformMesh, (..)

spline_cgvi(n) = CGVI(BSplineBasis(UniformMesh(n + 1, 0 .. 1), 3),
    CompositeQuadrature(GaussLegendreQuadrature(Float64, 4), n + 1))

const METHODS = vcat(
    [(family = "Lagrange", label = "P$(s)", m = 1, params = s + 1, method = cgvi(s)) for s in 3:6],
    [(family = "Lagrange", label = "P3 x$(m)", m = m, params = 3m + 1, method = cgvi(3)) for m in 2:4],
    [(family = "FixedSpline", label = "S3 n=$(n)", m = 1, params = 4 + n, method = spline_cgvi(n))
     for n in 1:3],
    [(family = "FreeSpline", label = "S3 free n=$(n)", m = 1, params = 4 + 2n,
         method = VISplineFree(3, n, GaussLegendreQuadrature(Float64, 4))) for n in 1:3])

"Step factors selected by `--c=…` (default all of STEP_FACTORS)."
function select_factors(args)
    sel = findfirst(a -> startswith(a, "--c="), args)
    sel === nothing ? STEP_FACTORS : parse.(Float64, split(split(args[sel], "=")[2], ","))
end

function run_case(case, factors)
    println("\n── $(case.name): T = $(case.T), Terr = $(case.Terr), h Ω ∈ $(factors)")
    ref = write_reference(case)
    foreach(M -> warmup(case, M.m, M.method), METHODS)

    invnames = join(string.(keys(case.invariants)), ",")
    open(joinpath(RESULTS_DIR, "$(case.name)_splines.csv"), "w") do io
        println(io, "case,c,h,family,label,m,params,steps,status,unconverged,fallbacks,max_res,q_err,$(invnames),iterations,secs")
        for c in factors, M in METHODS
            h = c / case.Ω
            r = run_one(case, ref, h, M.m, M.method)
            println(io, join((case.name, c, h, M.family, M.label, M.m, M.params, r.steps, r.status,
                    r.unconverged, r.fallbacks, csvnum(r.max_res), csvnum(r.q_err), csvnum.(r.inv)...,
                    r.iterations, csvnum(r.secs)), ","))
            flush(io)
            @printf("  c = %-4g %-14s %-11s fallbacks = %-5d q_err = %.2e  %.2f s\n", c, M.label,
                r.status, r.fallbacks, r.q_err, r.secs)
        end
    end
end

# ---- report ------------------------------------------------------------------------------------

"""
Per case and step: the q error and wall time of every method, and the verdict on FreeSpline with
the criterion of PLAN.md: better iff its q error is more than 3 times smaller than that of every
Lagrange / FixedSpline run that is not slower, its run converged, and at most 5% of its steps fell back.
"""
function report()
    files = filter(f -> endswith(f, "_splines.csv"), readdir(RESULTS_DIR))
    open(joinpath(RESULTS_DIR, "splines.md"), "w") do io
        println(io, "# Free-knot vs fixed-knot splines vs Lagrange polynomials (degree 3, Gauss(4) per cell)\n")
        println(io, "Entries: q error / wall time in s. `fb` = fallback steps of the knot equation. Verdict: FreeSpline is ",
            "**better** iff converged, fallbacks ≤ 5% of the steps, and its q error < 1/3 of every Lagrange or FixedSpline run ",
            "with wall time ≤ its own (at least one such run).\n")
        for file in files
            rows = [split(l, ",") for l in readlines(joinpath(RESULTS_DIR, file))]
            head, rows = rows[1], rows[2:end]
            col(r, k) = r[findfirst(==(k), head)]
            num(r, k) = parse(Float64, col(r, k))
            println(io, "## ", col(rows[1], "case"), "\n")
            labels = unique(col.(rows, "label"))
            println(io, "| h Ω | ", join(labels, " | "), " | verdict |")
            println(io, "|", repeat("---|", length(labels) + 2))
            for c in unique(col.(rows, "c"))
                rc = filter(r -> col(r, "c") == c, rows)
                cell(r) = (col(r, "status") == "ok" ? "" : col(r, "status") * " ") *
                          @sprintf("%.1e / %.2g", num(r, "q_err"), num(r, "secs")) *
                          (col(r, "family") == "FreeSpline" ? " (fb $(col(r, "fallbacks")))" : "")
                verdict = map(filter(r -> col(r, "family") == "FreeSpline", rc)) do f
                    ok = col(f, "status") == "ok" && num(f, "fallbacks") ≤ 0.05 * num(f, "steps")
                    rivals = filter(r -> col(r, "family") != "FreeSpline" && num(r, "secs") ≤ num(f, "secs"), rc)
                    better = ok && !isempty(rivals) && all(r -> !(num(r, "q_err") ≤ 3 * num(f, "q_err")), rivals)
                    better ? col(f, "label") : ""
                end
                v = filter(!isempty, verdict)
                println(io, "| ", c, " | ", join(cell.(rc), " | "), " | ",
                    isempty(v) ? "–" : "**better: " * join(v, ", ") * "**", " |")
            end
            println(io)
        end
    end
    println("wrote ", joinpath(RESULTS_DIR, "splines.md"))
end

function main(args)
    mkpath(RESULTS_DIR)
    !isempty(args) && args[1] == "report" && return report()
    factors = select_factors(args)
    foreach(case -> run_case(case, factors), select_cases(args))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
