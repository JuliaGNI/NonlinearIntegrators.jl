# ShallowNet(ReLUᵏ, S neurons, R) vs CGVI(P_k, R) for several k and S — does the S = 4, k = 3
# observation (“same error as CGVI”) hold for other powers and widths, as Theorem 1 predicts?
#
#   julia --project=benchmark benchmark/theory/relu_k_sweep.jl
#   julia --project=benchmark benchmark/theory/relu_k_sweep.jl --k=2,3,4,5 --S=auto --h=0.2,0.5,1,2,5
#   julia --project=benchmark benchmark/theory/relu_k_sweep.jl --bias=default     # OGA bias [−π, π]
#   julia --project=benchmark benchmark/theory/relu_k_sweep.jl --strict --final-time=40
#
# Options
#   --k=…           activation powers (default 2,3,4,5)
#   --S=auto|…      widths; `auto` = k+1, k+3, 2(k+1) for every k (default), or an explicit list
#   --R=auto|n      Gauss points; `auto` = k+1 (the CGVI(P_k) default, R = 4 for k = 3)
#   --h=…           step sizes (default 0.2,0.5,1,2,5,10);  --final-time=T (default 20)
#   --bias=kinkfree|default   OGA bias interval [1.1, π] (every seed kink outside [0,1],
#                   default) or [−π, π] (the package default — kinks may enter the interval)
#   --strict        only the residual stops the Newton solves (as in verify_conjectures.jl)
#   --problems=…    harmonic_oscillator,pendulum (default both)
#   --reg=λ         regularization_factor of the ShallowNet Newton solves (default 1e-5)
#   --plot-only     only redraw the figures from the existing relu_k_sweep_<solver>.csv
#   --solver=dogleg|newton   nonlinear solver of the ShallowNet steps (default dogleg):
#                   `dogleg` = SimpleSolvers.DogLeg() (trust region, no line search),
#                   `newton` = SimpleSolvers.Newton() with Backtracking line search.
#                   CGVI always uses the package default solver.
#   --act=relu,tanh activations (default both). tanh is compared, for every k, with CGVI(P_k)
#                   on the same R = k+1 Gauss points and with ReLU^k at the same width S.
#   --tanh-bias=default|lo,hi   OGA bias interval for tanh (default [−π, π], the package default)
#
# For every (problem, k, S, h) it reports the errors of ShallowNet and CGVI(P_k) against a
# Gauss(8) reference, their difference, the difference to CGVI(P_{k+1}) (to show the network
# matches P_k specifically, not “some CGVI”), and the per-step classification of the network
# (P_k-equiv / spline / degenerate, see verify_conjectures.jl). Theorem 1 predicts
# diff ≈ solver tolerance whenever every step is P_k-equiv and every Newton solve converged.
#
# It also reports the relative Hamiltonian error max_n |H(q_n, p_n) − H(q_0, p_0)| / |H(q_0, p_0)|.
#
# Writes benchmark/results/relu_k_sweep_<solver>.{md,csv,pdf} (error in q) and
# relu_k_sweep_<solver>_hamiltonian.pdf (relative Hamiltonian error), <solver> = dogleg | newton.

include(joinpath(@__DIR__, "theory_common.jl"))
include(joinpath(@__DIR__, "..", "shallownet_benchmark_common.jl"))
import GeometricIntegratorsBase
using GeometricProblems.HarmonicOscillator
using GeometricProblems.Pendulum
using CairoMakie
using Statistics: median
const CBF = GeometricIntegrators.Integrators.CompactBasisFunctions

# ---------------------------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------------------------

argval(name, default) = (a = findfirst(s -> startswith(s, "--$name="), ARGS);
    a === nothing ? default : String(split(ARGS[a], "="; limit = 2)[2]))
const KS = parse.(Int, split(argval("k", "2,3,4,5"), ","))
const S_ARG = argval("S", "auto")
const R_ARG = argval("R", "auto")
const HS = parse.(Float64, split(argval("h", "0.2,0.5,1,2,5,10"), ","))
const FINAL_TIME = parse(Float64, argval("final-time", "20"))
const BIAS = argval("bias", "kinkfree") == "default" ? (-pi, pi) : (1.1, pi)
const PROBLEM_NAMES = split(argval("problems", "harmonic_oscillator,pendulum"), ",")
const STRICT = "--strict" in ARGS
const ACTS = filter(a -> a in ("relu", "tanh"), split(argval("act", "relu,tanh"), ","))
const SOLVER = lowercase(argval("solver", "dogleg"))
SOLVER in ("dogleg", "newton") || error("--solver must be dogleg or newton, got $SOLVER")
const OUTBASE = "relu_k_sweep_$(SOLVER)"
const TANH_BIAS = argval("tanh-bias", "default") == "default" ? (-pi, pi) : Tuple(parse.(Float64, split(argval("tanh-bias", ""), ",")))

widths(k) = S_ARG == "auto" ? unique([k + 1, k + 3, 2 * (k + 1)]) : parse.(Int, split(S_ARG, ","))
quadpoints(k) = R_ARG == "auto" ? k + 1 : parse(Int, R_ARG)

const DT_REF = 0.005
const NVI_REG = parse(Float64, argval("reg", "1e-5"))   # regularization_factor of the network solves
const NVI_MAXIT = 1000

# ---------------------------------------------------------------------------------------------
# problems (as in verify_conjectures.jl)
# ---------------------------------------------------------------------------------------------

const ALL_PROBLEMS = [
    (name = "harmonic_oscillator", ics = ([0.5], [0.0]),
        build = (q0, p0, ts, h) -> HarmonicOscillator.lodeproblem(q0, p0; timespan = ts,
            timestep = h, parameters = HarmonicOscillator.default_parameters(Float64)),
        # H = p²/(2m) + k q²/2
        energy = (q, p) -> HarmonicOscillator.hamiltonian(0.0, q, p, HarmonicOscillator.default_parameters(Float64))),
    (name = "pendulum",
        ics = let d = Pendulum.iodeproblem()
            (collect(Float64.(d.ics.q)), collect(Float64.(d.ics.p)))
        end,
        build = (q0, p0, ts, h) -> Pendulum.iodeproblem(q0, p0; timespan = ts,
            timestep = h, parameters = Pendulum.default_parameters(Float64)),
        # degenerate IODE: q = (θ, θ̇), canonical momentum m l² θ̇ ⇒ H = m l² q₂²/2 + m g l cos q₁
        energy = (q, p) -> let c = Pendulum.default_parameters(Float64)
            c.m * c.l^2 * q[2]^2 / 2 + c.m * c.g * c.l * cos(q[1])
        end)
]
const PROBLEMS = filter(P -> P.name in PROBLEM_NAMES, ALL_PROBLEMS)

const STRICT_CANDIDATES = (x_abstol = -1.0, x_reltol = -1.0, x_suctol = -1.0,
    f_reltol = -1.0, f_suctol = -1.0)
const STRICT_OPTS = let accepted = Pair{Symbol, Float64}[]
    if STRICT
        P = ALL_PROBLEMS[1]
        prob = P.build(copy(P.ics[1]), copy(P.ics[2]), (0.0, 0.1), 0.1)
        m = CGVI(CBF.Lagrange(QuadratureRules.nodes(QuadratureRules.GaussLegendreQuadrature(Float64, 2))),
            QuadratureRules.GaussLegendreQuadrature(Float64, 2))
        for (key, v) in pairs(STRICT_CANDIDATES)
            try
                GeometricIntegrator(prob, m; key => v)
                push!(accepted, key => v)
            catch e
                @warn "solver option $key not accepted, skipped" exception = e
            end
        end
        println("strict mode: solver options ", accepted)
    end
    accepted
end

# ---------------------------------------------------------------------------------------------
# helpers (as in verify_conjectures.jl)
# ---------------------------------------------------------------------------------------------

tomat(sol) = reduce(hcat, [Float64.(collect(q)) for q in collect(sol.q[:])])
tomat_p(sol) = reduce(hcat, [Float64.(collect(p)) for p in collect(sol.p[:])])

"""max_n |H(q_n, p_n) − H(q_0, p_0)| / |H(q_0, p_0)| over the columns of Q, Pm."""
function rel_energy_error(P, Q, Pm)
    H = [P.energy(Q[:, n], Pm[:, n]) for n in axes(Q, 2)]
    maximum(abs, H .- H[1]) / max(abs(H[1]), eps())
end

nl_solver_kwargs() = SOLVER == "dogleg" ? (solver = SimpleSolvers.DogLeg(),) :
    (solver = SimpleSolvers.Newton(), linesearch = SimpleSolvers.Backtracking(Float64))

function theta_from_x(x, D, S, d)
    pack([x[D * (i - 1) + d] for i in 1:S],
        [x[D * (S + 1) + D * (i - 1) + d] for i in 1:S],
        [x[D * (2S + 1) + D * (i - 1) + d] for i in 1:S])
end

function classify_step(θ, k; TT = collect(range(0, 1; length = 201)))
    cls = neuron_classes(θ, k)
    any(==("interior"), cls) && return "spline"
    out = findall(==("outside"), cls)
    isempty(out) && return "degenerate"
    a, w, b = unpack(θ)
    θo = pack(a[out], w[out], b[out])
    numrank(nn_tangent(θo, TT, k)[1]; rtol = 1e-9) == k + 1 ? "P_k-equiv" : "degenerate"
end

function shallownet_stepwise(P, h, N, method, S)
    q, p = copy(P.ics[1]), copy(P.ics[2])
    D = length(q)
    Q = [copy(q)]
    Pm = [copy(p)]
    thetas = Vector{Vector{Vector{Float64}}}()
    maxed = 0
    unconverged = 0
    for _ in 1:N
        prob = P.build(q, p, (0.0, h), h)
        int = GeometricIntegrator(prob, method; nl_solver_kwargs()...,
            regularization_factor = NVI_REG, max_iterations = NVI_MAXIT, STRICT_OPTS...)
        itcap = SimpleSolvers.config(solver(int)).max_iterations
        res = integrate(int)
        sol = res isa Tuple ? first(res) : res
        if res isa Tuple && length(res) >= 3 && res[3] isa AbstractVector{Bool}
            unconverged += count(!, res[3])
        end
        x = copy(GeometricIntegratorsBase.nlsolution(int))
        it = try
            Float64(solverstate(int).iterations)
        catch
            NaN
        end
        (!isnan(it) && it >= itcap) && (maxed += 1)
        q = Float64.(collect(collect(sol.q[:])[end]))
        p = Float64.(collect(collect(sol.p[:])[end]))
        push!(Q, copy(q))
        push!(Pm, copy(p))
        push!(thetas, [theta_from_x(x, D, S, d) for d in 1:D])
    end
    reduce(hcat, Q), reduce(hcat, Pm), thetas, maxed, unconverged
end

function cgvi_package(P, h, s, R)
    quad = QuadratureRules.GaussLegendreQuadrature(Float64, R)
    nodes = QuadratureRules.nodes(QuadratureRules.GaussLegendreQuadrature(Float64, s + 1))
    prob = P.build(copy(P.ics[1]), copy(P.ics[2]), (0.0, FINAL_TIME), h)
    res = integrate(prob, CGVI(CBF.Lagrange(nodes), quad); STRICT_OPTS...)
    sol = res isa Tuple ? first(res) : res
    tomat(sol), tomat_p(sol)
end

# ---------------------------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------------------------

"""
Network for activation `act` (\"relu\" → ReLUᵏ with the chosen bias interval, \"tanh\" → tanh
with TANH_BIAS). For tanh, `k` only fixes R = k+1 and the CGVI(P_k) it is compared with.
"""
function build_method(act, k, S, R)
    σ, bias = act == "relu" ? (relu_k(k), BIAS) : (tanh, TANH_BIAS)
    basis = ShallowNetBasis{Float64}(σ, S)
    ShallowNet(basis, QuadratureRules.GaussLegendreQuadrature(Float64, R);
        show_status = false, bias_interval = [bias[1], bias[2]], dict_amount = DICT_AMOUNT)
end

actlabel(act, k) = act == "relu" ? "ReLU^$k" : "tanh"

function main()
    outdir = joinpath(@__DIR__, "..", "results")
    mkpath(outdir)
    println("ShallowNet solver: ", SOLVER)
    csv = open(joinpath(outdir, OUTBASE * ".csv"), "w")
    println(csv, "problem,activation,k,S,R,h,params_SN,params_CGVI,err_sn,err_cgvi,diff,diff_alt,err_relu_same_S,ratio,herr_sn,herr_cgvi,herr_ratio,n_equiv,n_spline,n_degen,maxiter,unconverged,verdict,sec_sn,sec_cgvi")
    rows = NamedTuple[]
    for P in PROBLEMS
        ref = integrate(P.build(copy(P.ics[1]), copy(P.ics[2]), (0.0, FINAL_TIME), DT_REF), Gauss(8))
        Qref = tomat(ref)
        scale = maximum(abs, Qref)
        for k in KS
            R = quadpoints(k)
            cg = Dict{Float64, Any}()          # CGVI trajectories per h, shared by all networks
            relu_err = Dict{Tuple{Int, Float64}, Float64}()
            for act in ACTS, S in widths(k)
                if act == "relu" && S < k + 1
                    @warn "S = $S < k+1 = $(k + 1): ReLU^$k cannot span P_k, skipped"
                    continue
                end
                method = build_method(act, k, S, R)
                for h in HS
                    N = round(Int, FINAL_TIME / h)
                    abs(N * h - FINAL_TIME) < 1e-9 || (@warn "T not a multiple of h = $h, skipped"; continue)
                    stride = round(Int, h / DT_REF)
                    qref = Qref[:, 1:stride:end]
                    row = try
                        t0 = time()
                        Qsn, Psn, thetas, maxed, unconv = shallownet_stepwise(P, h, N, method, S)
                        tsn = time() - t0
                        t0 = time()
                        (Qc, Pc), (Qa, _) = get!(cg, h) do
                            (cgvi_package(P, h, k, R), cgvi_package(P, h, k + 1, max(R, k + 2)))
                        end
                        tc = time() - t0
                        err_sn = maximum(abs, Qsn .- qref) / scale
                        err_c = maximum(abs, Qc .- qref) / scale
                        diff = maximum(abs, Qsn .- Qc) / scale
                        diff_alt = maximum(abs, Qsn .- Qa) / scale
                        herr_sn = rel_energy_error(P, Qsn, Psn)
                        herr_c = rel_energy_error(P, Qc, Pc)
                        if act == "relu"
                            relu_err[(S, h)] = err_sn
                            classes = [classify_step(θ, k) for θs in thetas for θ in θs]
                            ne = count(==("P_k-equiv"), classes)
                            ns = count(==("spline"), classes)
                            nd = count(==("degenerate"), classes)
                            predicted = ns == 0 && nd == 0 && maxed == 0 && unconv == 0
                            equal = diff <= max(1e-9, 0.01 * err_c)
                            verdict = predicted ? (equal ? "CONFIRMED" : "VIOLATED") :
                                      (equal ? "equal anyway" : "differs")
                        else
                            # no equivalence theorem for tanh: report how it compares
                            ne, ns, nd = 0, 0, 0
                            r = err_sn / err_c
                            verdict = unconv > 0 || maxed > 0 ? "tanh (unconverged steps)" :
                                      r < 0.5 ? "tanh better" : r > 2 ? "tanh worse" : "tanh ≈ CGVI"
                        end
                        (; problem = P.name, act, k, S, R, h, err_sn, err_c, diff, diff_alt,
                            err_relu = act == "tanh" ? get(relu_err, (S, h), NaN) : NaN,
                            ratio = err_sn / err_c, herr_sn, herr_c, herr_ratio = herr_sn / herr_c,
                            ne, ns, nd, maxed, unconv, verdict, tsn, tc)
                    catch e
                        e isa InterruptException && rethrow()
                        @warn "$(P.name) $(actlabel(act, k)) S=$S h=$h failed" exception = e
                        (; problem = P.name, act, k, S, R, h, err_sn = NaN, err_c = NaN, diff = NaN,
                            diff_alt = NaN, err_relu = NaN, ratio = NaN, herr_sn = NaN, herr_c = NaN,
                            herr_ratio = NaN, ne = 0, ns = 0, nd = 0,
                            maxed = 0, unconv = 0, verdict = "error: $(nameof(typeof(e)))", tsn = NaN, tc = NaN)
                    end
                    push!(rows, row)
                    f(x) = isnan(x) ? "—" : @sprintf("%.2e", x)
                    @printf("  %-19s %-6s R=%d S=%-2d h=%-4g  errSN=%-9s errCGVI(P%d)=%-9s diff=%-9s dH SN/CGVI=%s/%s classes=%d/%d/%d maxit=%d unconv=%d  %s\n",
                        P.name, actlabel(act, k), R, S, h, f(row.err_sn), k, f(row.err_c), f(row.diff),
                        f(row.herr_sn), f(row.herr_c),
                        row.ne, row.ns, row.nd, row.maxed, row.unconv, row.verdict)
                    println(csv, join((P.name, actlabel(act, k), k, S, R, h, 3S, k + 1, row.err_sn,
                            row.err_c, row.diff, row.diff_alt, row.err_relu, row.ratio,
                            row.herr_sn, row.herr_c, row.herr_ratio, row.ne, row.ns,
                            row.nd, row.maxed, row.unconv, row.verdict, row.tsn, row.tc), ","))
                    flush(csv)
                end
            end
        end
    end
    close(csv)
    write_markdown(rows, joinpath(outdir, OUTBASE * ".md"))
    write_plots(rows, outdir)
end

fmt(x) = isnan(x) ? "—" : @sprintf("%.2e", x)
fmtr(x) = isnan(x) ? "—" : @sprintf("%.4f", x)

function write_markdown(rows, path)
    open(path, "w") do io
        println(io, "# ShallowNet(ReLUᵏ / tanh, S) vs CGVI(P_k) — sweep over k and S\n")
        println(io, "t ∈ (0, $(FINAL_TIME)); R = $(R_ARG == "auto" ? "k+1" : R_ARG) Gauss points for every method; " *
                    "OGA bias interval ReLU [$(round(BIAS[1]; digits = 2)), $(round(BIAS[2]; digits = 2))], " *
                    "tanh [$(round(TANH_BIAS[1]; digits = 2)), $(round(TANH_BIAS[2]; digits = 2))]; " *
                    "strict tolerances: $(STRICT ? string(STRICT_OPTS) : "no"); regularization_factor = $(NVI_REG); " *
                    "ShallowNet solver: $(SOLVER == "dogleg" ? "DogLeg (trust region)" : "Newton + Backtracking").")
        println(io, "`err` = max over the grid ‖q − q_ref‖∞ / max‖q_ref‖∞ (reference Gauss(8), dt = $(DT_REF)); " *
                    "`diff` = same between the network and CGVI(P_k); `diff P_{k+1}` = network vs CGVI(P_{k+1}); " *
                    "`ΔH` = max_n |H_n − H_0| / |H_0| (relative Hamiltonian error).")
        println(io, "ReLU: `CONFIRMED` ⇔ every step P_k-equivalent, every Newton solve converged, and diff ≤ max(1e-9, 0.01·err_CGVI).")
        println(io, "tanh: no equivalence theorem applies; `tanh better/worse` ⇔ err_tanh/err_CGVI < 0.5 / > 2, " *
                    "compared with CGVI(P_k) on the same R Gauss points and with ReLU^k at the same S.\n")

        println(io, "## Summary per (activation, k, S)\n")
        println(io, "| problem | activation | R | S | params net / CGVI | runs | CONFIRMED / better | VIOLATED / worse | other | max diff (CONFIRMED) | median err_net/err_CGVI | median ΔH_net/ΔH_CGVI |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|")
        for (pn, act, k, S) in unique([(r.problem, r.act, r.k, r.S) for r in rows])
            rs = filter(r -> r.problem == pn && r.act == act && r.k == k && r.S == S, rows)
            good = act == "relu" ? filter(r -> r.verdict == "CONFIRMED", rs) : filter(r -> r.verdict == "tanh better", rs)
            bad = count(r -> r.verdict == (act == "relu" ? "VIOLATED" : "tanh worse"), rs)
            ratios = filter(isfinite, [r.ratio for r in rs])
            maxd = act == "relu" && !isempty(good) ? fmt(maximum(r.diff for r in good)) : "—"
            hratios = filter(isfinite, [r.herr_ratio for r in rs])
            println(io, "| $pn | $(actlabel(act, k)) | $(k + 1) | $S | $(3S) / $(k + 1) | $(length(rs)) | $(length(good)) | $bad | " *
                        "$(length(rs) - length(good) - bad) | $maxd | $(isempty(ratios) ? "—" : fmtr(median(ratios))) | " *
                        "$(isempty(hratios) ? "—" : fmtr(median(hratios))) |")
        end
        println(io)
        for pn in unique(r.problem for r in rows)
            println(io, "## $pn\n")
            println(io, "| activation | R | S | h | err net | err CGVI(P_k) | err_net/err_CGVI | err ReLU^k (same S) | diff | diff P_{k+1} | ΔH net | ΔH CGVI(P_k) | P_k-equiv / spline / degen | maxiter / unconv | verdict | s (net / CGVI) |")
            println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in filter(r -> r.problem == pn, rows)
                println(io, "| $(actlabel(r.act, r.k)) | $(r.R) | $(r.S) | $(r.h) | $(fmt(r.err_sn)) | $(fmt(r.err_c)) | " *
                            "$(fmtr(r.ratio)) | $(fmt(r.err_relu)) | $(fmt(r.diff)) | $(fmt(r.diff_alt)) | " *
                            "$(fmt(r.herr_sn)) | $(fmt(r.herr_c)) | " *
                            "$(r.act == "relu" ? "$(r.ne) / $(r.ns) / $(r.nd)" : "n/a") | $(r.maxed) / $(r.unconv) | $(r.verdict) | " *
                            "$(isnan(r.tsn) ? "—" : @sprintf("%.1f", r.tsn)) / $(isnan(r.tc) ? "—" : @sprintf("%.2f", r.tc)) |")
            end
            println(io)
        end
    end
    println("Wrote ", path)
end

"""
Figure with one panel per (problem, k): CGVI(P_k) as a black line, the networks as markers.
`fsn` / `fc` pick the plotted quantity of the network / of CGVI, `ylab` labels the y axis.
"""
function write_plot(rows, path; fsn = r -> r.err_sn, fc = r -> r.err_c, ylab = "max rel. error in q")
    pnames = unique(r.problem for r in rows)
    ks = unique(r.k for r in rows)
    markers = [:circle, :utriangle, :diamond, :rect, :star5, :hexagon]
    colors = Makie.wong_colors()
    fig = Figure(size = (340 * length(ks), 300 * length(pnames) + 90))
    # widths are drawn by their position in widths(k) so that one legend serves every panel
    for (i, pn) in enumerate(pnames), (j, k) in enumerate(ks)
        ax = Axis(fig[i, j]; xscale = log10, yscale = log10, xlabel = "h",
            ylabel = j == 1 ? ylab : "", title = "$pn, k = $k (R = $(k + 1))",titlesize = 20)
        rs = filter(r -> r.problem == pn && r.k == k, rows)
        isempty(rs) && continue
        base = Dict(r.h => fc(r) for r in rs if isfinite(fc(r)) && fc(r) > 0)
        if !isempty(base)
            hs = sort(collect(keys(base)))
            lines!(ax, hs, [base[h] for h in hs]; color = :black, linewidth = 2)
        end
        for act in ACTS, (w, S) in enumerate(widths(k))
            rr = sort(filter(r -> r.act == act && r.S == S && isfinite(fsn(r)) && fsn(r) > 0, rs); by = r -> r.h)
            isempty(rr) && continue
            m = markers[mod1(w, length(markers))]
            c = colors[mod1(w, length(colors))]
            if act == "relu"
                scatter!(ax, [r.h for r in rr], [fsn(r) for r in rr]; marker = m, markersize = 16, color = c)
            else
                scatter!(ax, [r.h for r in rr], [fsn(r) for r in rr]; marker = m, markersize = 16,
                    color = :transparent, strokecolor = c, strokewidth = 1.8)
            end
        end
    end
    # one legend for the whole figure, below the panels
    nw = maximum(length(widths(k)) for k in ks)
    wlabel(w) = S_ARG == "auto" ? ("S = k+1", "S = k+3", "S = 2(k+1)")[w] : "S = $(widths(first(ks))[w])"
    elems = Any[LineElement(color = :black, linewidth = 2)]
    labels = String["CGVI(P_k), R = k+1"]
    for act in ACTS, w in 1:nw
        m = markers[mod1(w, length(markers))]
        c = colors[mod1(w, length(colors))]
        if act == "relu"
            push!(elems, MarkerElement(marker = m, color = c, markersize = 10))
            push!(labels, "ReLU^k, " * wlabel(w))
        else
            push!(elems, MarkerElement(marker = m, color = :transparent, strokecolor = c,
                strokewidth = 1.8, markersize = 13))
            push!(labels, "tanh, " * wlabel(w))
        end
    end
    Legend(fig[length(pnames) + 1, :], elems, labels; orientation = :horizontal,
        nbanks = length(ACTS), framevisible = false, labelsize = 24)
    save(path, fig)
    println("Wrote ", path)
end

"""Both figures: error in q and relative Hamiltonian error."""
function write_plots(rows, outdir)
    for (name, kw) in ((OUTBASE * ".pdf", (;)),
                       (OUTBASE * "_hamiltonian.pdf", (fsn = r -> r.herr_sn, fc = r -> r.herr_c,
                            ylab = "max rel. Hamiltonian error")))
        try
            write_plot(rows, joinpath(outdir, name); kw...)
        catch e
            @warn "plot $name failed" exception = e
        end
    end
end

"""Re-draw the figures from an existing results/relu_k_sweep_<solver>.csv (no integration)."""
function plot_only()
    path = joinpath(@__DIR__, "..", "results", OUTBASE * ".csv")
    lines_ = readlines(path)
    hdr = split(lines_[1], ",")
    col(name) = findfirst(==(name), hdr)
    num(x) = x == "NaN" ? NaN : parse(Float64, x)
    getnum(f, name) = (c = col(name); c === nothing ? NaN : num(f[c]))   # older csv: no ΔH columns
    rows = NamedTuple[]
    for l in lines_[2:end]
        f = split(l, ",")
        push!(rows, (problem = String(f[col("problem")]),
            act = startswith(f[col("activation")], "ReLU") ? "relu" : "tanh",
            k = parse(Int, f[col("k")]), S = parse(Int, f[col("S")]), h = num(f[col("h")]),
            err_sn = num(f[col("err_sn")]), err_c = num(f[col("err_cgvi")]),
            herr_sn = getnum(f, "herr_sn"), herr_c = getnum(f, "herr_cgvi")))
    end
    write_plots(rows, joinpath(@__DIR__, "..", "results"))
end

"--plot-only" in ARGS ? plot_only() : main()
