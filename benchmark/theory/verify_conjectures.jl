# Numerical tests of the conjectures / predictions C1–C6 in `relu_cgvi_equivalence.md`.
#
#   julia --project=benchmark benchmark/theory/verify_conjectures.jl                 # everything
#   julia --project=benchmark benchmark/theory/verify_conjectures.jl C1 C3a          # some package configs
#   julia --project=benchmark benchmark/theory/verify_conjectures.jl --reference-only   # C4–C6 only
#   julia --project=benchmark benchmark/theory/verify_conjectures.jl --package-only --final-time=40
#
# Part A (C1–C3) runs the real `ShallowNet` of NonlinearIntegrators against `CGVI` of
# GeometricIntegrators, step by step, and reads the network parameters θ after every step
# (`nlsolution(int)`), so that every step can be classified by the theory:
#
#   P_k-equiv   every neuron is either kink-free (kink ∉ [0,1]) or has zero amplitude, and the
#               kink-free neurons span P_k (rank of their tangent vectors = k+1)
#               → Theorem 1 / Prop. 4.2 predict: this step is exactly the CGVI(P_k) step
#   spline      some neuron with non-zero amplitude has its kink inside (0,1)   → Theorem 3
#   degenerate  kink-free, but the tangent space is smaller than P_k (coalesced kinks) → Prop. 4.1
#
# and the prediction is compared with the measured difference to CGVI.
#
# Part B (C4–C6) uses the stdlib-only reference implementation in `theory_common.jl`.
#
# Writes benchmark/results/relu_cgvi_conjectures.csv and .md.

include(joinpath(@__DIR__, "theory_common.jl"))

const RUN_PACKAGE = !("--reference-only" in ARGS)
const RUN_REFERENCE = !("--package-only" in ARGS)
const FINAL_TIME = let a = findfirst(s -> startswith(s, "--final-time="), ARGS)
    a === nothing ? 20.0 : parse(Float64, split(ARGS[a], "=")[2])
end
const SELECTED = filter(s -> !startswith(s, "--"), ARGS)
const DTS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

const OUTDIR = joinpath(@__DIR__, "..", "results")
mkpath(OUTDIR)
const MD_LINES = String[]
md(s = "") = push!(MD_LINES, s)

Random.seed!(7)

# ===========================================================================
# Part A — the package
# ===========================================================================

if RUN_PACKAGE
    include(joinpath(@__DIR__, "..", "shallownet_benchmark_common.jl"))
    import GeometricIntegratorsBase
    using GeometricProblems.HarmonicOscillator
    using GeometricProblems.Pendulum
    const CBF = GeometricIntegrators.Integrators.CompactBasisFunctions
end

if RUN_PACKAGE

const DT_REF = 0.005
const NVI_REG = 1e-5
const NVI_MAXIT = 1000

# --strict: disable every stopping criterion except the residual one, so that "converged"
# means the equations are actually solved (f_abstol = the integrator's eps-scaled default).
# A negative tolerance never triggers. Options the solver does not know are dropped after a
# probe, so this works whatever the SimpleSolvers version accepts.
const STRICT = "--strict" in ARGS
const STRICT_CANDIDATES = (x_abstol = -1.0, x_reltol = -1.0, x_suctol = -1.0,
    f_reltol = -1.0, f_suctol = -1.0)

const PKG_PROBLEMS = [
    (name = "harmonic_oscillator", ics = ([0.5], [0.0]),
        build = (q0, p0, ts, h) -> HarmonicOscillator.lodeproblem(q0, p0; timespan = ts,
            timestep = h, parameters = HarmonicOscillator.default_parameters(Float64))),
    (name = "pendulum",
        ics = let d = Pendulum.iodeproblem()
            (collect(Float64.(d.ics.q)), collect(Float64.(d.ics.p)))
        end,
        build = (q0, p0, ts, h) -> Pendulum.iodeproblem(q0, p0; timespan = ts,
            timestep = h, parameters = Pendulum.default_parameters(Float64)))
]

const STRICT_OPTS = let accepted = Pair{Symbol, Float64}[]
    if STRICT
        P = PKG_PROBLEMS[1]
        prob = P.build(copy(P.ics[1]), copy(P.ics[2]), (0.0, 0.1), 0.1)
        m = CGVI(CBF.Lagrange(QuadratureRules.nodes(QuadratureRules.GaussLegendreQuadrature(Float64, 2))),
            QuadratureRules.GaussLegendreQuadrature(Float64, 2))
        for (k, v) in pairs(STRICT_CANDIDATES)
            try
                GeometricIntegrator(prob, m; k => v)
                push!(accepted, k => v)
            catch e
                @warn "solver option $k not accepted, skipped" exception = e
            end
        end
        println("strict mode: solver options ", accepted)
    end
    accepted
end

# id, activation power k, width S, OGA bias interval, CGVI degree predicted equal, a second
# CGVI degree predicted *different* (or nothing)
const PKG_CONFIGS = [
    (id = "C1", k = 3, S = 4, bias = (-pi, pi), s = 3, s_alt = nothing),
    (id = "C2a", k = 3, S = 6, bias = (-pi, pi), s = 3, s_alt = nothing),
    (id = "C2b", k = 3, S = 6, bias = (1.1, pi), s = 3, s_alt = nothing),
    (id = "C2c", k = 3, S = 8, bias = (-pi, pi), s = 3, s_alt = nothing),
    (id = "C2d", k = 3, S = 8, bias = (1.1, pi), s = 3, s_alt = nothing),
    (id = "C3a", k = 2, S = 4, bias = (-pi, pi), s = 2, s_alt = 3),
    (id = "C3b", k = 2, S = 4, bias = (1.1, pi), s = 2, s_alt = 3)
]

tomat(sol) = reduce(hcat, [Float64.(collect(q)) for q in collect(sol.q[:])])

function theta_from_x(x, D, S, d)
    pack([x[D * (i - 1) + d] for i in 1:S],
        [x[D * (S + 1) + D * (i - 1) + d] for i in 1:S],
        [x[D * (2S + 1) + D * (i - 1) + d] for i in 1:S])
end

"""Classify one (step, dimension) network θ, see the header."""
function classify_step(θ, k; TT = collect(range(0, 1; length = 201)))
    cls = neuron_classes(θ, k)
    any(==("interior"), cls) && return "spline"
    out = findall(==("outside"), cls)
    isempty(out) && return "degenerate"
    a, w, b = unpack(θ)
    θo = pack(a[out], w[out], b[out])
    numrank(nn_tangent(θo, TT, k)[1]; rtol = 1e-9) == k + 1 ? "P_k-equiv" : "degenerate"
end

"""ShallowNet one step at a time (the start value only depends on the current state)."""
function shallownet_stepwise(P, h, N, method, S)
    q, p = copy(P.ics[1]), copy(P.ics[2])
    D = length(q)
    Q = [copy(q)]
    thetas = Vector{Vector{Vector{Float64}}}()
    maxed = 0
    unconverged = 0
    for _ in 1:N
        prob = P.build(q, p, (0.0, h), h)
        int = GeometricIntegrator(prob, method; solver = SimpleSolvers.Newton(),
            linesearch = SimpleSolvers.Backtracking(Float64),
            regularization_factor = NVI_REG, max_iterations = NVI_MAXIT, STRICT_OPTS...)
        itcap = SimpleSolvers.config(solver(int)).max_iterations
        res = integrate(int)
        sol = res isa Tuple ? first(res) : res
        # ShallowNet's integrate! returns (sol, internal_values, solver_status_vector);
        # a `false` there means the Newton solve stopped without meeting its tolerance
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
        push!(thetas, [theta_from_x(x, D, S, d) for d in 1:D])
    end
    reduce(hcat, Q), thetas, maxed, unconverged
end

function cgvi_package(P, h, s, R)
    quad = QuadratureRules.GaussLegendreQuadrature(Float64, R)
    nodes = QuadratureRules.nodes(QuadratureRules.GaussLegendreQuadrature(Float64, s + 1))
    prob = P.build(copy(P.ics[1]), copy(P.ics[2]), (0.0, FINAL_TIME), h)
    res = integrate(prob, CGVI(CBF.Lagrange(nodes), quad); STRICT_OPTS...)
    tomat(res isa Tuple ? first(res) : res)
end

function run_package_part()
    configs = isempty(SELECTED) ? PKG_CONFIGS : filter(c -> c.id in SELECTED, PKG_CONFIGS)
    md("## Part A — ShallowNet (package) vs CGVI (package)")
    md()
    md("Strict solver tolerances: $(STRICT ? string(STRICT_OPTS) : "no (package defaults)").")
    md("t ∈ (0, $(FINAL_TIME)), R = 4 Gauss points for every method, reference Gauss(8) at dt = $(DT_REF).")
    md("`diff` = max over the grid of ‖q_SN − q_CGVI‖∞ / max‖q_ref‖∞; `err` the same against the reference.")
    md("`equal` ⇔ diff ≤ max(1e-9, 0.01·err_CGVI). Classes count (step, dof) pairs.")
    md()
    md("| id | problem | h | k | S | bias | err SN | err CGVI(P_s) | diff to P_s | diff to P_alt | P_k-equiv / spline / degenerate | maxiter / unconverged | diff per dof | prediction | verdict |")
    md("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    csv = open(joinpath(OUTDIR, "relu_cgvi_conjectures.csv"), "w")
    println(csv, "id,problem,h,k,S,bias_lo,bias_hi,err_sn,err_cgvi,diff_cgvi,diff_alt,n_equiv,n_spline,n_degen,maxiter,unconverged,diff_dof,prediction,verdict")
    for P in PKG_PROBLEMS
        ref = integrate(P.build(copy(P.ics[1]), copy(P.ics[2]), (0.0, FINAL_TIME), DT_REF), Gauss(8))
        Qref = tomat(ref)
        scale = maximum(abs, Qref)
        for cfg in configs
            basis = ShallowNetBasis{Float64}(relu_k(cfg.k), cfg.S)
            method = ShallowNet(basis, QuadratureRules.GaussLegendreQuadrature(Float64, 4);
                show_status = false, bias_interval = [cfg.bias[1], cfg.bias[2]],
                dict_amount = DICT_AMOUNT)
            for h in DTS
                N = round(Int, FINAL_TIME / h)
                stride = round(Int, h / DT_REF)
                qref = Qref[:, 1:stride:end]
                row = try
                    Qsn, thetas, maxed, unconv = shallownet_stepwise(P, h, N, method, cfg.S)
                    Qc = cgvi_package(P, h, cfg.s, 4)
                    diff = maximum(abs, Qsn .- Qc) / scale
                    # per degree of freedom (the pendulum IODE has a degenerate 2nd dof, p₂ = 0)
                    diff_dof = [maximum(abs, Qsn[d, :] .- Qc[d, :]) / scale for d in axes(Qsn, 1)]
                    diff_alt = cfg.s_alt === nothing ? NaN :
                               maximum(abs, Qsn .- cgvi_package(P, h, cfg.s_alt, 4)) / scale
                    err_sn = maximum(abs, Qsn .- qref) / scale
                    err_c = maximum(abs, Qc .- qref) / scale
                    classes = [classify_step(θ, cfg.k) for θs in thetas for θ in θs]
                    ne, ns, nd = count(==("P_k-equiv"), classes), count(==("spline"), classes),
                    count(==("degenerate"), classes)
                    predicted_equal = ns == 0 && nd == 0 && maxed == 0 && unconv == 0
                    equal = diff <= max(1e-9, 0.01 * err_c)
                    prediction = predicted_equal ? "= CGVI(P$(cfg.s))" : "none (not all steps P_k-equiv)"
                    verdict = predicted_equal ? (equal ? "CONFIRMED" : "VIOLATED") :
                              (equal ? "equal anyway" : "differs")
                    (; err_sn, err_c, diff, diff_alt, ne, ns, nd, maxed, unconv, diff_dof, prediction, verdict)
                catch e
                    e isa InterruptException && rethrow()
                    @warn "$(cfg.id) $(P.name) h=$h failed" exception = e
                    (; err_sn = NaN, err_c = NaN, diff = NaN, diff_alt = NaN, ne = 0, ns = 0, nd = 0,
                        maxed = 0, unconv = 0, diff_dof = Float64[], prediction = "—", verdict = "error: $(nameof(typeof(e)))")
                end
                f(x) = isnan(x) ? "—" : @sprintf("%.2e", x)
                @printf("  %-4s %-19s h=%-4g k=%d S=%d bias=[%.2f,%.2f]  errSN=%s errCG=%s diff=%s (per dof %s) diffAlt=%s classes=%d/%d/%d maxit=%d unconv=%d  %s\n",
                    cfg.id, P.name, h, cfg.k, cfg.S, cfg.bias[1], cfg.bias[2], f(row.err_sn),
                    f(row.err_c), f(row.diff), join(f.(row.diff_dof), "/"), f(row.diff_alt),
                    row.ne, row.ns, row.nd, row.maxed, row.unconv, row.verdict)
                md("| $(cfg.id) | $(P.name) | $(h) | $(cfg.k) | $(cfg.S) | [$(round(cfg.bias[1]; digits = 2)), $(round(cfg.bias[2]; digits = 2))] | " *
                   "$(f(row.err_sn)) | $(f(row.err_c)) | $(f(row.diff)) | $(f(row.diff_alt)) | " *
                   "$(row.ne) / $(row.ns) / $(row.nd) | $(row.maxed) / $(row.unconv) | $(join(f.(row.diff_dof), " / ")) | $(row.prediction) | $(row.verdict) |")
                println(csv, join((cfg.id, P.name, h, cfg.k, cfg.S, cfg.bias[1], cfg.bias[2],
                        row.err_sn, row.err_c, row.diff, row.diff_alt, row.ne, row.ns, row.nd,
                        row.maxed, row.unconv, join(row.diff_dof, ";"), row.prediction, row.verdict), ","))
                flush(csv)
                # the theory makes a hard prediction only for all-P_k-equivalent runs
                row.verdict == "VIOLATED" &&
                    vcheck("A-" * cfg.id, "$(P.name) h=$h: predicted = CGVI(P$(cfg.s)) but diff", row.diff,
                        max(1e-9, 0.01 * row.err_c))
                row.verdict == "CONFIRMED" &&
                    vcheck("A-" * cfg.id, "$(P.name) h=$h: all steps P_k-equiv ⇒ = CGVI(P$(cfg.s))", row.diff,
                        max(1e-9, 0.01 * row.err_c))
                (cfg.s_alt !== nothing && row.verdict == "CONFIRMED") &&
                    vcheck_ge("A-" * cfg.id * "'", "$(P.name) h=$h: and ≠ CGVI(P$(cfg.s_alt))",
                        row.diff_alt, 10 * max(row.diff, 1e-12))
            end
        end
    end
    close(csv)
    md()
end

end # if RUN_PACKAGE

# ===========================================================================
# Part B — reference implementation
# ===========================================================================

order_est(e1, e2, h1, h2) = log(e2 / e1) / log(h2 / h1)

function err_curve(stepper, hs, Tf; q0 = 0.5, p0 = 0.0)
    map(hs) do h
        N = round(Int, Tf / h)
        q = [q0]
        p = [p0]
        for _ in 1:N
            q1, p1 = stepper(q[end], p[end], h)
            push!(q, q1)
            push!(p, p1)
        end
        t = (0:N) .* h
        maximum(abs.(q .- ho_exact.(q0, p0, t))) / q0
    end
end

lin_stepper(phi, quad) = (q, p, h) -> (s = linear_step(q, p, h, phi, quad, TOY_HO);
    s.converged || error("step failed"); (s.q1, s.p1))

function substep_stepper(phi, quad, nsub)
    (q, p, h) -> begin
        for _ in 1:nsub
            s = linear_step(q, p, h / nsub, phi, quad, TOY_HO)
            s.converged || error("step failed")
            q, p = s.q1, s.p1
        end
        (q, p)
    end
end

fmt_row(v) = join([@sprintf("%.2e", x) for x in v], " | ")

function run_reference_part()
    hs = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    Tf = 10.0
    md("## Part B — reference implementation (harmonic oscillator, exact solution, T = $(Tf))")
    md()

    # ---- C4 — frozen interior knots: spline Galerkin VI -----------------------
    section("C4  frozen interior knots (spline Galerkin VI), global vs composite quadrature")
    md("### C4 — frozen knots: max relative error in q")
    md()
    md("| method | " * join(["h=$(h)" for h in hs], " | ") * " |")
    md("|---|" * repeat("---|", length(hs)))
    curves = Dict{String, Vector{Float64}}()
    k = 3
    specs = Any[("CGVI P3 R4", lin_stepper(cgvi_basis(3), QRule01(4)))]
    for m in (1, 3)
        Z = collect((1:m) ./ (m + 1))
        push!(specs, ("spline m=$m, global R4", lin_stepper(truncated_power_basis(k, Z), QRule01(4))))
        push!(specs, ("spline m=$m, global R$(4(m + 1))",
            lin_stepper(truncated_power_basis(k, Z), QRule01(4 * (m + 1)))))
        push!(specs, ("spline m=$m, composite R4", lin_stepper(truncated_power_basis(k, Z), composite_rule(4, Z))))
        wk, bk = kinkfree_neurons(k + 1)
        push!(specs, ("frozen network m=$m, composite R4",
            lin_stepper(neuron_basis(k, vcat(wk, ones(m)), vcat(bk, -Z)), composite_rule(4, Z))))
        push!(specs, ("CGVI P3 R4 on $(m + 1) substeps", substep_stepper(cgvi_basis(3), QRule01(4), m + 1)))
    end
    for (name, st) in specs
        e = try
            err_curve(st, hs, Tf)
        catch err
            err isa InterruptException && rethrow()
            fill(NaN, length(hs))
        end
        curves[name] = e
        @printf("  %-36s %s\n", name, join([@sprintf("%9.2e", x) for x in e], " "))
        md("| $(name) | " * fmt_row(e) * " |")
    end
    md()
    for m in (1, 3)
        sp = curves["spline m=$m, composite R4"]
        fr = curves["frozen network m=$m, composite R4"]
        vcheck("C4a", "m=$m: frozen network ≡ spline Galerkin VI (max |Δerr| over h)",
            maximum(abs.(sp .- fr)), 1e-11)
        vcheck("C4b", "m=$m: composite quadrature keeps order 6 (|p−6|, h=0.2→0.5)",
            abs(order_est(sp[2], sp[3], hs[2], hs[3]) - 6), 0.4)
        sub = curves["CGVI P3 R4 on $(m + 1) substeps"]
        ratio = sp[2] / sub[2]
        @printf("      m=%d: err(spline) / err(CGVI on %d substeps) at h=0.2: %.3f\n", m, m + 1, ratio)
        vcheck("C4c", "m=$m: spline error ≈ CGVI on m+1 substeps (|log10 ratio|, h=0.2)",
            abs(log10(ratio)), 0.7)
    end
    g1 = curves["spline m=1, global R4"]
    vcheck("C4d", "global R4 quadrature destroys the order (observed order h=0.1→0.2)",
        order_est(g1[1], g1[2], hs[1], hs[2]), 3.0)

    # ---- C5 — order of CGVI(P_s, R) -----------------------------------------
    section("C5  convergence order of CGVI(P_s) with R Gauss points: 2s for R ≥ s")
    md("### C5 — CGVI(P_s, R): observed order (h = 0.2 → 0.5)")
    md()
    md("| s | R | predicted | observed | errors (h = " * join(hs, ", ") * ") |")
    md("|---|---|---|---|---|")
    for (s, R) in ((2, 3), (2, 4), (3, 3), (3, 4), (4, 4), (4, 5), (4, 3), (3, 2))
        e = try
            err_curve(lin_stepper(cgvi_basis(s), QRule01(R)), hs, Tf)
        catch err
            err isa InterruptException && rethrow()
            fill(NaN, length(hs))
        end
        o = order_est(e[2], e[3], hs[2], hs[3])
        pred = R >= s ? string(2s) : "(outside hypothesis)"
        @printf("  s=%d R=%d  predicted %-22s observed %.3f\n", s, R, pred, o)
        md("| $s | $R | $pred | $(@sprintf("%.3f", o)) | $(fmt_row(e)) |")
        R >= s && vcheck("C5", "CGVI(P$s, R=$R): order 2s = $(2s)", abs(o - 2s), 0.3)
    end
    md()

    # ---- C6 — free knots: what the network equations converge to --------------
    section("C6  free knots (k=3, S=5, R=8, pendulum h=1): classify every converged LM solve")
    md("### C6 — free-knot network solves (k = 3, S = 5, global R = 8, pendulum, h = 1)")
    md()
    md("| start kink z₀ | start a₅ | converged | class | final interior kink | \\|q₁ − q₁(CGVI P₃)\\| | theory check | value |")
    md("|---|---|---|---|---|---|---|---|")
    let quad = QRule01(8), prob = TOY_PEND, h = 1.0, q0 = 0.5, p0 = 0.1
        cg = linear_step(q0, p0, h, cgvi_basis(3), quad, prob)
        wk, bk = kinkfree_neurons(4)
        base = fit_output_weights(TT_FIT, q0 .+ h * p0 .* TT_FIT, wk, bk, k)
        a4, _, _ = unpack(base)
        for z0 in (0.2, 0.4, 0.6, 0.8), a5 in (1e-3, 0.1)
            θ0 = pack(vcat(a4, a5), vcat(wk, 1.0), vcat(bk, -z0))
            sol = nn_step(q0, p0, h, k, quad, prob, vcat(θ0, p0))
            cls = neuron_classes(sol.θ, k)
            a, w, b = unpack(sol.θ)
            zi = findfirst(c -> c == "interior" || c == "interior0", cls)
            zfinal = zi === nothing ? NaN : -b[zi] / w[zi]
            class = any(==("interior"), cls) ? "spline" :
                    (numrank(nn_tangent(pack(a[cls .== "outside"], w[cls .== "outside"],
                            b[cls .== "outside"]), TT_FIT, k)[1]; rtol = 1e-9) == k + 1 ? "P_k-equiv" :
                     "degenerate")
            dq = abs(sol.q1 - cg.q1)
            what, val, ok = "", NaN, true
            if !sol.converged
                what = "not converged"
            elseif class == "P_k-equiv"
                what = "= CGVI(P₃)"
                val = max(dq, abs(sol.p1 - cg.p1))
                ok = vcheck("C6", "z₀=$z0 a₅=$a5: P_k-equivalent solution equals CGVI", val, 1e-9)
            elseif class == "spline"
                # Theorem 3: the step is the fixed-knot (type-I) Galerkin step on
                # S_3(Z) = P_3 ⊕ span{(τ − zⱼ)₊³} with the network's interior kinks Z, and every
                # interior knot is stationary for L_d^Z(qₙ, qₙ₊₁).
                out = cls .== "outside"
                Zint = [-b[i] / w[i] for i in eachindex(cls) if cls[i] == "interior"]
                if numrank(nn_tangent(pack(a[out], w[out], b[out]), TT_FIT, k)[1]; rtol = 1e-9) != k + 1
                    what = "interior kink on a degenerate kink-free part (not covered)"
                else
                    e = 1e-6
                    T1(Z) = typeI_step(q0, sol.q1, h, truncated_power_basis(k, Z), quad, prob)
                    s1 = T1(Zint)
                    grads = map(eachindex(Zint)) do j
                        Zp = copy(Zint); Zp[j] += e
                        Zm = copy(Zint); Zm[j] -= e
                        (T1(Zp).Ld - T1(Zm).Ld) / (2 * e)
                    end
                    val = max(abs(s1.p_a - p0), abs(s1.p_b - sol.p1), maximum(abs, grads))
                    what = "type-I on S₃(Z) reproduces the step, ∇_Z L_d = 0"
                    ok = vcheck("C6", "z₀=$z0 a₅=$a5: spline solution satisfies Theorem 3", val, 1e-6)
                end
            else
                what = "degenerate stratum (Prop. 4.1)"
            end
            md("| $z0 | $a5 | $(sol.converged) | $class | $(isnan(zfinal) ? "—" : @sprintf("%.4f", zfinal)) | " *
               "$(@sprintf("%.2e", dq)) | $what | $(isnan(val) ? "—" : @sprintf("%.2e", val)) |")
            @printf("      z₀=%.1f a₅=%-6g converged=%-5s class=%-10s z_final=%8.4f |Δq₁|=%.2e\n",
                z0, a5, sol.converged, class, zfinal, dq)
        end
    end
    md()
end

# ===========================================================================

md("# ReLUᵏ ShallowNet ⇔ CGVI — conjecture tests")
md()
md("Generated by `benchmark/theory/verify_conjectures.jl`.")
md()
RUN_PACKAGE && run_package_part()
RUN_REFERENCE && run_reference_part()

open(joinpath(OUTDIR, "relu_cgvi_conjectures.md"), "w") do io
    foreach(l -> println(io, l), MD_LINES)
end
println("Wrote ", joinpath(OUTDIR, "relu_cgvi_conjectures.md"))
ok = summarize_checks()
ok || exit(1)
