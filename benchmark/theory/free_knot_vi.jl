# Free-knot spline variational integrator (Theorem 3 of relu_cgvi_equivalence.md) and a
# comparison with CGVI, CGVI on substeps and the fixed-knot spline VI: parameters, FLOPs, errors.
#
#   julia benchmark/theory/free_knot_vi.jl                    # defaults below
#   julia benchmark/theory/free_knot_vi.jl --T=40 --k=3 --R=4 --h=0.5,1,2,5 --m=1,2,3
#
# Needs only the standard library (+ theory_common.jl). Writes
# benchmark/results/free_knot_vi.{md,csv}.
#
# ── The method ───────────────────────────────────────────────────────────────────────────────
# One step [tₙ, tₙ+h], τ ∈ [0,1]. Trial space S_k(Z) = P_k ⊕ span{(τ − zᵢ)₊ᵏ}, i = 1..m.
#
#   inner  (fixed knots Z):  Galerkin VI on S_k(Z), composite R-point Gauss rule split at Z,
#          unknowns (c, pₙ₊₁), Newton with the analytic Jacobian. Regular — no gauge freedom.
#   outer  (knots):          solve g(Z) = ∂L_d^{Z}(qₙ, qₙ₊₁(Z))/∂Z = 0 (Theorem 3), with a
#          safeguarded Newton; g is analytic (envelope formula + the motion of the composite
#          quadrature nodes), its Jacobian by forward differences (m extra inner solves).
#
# The network (ShallowNet with k+1+m ReLUᵏ neurons, m of them with interior kinks) solves the
# same equations in one Newton over 3(k+1+m)+1 unknowns; this is the same solution split into
# a well-posed inner problem and a small outer one.
#
# ── Compared methods (same k, same R) ───────────────────────────────────────────────────────
#   CGVI            P_k, R Gauss points                                   (Theorem 1: = kink-free net)
#   CGVI substeps   P_k on m+1 substeps of h/(m+1)
#   fixed spline    S_k(Z), Z uniform, composite R                        (Theorem 2: = frozen-knot net)
#   free spline     S_k(Z*), Z* variational, composite R moving with Z*   (Theorem 3: = free-knot net)
#
# ── FLOP model ─────────────────────────────────────────────────────────────────────────────
# FLOPs are *counted by the code* with the following per-operation model (an efficient
# implementation with analytic Jacobians; the counts are not wall-clock measurements):
#   basis at N quadrature nodes (values, 1st, 2nd derivative), n functions : 3·(k+2)·N·n
#   residual        : 8·N·n + N·c_f                (q, q′ at nodes, force, two projections)
#   Jacobian        : 4·N·n² + 2·N·n + N·c_f
#   LU solve (u = n+1 unknowns) : (2/3)u³ + 2u²
#   start value (least squares fit, 21 points) : 2·21·n²
#   knot gradient g : m·N·(4(k+2) + 30)
# c_f = cost of one V′/V″ evaluation: 1 (harmonic oscillator), 10 (pendulum, sin/cos).
# Fixed-knot methods evaluate their basis once per run (it does not change); the free-knot
# method re-evaluates it every time the knots move.

include(joinpath(@__DIR__, "theory_common.jl"))

# ---------------------------------------------------------------------------------------------
# problems  L = v²/2 − V(q),  H = p²/2 + V(q)
# ---------------------------------------------------------------------------------------------

struct FKProblem
    name::String
    V::Function
    dV::Function
    d2V::Function
    cf::Int
end
const FK_HO = FKProblem("harmonic_oscillator", q -> q^2 / 2, q -> q, q -> one(q), 1)
const FK_PEND = FKProblem("pendulum", q -> 1 - cos(q), q -> sin(q), q -> cos(q), 10)
energy(prob::FKProblem, q, p) = p^2 / 2 + prob.V(q)

# ---------------------------------------------------------------------------------------------
# FLOP counter
# ---------------------------------------------------------------------------------------------

mutable struct Counter
    flops::Float64
    newton::Int      # inner Newton iterations
    inner::Int       # inner solves
    outer::Int       # outer (knot) iterations
    fallback::Int    # steps whose knot equation did not converge
end
const CNT = Counter(0, 0, 0, 0, 0)
reset!(c::Counter) = (c.flops = 0; c.newton = 0; c.inner = 0; c.outer = 0; c.fallback = 0; c)
addf!(x) = (CNT.flops += x)

# ---------------------------------------------------------------------------------------------
# S_k(Z) in the truncated-power basis 1, τ, …, τᵏ, (τ − zᵢ)₊ᵏ  (Z = [] gives P_k, i.e. CGVI:
# the Galerkin solution does not depend on the basis, Theorem 2 / check V5.3)
# ---------------------------------------------------------------------------------------------

function spline_PDE(k, Z, t::AbstractVector)
    n = k + 1 + length(Z)
    P = zeros(length(t), n)
    D = zeros(length(t), n)
    E = zeros(length(t), n)
    for (r, τ) in enumerate(t)
        for j in 0:k
            P[r, j + 1] = τ^j
            D[r, j + 1] = j == 0 ? 0.0 : j * τ^(j - 1)
            E[r, j + 1] = j <= 1 ? 0.0 : j * (j - 1) * τ^(j - 2)
        end
        for (i, z) in enumerate(Z)
            P[r, k + 1 + i] = rp(τ - z, k)
            D[r, k + 1 + i] = k * rp(τ - z, k - 1)
            E[r, k + 1 + i] = k * (k - 1) * rp(τ - z, k - 2)
        end
    end
    addf!(3 * (k + 2) * length(t) * n)
    P, D, E
end

"""Everything the inner solver needs for a given knot set: basis at nodes and end points."""
struct StepSpace
    k::Int
    Z::Vector{Float64}
    c::Vector{Float64}          # quadrature nodes
    b::Vector{Float64}          # weights
    piece::Vector{Int}          # which sub-interval each node belongs to
    c0::Vector{Float64}         # reference Gauss nodes/weights on [0,1]
    b0::Vector{Float64}
    P::Matrix{Float64}
    D::Matrix{Float64}
    E::Matrix{Float64}
    r0::Vector{Float64}
    r1::Vector{Float64}
end

function StepSpace(k, Z, R; composite = true)
    Z = sort(collect(Float64, Z))
    c0, b0 = gauss_legendre01(R)
    br = composite ? vcat(0.0, Z, 1.0) : [0.0, 1.0]
    c = Float64[]
    b = Float64[]
    piece = Int[]
    for i in 1:(length(br) - 1)
        l, r = br[i], br[i + 1]
        append!(c, l .+ (r - l) .* c0)
        append!(b, (r - l) .* b0)
        append!(piece, fill(i, R))
    end
    P, D, E = spline_PDE(k, Z, c)
    r0 = vec(spline_PDE(k, Z, [0.0])[1])
    r1 = vec(spline_PDE(k, Z, [1.0])[1])
    StepSpace(k, Z, c, b, piece, c0, b0, P, D, E, r0, r1)
end
nfun(sp::StepSpace) = size(sp.P, 2)

# ---------------------------------------------------------------------------------------------
# inner solver: Galerkin VI step on a fixed space, analytic Jacobian
# ---------------------------------------------------------------------------------------------

function start_value(sp::StepSpace, q0, p0, h)
    tt = collect(range(0, 1; length = 21))
    Pt = spline_PDE(sp.k, sp.Z, tt)[1]
    n = nfun(sp)
    addf!(2 * 21 * n^2)
    vcat(Pt \ (q0 .+ h * p0 .* tt), p0)
end

"""
Solve the type-II step on `sp`. Returns (x = [c; pₙ₊₁], converged). `x0` is the start value.
"""
function inner_solve(sp::StepSpace, q0, p0, h, prob::FKProblem, x0; tol = 1e-13, maxit = 50)
    n = nfun(sp)
    N = length(sp.c)
    x = copy(x0)
    CNT.inner += 1
    for _ in 1:maxit
        X = view(x, 1:n)
        p1 = x[n + 1]
        Q = sp.P * X
        V = (sp.D * X) ./ h
        f = -prob.dV.(Q)
        res = vcat(sp.r1 .* p1 .- sp.r0 .* p0 .-
                   (sp.P' * (sp.b .* h .* f) .+ sp.D' * (sp.b .* V)),
            q0 - dot(sp.r0, X))
        addf!(8 * N * n + N * prob.cf)
        scale = max(1.0, abs(p0), abs(q0))
        norm(res, Inf) < tol * scale && return x, true
        # Jacobian:  ∂res/∂X = −(Pᵀ diag(b h (−V″)) P + Dᵀ diag(b/h) D),  ∂res/∂p₁ = r₁
        α = sp.b .* h .* (-prob.d2V.(Q))
        J = zeros(n + 1, n + 1)
        J[1:n, 1:n] .= -(sp.P' * (α .* sp.P) .+ sp.D' * ((sp.b ./ h) .* sp.D))
        J[1:n, n + 1] .= sp.r1
        J[n + 1, 1:n] .= -sp.r0
        addf!(4 * N * n^2 + 2 * N * n + N * prob.cf)
        u = n + 1
        addf!((2 / 3) * u^3 + 2 * u^2)
        x .-= J \ res
        CNT.newton += 1
    end
    x, false
end

# ---------------------------------------------------------------------------------------------
# knot gradient g(Z) = dL_d^Z(qₙ, qₙ₊₁)/dZ at a converged inner solution (Theorem 3 + node motion)
# ---------------------------------------------------------------------------------------------

function knot_gradient(sp::StepSpace, x, q0, p0, h, prob::FKProblem)
    k = sp.k
    m = length(sp.Z)
    n = nfun(sp)
    X = x[1:n]
    p1 = x[n + 1]
    Q = sp.P * X
    Qp = sp.D * X
    Qpp = sp.E * X
    V = Qp ./ h
    f = -prob.dV.(Q)
    L = 0.5 .* V .^ 2 .- prob.V.(Q)
    dLdτ = V .* Qpp ./ h .- prob.dV.(Q) .* Qp
    R = length(sp.c0)
    g = zeros(m)
    for i in 1:m
        z = sp.Z[i]
        ci = X[k + 1 + i]
        dq = -k * ci .* rp.(sp.c .- z, k - 1)                 # ∂q/∂zᵢ at fixed coefficients
        dqp = -k * (k - 1) * ci .* rp.(sp.c .- z, k - 2)
        explicit = sum(sp.b .* (h .* f .* dq .+ V .* dqp)) - p1 * (-k * ci * rp(1 - z, k - 1))
        # motion of the composite nodes: knot i closes piece i and opens piece i+1
        left = sp.piece .== i
        right = sp.piece .== i + 1
        motion = h * sum(sp.b0 .* L[left] .+ sp.b[left] .* sp.c0 .* dLdτ[left]) +
                 h * sum(-sp.b0 .* L[right] .+ sp.b[right] .* (1 .- sp.c0) .* dLdτ[right])
        g[i] = explicit + motion
    end
    addf!(m * length(sp.c) * (4 * (k + 2) + 30))
    g
end

"""Discrete Lagrangian L_d^Z(qa, qb) (type I, composite rule moving with Z) — for checks only."""
function discrete_lagrangian(k, Z, R, qa, qb, h, prob::FKProblem)
    sp = StepSpace(k, Z, R)
    n = nfun(sp)
    function F(y)
        X = y[1:n]
        Q = sp.P * X
        V = (sp.D * X) ./ h
        gA = sp.P' * (sp.b .* h .* (-prob.dV.(Q))) .+ sp.D' * (sp.b .* V)
        vcat(gA .- y[n + 1] .* sp.r0 .- y[n + 2] .* sp.r1, dot(sp.r0, X) - qa, dot(sp.r1, X) - qb)
    end
    tt = collect(range(0, 1; length = 21))
    y0 = vcat(spline_PDE(k, sp.Z, tt)[1] \ (qa .+ (qb - qa) .* tt), 0.0, 0.0)
    y = newton_solve(F, y0).x
    X = y[1:n]
    h * sum(sp.b .* (0.5 .* ((sp.D * X) ./ h) .^ 2 .- prob.V.(sp.P * X)))
end

# ---------------------------------------------------------------------------------------------
# one step of each method
# ---------------------------------------------------------------------------------------------

function fixed_step(sp::StepSpace, q0, p0, h, prob; x0 = nothing)
    x0 === nothing && (x0 = start_value(sp, q0, p0, h))
    x, ok = inner_solve(sp, q0, p0, h, prob, x0)
    ok || error("inner Newton did not converge")
    n = nfun(sp)
    dot(sp.r1, x[1:n]), x[n + 1], x
end

"""Clamp knots into (δ, 1−δ), sorted, with a minimum gap."""
function sanitize(Z; δ = 0.02, gap = 0.02)
    Z = sort(clamp.(Z, δ, 1 - δ))
    for i in 2:length(Z)
        Z[i] = max(Z[i], Z[i - 1] + gap)
    end
    for i in (length(Z) - 1):-1:1
        Z[i] = min(Z[i], Z[i + 1] - gap)
    end
    Z
end

"""
Free-knot step: outer safeguarded Newton on g(Z) = 0 starting from Z0.
Returns (q₁, p₁, Z*, converged).
"""
function free_step(k, R, q0, p0, h, prob, Z0; gtol = 1e-12, maxouter = 30, dz = 1e-6, maxmove = 0.1)
    Z = sanitize(copy(Z0))
    m = length(Z)
    sp = StepSpace(k, Z, R)
    x, ok = inner_solve(sp, q0, p0, h, prob, start_value(sp, q0, p0, h))
    ok || error("inner Newton did not converge")
    best = (Z = copy(Z), x = copy(x), sp = sp, gn = Inf)
    converged = false
    for _ in 1:maxouter
        g = knot_gradient(sp, x, q0, p0, h, prob)
        gn = norm(g, Inf)
        gn < best.gn && (best = (Z = copy(Z), x = copy(x), sp = sp, gn = gn))
        if gn < gtol * max(1.0, abs(energy(prob, q0, p0)) * h)
            converged = true
            break
        end
        CNT.outer += 1
        J = zeros(m, m)
        for j in 1:m
            Zp = copy(Z)
            Zp[j] += dz
            spp = StepSpace(k, Zp, R)
            xp, okp = inner_solve(spp, q0, p0, h, prob, x)
            okp || break
            J[:, j] = (knot_gradient(spp, xp, q0, p0, h, prob) .- g) ./ dz
        end
        dZ = try
            -(J \ g)
        catch
            -sign.(g) .* 0.01            # singular knot Jacobian: small gradient-sign step
        end
        dZ = clamp.(dZ, -maxmove, maxmove)
        Znew = sanitize(Z .+ dZ)
        spn = StepSpace(k, Znew, R)
        xn, okn = inner_solve(spn, q0, p0, h, prob, x)
        okn || break
        Z, sp, x = Znew, spn, xn
    end
    if !converged
        CNT.fallback += 1
        Z, x, sp = best.Z, best.x, best.sp     # best knot found (smallest |g|)
    end
    n = nfun(sp)
    dot(sp.r1, x[1:n]), x[n + 1], Z, converged
end

# ---------------------------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------------------------

"""Run a method over N steps; returns (q, p, extra) with q, p of length N+1."""
function run_method(method::Symbol, prob, k, R, m, h, N, q0, p0)
    q = [q0]
    p = [p0]
    if method == :cgvi
        sp = StepSpace(k, Float64[], R)
        for _ in 1:N
            q1, p1, _ = fixed_step(sp, q[end], p[end], h, prob)
            push!(q, q1)
            push!(p, p1)
        end
    elseif method == :substeps
        sp = StepSpace(k, Float64[], R)
        hs = h / (m + 1)
        for _ in 1:N
            qq, pp = q[end], p[end]
            for _ in 1:(m + 1)
                qq, pp, _ = fixed_step(sp, qq, pp, hs, prob)
            end
            push!(q, qq)
            push!(p, pp)
        end
    elseif method == :fixed
        sp = StepSpace(k, collect((1:m) ./ (m + 1)), R)
        for _ in 1:N
            q1, p1, _ = fixed_step(sp, q[end], p[end], h, prob)
            push!(q, q1)
            push!(p, p1)
        end
    elseif method == :free
        Z = collect((1:m) ./ (m + 1))
        zs = Vector{Vector{Float64}}()
        for _ in 1:N
            q1, p1, Z, _ = free_step(k, R, q[end], p[end], h, prob, Z)   # warm start from last Z
            push!(q, q1)
            push!(p, p1)
            push!(zs, copy(Z))
        end
        return q, p, zs
    else
        error("unknown method $method")
    end
    q, p, nothing
end

"""Reference trajectory: CGVI(P₇, R = 8) at a step h/sub, sampled on the coarse grid."""
function reference(prob, h, N, q0, p0; dtmax = 0.01)
    sub = max(1, ceil(Int, h / dtmax))
    sp = StepSpace(7, Float64[], 8)
    q = [q0]
    p = [p0]
    qq, pp = q0, p0
    for _ in 1:N
        for _ in 1:sub
            qq, pp, _ = fixed_step(sp, qq, pp, h / sub, prob)
        end
        push!(q, qq)
        push!(p, pp)
    end
    q, p
end

# parameters per step
params(method, k, m) = method == :cgvi ? k + 1 :
                       method == :substeps ? (m + 1) * (k + 1) :
                       method == :fixed ? k + 1 + m : k + 1 + 2m
unknowns(method, k, m) = method == :cgvi ? "$(k + 2)" :
                         method == :substeps ? "$(m + 1)×$(k + 2)" :
                         method == :fixed ? "$(k + 2 + m)" : "$(k + 2 + m) (+$m knots outer)"
label(method, k, R, m) = method == :cgvi ? "CGVI P$k R$R" :
                         method == :substeps ? "CGVI P$k R$R, $(m + 1) substeps" :
                         method == :fixed ? "fixed spline m=$m (uniform), composite R$R" :
                         "free spline m=$m (variational), composite R$R"

# ---------------------------------------------------------------------------------------------
# checks: analytic knot gradient vs FD of L_d, and area preservation of the free-knot step
# ---------------------------------------------------------------------------------------------

function run_checks(k, R)
    section("checks")
    for prob in (FK_HO, FK_PEND), Z in ([0.4], [0.3, 0.7])
        q0, p0, h = 0.5, 0.1, 2.0
        sp = StepSpace(k, Z, R)
        x, _ = inner_solve(sp, q0, p0, h, prob, start_value(sp, q0, p0, h))
        q1 = dot(sp.r1, x[1:nfun(sp)])
        ga = knot_gradient(sp, x, q0, p0, h, prob)
        e = 1e-6
        gf = [(discrete_lagrangian(k, [j == i ? z + e : z for (j, z) in enumerate(Z)], R, q0, q1, h, prob) -
               discrete_lagrangian(k, [j == i ? z - e : z for (j, z) in enumerate(Z)], R, q0, q1, h, prob)) / (2 * e)
              for i in eachindex(Z)]
        vcheck("F1", "$(prob.name) Z=$Z: analytic dL_d/dZ = FD of L_d (rel.)",
            norm(ga .- gf) / max(norm(gf), 1e-300), 1e-5)
    end
    # the free-knot step map is area preserving (det = 1) — Corollary 5.3
    for prob in (FK_HO, FK_PEND)
        q0, p0, h = 0.5, 0.1, 2.0
        _, _, Zs, conv = free_step(k, R, q0, p0, h, prob, [0.5])
        e = 1e-5
        M = zeros(2, 2)
        for (j, (dq, dp)) in enumerate(((e, 0.0), (0.0, e)))
            a = free_step(k, R, q0 + dq, p0 + dp, h, prob, Zs)
            b = free_step(k, R, q0 - dq, p0 - dp, h, prob, Zs)
            M[:, j] = [(a[1] - b[1]) / (2 * e), (a[2] - b[2]) / (2 * e)]
        end
        vcheck("F2", "$(prob.name): free-knot step map |det − 1| (knot converged: $conv)", abs(det(M) - 1), 1e-7)
    end
    # the monomial-basis CGVI equals the Lagrange-basis CGVI of theory_common (basis independence)
    let sp = StepSpace(k, Float64[], R)
        a = fixed_step(sp, 0.5, 0.1, 1.0, FK_PEND)
        b = linear_step(0.5, 0.1, 1.0, cgvi_basis(k), QRule01(R), TOY_PEND)
        vcheck("F3", "CGVI in monomial basis = CGVI in Lagrange basis", max(abs(a[1] - b.q1), abs(a[2] - b.p1)), 1e-12)
    end
end

# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------

argval(name, default) = (a = findfirst(s -> startswith(s, "--$name="), ARGS);
    a === nothing ? default : split(ARGS[a], "=")[2])
parse_list(s) = parse.(Float64, split(s, ","))

function main()
    T = parse(Float64, argval("T", "20"))
    k = parse(Int, argval("k", "3"))
    R = parse(Int, argval("R", "4"))
    hs = parse_list(argval("h", "0.5,1,2,5"))
    ms = Int.(parse_list(argval("m", "1,2,3")))
    q0, p0 = 0.5, 0.0

    run_checks(k, R)

    outdir = joinpath(@__DIR__, "..", "results")
    mkpath(outdir)
    csv = open(joinpath(outdir, "free_knot_vi.csv"), "w")
    println(csv, "problem,h,m,method,params_per_step,unknowns,q_err,energy_err,flops_total,flops_per_step,newton_iters,inner_solves,outer_iters,knot_fallbacks,seconds")
    md = String[]
    push!(md, "# Free-knot spline VI vs CGVI — parameters, FLOPs, errors\n")
    push!(md, "k = $k, R = $R Gauss points per (sub)interval, t ∈ (0, $T), (q₀, p₀) = ($q0, $p0).")
    push!(md, "`q err` = max over the grid |q − q_ref| / max|q_ref|; `H err` = max |H − H₀| / |H₀|.")
    push!(md, "FLOPs are counted with the model in the header of `free_knot_vi.jl` (analytic Jacobians);")
    push!(md, "`fallbacks` = steps where the knot equation did not converge (best knot used instead —")
    push!(md, "such steps are no longer exactly variational). Reference: CGVI(P₇, R = 8) at dt ≤ 0.01.\n")

    for prob in (FK_HO, FK_PEND)
        push!(md, "## $(prob.name)\n")
        push!(md, "| h | method | params/step | unknowns per solve | q err | H err | FLOPs total | FLOPs/step | Newton its | outer its | fallbacks | s |")
        push!(md, "|---|---|---|---|---|---|---|---|---|---|---|---|")
        section(prob.name)
        for h in hs
            N = round(Int, T / h)
            abs(N * h - T) < 1e-9 || (@warn "T not a multiple of h=$h, skipped"; continue)
            qr, pr = reference(prob, h, N, q0, p0)
            scale = maximum(abs, qr)
            H0 = energy(prob, q0, p0)
            rows = Tuple{Symbol, Int}[(:cgvi, 0)]
            for m in ms
                append!(rows, [(:substeps, m), (:fixed, m), (:free, m)])
            end
            for (method, m) in rows
                reset!(CNT)
                t0 = time()
                res = try
                    run_method(method, prob, k, R, m, h, N, q0, p0)
                catch e
                    e isa InterruptException && rethrow()
                    @warn "$(prob.name) h=$h $(label(method, k, R, m)) failed" exception = e
                    nothing
                end
                secs = time() - t0
                if res === nothing
                    qerr, herr = NaN, NaN
                else
                    q, p, _ = res
                    qerr = maximum(abs.(q .- qr)) / scale
                    herr = maximum(abs.(energy.(Ref(prob), q, p) .- H0)) / abs(H0)
                end
                lab = label(method, k, R, m)
                fmt(x) = isnan(x) ? "—" : @sprintf("%.2e", x)
                @printf("  h=%-4g %-48s params=%-3d qerr=%-9s Herr=%-9s flops=%-9s newton=%-5d outer=%-4d fallback=%d\n",
                    h, lab, params(method, k, m), fmt(qerr), fmt(herr), fmt(CNT.flops), CNT.newton, CNT.outer, CNT.fallback)
                push!(md, "| $h | $lab | $(params(method, k, m)) | $(unknowns(method, k, m)) | $(fmt(qerr)) | $(fmt(herr)) | " *
                          "$(fmt(CNT.flops)) | $(fmt(CNT.flops / N)) | $(CNT.newton) | $(CNT.outer) | $(CNT.fallback) | $(@sprintf("%.2f", secs)) |")
                println(csv, join((prob.name, h, m, method, params(method, k, m), unknowns(method, k, m), qerr, herr,
                        CNT.flops, CNT.flops / N, CNT.newton, CNT.inner, CNT.outer, CNT.fallback, secs), ","))
                flush(csv)
            end
        end
        push!(md, "")
    end
    push!(md, "ShallowNet with the same function class uses 3(k+1+m) parameters per step (k+1+2m effective,")
    push!(md, "the rest is gauge freedom) and solves all of them in one Newton with a singular Jacobian.")
    close(csv)
    open(joinpath(outdir, "free_knot_vi.md"), "w") do io
        foreach(l -> println(io, l), md)
    end
    println("Wrote ", joinpath(outdir, "free_knot_vi.md"))
    summarize_checks()
end

main()
