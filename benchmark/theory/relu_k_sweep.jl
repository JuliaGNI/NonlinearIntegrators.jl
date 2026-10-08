# ShallowNet(ReLUᵏ / tanh, S neurons) vs CGVI for several k, S, quadratures, problems and ShallowNet
# variants — does the S = 4, k = 3 observation ("same error as CGVI") hold for other powers, widths and
# composite rules, as Theorem 1 predicts? And where do tanh networks stand at large step sizes?
#
# Laid out like scripts/run_nvi.jl: a registry of runs named by stem, a driver that solves and archives
# each run on its own, and a report that reads the archives. A run is one configuration
# (problem, variant, activation, k, S, M) over all its step sizes h; it writes one csv,
# results/relu_k_sweep_<solver>/<stem>.csv, so runs can go to separate processes on a server:
#
#   julia --project=benchmark benchmark/theory/relu_k_sweep.jl --list [options] > runs.txt
#   xargs -P 16 -I{} julia --project=benchmark benchmark/theory/relu_k_sweep.jl {} [options] < runs.txt
#   julia --project=benchmark benchmark/theory/relu_k_sweep.jl --report [options]
#
# or everything in one process:
#
#   julia --project=benchmark benchmark/theory/relu_k_sweep.jl [options]            # all runs, then the report
#   julia --project=benchmark benchmark/theory/relu_k_sweep.jl kepler-shallownet-tanh-k3-S4-M2 --r=4
#
# A stem is problem-variant-activation-kK-SS-MM, e.g. `harmonic_oscillator-reversible-relu3-k3-S4-M2`.
# The options that a stem does not name (--r, --h, --final-time, --bias, --tanh-bias, --strict, --reg,
# --solver) must be the same for every run of one sweep. Every process compiles and computes the
# Gauss(8) reference of its problem once; for fmo (T = 600) that reference is the larger part of a
# short run.
#
# Options
#   --list          print the stems of the selected runs, one per line, and stop
#   --report        read every csv in results/relu_k_sweep_<solver>/ and write the md, the merged csv
#                   and the figures; no integration
#   --k=…           activation powers (default 3,5)
#   --S=auto|…      widths; `auto` = k+1, 2k, 3k for every k (default), or an explicit list. Every width is
#                   rounded up to an even number, for every variant: the reversible variant needs mirrored
#                   neuron pairs, and the variants are compared at the same width.
#   --r=auto|n      Gauss points per sub-interval; `auto` = k+1 (the CGVI(P_k) default, R = 4 for k = 3)
#   --M=…           sub-intervals of the composite rule CompositeQuadrature(GaussLegendre(r), M) for
#                   h < 10 (default 1, the single Gauss rule of the earlier runs)
#   --M-large=n     sub-intervals for h ≥ 10 (default 10: sub-intervals of length ≤ 1, r points each);
#                   a run with M = n covers only those h
#   --variants=…    shallownet,reversible,autodiff (default all three): ShallowNet, ShallowNetReversible,
#                   ShallowNetAutodiff
#   --h=…           step sizes (default: per problem, see ALL_PROBLEMS); --final-time=T (default: per problem)
#   --bias=kinkfree|default   OGA bias interval [1.1, π] (every seed kink outside [0,1],
#                   default) or [−π, π] (the package default — kinks may enter the interval)
#   --strict        only the residual stops the Newton solves (as in verify_conjectures.jl)
#   --problems=…    harmonic_oscillator, pendulum, double_pendulum, toda_lattice, fmo, kepler,
#                   henon_heiles (default all)
#   --reg=λ         regularization_factor of the ShallowNet Newton solves (default 1e-5)
#   --solver=dogleg|newton   nonlinear solver of the ShallowNet steps (default dogleg):
#                   `dogleg` = SimpleSolvers.DogLeg() (trust region, no line search),
#                   `newton` = SimpleSolvers.Newton() with Backtracking line search.
#                   CGVI always uses the package default solver.
#   --act=relu,tanh activations (default both). tanh is compared, for every k, with CGVI(P_k)
#                   on the same quadrature and with ReLU^k at the same width S.
#   --tanh-bias=default|lo,hi   OGA bias interval for tanh (default [−π, π], the package default)
#
# For every h of a run it reports the errors of the network and of three CGVI baselines against a
# Gauss(8) reference:
#   CGVI(P_k) on Gauss(k+1)                                         `err_cgvi`      (the earlier baseline)
#   CGVI(P_k) on the network's rule CompositeQuadrature(GL(r), M)   `err_cgvi_comp`
#   CGVI on the spline space S_k with M−1 uniform interior knots, same rule   `err_spline`
# their differences, the difference to CGVI(P_{k+1}), and the per-step classification of the network
# (P_k-equiv / spline / degenerate, see verify_conjectures.jl). Theorem 1 is a statement about the trial
# space, so the ReLU verdict compares with CGVI(P_k) on the same quadrature: it predicts
# diff ≈ solver tolerance whenever every step is P_k-equiv and the solver reports every step converged
# (`SimpleSolvers.isconverged` of its status; the largest residual ‖r‖∞ is reported alongside).
#
# It also reports the relative error of the energy, max_n |E(t_n, q_n, p_n) − E_0| / |E_0|, where E is the
# Hamiltonian, except for the non-autonomous frequency-modulated oscillator, where it is the adiabatic
# invariant J = H/ω.
#
# The report writes results/relu_k_sweep_<solver>.{md,csv} and, per problem <p>,
# relu_k_sweep_<solver>_<p>_qerr.pdf (error in q) and relu_k_sweep_<solver>_<p>_energy.pdf.

include(joinpath(@__DIR__, "theory_common.jl"))
include(joinpath(@__DIR__, "..", "shallownet_benchmark_common.jl"))
# in a module of its own: integration.jl defines a `RESULTS_DIR` of its own
module HardProblems
include(joinpath(@__DIR__, "..", "hard_problems", "integration.jl"))
end
using .HardProblems: integrate_counting, WarnCounter
using Logging: with_logger
import GeometricIntegratorsBase
using GeometricProblems.HarmonicOscillator
using GeometricProblems.Pendulum
using GeometricProblems.DoublePendulum
using GeometricProblems.TodaLattice
import GeometricProblems.FrequencyModulatedOscillator as fmo
import GeometricProblems.KeplerProblem as kepler
import GeometricProblems.HenonHeilesPotential as hh
using SimpleSplines: BSplineBasis, UniformMesh, (..)
using CairoMakie
using Statistics: median
const CBF = GeometricIntegrators.Integrators.CompactBasisFunctions

# ---------------------------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------------------------

argval(name, default) = (a = findfirst(s -> startswith(s, "--$name="), ARGS);
    a === nothing ? default : String(split(ARGS[a], "="; limit = 2)[2]))
const KS = parse.(Int, split(argval("k", "3,5"), ","))
const S_ARG = argval("S", "auto")
const R_ARG = argval("r", "auto")
const M_LIST = parse.(Int, split(argval("M", "1"), ","))
const H_LARGE = 10                                          # steps h ≥ H_LARGE use M_LARGE instead
const M_LARGE = parse(Int, argval("M-large", "10"))
const H_ARG = argval("h", nothing)
const T_ARG = argval("final-time", nothing)
const BIAS = argval("bias", "kinkfree") == "default" ? (-pi, pi) : (1.1, pi)
const PROBLEM_NAMES = split(argval("problems", "harmonic_oscillator,pendulum,double_pendulum,toda_lattice,fmo,kepler,henon_heiles"), ",")
const STRICT = "--strict" in ARGS
# relu before tanh whatever the order given (and in the report, which reads the ReLU error at the same S)
const ACTS = filter(in(split(argval("act", "relu,tanh"), ",")), ["relu", "tanh"])
const VARIANTS = (shallownet = ShallowNet, reversible = ShallowNetReversible, autodiff = ShallowNetAutodiff)
const VARIANT_NAMES = split(argval("variants", "shallownet,reversible,autodiff"), ",")
all(v -> Symbol(v) in keys(VARIANTS), VARIANT_NAMES) || error("--variants must be among $(keys(VARIANTS))")
const SOLVER = lowercase(argval("solver", "dogleg"))
SOLVER in ("dogleg", "newton") || error("--solver must be dogleg or newton, got $SOLVER")
const OUTBASE = "relu_k_sweep_$(SOLVER)"
const RUNS_DIR = joinpath(@__DIR__, "..", "results", OUTBASE)
const TANH_BIAS = argval("tanh-bias", "default") == "default" ? (-pi, pi) : Tuple(parse.(Float64, split(argval("tanh-bias", ""), ",")))

even(S) = S + isodd(S)
widths(k) = unique(even.(S_ARG == "auto" ? [k + 1, 2 * k, 3 * k] : parse.(Int, split(S_ARG, ","))))
rpoints(k) = R_ARG == "auto" ? k + 1 : parse(Int, R_ARG)
quadrature(k, M) = CompositeQuadrature(GaussLegendreQuadrature(Float64, rpoints(k)), M)

const DT_REF = 0.005
const NVI_REG = parse(Float64, argval("reg", "1e-5"))   # regularization_factor of the network solves
const NVI_MAXIT = 1000

# ---------------------------------------------------------------------------------------------
# problems: `energy(t, q, p)` is the invariant whose relative error is reported, `T` and `hs` the
# default final time and step sizes (overridden by --final-time and --h). The hard problems follow
# hard_problems/setup.jl (h·Ω up to 10); T is a multiple of every h there.
# ---------------------------------------------------------------------------------------------

const DEFAULT_HS = [0.2, 0.5, 1, 2, 5, 10]
const LARGE_HS = [0.3, 1, 3, 10]

const ALL_PROBLEMS = [
    (name = "harmonic_oscillator", ics = ([0.5], [0.0]),
        build = (q0, p0, ts, h) -> HarmonicOscillator.lodeproblem(q0, p0; timespan = ts,
            timestep = h, parameters = HarmonicOscillator.default_parameters(Float64)),
        # H = p²/(2m) + k q²/2
        energy = (t, q, p) -> HarmonicOscillator.hamiltonian(t, q, p, HarmonicOscillator.default_parameters(Float64)),
        T = 20.0, hs = DEFAULT_HS),
    (name = "pendulum",
        ics = let d = Pendulum.iodeproblem()
            (collect(Float64.(d.ics.q)), collect(Float64.(d.ics.p)))
        end,
        build = (q0, p0, ts, h) -> Pendulum.iodeproblem(q0, p0; timespan = ts,
            timestep = h, parameters = Pendulum.default_parameters(Float64)),
        # degenerate IODE: q = (θ, θ̇), canonical momentum m l² θ̇ ⇒ H = m l² q₂²/2 + m g l cos q₁
        energy = (t, q, p) -> let c = Pendulum.default_parameters(Float64)
            c.m * c.l^2 * q[2]^2 / 2 + c.m * c.g * c.l * cos(q[1])
        end,
        T = 20.0, hs = DEFAULT_HS),
    (name = "double_pendulum",
        ics = let d = DoublePendulum.lodeproblem()
            (collect(Float64.(d.ics.q)), collect(Float64.(d.ics.p)))
        end,
        build = (q0, p0, ts, h) -> DoublePendulum.lodeproblem(q0, p0; timespan = ts,
            timestep = h, parameters = DoublePendulum.default_parameters(Float64)),
        energy = (t, q, p) -> DoublePendulum.hamiltonian(t, q, p, DoublePendulum.default_parameters(Float64)),
        T = 20.0, hs = DEFAULT_HS),
    (name = "toda_lattice",
        ics = let d = TodaLattice.lodeproblem(4)
            (collect(Float64.(d.ics.q)), collect(Float64.(d.ics.p)))
        end,
        build = (q0, p0, ts, h) -> TodaLattice.lodeproblem(q0, p0; timespan = ts,
            timestep = h, parameters = TodaLattice.default_parameters(Float64)),
        energy = (t, q, p) -> TodaLattice.hamiltonian(t, q, p, TodaLattice.default_parameters(Float64), 4),
        T = 20.0, hs = DEFAULT_HS),
    (name = "fmo", ics = ([1.0], [0.0]),       # ε = 1e-2; H is not conserved, J = H/ω is adiabatic
        build = (q0, p0, ts, h) -> fmo.lodeproblem(q0, p0; timespan = ts, timestep = h,
            parameters = merge(fmo.default_parameters(), (ε = 1e-2,))),
        energy = (t, q, p) -> fmo.adiabatic_invariant(t, q, p, merge(fmo.default_parameters(), (ε = 1e-2,))),
        T = 600.0, hs = LARGE_HS),
    (name = "kepler", ics = kepler.initial_condition(0.9),
        build = (q0, p0, ts, h) -> kepler.lodeproblem(q0, p0; timespan = ts, timestep = h,
            parameters = kepler.default_parameters()),
        energy = (t, q, p) -> kepler.hamiltonian(t, q, p, kepler.default_parameters()),
        T = 60.0, hs = LARGE_HS),
    (name = "henon_heiles", ics = ([0.1, 0.1], [0.1, 0.1]),   # E ≈ 0.02, quasi-periodic
        build = (q0, p0, ts, h) -> hh.lodeproblem(q0, p0; timespan = ts, timestep = h,
            parameters = hh.default_parameters()),
        energy = (t, q, p) -> hh.hamiltonian(t, q, p, hh.default_parameters()),
        T = 90.0, hs = LARGE_HS)
]
const PROBLEMS = filter(P -> P.name in PROBLEM_NAMES, ALL_PROBLEMS)
length(PROBLEMS) == length(PROBLEM_NAMES) ||
    error("unknown problem in --problems; available: $(join((P.name for P in ALL_PROBLEMS), ", "))")

final_time(P) = T_ARG === nothing ? P.T : parse(Float64, T_ARG)
step_sizes(P) = H_ARG === nothing ? P.hs : parse.(Float64, split(H_ARG, ","))
"The step sizes of a run with M sub-intervals: M_LIST for h < H_LARGE, M_LARGE for h ≥ H_LARGE."
run_steps(P, M) = filter(h -> h >= H_LARGE ? M == M_LARGE : M in M_LIST, step_sizes(P))

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
        println(stderr, "strict mode: solver options ", accepted)   # stderr: stdout carries --list
    end
    accepted
end

# ---------------------------------------------------------------------------------------------
# helpers (as in verify_conjectures.jl)
# ---------------------------------------------------------------------------------------------

tomat(sol) = reduce(hcat, [Float64.(collect(q)) for q in collect(sol.q[:])])
tomat_p(sol) = reduce(hcat, [Float64.(collect(p)) for p in collect(sol.p[:])])

"""max_n |E(t_n, q_n, p_n) − E_0| / |E_0| over the columns of Q, Pm, with t_n = (n − 1) h."""
function rel_energy_error(P, Q, Pm, h)
    E = [P.energy((n - 1) * h, Q[:, n], Pm[:, n]) for n in axes(Q, 2)]
    maximum(abs, E .- E[1]) / max(abs(E[1]), eps())
end

nl_solver_kwargs() = SOLVER == "dogleg" ? (solver = SimpleSolvers.DogLeg(),) :
    (solver = SimpleSolvers.Newton(), linesearch = SimpleSolvers.Backtracking(Float64))

function classify_step(θ, k; TT = collect(range(0, 1; length = 201)))
    cls = neuron_classes(θ, k)
    any(==("interior"), cls) && return "spline"
    out = findall(==("outside"), cls)
    isempty(out) && return "degenerate"
    a, w, b = unpack(θ)
    θo = pack(a[out], w[out], b[out])
    numrank(nn_tangent(θo, TT, k)[1]; rtol = 1e-9) == k + 1 ? "P_k-equiv" : "degenerate"
end

"""
The solver's own verdict on every network step: `SimpleSolvers.isconverged` of the status that the
network integrators' `integrate_step!` passes to this hook of GeometricIntegratorsBase. A step counts
as unconverged exactly when the solver says so.
"""
const STEP_CONVERGED = Bool[]
function GeometricIntegratorsBase.check_solver_status(status, ::GeometricIntegrator{<:ShallowNetMethod})
    push!(STEP_CONVERGED, SimpleSolvers.isconverged(status))
    status
end

"CGVI(P_s) with the Lagrange basis on the s+1 Gauss points, on the rule `quad`."
lagrange_cgvi(s, quad) = CGVI(CBF.Lagrange(QuadratureRules.nodes(GaussLegendreQuadrature(Float64, s + 1))), quad)

"CGVI on the degree-k splines with M − 1 uniform interior knots (one cell per sub-interval of `quad`)."
spline_cgvi(k, M, quad) = CGVI(BSplineBasis(UniformMesh(M, 0 .. 1), k), quad)

"Positions and momenta of the CGVI `method` over (0, T), or `nothing` (with a warning) if a step throws."
function cgvi_run(P, h, T, method)
    res = try
        integrate(P.build(copy(P.ics[1]), copy(P.ics[2]), (0.0, T), h), method; STRICT_OPTS...)
    catch e
        e isa InterruptException && rethrow()
        @warn "$(P.name): CGVI baseline at h = $h failed" exception = e
        return nothing
    end
    sol = res isa Tuple ? first(res) : res
    tomat(sol), tomat_p(sol)
end

"Gauss(8) reference on the grid n DT_REF, and its relative difference to the run with DT_REF / 2."
function reference(P, T)
    Q₁ = tomat(integrate(P.build(copy(P.ics[1]), copy(P.ics[2]), (0.0, T), DT_REF), Gauss(8)))
    Q₂ = tomat(integrate(P.build(copy(P.ics[1]), copy(P.ics[2]), (0.0, T), DT_REF / 2), Gauss(8)))[:, 1:2:end]
    Q₁, maximum(abs, Q₁ .- Q₂) / maximum(abs, Q₁)
end

actlabel(act, k) = act == "relu" ? "ReLU^$k" : "tanh"

# ---------------------------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------------------------

"Every selected configuration; a ReLUᵏ network narrower than k+1 cannot span P_k and is left out."
const SWEEP_RUNS = [(problem = P.name, variant, act, k, S, M)
                    for P in PROBLEMS for k in KS for M in unique([M_LIST; M_LARGE]) for act in ACTS for S in widths(k)
                    for variant in VARIANT_NAMES if !(act == "relu" && S < k + 1) && !isempty(run_steps(P, M))]

stem(run) = join((run.problem, run.variant, run.act == "relu" ? "relu$(run.k)" : "tanh",
    "k$(run.k)", "S$(run.S)", "M$(run.M)"), "-")

"The run a stem names; a stem is read, not looked up, so a server job needs no selection options."
function sweep_run(s)
    f = split(s, "-")
    run = length(f) == 6 && any(P -> P.name == f[1], ALL_PROBLEMS) && Symbol(f[2]) in keys(VARIANTS) ?
          (problem = String(f[1]), variant = String(f[2]), act = startswith(f[3], "relu") ? "relu" : "tanh",
              k = parse(Int, f[4][2:end]), S = parse(Int, f[5][2:end]), M = parse(Int, f[6][2:end])) : nothing
    run !== nothing && stem(run) == s && (run.variant != "reversible" || iseven(run.S)) ||
        throw(ArgumentError("`$s` is not a stem problem-variant-activation-kK-SS-MM with an even S for " *
                            "reversible, e.g. harmonic_oscillator-reversible-relu3-k3-S4-M2"))
    run
end

"""
Solve one run at every step size of its problem and write its csv. Per h: the network, stepped by
`integrate_counting` (which records the nonlinear solution of every step and its residual), with the
solver warnings counted rather than printed, as in hard_problems/integration.jl; and the three CGVI
baselines. A failed baseline is NaN and leaves the network result in place.
"""
function run_sweep(run)
    (; variant, act, k, S, M) = run
    P = only(filter(P -> P.name == run.problem, ALL_PROBLEMS))
    T, hs = final_time(P), run_steps(P, M)
    D = length(P.ics[1])
    quad = quadrature(k, M)
    σ, bias = act == "relu" ? (relu_k(k), BIAS) : (tanh, TANH_BIAS)
    method = VARIANTS[Symbol(variant)](ShallowNetBasis{Float64}(σ, S), quad;
        show_status = false, bias_interval = [bias[1], bias[2]], dict_amount = DICT_AMOUNT)
    Qref, refcheck = reference(P, T)
    scale = maximum(abs, Qref)
    @printf("%s: T = %g, Gauss(8) reference dt = %g, |dt − dt/2| = %.2e\n", stem(run), T, DT_REF, refcheck)
    mkpath(RUNS_DIR)
    path = joinpath(RUNS_DIR, stem(run) * ".csv")
    io = open(path, "w")      # written row by row, so a run stopped by a time limit keeps its rows
    for h in hs
        N = round(Int, T / h)
        stride = round(Int, h / DT_REF)
        abs(N * h - T) < 1e-9 && abs(stride * DT_REF - h) < 1e-12 ||
            (@warn "h = $h does not divide T = $T or is no multiple of $DT_REF, skipped"; continue)
        qref = Qref[:, 1:stride:end]
        try
            # the network
            prob = P.build(copy(P.ics[1]), copy(P.ics[2]), (0.0, T), h)
            logger = WarnCounter()
            empty!(STEP_CONVERGED)
            tsn = @elapsed res = with_logger(logger) do
                integrate_counting(prob, method; record = true, nl_solver_kwargs()...,
                    regularization_factor = NVI_REG, max_iterations = NVI_MAXIT, STRICT_OPTS...)
            end
            done = res.status == "ok"
            Q, Pm = tomat(res.sol), tomat_p(res.sol)
            # the baselines; a failed one is `nothing`, and everything computed from it NaN
            tc = @elapsed begin
                cg = cgvi_run(P, h, T, lagrange_cgvi(k, GaussLegendreQuadrature(Float64, k + 1)))
                ca = cgvi_run(P, h, T, lagrange_cgvi(k + 1, GaussLegendreQuadrature(Float64, max(rpoints(k), k + 2))))
                cc = cgvi_run(P, h, T, lagrange_cgvi(k, quad))
                cs = cgvi_run(P, h, T, spline_cgvi(k, M, quad))
            end
            relerr(X) = X === nothing ? NaN : maximum(abs, X[1] .- qref) / scale
            herr(X) = X === nothing ? NaN : rel_energy_error(P, X[1], X[2], h)
            diffto(X) = done && X !== nothing ? maximum(abs, Q .- X[1]) / scale : NaN
            net = done ? (Q, Pm) : nothing
            err_sn, err_cc, herr_sn, herr_cc = relerr(net), relerr(cc), herr(net), herr(cc)
            diff = diffto(cc)
            unconverged = count(!, STEP_CONVERGED)
            if act == "relu"
                # θ = (a, w, b) of every step and dimension: column i of reshape(x, D, :) is the i-th unknown
                # of every dimension — a₁…a_S, p, then w and b; ShallowNetReversible stores the S/2
                # independent neurons, each mirrored as w₂ᵢ = −w₂ᵢ₋₁, b₂ᵢ = w₂ᵢ₋₁ + b₂ᵢ₋₁
                classes = map(Iterators.product(res.x, 1:D)) do (x, d)
                    all(isfinite, x) || return "degenerate"     # a diverged last step
                    X = reshape(x, D, :)
                    θ = if variant == "reversible"
                        w, b = X[d, (S + 2):(S + 1 + S ÷ 2)], X[d, (S + 2 + S ÷ 2):end]
                        vcat(X[d, 1:S], vec([w -w]'), vec([b w .+ b]'))
                    else
                        vcat(X[d, 1:S], X[d, (S + 2):end])
                    end
                    classify_step(θ, k)
                end
                ne, ns, nd = count(==("P_k-equiv"), classes), count(==("spline"), classes), count(==("degenerate"), classes)
                equal = diff <= max(1e-9, 0.01 * err_cc)
                verdict = done && ns == 0 && nd == 0 && unconverged == 0 ? (equal ? "CONFIRMED" : "VIOLATED") :
                          (equal ? "equal anyway" : "differs")
            else
                # no equivalence theorem for tanh: report how it compares
                ne, ns, nd = 0, 0, 0
                ratio = err_sn / err_cc
                verdict = !done || unconverged > 0 ? "tanh (unconverged steps)" :
                          ratio < 0.5 ? "tanh better" : ratio > 2 ? "tanh worse" : "tanh ≈ CGVI"
            end
            row = (problem = P.name, variant, activation = actlabel(act, k), k, S, r = rpoints(k), M, h, T,
                params_SN = variant == "reversible" ? 2S + 1 : 3S + 1, params_CGVI = k + 1, params_spline = k + M,
                err_sn, err_cgvi = relerr(cg), err_cgvi_comp = err_cc, err_spline = relerr(cs), diff,
                diff_alt = diffto(ca), ratio = err_sn / err_cc, herr_sn, herr_cgvi = herr(cg), herr_cgvi_comp = herr_cc,
                herr_spline = herr(cs), herr_ratio = herr_sn / herr_cc, n_equiv = ne, n_spline = ns, n_degen = nd,
                unconverged, max_res = res.max_res, iterations = res.iterations, warnings = logger.count,
                status = res.status, verdict, sec_sn = tsn, sec_cgvi = tc, refcheck)
            position(io) == 0 && println(io, join(keys(row), ","))
            println(io, join(values(row), ","))
            flush(io)
            f(x) = isnan(x) ? "—" : @sprintf("%.2e", x)
            @printf("  h=%-5g err net=%-9s CGVI(P%d)=%-9s comp=%-9s spline=%-9s diff=%-9s dE net/comp=%s/%s classes=%d/%d/%d unconv=%d  %s (%.1f s)\n",
                h, f(err_sn), k, f(row.err_cgvi), f(err_cc), f(row.err_spline), f(diff), f(herr_sn), f(herr_cc),
                ne, ns, nd, unconverged, verdict, tsn)
        catch e
            e isa InterruptException && rethrow()
            @warn "$(stem(run)) h = $h failed, no row written" exception = e
        end
    end
    close(io)
    println("Wrote ", path)
end

# ---------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------

const STRING_COLUMNS = (:problem, :variant, :activation, :status, :verdict)
const INT_COLUMNS = (:k, :S, :r, :M, :params_SN, :params_CGVI, :params_spline, :n_equiv, :n_spline, :n_degen,
    :unconverged, :iterations, :warnings)

"Every row of every run csv, with `act` (relu / tanh) and `err_relu`: the ReLUᵏ error at the same S, M, h."
function read_runs()
    rows = NamedTuple[]
    for file in sort(filter(endswith(".csv"), readdir(RUNS_DIR; join = true)))
        lines = readlines(file)
        length(lines) < 2 && continue
        names = Symbol.(split(lines[1], ","))
        cell(n, x) = n in STRING_COLUMNS ? String(x) : n in INT_COLUMNS ? parse(Int, x) : parse(Float64, x)
        for l in lines[2:end]
            row = NamedTuple{Tuple(names)}(Tuple(cell(n, x) for (n, x) in zip(names, split(l, ","))))
            push!(rows, merge(row, (act = startswith(row.activation, "ReLU") ? "relu" : "tanh",)))
        end
    end
    relu = Dict((r.problem, r.variant, r.k, r.S, r.M, r.h) => r.err_sn for r in rows if r.act == "relu")
    [merge(r, (err_relu = r.act == "tanh" ? get(relu, (r.problem, r.variant, r.k, r.S, r.M, r.h), NaN) : NaN,))
     for r in rows]
end

function report()
    rows = read_runs()
    isempty(rows) && error("no run csv in $RUNS_DIR")
    outdir = dirname(RUNS_DIR)
    open(joinpath(outdir, OUTBASE * ".csv"), "w") do io
        println(io, join(keys(first(rows)), ","))
        foreach(row -> println(io, join(values(row), ",")), rows)
    end
    write_markdown(rows, joinpath(outdir, OUTBASE * ".md"))
    write_plots(rows, outdir)
end

fmt(x) = isnan(x) ? "—" : @sprintf("%.2e", x)
fmtr(x) = isnan(x) ? "—" : @sprintf("%.4f", x)

function write_markdown(rows, path)
    open(path, "w") do io
        println(io, "# ShallowNet variants (ReLUᵏ / tanh, S) vs CGVI — sweep over k, S, the composite rule and h\n")
        println(io, "r Gauss points on each of M sub-intervals; " *
                    "OGA bias interval ReLU [$(round(BIAS[1]; digits = 2)), $(round(BIAS[2]; digits = 2))], " *
                    "tanh [$(round(TANH_BIAS[1]; digits = 2)), $(round(TANH_BIAS[2]; digits = 2))]; " *
                    "strict tolerances: $(STRICT ? string(STRICT_OPTS) : "no"); regularization_factor = $(NVI_REG); " *
                    "ShallowNet solver: $(SOLVER == "dogleg" ? "DogLeg (trust region)" : "Newton + Backtracking") " *
                    "(as given to the report; the runs must have used the same).")
        refchecks = unique((r.problem, r.refcheck) for r in rows)
        println(io, "`err` = max over the grid ‖q − q_ref‖∞ / max‖q_ref‖∞ (reference Gauss(8), dt = $(DT_REF); " *
                    "relative difference to dt/2: " * join(("$p $(fmt(v))" for (p, v) in refchecks), ", ") * "). " *
                    "Baselines: `CGVI` = CGVI(P_k) on Gauss(k+1); `comp` = CGVI(P_k) on the network's rule; " *
                    "`spline` = CGVI on S_k with M−1 uniform interior knots on the network's rule. " *
                    "`diff` = network vs `comp`; `diff P_{k+1}` = network vs CGVI(P_{k+1}); " *
                    "`ΔE` = max_n |E_n − E_0| / |E_0| (E = H, J = H/ω for fmo).")
        println(io, "ReLU: `CONFIRMED` ⇔ every step P_k-equivalent, every step reported converged by the solver " *
                    "(`unconv` counts the steps it does not; `max ‖r‖` is the largest step residual), and " *
                    "diff ≤ max(1e-9, 0.01·err_comp).")
        println(io, "tanh: no equivalence theorem applies; `tanh better/worse` ⇔ err_tanh/err_comp < 0.5 / > 2. " *
                    "Wall times: the network's includes compilation for the first h of a run; the CGVI time is " *
                    "that of the four baselines at this h.\n")

        println(io, "## Summary per (problem, variant, activation, k, S, M)\n")
        println(io, "| problem | variant | activation | r | M | S | runs | CONFIRMED / better | VIOLATED / worse | other | max diff (CONFIRMED) | median err_net/err_comp | median ΔE_net/ΔE_comp |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for (pn, v, act, k, S, M) in unique([(r.problem, r.variant, r.act, r.k, r.S, r.M) for r in rows])
            rs = filter(r -> (r.problem, r.variant, r.act, r.k, r.S, r.M) == (pn, v, act, k, S, M), rows)
            good = filter(r -> r.verdict == (act == "relu" ? "CONFIRMED" : "tanh better"), rs)
            bad = count(r -> r.verdict == (act == "relu" ? "VIOLATED" : "tanh worse"), rs)
            ratios = filter(isfinite, [r.ratio for r in rs])
            maxd = act == "relu" && !isempty(good) ? fmt(maximum(r.diff for r in good)) : "—"
            hratios = filter(isfinite, [r.herr_ratio for r in rs])
            println(io, "| $pn | $v | $(actlabel(act, k)) | $(first(rs).r) | $M | $S | $(length(rs)) | $(length(good)) | $bad | " *
                        "$(length(rs) - length(good) - bad) | $maxd | $(isempty(ratios) ? "—" : fmtr(median(ratios))) | " *
                        "$(isempty(hratios) ? "—" : fmtr(median(hratios))) |")
        end
        println(io)
        for pn in unique(r.problem for r in rows)
            println(io, "## $pn\n")
            println(io, "| variant | activation | r | M | S | h | err net | err CGVI | err comp | err spline | err ReLU^k (same S) | diff | diff P_{k+1} | ΔE net | ΔE CGVI | ΔE comp | ΔE spline | P_k-equiv / spline / degen | unconv / max ‖r‖ | verdict | s (net / CGVI) |")
            println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in filter(r -> r.problem == pn, rows)
                println(io, "| $(r.variant) | $(r.activation) | $(r.r) | $(r.M) | $(r.S) | $(r.h) | $(fmt(r.err_sn)) | " *
                            "$(fmt(r.err_cgvi)) | $(fmt(r.err_cgvi_comp)) | $(fmt(r.err_spline)) | $(fmt(r.err_relu)) | $(fmt(r.diff)) | " *
                            "$(fmt(r.diff_alt)) | $(fmt(r.herr_sn)) | $(fmt(r.herr_cgvi)) | $(fmt(r.herr_cgvi_comp)) | $(fmt(r.herr_spline)) | " *
                            "$(r.act == "relu" ? "$(r.n_equiv) / $(r.n_spline) / $(r.n_degen)" : "n/a") | $(r.unconverged) / $(fmt(r.max_res)) | $(r.verdict) | " *
                            "$(isnan(r.sec_sn) ? "—" : @sprintf("%.1f", r.sec_sn)) / $(isnan(r.sec_cgvi) ? "—" : @sprintf("%.2f", r.sec_cgvi)) |")
            end
            println(io)
        end
    end
    println("Wrote ", path)
end

"""
One figure for problem `pn`: one panel per (variant, k). Lines are the CGVI baselines — Gauss(k+1) solid
black; for every M > 1, CGVI(P_k) on the composite rule dashed (× at a single h) and CGVI(S_k) dotted
(+ at a single h), in the colour of M — and markers the networks: shape = width S, colour = M,
filled = ReLUᵏ, open = tanh. Every panel has the same fixed size, so the columns have equal width. `fsn` and the three
`fc*` pick the plotted quantity of the network and of the baselines.
"""
function write_plot(rows, path, pn; fsn = r -> r.err_sn, fc = r -> r.err_cgvi, fcc = r -> r.err_cgvi_comp,
        fsp = r -> r.err_spline, ylab = "max rel. error in q")
    rs = filter(r -> r.problem == pn, rows)
    vs, ks, Ms = unique(r.variant for r in rs), unique(r.k for r in rs), sort(unique(r.M for r in rs))
    Ss = sort(unique(r.S for r in rs))
    markers = [:circle, :utriangle, :diamond, :rect, :star5, :hexagon]
    colors = Makie.wong_colors()
    good(x) = isfinite(x) && x > 0
    function baseline!(ax, rr, f; marker = :hline, color, kwargs...)
        base = Dict(r.h => f(r) for r in rr if good(f(r)))
        hs = sort(collect(keys(base)))
        length(hs) == 1 && scatter!(ax, hs, [base[h] for h in hs]; marker, markersize = 18, color)
        length(hs) > 1 && lines!(ax, hs, [base[h] for h in hs]; linewidth = 2, color, kwargs...)
    end
    fig = Figure()
    Label(fig[0, :], pn; fontsize = 22, font = :bold)
    for (i, v) in enumerate(vs), (j, k) in enumerate(ks)
        ax = Axis(fig[i, j]; width = 330, height = 240, xscale = log10, yscale = log10, xlabel = "h",
            ylabel = j == 1 ? ylab : "", title = "$v, k = $k", titlesize = 18)
        rk = filter(r -> r.variant == v && r.k == k, rs)
        isempty(rk) && continue
        baseline!(ax, rk, fc; color = :black)
        for (m, M) in enumerate(Ms)
            rm = filter(r -> r.M == M, rk)
            if M > 1
                baseline!(ax, rm, fcc; color = colors[m], linestyle = :dash, marker = :xcross)
                baseline!(ax, rm, fsp; color = colors[m], linestyle = :dot, marker = :cross)
            end
            for act in unique(r.act for r in rs), (w, S) in enumerate(Ss)
                rr = sort(filter(r -> r.act == act && r.S == S && good(fsn(r)), rm); by = r -> r.h)
                isempty(rr) && continue
                mk = markers[mod1(w, length(markers))]
                if act == "relu"
                    scatter!(ax, [r.h for r in rr], [fsn(r) for r in rr]; marker = mk, markersize = 14, color = colors[m])
                else
                    scatter!(ax, [r.h for r in rr], [fsn(r) for r in rr]; marker = mk, markersize = 14,
                        color = :transparent, strokecolor = colors[m], strokewidth = 1.8)
                end
            end
        end
    end
    # one legend for the whole figure, below the panels
    elems = Any[LineElement(color = :black, linewidth = 2)]
    labels = String["CGVI(P_k), Gauss(k+1)"]
    for (m, M) in enumerate(Ms)
        push!(elems, MarkerElement(marker = :rect, color = colors[m], markersize = 12))
        push!(labels, "M = $M")
        M > 1 || continue
        push!(elems, [LineElement(color = colors[m], linewidth = 2, linestyle = :dash),
                      MarkerElement(marker = :xcross, color = colors[m], markersize = 12)])
        push!(labels, "CGVI(P_k), r×$M")
        push!(elems, [LineElement(color = colors[m], linewidth = 2, linestyle = :dot),
                      MarkerElement(marker = :cross, color = colors[m], markersize = 12)])
        push!(labels, "CGVI(S_k), $(M - 1) knots")
    end
    for (w, S) in enumerate(Ss)
        push!(elems, MarkerElement(marker = markers[mod1(w, length(markers))], color = :gray, markersize = 12))
        push!(labels, "S = $S")
    end
    any(r -> r.act == "relu", rs) && (push!(elems, MarkerElement(marker = :circle, color = :gray, markersize = 12)); push!(labels, "filled: ReLU^k"))
    any(r -> r.act == "tanh", rs) && (push!(elems, MarkerElement(marker = :circle, color = :transparent, strokecolor = :gray,
        strokewidth = 1.8, markersize = 12)); push!(labels, "open: tanh"))
    Legend(fig[length(vs) + 1, :], elems, labels; orientation = :horizontal, nbanks = 3,
        framevisible = false, labelsize = 15)
    resize_to_layout!(fig)
    save(path, fig)
    println("Wrote ", path)
end

"Per problem: error in q and relative energy error."
function write_plots(rows, outdir)
    for pn in unique(r.problem for r in rows)
        for (name, kw) in (("_qerr.pdf", (;)),
                           ("_energy.pdf", (fsn = r -> r.herr_sn, fc = r -> r.herr_cgvi, fcc = r -> r.herr_cgvi_comp,
                                fsp = r -> r.herr_spline, ylab = "max rel. energy error")))
            try
                write_plot(rows, joinpath(outdir, OUTBASE * "_" * pn * name), pn; kw...)
            catch e
                @warn "plot $pn$name failed" exception = e
            end
        end
    end
end

# ---------------------------------------------------------------------------------------------
# driver (as scripts/run_nvi.jl): one failing run must not cost the others; the process exits
# non-zero if any failed, so that a server job that archived nothing does not look like a success
# ---------------------------------------------------------------------------------------------

function main(args)
    "--list" in args && (foreach(r -> println(stem(r)), SWEEP_RUNS); return true)
    "--report" in args && (report(); return true)
    stems = filter(a -> !startswith(a, "--"), args)
    runs = isempty(stems) ? SWEEP_RUNS : map(sweep_run, stems)
    failed = String[]
    for run in runs
        try
            run_sweep(run)
        catch e
            e isa InterruptException && rethrow()
            push!(failed, stem(run))
            @error "$(stem(run)) failed" exception = (e, catch_backtrace())
        end
    end
    isempty(failed) || println("$(length(failed)) of $(length(runs)) runs failed:\n  ", join(failed, "\n  "))
    isempty(stems) && report()
    isempty(failed)
end

main(ARGS) || exit(1)
