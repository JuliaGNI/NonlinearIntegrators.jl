# Nonlinear variational integrators of the package on the hard problems (PLAN.md §2, phase 2):
#
#   N1  ShallowNet(ReLU³, S = 4, R = 4), kink-free OGA bias interval [1.1, π]: equivalent to
#       CGVI(P₃, R = 4) by Theorem 1 as long as no kink enters a step
#   N2  ShallowNet(ReLU³, S = 4, R = 4) with the default bias interval [-π, π], and
#       ShallowNet(tanh) with S = R = 4 and S = R = 8
#
# N3 (VISE) of the plan is left out: it is a separate study.
#
#   julia --project=benchmark benchmark/hard_problems/run_nvi.jl            # quick cases
#   julia --project=benchmark benchmark/hard_problems/run_nvi.jl full --cases=FrequencyModulatedOscillator_eps0.01
#
# The networks run on the same cases, steps and metrics as run_baselines.jl (see its header),
# with the nonlinear solver of benchmark/theory/relu_k_sweep.jl (DogLeg, regularisation 1e-5,
# at most 1000 iterations). Additionally recorded:
#   n_equiv, n_spline, n_degen   per step and degree of freedom, the ReLU network classified as in
#                                relu_k_sweep.jl: P_k-equivalent / kink inside the step / degenerate
#   diff_cgvi3                   max_n |q_n - q_n^CGVI(P₃)|∞ / max_n |q_n^CGVI(P₃)|∞ over (0, T)
#   secs_cgvi3                   wall time of that CGVI(P₃) run, timed in the same process
#
# Writes results/<case>_nvi.csv.

include(joinpath(@__DIR__, "integration.jl"))

# the stdlib-only reference implementation of the ReLU network (neuron_classes, nn_tangent, …), in
# a module of its own so that its names cannot clash with the ones defined here
module TheoryCommon
include(joinpath(@__DIR__, "..", "theory", "theory_common.jl"))
end
import .TheoryCommon as TC

using NonlinearIntegrators
import SimpleSolvers

const NVI_OPTIONS = (solver = SimpleSolvers.DogLeg(), regularization_factor = 1E-5,
    max_iterations = 1000)
const DICT_AMOUNT = 4000

relu3(x) = max(zero(x), x)^3

function shallownet(σ, S, R, bias)
    ShallowNet(ShallowNetBasis{Float64}(σ, S), GaussLegendreQuadrature(Float64, R);
        show_status = false, bias_interval = collect(bias), dict_amount = DICT_AMOUNT)
end

# k is the power of the ReLU network used to classify its steps, `nothing` for tanh
const NETWORKS = [
    (label = "N1", activation = "ReLU3 kinkfree", S = 4, R = 4, k = 3,
        method = shallownet(relu3, 4, 4, (1.1, π))),
    (label = "N2", activation = "ReLU3", S = 4, R = 4, k = 3,
        method = shallownet(relu3, 4, 4, (-π, π))),
    (label = "N2", activation = "tanh", S = 4, R = 4, k = nothing,
        method = shallownet(tanh, 4, 4, (-π, π))),
    (label = "N2", activation = "tanh", S = 8, R = 8, k = nothing,
        method = shallownet(tanh, 8, 8, (-π, π)))]

# ---- classification of the ReLU steps (as in relu_k_sweep.jl) --------------------------------

# network parameters θ = [a; w; b] of degree of freedom d in the nonlinear solution x of a step,
# laid out as [a (S); p (1); w (S); b (S)], each entry repeated over the D degrees of freedom
function theta_from_x(x, D, S, d)
    TC.pack([x[D * (i - 1) + d] for i in 1:S],
        [x[D * (S + 1) + D * (i - 1) + d] for i in 1:S],
        [x[D * (2S + 1) + D * (i - 1) + d] for i in 1:S])
end

function classify_step(θ, k; TT = collect(range(0, 1; length = 201)))
    cls = TC.neuron_classes(θ, k)
    any(==("interior"), cls) && return "spline"
    out = findall(==("outside"), cls)
    isempty(out) && return "degenerate"
    a, w, b = TC.unpack(θ)
    θo = TC.pack(a[out], w[out], b[out])
    TC.numrank(TC.nn_tangent(θo, TT, k)[1]; rtol = 1e-9) == k + 1 ? "P_k-equiv" : "degenerate"
end

function step_classes(x, D, S, k)
    k === nothing && return (0, 0, 0)
    cls = [classify_step(theta_from_x(xn, D, S, d), k) for xn in x if all(isfinite, xn)
           for d in 1:D]
    (count(==("P_k-equiv"), cls), count(==("spline"), cls), count(==("degenerate"), cls))
end

# ---- networks on the cases ---------------------------------------------------------------------

function run_case(case)
    println("\n── $(case.name): T = $(case.T), Terr = $(case.Terr), h Ω ∈ $(STEP_FACTORS)")
    ref = write_reference(case)
    foreach(N -> warmup(case, 1, N.method; NVI_OPTIONS...), NETWORKS)
    warmup(case, 1, cgvi(3))
    D = length(case.build((0.0, 1.0), 1.0).ics.q)

    invnames = join(string.(keys(case.invariants)), ",")
    open(joinpath(RESULTS_DIR, "$(case.name)_nvi.csv"), "w") do io
        println(io, "case,c,h,method,activation,S,R,steps,status,unconverged,max_res,warnings,q_err,",
            "$(invnames),iterations,secs,n_equiv,n_spline,n_degen,diff_cgvi3,secs_cgvi3")
        for c in STEP_FACTORS
            h = c / case.Ω
            cg = run_one(case, ref, h, 1, cgvi(3))
            for N in NETWORKS
                r = run_one(case, ref, h, 1, N.method; record = true, NVI_OPTIONS...)
                ne, ns, nd = step_classes(r.x, D, N.S, N.k)
                both = r.status in ("ok", "unconverged") && cg.status in ("ok", "unconverged")
                diff = both ? relerr(r.Q, cg.Q) : NaN
                println(io, join((case.name, c, h, N.label, N.activation, N.S, N.R, r.steps,
                        r.status, r.unconverged, csvnum(r.max_res), r.warnings, csvnum(r.q_err), csvnum.(r.inv)...,
                        r.iterations, csvnum(r.secs), ne, ns, nd, csvnum(diff), csvnum(cg.secs)), ","))
                flush(io)
                @printf("  c = %-4g %s %-14s S = %d  %-11s q_err = %.2e  diff = %.2e  %d/%d/%d  %.1f s\n",
                    c, N.label, N.activation, N.S, r.status, r.q_err, diff, ne, ns, nd, r.secs)
            end
        end
    end
end

function main(args)
    mkpath(RESULTS_DIR)
    foreach(run_case, select_cases(args))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
