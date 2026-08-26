using GeometricIntegratorsBase
using NonlinearIntegrators
using QuadratureRules
using CompactBasisFunctions
using GeometricProblems
using CairoMakie
using GeometricSolutions: relative_maximum_error
using GeometricIntegrators
using JLD2
using SimpleSolvers
using Logging

# Loads max_iterations, dict_amount, figure style,
# and helper functions plot_1d!, plot_2d!, save_1d_jld2, save_2d_jld2.
include(joinpath(@__DIR__, "run_config.jl"))

dtype_str    = length(ARGS) >= 1 ? ARGS[1] : "Float64"        # "Float16", "Float32", or "Float64"
T            = eval(Meta.parse(dtype_str))
int_step     = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : T(0.1)
reg_factor   = length(ARGS) >= 3 ? eval(Meta.parse(ARGS[3])) : T(1e-5)
f_abstol     = length(ARGS) >= 4 ? eval(Meta.parse(ARGS[4])) : SimpleSolvers.absolute_tolerance(T) 
x_suctol     = length(ARGS) >= 5 ? eval(Meta.parse(ARGS[5])) : SimpleSolvers.default_tolerance(T)
int_timespan = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : T(100.0)
solver_name  = length(ARGS) >= 7 ? ARGS[7] : "backtracking"  # "backtracking" or "dogleg"
R      = length(ARGS) >= 8  ? parse(Int, ARGS[8])  : 4
S      = length(ARGS) >= 9  ? parse(Int, ARGS[9])  : 4
k_relu = length(ARGS) >= 10 ? parse(Int, ARGS[10]) : 3
run_dp       = "--double-pendulum" in ARGS

outdir = joinpath(@__DIR__, "results", "shallownet")
mkpath(outdir)

# ── Harmonic Oscillator setup ────────────────────────────────────────────────
HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(timestep=int_step, timespan=(0, int_timespan))
HO_initial_hamiltonian = GeometricProblems.HarmonicOscillator.hamiltonian(
    0.0, HO_lode.ics.q, HO_lode.ics.p, HO_lode.parameters)
HO_ref = GeometricProblems.HarmonicOscillator.exact_solution(
    GeometricProblems.HarmonicOscillator.podeproblem(timestep=int_step, timespan=(0, int_timespan)))
ts_HO = collect(0:int_step:int_timespan)

# ── Block 1: HO + ReLU ───────────────────────────────────────────────────────
QGau = QuadratureRules.GaussLegendreQuadrature(R)
try
    relu = x -> max(zero(T), x)^k_relu
    net = ShallowNetBasis{T}(relu, S)
    nlmethod = ShallowNet(net, QGau, bias_interval=[T(-pi), T(pi)], dict_amount=dict_amount)

    HO_sol, HO_internal = integrate(HO_lode, nlmethod,regularization_factor = reg_factor, max_iterations = max_iterations,f_abstol = f_abstol,x_suctol = x_suctol,solver = SimpleSolvers.DogLeg())
    qend = HO_sol.q[end]
    if !(eltype(qend) === T)
        @warn "upcast from $(T) for HO ReLU h=$(int_step) S=$(S) R=$(R) k=$(k_relu)"
    elseif any(!isfinite, qend)
        @warn "nonfinite for HO ReLU h=$(int_step) S=$(S) R=$(R) k=$(k_relu)"
    else
        HO_qerror = relative_maximum_error(HO_sol.q, HO_ref.q)
        hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters)
                for (q, p) in zip(collect(HO_sol.q[:]), collect(HO_sol.p[:]))]
        hams_err = abs.((hams .- HO_initial_hamiltonian) / HO_initial_hamiltonian)

        fname = "NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)reg=$(reg_factor)fabs=$(f_abstol)_xsuc=$(x_suctol)_solver=default_$(dtype_str)"
        plot_1d!(outdir, fname, ts_HO, collect(HO_sol.q[:, 1]), collect(HO_sol.p[:, 1]), hams_err,
                 "HO ReLU k=$(k_relu) S$(S)R$(R) h=$(int_step) $(dtype_str)")
        save_1d_jld2(outdir, fname, collect(HO_sol.q[:, 1]), collect(HO_sol.p[:, 1]),
                     HO_internal, HO_qerror, hams_err; prefix="HO")
    end
catch e
    println("Error HO ReLU h=$(int_step) S=$(S) R=$(R) k=$(k_relu): ", e)
end

# ── Block 2: HO + tanh ───────────────────────────────────────────────────────
try
    net = ShallowNetBasis{T}(tanh, S)
    nlmethod = ShallowNet(net, QGau, bias_interval=[T(-pi), T(pi)], dict_amount=dict_amount)

    HO_sol, HO_internal = integrate(HO_lode, nlmethod)
    qend = HO_sol.q[end]
    if !(eltype(qend) === T)
        @warn "upcast from $(T) for HO tanh h=$(int_step) S=$(S) R=$(R)"
    elseif any(!isfinite, qend)
        @warn "nonfinite for HO tanh h=$(int_step) S=$(S) R=$(R)"
    else
        HO_qerror = relative_maximum_error(HO_sol.q, HO_ref.q)
        hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters)
                for (q, p) in zip(collect(HO_sol.q[:]), collect(HO_sol.p[:]))]
        hams_err = abs.((hams .- HO_initial_hamiltonian) / HO_initial_hamiltonian)

        fname = "NVI_HO_h$(int_step)S$(S)R$(R)tanh_reg=$(reg_factor)fabs=$(f_abstol)xsuc=$(x_suctol)_$(solver_name)_$(dtype_str)"
        plot_1d!(outdir, fname, ts_HO, collect(HO_sol.q[:, 1]), collect(HO_sol.p[:, 1]), hams_err,
                 "HO tanh S$(S)R$(R) h=$(int_step) $(dtype_str)")
        save_1d_jld2(outdir, fname, collect(HO_sol.q[:, 1]), collect(HO_sol.p[:, 1]),
                     HO_internal, HO_qerror, hams_err; prefix="HO")
    end
catch e
    println("Error HO tanh h=$(int_step) S=$(S) R=$(R): ", e)
end

# ── Double Pendulum (optional) ────────────────────────────────────────────────
if run_dp
    DP_params = (l₁=1.0, l₂=1.0, m₁=1.0, m₂=1.0, g=1.0)
    DP_ics = (t=0.0, q=[0.7853981633974483, 1.5707963267948966],
               p=[0.2776801836348979, 0.39269908169872414],
               v=[0.0, 0.39269908169872414])
    DP_lode = GeometricProblems.DoublePendulum.lodeproblem(
        DP_ics.q, DP_ics.p; timestep=int_step, timespan=(0, int_timespan), parameters=DP_params)
    DP_initial_hamiltonian = GeometricProblems.DoublePendulum.hamiltonian(
        0.0, DP_lode.ics.q, DP_lode.ics.p, DP_lode.parameters)
    DP_ref = integrate(DP_lode, Gauss(8))
    ts_DP = collect(0:int_step:int_timespan)

    # Block 3: DP + ReLU
    QLob = QuadratureRules.LobattoLegendreQuadrature(R)
    try
        relu = x -> max(zero(T), x)^k_relu
        net = ShallowNetBasis{T}(relu, S)
        nlmethod = ShallowNet(net, QLob, show_status=false, bias_interval=[T(-pi), T(pi)], dict_amount=dict_amount)

        DP_sol, DP_internal = integrate(DP_lode, nlmethod)
        qend = DP_sol.q[end]
        if !(eltype(qend) === T)
            @warn "upcast from $(T) for DP ReLU h=$(int_step) S=$(S) R=$(R) k=$(k_relu)"
        elseif any(!isfinite, qend)
            @warn "nonfinite for DP ReLU h=$(int_step) S=$(S) R=$(R) k=$(k_relu)"
        else
            DP_qerror = relative_maximum_error(DP_sol.q, DP_ref.q)
            DP_hams = [GeometricProblems.DoublePendulum.hamiltonian(0, q, p, DP_lode.parameters)
                       for (q, p) in zip(collect(DP_sol.q[:]), collect(DP_sol.p[:]))]
            DP_hams_err = abs.((DP_hams .- DP_initial_hamiltonian) / DP_initial_hamiltonian)

            fname = "NVI_DP_h$(int_step)S$(S)R$(R)reluk=$(k_relu)reg=$(reg_factor)fabs=$(f_abstol)xsuc=$(x_suctol)_$(solver_name)_$(dtype_str)"
            plot_2d!(outdir, fname, ts_DP,
                     collect(DP_sol.q[:, 1]), collect(DP_sol.q[:, 2]),
                     collect(DP_sol.p[:, 1]), collect(DP_sol.p[:, 2]),
                     DP_hams_err, "DP ReLU k=$(k_relu) S$(S)R$(R) h=$(int_step) $(dtype_str)")
            save_2d_jld2(outdir, fname,
                         collect(DP_sol.q[:, 1]), collect(DP_sol.q[:, 2]),
                         collect(DP_sol.p[:, 1]), collect(DP_sol.p[:, 2]),
                         DP_internal, DP_qerror, DP_hams_err; prefix="DP")
        end
    catch e
        println("Error DP ReLU h=$(int_step) S=$(S) R=$(R) k=$(k_relu): ", e)
    end

    # Block 4: DP + tanh
    try
        net = ShallowNetBasis{T}(tanh, S)
        nlmethod = ShallowNet(net, QLob, show_status=false, bias_interval=[T(-pi), T(pi)], dict_amount=dict_amount)

        DP_sol, DP_internal = integrate(DP_lode, nlmethod)
        qend = DP_sol.q[end]
        if !(eltype(qend) === T)
            @warn "upcast from $(T) for DP tanh h=$(int_step) S=$(S) R=$(R)"
        elseif any(!isfinite, qend)
            @warn "nonfinite for DP tanh h=$(int_step) S=$(S) R=$(R)"
        else
            DP_qerror = relative_maximum_error(DP_sol.q, DP_ref.q)
            DP_hams = [GeometricProblems.DoublePendulum.hamiltonian(0, q, p, DP_lode.parameters)
                       for (q, p) in zip(collect(DP_sol.q[:]), collect(DP_sol.p[:]))]
            DP_hams_err = abs.((DP_hams .- DP_initial_hamiltonian) / DP_initial_hamiltonian)

            fname = "NVI_DP_h$(int_step)S$(S)R$(R)tanh_reg=$(reg_factor)fabs=$(f_abstol)xsuc=$(x_suctol)_$(solver_name)_$(dtype_str)"
            plot_2d!(outdir, fname, ts_DP,
                     collect(DP_sol.q[:, 1]), collect(DP_sol.q[:, 2]),
                     collect(DP_sol.p[:, 1]), collect(DP_sol.p[:, 2]),
                     DP_hams_err, "DP tanh S$(S)R$(R) h=$(int_step) $(dtype_str)")
            save_2d_jld2(outdir, fname,
                         collect(DP_sol.q[:, 1]), collect(DP_sol.q[:, 2]),
                         collect(DP_sol.p[:, 1]), collect(DP_sol.p[:, 2]),
                         DP_internal, DP_qerror, DP_hams_err; prefix="DP")
        end
    catch e
        println("Error DP tanh h=$(int_step) S=$(S) R=$(R): ", e)
    end
end
