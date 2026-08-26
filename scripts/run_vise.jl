using Symbolics
using NonlinearIntegrators
using QuadratureRules
using GeometricProblems
using CairoMakie
using GeometricSolutions: relative_maximum_error
using GeometricIntegrators
using JLD2
using Logging

# Loads figure style and helper functions plot_1d!, plot_2d!, save_1d_jld2, save_2d_jld2.
# Note: R_list / S_list / k_list / dict_amount are not used for VISE.
include(joinpath(@__DIR__, "run_config.jl"))

dtype_str    = length(ARGS) >= 4 ? ARGS[4] : "Float64"  # "Float16", "Float32", or "Float64"
T            = eval(Meta.parse(dtype_str))
int_step     = parse(Float64, ARGS[1])
R            = parse(Int,     ARGS[2])
int_timespan = parse(Float64, ARGS[3])
reg_factor   = length(ARGS) >= 3  ? eval(Meta.parse(ARGS[4])) : T(1e-7)



outdir = joinpath(@__DIR__, "results", "vise")
mkpath(outdir)

QGau = QuadratureRules.GaussLegendreQuadrature(R)

# ── Harmonic Oscillator ───────────────────────────────────────────────────────
# Ansatz: q(t) = W[1] * sin(W[2]*t + W[3])
# Initial guess from test_vise.jl.
begin
    @variables W[1:3] ttt
    q_expr = W[1] * sin(W[2] * ttt + W[3])
    vise_basis = VISEBasis{T}([q_expr], [W], ttt, 1)
    vise_method = VISE(vise_basis, QGau, [T.([-0.5000433352162222, 0.705350078478666, -1.5678140333370576])])

    HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(timestep=int_step, timespan=(0, int_timespan))
    HO_initial_hamiltonian = GeometricProblems.HarmonicOscillator.hamiltonian(
        0.0, HO_lode.ics.q, HO_lode.ics.p, HO_lode.parameters)
    HO_ref = GeometricProblems.HarmonicOscillator.exact_solution(
        GeometricProblems.HarmonicOscillator.podeproblem(timestep=int_step, timespan=(0, int_timespan)))
    ts_HO = collect(0:int_step:int_timespan)

    try
        HO_sol, HO_internal, HO_x_list = integrate(HO_lode, vise_method, regularization_factor = reg_factor, max_iterations = max_iterations)
        qend = HO_sol.q[end]
        if !(eltype(qend) === T)
            @warn "upcast from $(T) for HO VISE h=$(int_step) R=$(R)"
        elseif any(!isfinite, qend)
            @warn "nonfinite for HO VISE h=$(int_step) R=$(R)"
        else
            HO_qerror = relative_maximum_error(HO_sol.q, HO_ref.q)
            hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters)
                    for (q, p) in zip(collect(HO_sol.q[:]), collect(HO_sol.p[:]))]
            hams_err = abs.((hams .- HO_initial_hamiltonian) / HO_initial_hamiltonian)

            fname = "VISE_HO_h$(int_step)R$(R)_$(dtype_str)"
            plot_1d!(outdir, fname, ts_HO, collect(HO_sol.q[:, 1]), collect(HO_sol.p[:, 1]), hams_err,
                     "HO VISE R=$(R) h=$(int_step) $(dtype_str)")
            save(joinpath(outdir, "$(fname).jld2"), Dict(
                "HO_sol_q"        => collect(HO_sol.q[:, 1]),
                "HO_sol_p"        => collect(HO_sol.p[:, 1]),
                "HO_internal_sol" => HO_internal,
                "HO_x_list"       => HO_x_list,
                "HO_qerror"       => HO_qerror,
                "HO_hams_err"     => hams_err,
                "HO_max_hams_err" => maximum(hams_err),
            ))
        end
    catch e
        println("Error HO VISE h=$(int_step) R=$(R): ", e)
    end
end

# ── Perturbed Pendulum ────────────────────────────────────────────────────────
# Ansatz: q(t) = W[1] * cos(W[2]*t + W[3])
# Initial guess from test_vise.jl.
begin
    @variables W[1:3] ttt
    q_expr = W[1] * cos(W[2] * ttt + W[3])
    vise_basis = VISEBasis{T}([q_expr], [W], ttt, 1)
    vise_method = VISE(vise_basis, QGau, [T.([-0.51941, -0.47405, 2.8713])])

    PP_lode = GeometricProblems.PerturbedPendulum.lodeproblem(timestep=int_step, timespan=(0, int_timespan))
    PP_ref  = integrate(PP_lode, Gauss(8))
    PP_initial_hamiltonian = GeometricProblems.PerturbedPendulum.hamiltonian(
        0.0, PP_lode.ics.q, PP_lode.ics.p, PP_lode.parameters)
    ts_PP = collect(0:int_step:int_timespan)

    try
        PP_sol, PP_internal, PP_x_list = integrate(PP_lode, vise_method)
        qend = PP_sol.q[end]
        if !(eltype(qend) === T)
            @warn "upcast from $(T) for PP VISE h=$(int_step) R=$(R)"
        elseif any(!isfinite, qend)
            @warn "nonfinite for PP VISE h=$(int_step) R=$(R)"
        else
            PP_qerror = relative_maximum_error(PP_sol.q, PP_ref.q)
            PP_hams = [GeometricProblems.PerturbedPendulum.hamiltonian(0.0, q, p, PP_lode.parameters)
                       for (q, p) in zip(collect(PP_sol.q[:]), collect(PP_sol.p[:]))]
            PP_hams_err = abs.((PP_hams .- PP_initial_hamiltonian) / PP_initial_hamiltonian)

            fname = "VISE_PP_h$(int_step)R$(R)_$(dtype_str)"
            plot_1d!(outdir, fname, ts_PP, collect(PP_sol.q[:, 1]), collect(PP_sol.p[:, 1]), PP_hams_err,
                     "Perturbed Pendulum VISE R=$(R) h=$(int_step) $(dtype_str)")
            save(joinpath(outdir, "$(fname).jld2"), Dict(
                "PP_sol_q"        => collect(PP_sol.q[:, 1]),
                "PP_sol_p"        => collect(PP_sol.p[:, 1]),
                "PP_internal_sol" => PP_internal,
                "PP_x_list"       => PP_x_list,
                "PP_qerror"       => PP_qerror,
                "PP_hams_err"     => PP_hams_err,
                "PP_max_hams_err" => maximum(PP_hams_err),
            ))
        end
    catch e
        println("Error PerturbedPendulum VISE h=$(int_step) R=$(R): ", e)
    end
end

# ── Hénon-Heiles Potential ────────────────────────────────────────────────────
# Ansatz: q₁(t) = W1[1]*cos(W1[2]*t + W1[3]) + W1[4]
#         q₂(t) = W2[1]*cos(W2[2]*t + W2[3]) + W2[4]
# Initial guess from test_vise.jl.
begin
    @variables W1[1:4] W2[1:4] ttt
    q1_expr = W1[1] * cos(W1[2] * ttt + W1[3]) + W1[4]
    q2_expr = W2[1] * cos(W2[2] * ttt + W2[3]) + W2[4]
    vise_basis = VISEBasis{T}([q1_expr, q2_expr], [W1, W2], ttt, 2)
    vise_method = VISE(vise_basis, QGau, [
        T.([0.14831, 1.0, -0.64812, -0.018712]),
        T.([0.14298, -0.97215, 0.7615, -0.0013983]),
    ])

    HH_lode = GeometricProblems.HenonHeilesPotential.lodeproblem(
        [0.1, 0.1], [0.1, 0.1]; timestep=int_step, timespan=(0, int_timespan))
    HH_ref = integrate(HH_lode, Gauss(8))
    HH_initial_hamiltonian = GeometricProblems.HenonHeilesPotential.hamiltonian(
        0.0, HH_lode.ics.q, HH_lode.ics.p, HH_lode.parameters)
    ts_HH = collect(0:int_step:int_timespan)

    try
        HH_sol, HH_internal, HH_x_list = integrate(HH_lode, vise_method)
        qend = HH_sol.q[end]
        if !(eltype(qend) === T)
            @warn "upcast from $(T) for HH VISE h=$(int_step) R=$(R)"
        elseif any(!isfinite, qend)
            @warn "nonfinite for HH VISE h=$(int_step) R=$(R)"
        else
            HH_qerror = relative_maximum_error(HH_sol.q, HH_ref.q)
            HH_hams = [GeometricProblems.HenonHeilesPotential.hamiltonian(0.0, q, p, HH_lode.parameters)
                       for (q, p) in zip(collect(HH_sol.q[:]), collect(HH_sol.p[:]))]
            HH_hams_err = abs.((HH_hams .- HH_initial_hamiltonian) / HH_initial_hamiltonian)

            fname = "VISE_HH_h$(int_step)R$(R)_$(dtype_str)"
            plot_2d!(outdir, fname, ts_HH,
                     collect(HH_sol.q[:, 1]), collect(HH_sol.q[:, 2]),
                     collect(HH_sol.p[:, 1]), collect(HH_sol.p[:, 2]),
                     HH_hams_err, "Hénon-Heiles VISE R=$(R) h=$(int_step) $(dtype_str)")
            save(joinpath(outdir, "$(fname).jld2"), Dict(
                "HH_sol_q1"       => collect(HH_sol.q[:, 1]),
                "HH_sol_q2"       => collect(HH_sol.q[:, 2]),
                "HH_sol_p1"       => collect(HH_sol.p[:, 1]),
                "HH_sol_p2"       => collect(HH_sol.p[:, 2]),
                "HH_internal_sol" => HH_internal,
                "HH_x_list"       => HH_x_list,
                "HH_qerror"       => HH_qerror,
                "HH_hams_err"     => HH_hams_err,
                "HH_max_hams_err" => maximum(HH_hams_err),
            ))
        end
    catch e
        println("Error HenonHeiles VISE h=$(int_step) R=$(R): ", e)
    end
end
