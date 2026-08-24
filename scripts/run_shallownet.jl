using GeometricIntegratorsBase
using NonlinearIntegrators
using QuadratureRules
using CompactBasisFunctions
using GeometricProblems
# using BenchmarkTools
using CairoMakie
using GeometricSolutions:relative_maximum_error
using GeometricIntegrators
using JLD2
using SimpleSolvers

# int_step = parse(Float64,ARGS[1])
# reg_factor = eval(Meta.parse(ARGS[2]))

int_step = 0.1
reg_factor = 1e-7

GeometricIntegratorsBase.default_options(method::ShallowNet) = (
    max_iterations = 10000,
    regularization_factor = reg_factor,
    linesearch=GeometricIntegratorsBase.default_linesearch(method),
    # linesearch=SimpleSolvers.Static(),
)

R_list = [8,]#16,4
S_list = [6,]#6,8
k_list = [3,]#2,3,4

# ============================================================
# Block 1: Harmonic Oscillator + relu
# ============================================================
int_timespan = 10.0
HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(timestep=int_step, timespan=(0,int_timespan))
initial_hamiltonian = GeometricProblems.HarmonicOscillator.hamiltonian(0.0, HO_lode.ics.q, HO_lode.ics.p, HO_lode.parameters)
HO_ref = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(timestep=int_step, timespan=(0,int_timespan)))

for R in R_list
    Q = 2 * R
    QGau = QuadratureRules.GaussLegendreQuadrature(R)
    for S in S_list
        for k_relu in k_list
            try
                record_results = Dict()

                relu = x -> max(0.0, x) ^ k_relu
                net = ShallowNetBasis{Float64}(relu, S)
                nlmethod = ShallowNet(net, QGau, bias_interval=[-pi,pi], dict_amount=400000)

                HO_sol, HO_internal_values = integrate(HO_lode, nlmethod)
                HO_qerror = relative_maximum_error(HO_sol.q, HO_ref.q)
                hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_sol.q[:]), collect(HO_sol.p[:]))]
                relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)

                ts = collect(0:int_step:int_timespan)
                fig = Figure(size=(800, 900))
                ax1 = Axis(fig[1,1], xlabel="Time", ylabel="q(t)", title="HO relu k=$(k_relu), S$(S)R$(R)")
                lines!(ax1, ts, collect(HO_sol.q[:, 1]), label="S$(S)R$(R)reluk$(k_relu)")
                ax2 = Axis(fig[2,1], xlabel="Time", ylabel="p(t)")
                lines!(ax2, ts, collect(HO_sol.p[:, 1]), label="S$(S)R$(R)reluk$(k_relu)")
                ax3 = Axis(fig[3,1], xlabel="Time", ylabel="Relative Hamiltonian Error")
                lines!(ax3, ts, relative_hams_err, label="S$(S)R$(R)reluk$(k_relu)")
                save("NVI_HO_h$(int_step)S$(S)R$(R)reluk$(k_relu)_reg_factor=$(reg_factor).pdf", fig)

                record_results["HO_sol_q"] = collect(HO_sol.q[:,1])
                record_results["HO_sol_p"] = collect(HO_sol.p[:,1])
                record_results["HO_internal_sol"] = HO_internal_values
                record_results["HO_qerror"] = HO_qerror
                record_results["HO_hams_err"] = relative_hams_err
                record_results["HO_max_hams_err"] = maximum(relative_hams_err)
                save("NVI_HO_h$(int_step)S$(S)R$(R)reluk$(k_relu)_reg_factor=$(reg_factor).jld2", record_results)
            catch e
                println("Error on HO relu, NVI_HO_h$(int_step)S$(S)R$(R)reluk$(k_relu)_reg_factor=$(reg_factor): ", e)
                continue
            end
        end
    end
end

# ============================================================
# Block 2: Harmonic Oscillator + tanh
# ============================================================
int_timespan = 10.0
HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(timestep=int_step, timespan=(0,int_timespan))
initial_hamiltonian = GeometricProblems.HarmonicOscillator.hamiltonian(0.0, HO_lode.ics.q, HO_lode.ics.p, HO_lode.parameters)
HO_ref = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(timestep=int_step, timespan=(0,int_timespan)))

for R in R_list
    Q = 2 * R
    QGau = QuadratureRules.GaussLegendreQuadrature(R)
    for S in S_list
        try
            record_results = Dict()

            net = ShallowNetBasis{Float64}(tanh, S)
            nlmethod = ShallowNet(net, QGau, bias_interval=[-pi,pi], dict_amount=400000)

            HO_sol, HO_internal_values = integrate(HO_lode, nlmethod)
            HO_qerror = relative_maximum_error(HO_sol.q, HO_ref.q)
            hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_sol.q[:]), collect(HO_sol.p[:]))]
            relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)

            ts = collect(0:int_step:int_timespan)
            fig = Figure(size=(800, 900))
            ax1 = Axis(fig[1,1], xlabel="Time", ylabel="q(t)", title="HO tanh, S$(S)R$(R)")
            lines!(ax1, ts, collect(HO_sol.q[:, 1]), label="S$(S)R$(R)tanh")
            ax2 = Axis(fig[2,1], xlabel="Time", ylabel="p(t)")
            lines!(ax2, ts, collect(HO_sol.p[:, 1]), label="S$(S)R$(R)tanh")
            ax3 = Axis(fig[3,1], xlabel="Time", ylabel="Relative Hamiltonian Error")
            lines!(ax3, ts, relative_hams_err, label="S$(S)R$(R)tanh")
            save("NVI_HO_h$(int_step)S$(S)R$(R)tanh_reg_factor=$(reg_factor).pdf", fig)

            record_results["HO_sol_q"] = collect(HO_sol.q[:,1])
            record_results["HO_sol_p"] = collect(HO_sol.p[:,1])
            record_results["HO_internal_sol"] = HO_internal_values
            record_results["HO_qerror"] = HO_qerror
            record_results["HO_hams_err"] = relative_hams_err
            record_results["HO_max_hams_err"] = maximum(relative_hams_err)
            save("NVI_HO_h$(int_step)S$(S)R$(R)tanh_reg_factor=$(reg_factor).jld2", record_results)
        catch e
            println("Error on HO tanh, NVI_HO_h$(int_step)S$(S)R$(R)tanh_reg_factor=$(reg_factor): ", e)
            continue
        end
    end
end

# ============================================================
# Block 3: Double Pendulum + relu
# ============================================================
int_timespan = 10.0
DP_params = (
    l₁ = 1.0,
    l₂ = 1.0,
    m₁ = 1.0,
    m₂ = 1.0,
    g  = 1.0,
)
DP_ics = (t=0.0, q=[0.7853981633974483, 1.5707963267948966], p=[0.2776801836348979, 0.39269908169872414], v=[0.0, 0.39269908169872414])

DP_lode = GeometricProblems.DoublePendulum.lodeproblem(DP_ics.q, DP_ics.p; timestep=int_step, timespan=(0,int_timespan), parameters=DP_params)
DP_initial_hamiltonian = GeometricProblems.DoublePendulum.hamiltonian(0.0, DP_lode.ics.q, DP_lode.ics.p, DP_lode.parameters)
DP_ref = integrate(DP_lode, Gauss(8))
pref_lode = GeometricProblems.DoublePendulum.lodeproblem(DP_ics.q, DP_ics.p; timestep=int_step/40, timespan=(0,int_timespan), parameters=DP_params)
DP_pref = integrate(pref_lode, Gauss(8))

for R in R_list
    Q = 2 * R
    QGau = QuadratureRules.LobattoLegendreQuadrature(R)
    for S in S_list
        for k_relu in k_list
            try
                record_results = Dict()

                relu = x -> max(0.0, x) ^ k_relu
                net = ShallowNetBasis{Float64}(relu, S)
                nlmethod = ShallowNet(net, QGau, show_status=false, bias_interval=[-pi,pi], dict_amount=400000)

                DP_sol, DP_internal = integrate(DP_lode, nlmethod)
                DP_qerror = relative_maximum_error(DP_sol.q, DP_ref.q)
                DP_hams = [GeometricProblems.DoublePendulum.hamiltonian(0, q, p, DP_lode.parameters) for (q, p) in zip(collect(DP_sol.q[:]), collect(DP_sol.p[:]))]
                DP_relative_hams_err = abs.((DP_hams .- DP_initial_hamiltonian) / DP_initial_hamiltonian)

                DP_internal_q1 = [DP_internal[i][:,1] for i in 1:Int(int_timespan/int_step)]
                DP_internal_q2 = [DP_internal[i][:,2] for i in 1:Int(int_timespan/int_step)]
                ts_fine = collect(int_step/40:int_step/40:int_timespan)
                ts_coarse = collect(0:int_step:int_timespan)

                fig = Figure(size=(1200, 900))
                ax1 = Axis(fig[1,1], xlabel="Time", ylabel="q₁", title="DP relu k=$(k_relu), S$(S)R$(R)")
                lines!(ax1, ts_fine, vcat(hcat(DP_internal_q1...)[2:end,:]...), label="S$(S)R$(R)reluk$(k_relu)")
                lines!(ax1, collect(0:int_step/40:int_timespan), collect(DP_pref.q[:, 1]), label="Reference Solution")
                ylims!(ax1, -2, 2)
                ax2 = Axis(fig[1,2], xlabel="Time", ylabel="q₂")
                lines!(ax2, ts_fine, vcat(hcat(DP_internal_q2...)[2:end,:]...), label="S$(S)R$(R)reluk$(k_relu)")
                lines!(ax2, collect(0:int_step/40:int_timespan), collect(DP_pref.q[:, 2]), label="Reference Solution")
                ylims!(ax2, -2, 2)
                ax3 = Axis(fig[2,1], xlabel="Time", ylabel="p₁")
                lines!(ax3, ts_coarse, collect(DP_sol.p[:, 1]), label="S$(S)R$(R)reluk$(k_relu)")
                lines!(ax3, collect(0:int_step/40:int_timespan), collect(DP_pref.p[:, 1]), label="Reference Solution")
                ylims!(ax3, -3, 3)
                ax4 = Axis(fig[2,2], xlabel="Time", ylabel="p₂")
                lines!(ax4, ts_coarse, collect(DP_sol.p[:, 2]), label="S$(S)R$(R)reluk$(k_relu)")
                lines!(ax4, collect(0:int_step/40:int_timespan), collect(DP_pref.p[:, 2]), label="Reference Solution")
                ylims!(ax4, -3, 3)
                ax5 = Axis(fig[3,1:2], xlabel="Time", ylabel="Relative Hamiltonian Error")
                lines!(ax5, ts_coarse, DP_relative_hams_err, label="S$(S)R$(R)reluk$(k_relu)")
                save("NVI_DP_h$(int_step)S$(S)R$(R)reluk$(k_relu)_reg_factor=$(reg_factor).pdf", fig)

                record_results["DP_sol_q1"] = collect(DP_sol.q[:,1])
                record_results["DP_sol_q2"] = collect(DP_sol.q[:,2])
                record_results["DP_sol_p1"] = collect(DP_sol.p[:,1])
                record_results["DP_sol_p2"] = collect(DP_sol.p[:,2])
                record_results["DP_internal_sol"] = DP_internal
                record_results["DP_qerror"] = DP_qerror
                record_results["DP_hams_err"] = DP_relative_hams_err
                record_results["DP_max_hams_err"] = maximum(DP_relative_hams_err)
                save("NVI_DP_h$(int_step)S$(S)R$(R)reluk$(k_relu)_reg_factor=$(reg_factor).jld2", record_results)
            catch e
                println("Error on DP relu, NVI_DP_h$(int_step)S$(S)R$(R)reluk$(k_relu)_reg_factor=$(reg_factor): ", e)
                continue
            end
        end
    end
end

# ============================================================
# Block 4: Double Pendulum + tanh
# ============================================================
int_timespan = 10.0
DP_params = (
    l₁ = 1.0,
    l₂ = 1.0,
    m₁ = 1.0,
    m₂ = 1.0,
    g  = 1.0,
)
DP_ics = (t=0.0, q=[0.7853981633974483, 1.5707963267948966], p=[0.2776801836348979, 0.39269908169872414], v=[0.0, 0.39269908169872414])

DP_lode = GeometricProblems.DoublePendulum.lodeproblem(DP_ics.q, DP_ics.p; timestep=int_step, timespan=(0,int_timespan), parameters=DP_params)
DP_initial_hamiltonian = GeometricProblems.DoublePendulum.hamiltonian(0.0, DP_lode.ics.q, DP_lode.ics.p, DP_lode.parameters)
DP_ref = integrate(DP_lode, Gauss(8))
pref_lode = GeometricProblems.DoublePendulum.lodeproblem(DP_ics.q, DP_ics.p; timestep=int_step/40, timespan=(0,int_timespan), parameters=DP_params)
DP_pref = integrate(pref_lode, Gauss(8))

for R in R_list
    Q = 2 * R
    QGau = QuadratureRules.LobattoLegendreQuadrature(R)
    for S in S_list
        try
            record_results = Dict()

            net = ShallowNetBasis{Float64}(tanh, S)
            nlmethod = ShallowNet(net, QGau, show_status=false, bias_interval=[-pi,pi], dict_amount=400000)

            DP_sol, DP_internal = integrate(DP_lode, nlmethod)
            DP_qerror = relative_maximum_error(DP_sol.q, DP_ref.q)
            DP_hams = [GeometricProblems.DoublePendulum.hamiltonian(0, q, p, DP_lode.parameters) for (q, p) in zip(collect(DP_sol.q[:]), collect(DP_sol.p[:]))]
            DP_relative_hams_err = abs.((DP_hams .- DP_initial_hamiltonian) / DP_initial_hamiltonian)

            DP_internal_q1 = [DP_internal[i][:,1] for i in 1:Int(int_timespan/int_step)]
            DP_internal_q2 = [DP_internal[i][:,2] for i in 1:Int(int_timespan/int_step)]
            ts_fine = collect(int_step/40:int_step/40:int_timespan)
            ts_coarse = collect(0:int_step:int_timespan)

            fig = Figure(size=(1200, 900))
            ax1 = Axis(fig[1,1], xlabel="Time", ylabel="q₁", title="DP tanh, S$(S)R$(R)")
            lines!(ax1, ts_fine, vcat(hcat(DP_internal_q1...)[2:end,:]...), label="S$(S)R$(R)tanh")
            lines!(ax1, collect(0:int_step/40:int_timespan), collect(DP_pref.q[:, 1]), label="Reference Solution")
            ylims!(ax1, -2, 2)
            ax2 = Axis(fig[1,2], xlabel="Time", ylabel="q₂")
            lines!(ax2, ts_fine, vcat(hcat(DP_internal_q2...)[2:end,:]...), label="S$(S)R$(R)tanh")
            lines!(ax2, collect(0:int_step/40:int_timespan), collect(DP_pref.q[:, 2]), label="Reference Solution")
            ylims!(ax2, -2, 2)
            ax3 = Axis(fig[2,1], xlabel="Time", ylabel="p₁")
            lines!(ax3, ts_coarse, collect(DP_sol.p[:, 1]), label="S$(S)R$(R)tanh")
            lines!(ax3, collect(0:int_step/40:int_timespan), collect(DP_pref.p[:, 1]), label="Reference Solution")
            ylims!(ax3, -3, 3)
            ax4 = Axis(fig[2,2], xlabel="Time", ylabel="p₂")
            lines!(ax4, ts_coarse, collect(DP_sol.p[:, 2]), label="S$(S)R$(R)tanh")
            lines!(ax4, collect(0:int_step/40:int_timespan), collect(DP_pref.p[:, 2]), label="Reference Solution")
            ylims!(ax4, -3, 3)
            ax5 = Axis(fig[3,1:2], xlabel="Time", ylabel="Relative Hamiltonian Error")
            lines!(ax5, ts_coarse, DP_relative_hams_err, label="S$(S)R$(R)tanh")
            save("NVI_DP_h$(int_step)S$(S)R$(R)tanh_reg_factor=$(reg_factor).pdf", fig)

            record_results["DP_sol_q1"] = collect(DP_sol.q[:,1])
            record_results["DP_sol_q2"] = collect(DP_sol.q[:,2])
            record_results["DP_sol_p1"] = collect(DP_sol.p[:,1])
            record_results["DP_sol_p2"] = collect(DP_sol.p[:,2])
            record_results["DP_internal_sol"] = DP_internal
            record_results["DP_qerror"] = DP_qerror
            record_results["DP_hams_err"] = DP_relative_hams_err
            record_results["DP_max_hams_err"] = maximum(DP_relative_hams_err)
            save("NVI_DP_h$(int_step)S$(S)R$(R)tanh_reg_factor=$(reg_factor).jld2", record_results)
        catch e
            println("Error on DP tanh, NVI_DP_h$(int_step)S$(S)R$(R)tanh_reg_factor=$(reg_factor): ", e)
            continue
        end
    end
end