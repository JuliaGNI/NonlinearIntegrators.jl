using GeometricIntegratorsBase
using NonlinearIntegrators
using QuadratureRules
using CompactBasisFunctions
using GeometricProblems
# using BenchmarkTools
using Plots
using GeometricSolutions:relative_maximum_error
using GeometricIntegrators
using JLD2
using SimpleSolvers

int_step = parse(Float64,ARGS[1])
f_abs = eval(Meta.parse(ARGS[2]))
x_suc = eval(Meta.parse(ARGS[3]))

# int_step = 0.1
# f_abs = 2.0
# x_abs = 2.0

GeometricIntegratorsBase.default_options(method::NonLinear_OneLayer_GML) = (
    x_suctol = x_suc * eps(),
    f_abstol = f_abs * eps(),
    max_iterations = 10000,
    # linesearch=GeometricIntegratorsBase.default_linesearch(method), 
    linesearch=SimpleSolvers.Static(), 
)
# SimpleSolvers.Backtracking() # The default linear search method is Backtracking()
# # GeometricIntegrators.Integrators.default_linesearch(method::PR_Integrator) =SimpleSolvers.Quadratic()
# SimpleSolvers.Bisection()
# SimpleSolvers.Static()

R_list = [8,16,4]#
S_list = [4,6,8]# 
k_list = [2,3,4]# 

# Set up the Harmonic Oscillator problem
int_timespan = 1000.0
HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(timestep=int_step,timespan=(0,int_timespan))
initial_hamiltonian = GeometricProblems.HarmonicOscillator.hamiltonian(0.0, HO_lode.ics.q, HO_lode.ics.p, HO_lode.parameters)

HO_ref = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(timestep=int_step,timespan=(0,int_timespan)))
HO_pref = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(timestep=int_step/40,timespan=(0,int_timespan)))

for R in R_list
    Q = 2 * R
    QGau = QuadratureRules.GaussLegendreQuadrature(R)
    for S in S_list
        for k_relu in k_list
            try
                # log_file="default_linesearch_073_gau/NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)fabs$(f_abs)xsuc$(x_suc).txt"
                # log_file="default_linesearch_073_gau/NVI_HO_h$(int_step)S$(S)R$(R)fabs$(f_abs)xsuc$(x_suc)tanh.txt"

                # open(log_file, "w") do io
                #     redirect_stdio(stdout=log_file, stderr=log_file) do
                        record_results = Dict()

                        relu = x->max(0.0,x) ^ k_relu
                        OLnetwork = OneLayerNetwork_GML{Float64}(relu,S)
                        NLOLCGVNI_Gml = NonLinear_OneLayer_GML(OLnetwork, QGau, show_status = false, bias_interval = [-pi,pi], dict_amount = 400000)
                    
                        #HarmonicOscillator
                        HO_NLOLsol,internal_values = integrate(HO_lode, NLOLCGVNI_Gml)
                        HO_qerror = relative_maximum_error(HO_NLOLsol.q,HO_ref.q)
                        hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_NLOLsol.q[:]), collect(HO_NLOLsol.p[:]))]
                        relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)

                        ### Figures in the paper
                        p = plot(layout=@layout([a; b; c]), label="", size=(700, 700), plot_title="HarmonicOscillator,h = $(int_step)")
                        plot!(p[1], int_step/40:int_step/40:int_timespan, vcat(hcat(internal_values...)[2:end,:]...), label="S$(S)R$(R)k$(k_relu)", ylims=(-0.6, 0.6))
                        plot!(p[1], int_step/40:int_step/40:int_timespan, collect(HO_pref.q[:, 1])[2:end], label="Analytic Solution", xaxis="time", yaxis="q₁")
                        plot!(p[2], 0:int_step:int_timespan, collect(HO_NLOLsol.p[:, 1]), label="S$(S)R$(R)k$(k_relu)", ylims=(-0.6, 0.6))
                        plot!(p[2], 0:int_step/40:int_timespan, collect(HO_pref.p[:, 1]), label="Analytic Solution", xaxis="time", yaxis="p₁")
                        plot!(p[3], 0:int_step:int_timespan, relative_hams_err, label="S$(S)R$(R)k$(k_relu)", xaxis="time", yaxis="Relative Hamiltonian error")
                        savefig(p, "default_linesearch_073_gau/NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)fabs$(f_abs)xsuc$(x_suc).pdf")

                        # save results
                        record_results[("HO_sol_q")] = collect(HO_NLOLsol.q[:,1])
                        record_results[("HO_sol_p")] = collect(HO_NLOLsol.p[:,1])
                        record_results[("HO_internal_sol")] = internal_values
                        record_results[("HO_qerror")] = HO_qerror
                        record_results[("HO_hams_err")] = relative_hams_err
                        record_results[("HO_max_hams_err")] = maximum(relative_hams_err)
                        save("default_linesearch_073_gau/NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)fabs$(f_abs)xsuc$(x_suc).jld2",record_results)

                        # # figure for q
                        # plot(int_step/40:int_step/40:int_timespan, vcat(hcat(internal_values...)[2:end,:]...))
                        # plot!(int_step/40:int_step/40:int_timespan, collect(HO_pref.q[:, 1])[2:end], label="Truth", linestyle=:dash, linecolor=:black)
                        # scatter!(collect(0:int_step:int_timespan), collect(HO_NLOLsol.q[:, 1]), label="Discrete solution")
                        # savefig("result_figures/nn_harmonic_oscillator_solution.png")
                #     end
                # end
            catch e
                println("Error on Harmonic Oscillator, NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)fabs$(f_abs)xsuc$(x_suc)",e)
                continue
            end
        end
    end
end

# # DoublePendulum
# # int_step = 1.0
# int_timespan = 200.0
# DP_params = (
#     l₁ = 1.0,
#     l₂ = 1.0,
#     m₁ = 1.0,
#     m₂ = 1.0,
#     g = 1.0,
#     )

# DP_ics = (t = 0.0, q = [0.7853981633974483, 1.5707963267948966], p = [0.2776801836348979, 0.39269908169872414], v = [0.0, 0.39269908169872414])

# DP_lode = GeometricProblems.DoublePendulum.lodeproblem(DP_ics.q, DP_ics.p; timestep = int_step, timespan = (0,int_timespan), parameters = DP_params)
# DP_initial_hamiltonian = GeometricProblems.DoublePendulum.hamiltonian(0.0, DP_lode.ics.q, DP_lode.ics.p, DP_lode.parameters)
# DP_ref = integrate(DP_lode, Gauss(8))

# pref_lode = GeometricProblems.DoublePendulum.lodeproblem(DP_ics.q, DP_ics.p; timestep = int_step/40, timespan = (0,int_timespan), parameters = DP_params)
# DP_pref= integrate(pref_lode, Gauss(8))

# DP_internal_q1 = Array{Vector}(undef,Int(int_timespan/int_step))
# DP_internal_q2 = Array{Vector}(undef,Int(int_timespan/int_step))

# for R in R_list
#     Q = 2 * R
#     QGau = QuadratureRules.LobattoLegendreQuadrature(R)

#     for S = S_list
#         for k_relu in k_list
#             try
#             # log_file="default_linesearch_073_gau/NVI_DP_h$(int_step)S$(S)R$(R)reluk$(k_relu)fabs$(f_abs)xsuc$(x_suc).txt"
#             # open(log_file, "w") do io
#             #     redirect_stdio(stdout=log_file, stderr=log_file) do
#                     record_results = Dict()
                
#                     relu = x->max(0.0,x) ^ k_relu
#                     OLnetwork = OneLayerNetwork_GML{Float64}(relu,S)
#                     NLOLCGVNI_Gml = NonLinear_OneLayer_GML(OLnetwork, QGau, show_status = false, bias_interval = [-pi,pi], dict_amount = 400000)

#                     DP_NLOLsol,DP_internal = integrate(DP_lode, NLOLCGVNI_Gml)
                    
#                     # Figures for the paper
#                     for i in 1:Int(int_timespan/int_step)
#                         DP_internal_q1[i] = DP_internal[i][:,1]
#                         DP_internal_q2[i] = DP_internal[i][:,2]
#                     end

#                     p = plot(layout=@layout([a b; c d; e]), label="", size=(700, 700), plot_title="S$(S)R$(R)k$(k_relu)")# d;e

#                     plot!(p[1], int_step/40:int_step/40:int_timespan, vcat(hcat(DP_internal_q1...)[2:end,:]...), label="S$(S)R$(R)k$(k_relu)", xaxis="time", yaxis="q₁")
#                     plot!(p[1], 0:int_step/40:int_timespan, collect(DP_pref.q[:, 1]), label="Reference Solution", ylims=(-2, 2))

#                     plot!(p[2], int_step/40:int_step/40:int_timespan, vcat(hcat(DP_internal_q2...)[2:end,:]...), label="S$(S)R$(R)k$(k_relu)", xaxis="time", yaxis="q₂")
#                     plot!(p[2], 0:int_step/40:int_timespan, collect(DP_pref.q[:, 2]), label="Reference Solution", ylims=(-2, 2))

#                     plot!(p[3], 0:int_step:int_timespan, collect(DP_NLOLsol.p[:, 1]), label="S$(S)R$(R)k$(k_relu)", xaxis="time", yaxis="p₁")
#                     plot!(p[3], 0:int_step/40:int_timespan, collect(DP_pref.p[:, 1]), label="Reference Solution", ylims=(-3, 3))

#                     plot!(p[4], 0:int_step:int_timespan, collect(DP_NLOLsol.p[:, 2]), label="S$(S)R$(R)k$(k_relu)", xaxis="time", yaxis="p₂")
#                     plot!(p[4], 0:int_step/40:int_timespan, collect(DP_pref.p[:, 2]), label="Reference Solution", ylims=(-3, 3))

#                     plot!(p[5], 0:int_step:int_timespan, DP_relative_hams_err, label="S$(S)R$(R)k$(k_relu)", xaxis="time", yaxis="Relative Hamiltonian error")
#                     savefig(p, "default_linesearch_073_gau/NVI_DP_h$(int_step)S$(S)R$(R)reluk$(k_relu)fabs$(f_abs)xsuc$(x_suc).pdf")

#                     #save results
#                     DP_qerror = relative_maximum_error(DP_NLOLsol.q,DP_ref.q)
#                     DP_hams = [GeometricProblems.DoublePendulum.hamiltonian(0, q, p, DP_lode.parameters) for (q, p) in zip(collect(DP_NLOLsol.q[:]), collect(DP_NLOLsol.p[:]))]
#                     DP_relative_hams_err = abs.((DP_hams .- DP_initial_hamiltonian) / DP_initial_hamiltonian)

#                     record_results[("DP_sol_q1")] = collect(DP_NLOLsol.q[:,1])
#                     record_results[("DP_sol_q2")] = collect(DP_NLOLsol.q[:,2])
#                     record_results[("DP_sol_p1")] = collect(DP_NLOLsol.p[:,1])
#                     record_results[("DP_sol_p2")] = collect(DP_NLOLsol.p[:,2])
#                     record_results[("DP_internal_sol")] = DP_internal

#                     record_results[("DP_qerror")] = DP_qerror
#                     record_results[("DP_hams_err")] = DP_relative_hams_err
#                     record_results[("DP_max_hams_err")] = maximum(DP_relative_hams_err)

#                     save("default_linesearch_073_gau/NVI_DP_h$(int_step)S$(S)R$(R)reluk$(k_relu)fabs$(f_abs)xsuc$(x_suc).jld2",record_results)
#                 #     end
#                 # end
#             catch e
#                 println("Error on Double Pendulum, NVI_DP_h$(int_step)S$(S)R$(R)reluk$(k_relu)fabs$(f_abs)xsuc$(x_suc)",e)
#                 continue
#             end
#         end
#     end
# end
