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
using Infiltrator

# int_step = parse(Float64,ARGS[1])
# f_abs = eval(Meta.parse(ARGS[2]))
# x_suc = eval(Meta.parse(ARGS[3]))

int_step = 0.1
f_abs = 0.0
x_suc = 0.0

GeometricIntegratorsBase.default_options(method::NonLinear_OneLayer_GML) = (
    # x_suctol = x_suc * eps(),
    # f_abstol = f_abs * eps(),
    max_iterations = 10000,
    linesearch=GeometricIntegratorsBase.default_linesearch(method),
    regularization_factor = 1e-5, 
)
# SimpleSolvers.Backtracking() # The default linear search method is Backtracking()
# # GeometricIntegrators.Integrators.default_linesearch(method::PR_Integrator) =SimpleSolvers.Quadratic()
# SimpleSolvers.Bisection()
# SimpleSolvers.Static()

# R_list = [8,16,4]#
# S_list = [4,6,8]# 
# k_list = [2,3,4]# 

S = 4
R = 4
k_relu = 3

# Set up the Harmonic Oscillator problem
int_timespan = 10.0
HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(timestep=int_step,timespan=(0,int_timespan))
initial_hamiltonian = GeometricProblems.HarmonicOscillator.hamiltonian(0.0, HO_lode.ics.q, HO_lode.ics.p, HO_lode.parameters)

HO_ref = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(timestep=int_step,timespan=(0,int_timespan)))
HO_pref = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(timestep=int_step/40,timespan=(0,int_timespan)))

# for bias_interval in [[0.0,pi],[0.0,1.0],[ -1.0,1.0],[-pi,pi]]
    Q = 2 * R
    QGau = QuadratureRules.GaussLegendreQuadrature(R)
    # QGau = QuadratureRules.LobattoLegendreQuadrature(R)

    # for S in S_list
    #     for k_relu in k_list
            # try
            # log_file="HC_int_072/NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)fabs$(f_abs)xsuc$(x_suc).txt"
            # log_file="HC_int_072/NVI_HO_h$(int_step)S$(S)R$(R)fabs$(f_abs)xsuc$(x_suc)tanh.txt"

            # open(log_file, "w") do io
            #     redirect_stdio(stdout=log_file, stderr=log_file) do
                    record_results = Dict()
                    # println("bias_interval=",bias_interval,", S=",S, " k_relu=",k_relu)
                    relu = x->max(0.0,x) ^ k_relu
                    OLnetwork = OneLayerNetwork_GML{Float64}(relu,S)
                    NLOLCGVNI_Gml = NonLinear_OneLayer_GML(OLnetwork, QGau, show_status = false, bias_interval = [-pi,pi], dict_amount = 400000)
                
                    #HarmonicOscillator
                    HO_NLOLsol = integrate(HO_lode, NLOLCGVNI_Gml)
                    HO_qerror = relative_maximum_error(HO_NLOLsol.sol.q,HO_ref.q)
                    hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_NLOLsol.sol.q[:]), collect(HO_NLOLsol.sol.p[:]))]
                    relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)

                    ### Figures in the paper
                    p = plot(layout=@layout([a; b; c]), label="", size=(700, 700), plot_title="HarmonicOscillator,h = $(int_step)")
                    plot!(p[1], int_step/40:int_step/40:int_timespan, vcat(hcat(HO_NLOLsol.internal_values...)[2:end,:]...), label="S$(S)R$(R)k$(k_relu)", ylims=(-0.6, 0.6))
                    plot!(p[1], int_step/40:int_step/40:int_timespan, collect(HO_pref.q[:, 1])[2:end], label="Analytic Solution", xaxis="time", yaxis="q₁")
                    plot!(p[2], 0:int_step:int_timespan, collect(HO_NLOLsol.sol.p[:, 1]), label="S$(S)R$(R)k$(k_relu)", ylims=(-0.6, 0.6))
                    plot!(p[2], 0:int_step/40:int_timespan, collect(HO_pref.p[:, 1]), label="Analytic Solution", xaxis="time", yaxis="p₁")
                    plot!(p[3], 0:int_step:int_timespan, relative_hams_err, label="S$(S)R$(R)k$(k_relu)", xaxis="time", yaxis="Relative Hamiltonian error")
                    savefig(p, "NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)fabs$(f_abs)xsuc$(x_suc)_T$(int_timespan)_Dogleg078_reg.pdf")

                    # save results
                    record_results[("HO_sol_q")] = collect(HO_NLOLsol.sol.q[:,1])
                    record_results[("HO_sol_p")] = collect(HO_NLOLsol.sol.p[:,1])
                    record_results[("HO_internal_sol")] = HO_NLOLsol.internal_values
                    record_results[("HO_qerror")] = HO_qerror
                    record_results[("HO_hams_err")] = relative_hams_err
                    record_results[("HO_max_hams_err")] = maximum(relative_hams_err)
                    save("NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)fabs$(f_abs)xsuc$(x_suc)_T$(int_timespan)_Dogleg078_reg.jld2",record_results)

                    # # figure for q
                    # plot(int_step/40:int_step/40:int_timespan, vcat(hcat(internal_values...)[2:end,:]...))
                    # plot!(int_step/40:int_step/40:int_timespan, collect(HO_pref.q[:, 1])[2:end], label="Truth", linestyle=:dash, linecolor=:black)
                    # scatter!(collect(0:int_step:int_timespan), collect(HO_NLOLsol.sol.q[:, 1]), label="Discrete solution")
                    # savefig("result_figures/nn_harmonic_oscillator_solution.png")

                #     end
                # end
            # catch e
            #     println("Error on Harmonic Oscillator, NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)fabs$(f_abs)xsuc$(x_suc)",e)
            #     # continue
            # end
#         end
#     end
# end