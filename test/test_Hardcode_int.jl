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
int_step = parse(Float64,ARGS[1])
f_abs = eval(Meta.parse(ARGS[2]))
x_suc = eval(Meta.parse(ARGS[3]))

# int_step = 0.1
# f_abs = 2.0
# x_abs = 2.0

GeometricIntegratorsBase.default_options(method::Hardcode_int) = (
    x_suctol = x_suc * eps(),
    f_abstol = f_abs * eps(),
    max_iterations = 10000,
    linesearch=GeometricIntegratorsBase.default_linesearch(method), 
    # linesearch=SimpleSolvers.Bisection(), 
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
    # QGau = QuadratureRules.GaussLegendreQuadrature(R)
    QGau = QuadratureRules.LobattoLegendreQuadrature(R)

    for S in S_list
        for k_relu in k_list
            try
            # log_file="HC_int_072/NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)fabs$(f_abs)xsuc$(x_suc).txt"
            # log_file="HC_int_072/NVI_HO_h$(int_step)S$(S)R$(R)fabs$(f_abs)xsuc$(x_suc)tanh.txt"

            # open(log_file, "w") do io
            #     redirect_stdio(stdout=log_file, stderr=log_file) do
                    record_results = Dict()

                    relu = x->max(0.0,x) ^ k_relu
                    OLnetwork = OneLayerNetwork_GML{Float64}(relu,S)
                    NLOLCGVNI_Gml = Hardcode_int(OLnetwork, QGau, show_status = false, bias_interval = [-pi,pi], dict_amount = 400000)
                
                    #HarmonicOscillator
                    HO_NLOLsol,internal_values = integrate(HO_lode, NLOLCGVNI_Gml)
                    
                    ### Figures in the paper
                    p = plot(layout=@layout([a; b; c]), label="", size=(700, 700), plot_title="HarmonicOscillator,h = $(int_step)")
                    plot!(p[1], int_step/40:int_step/40:int_timespan, vcat(hcat(internal_values...)[2:end,:]...), label="S$(S)R$(R)k$(k_relu)", ylims=(-0.6, 0.6))
                    plot!(p[1], int_step/40:int_step/40:int_timespan, collect(HO_pref.q[:, 1])[2:end], label="Analytic Solution", xaxis="time", yaxis="q₁")
                    plot!(p[2], 0:int_step:int_timespan, collect(HO_NLOLsol.p[:, 1]), label="S$(S)R$(R)k$(k_relu)", ylims=(-0.6, 0.6))
                    plot!(p[2], 0:int_step/40:int_timespan, collect(HO_pref.p[:, 1]), label="Analytic Solution", xaxis="time", yaxis="p₁")
                    plot!(p[3], 0:int_step:int_timespan, relative_hams_err, label="S$(S)R$(R)k$(k_relu)", xaxis="time", yaxis="Relative Hamiltonian error")
                    savefig(p, "HC_int_072/NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)fabs$(f_abs)xsuc$(x_suc).pdf")

                    # save results
                    HO_qerror = relative_maximum_error(HO_NLOLsol.q,HO_ref.q)
                    hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_NLOLsol.q[:]), collect(HO_NLOLsol.p[:]))]
                    relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)
                    record_results[("HO_sol_q")] = collect(HO_NLOLsol.q[:,1])
                    record_results[("HO_sol_p")] = collect(HO_NLOLsol.p[:,1])
                    record_results[("HO_internal_sol")] = internal_values
                    record_results[("HO_qerror")] = HO_qerror
                    record_results[("HO_hams_err")] = relative_hams_err
                    record_results[("HO_max_hams_err")] = maximum(relative_hams_err)
                    save("HC_int_072/NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)fabs$(f_abs)xsuc$(x_suc).jld2",record_results)

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