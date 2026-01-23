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

GeometricIntegratorsBase.default_options(method::NonLinear_OneLayer_GML) = (
    max_iterations = 10000,
    f_abstol = 8eps(),
    x_suctol = 2eps(),
    linesearch=GeometricIntegratorsBase.default_linesearch(method), 
)

int_step = 0.1
S = 4
R = 8
k_relu = 3 

# Set up the Harmonic Oscillator problem
int_timespan = 1000.0
HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(timestep=int_step,timespan=(0,int_timespan))
initial_hamiltonian = GeometricProblems.HarmonicOscillator.hamiltonian(0.0, HO_lode.ics.q, HO_lode.ics.p, HO_lode.parameters)

HO_ref = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(timestep=int_step,timespan=(0,int_timespan)))
HO_pref = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(timestep=int_step/40,timespan=(0,int_timespan)))

Q = 2 * R
# QGau = QuadratureRules.GaussLegendreQuadrature(R)
# log_file="NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)_GaussLegendre.txt"
# record_results = Dict()

# relu = x->max(0.0,x) ^ k_relu
# OLnetwork = OneLayerNetwork_GML{Float64}(relu,S)
# NLOLCGVNI_Gml = NonLinear_OneLayer_GML(OLnetwork, QGau, show_status = false, bias_interval = [-pi,pi], dict_amount = 400000)

# #HarmonicOscillator
# open(log_file, "w") do io
#     redirect_stdio(stdout=log_file, stderr=log_file) do
#         HO_NLOLsol,internal_values = integrate(HO_lode, NLOLCGVNI_Gml)
#         HO_qerror = relative_maximum_error(HO_NLOLsol.q,HO_ref.q)
#         hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_NLOLsol.q[:]), collect(HO_NLOLsol.p[:]))]
#         relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)

#         record_results[("HO_sol_q")] = collect(HO_NLOLsol.q[:,1])
#         record_results[("HO_sol_p")] = collect(HO_NLOLsol.p[:,1])
#         record_results[("HO_internal_sol")] = internal_values 

#         record_results[("HO_qerror")] = HO_qerror
#         record_results[("HO_hams_err")] = relative_hams_err
#         record_results[("HO_max_hams_err")] = maximum(relative_hams_err)

#         save("NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)_GaussLegendre.jld2",record_results)

#         ### Figures in the paper
#         p = plot(layout=@layout([a; b; c]), label="", size=(700, 700), plot_title="HarmonicOscillator,h = $(int_step)")

#         plot!(p[1], int_step/40:int_step/40:int_timespan, vcat(hcat(internal_values...)[2:end,:]...), label="S$(S)R$(R)Q$(Q)reluk=$(k_relu)", ylims=(-0.6, 0.6))
#         plot!(p[1], int_step/40:int_step/40:int_timespan, collect(HO_pref.q[:, 1])[2:end], label="Analytic Solution", xaxis="time", yaxis="q₁")

#         plot!(p[2], 0:int_step:int_timespan, collect(HO_NLOLsol.p[:, 1]), label="S$(S)R$(R)Q$(Q)reluk=$(k_relu)", ylims=(-0.6, 0.6))
#         plot!(p[2], 0:int_step/40:int_timespan, collect(HO_pref.p[:, 1]), label="Analytic Solution", xaxis="time", yaxis="p₁")

#         plot!(p[3], 0:int_step:int_timespan, relative_hams_err, label="S$(S)R$(R)Q$(Q)reluk=$(k_relu)", xaxis="time", yaxis="Relative Hamiltonian error")
#         savefig(p, "NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)_GaussLegendre.pdf")
#     end
# end

# QGau = QuadratureRules.LobattoChebyshevQuadrature(R)
# log_file="NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)_LobattoChebyshev.txt"
# record_results = Dict()

# relu = x->max(0.0,x) ^ k_relu
# OLnetwork = OneLayerNetwork_GML{Float64}(relu,S)
# NLOLCGVNI_Gml = NonLinear_OneLayer_GML(OLnetwork, QGau, show_status = true, bias_interval = [-pi,pi], dict_amount = 400000)

# #HarmonicOscillator
# open(log_file, "w") do io
#     redirect_stdio(stdout=log_file, stderr=log_file) do
#         HO_NLOLsol,internal_values = integrate(HO_lode, NLOLCGVNI_Gml)
#         HO_qerror = relative_maximum_error(HO_NLOLsol.q,HO_ref.q)
#         hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_NLOLsol.q[:]), collect(HO_NLOLsol.p[:]))]
#         relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)

#         record_results[("HO_sol_q")] = collect(HO_NLOLsol.q[:,1])
#         record_results[("HO_sol_p")] = collect(HO_NLOLsol.p[:,1])
#         record_results[("HO_internal_sol")] = internal_values 

#         record_results[("HO_qerror")] = HO_qerror
#         record_results[("HO_hams_err")] = relative_hams_err
#         record_results[("HO_max_hams_err")] = maximum(relative_hams_err)

#         save("NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)_LobattoChebyshev.jld2",record_results)

#         ### Figures in the paper
#         p = plot(layout=@layout([a; b; c]), label="", size=(700, 700), plot_title="HarmonicOscillator,h = $(int_step)")

#         plot!(p[1], int_step/40:int_step/40:int_timespan, vcat(hcat(internal_values...)[2:end,:]...), label="S$(S)R$(R)Q$(Q)reluk=$(k_relu)", ylims=(-0.6, 0.6))
#         plot!(p[1], int_step/40:int_step/40:int_timespan, collect(HO_pref.q[:, 1])[2:end], label="Analytic Solution", xaxis="time", yaxis="q₁")

#         plot!(p[2], 0:int_step:int_timespan, collect(HO_NLOLsol.p[:, 1]), label="S$(S)R$(R)Q$(Q)reluk=$(k_relu)", ylims=(-0.6, 0.6))
#         plot!(p[2], 0:int_step/40:int_timespan, collect(HO_pref.p[:, 1]), label="Analytic Solution", xaxis="time", yaxis="p₁")

#         plot!(p[3], 0:int_step:int_timespan, relative_hams_err, label="S$(S)R$(R)Q$(Q)reluk=$(k_relu)", xaxis="time", yaxis="Relative Hamiltonian error")
#         savefig(p, "NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)_LobattoChebyshev.pdf")
#     end
# end


QGau = QuadratureRules.LobattoLegendreQuadrature(R)
log_file="TR_NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)_LobattoLegendre.txt"
record_results = Dict()

relu = x->max(0.0,x) ^ k_relu
OLnetwork = OneLayerNetwork_GML{Float64}(relu,S)
NLOLCGVNI_Gml = Time_reversible_OneLayer(OLnetwork, QGau, show_status = false, bias_interval = [-pi,pi], dict_amount = 400000)

#HarmonicOscillator
# open(log_file, "w") do io
#     redirect_stdio(stdout=log_file, stderr=log_file) do
        HO_NLOLsol,internal_values = integrate(HO_lode, NLOLCGVNI_Gml)
        HO_qerror = relative_maximum_error(HO_NLOLsol.q,HO_ref.q)
        hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_NLOLsol.q[:]), collect(HO_NLOLsol.p[:]))]
        relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)

        record_results[("HO_sol_q")] = collect(HO_NLOLsol.q[:,1])
        record_results[("HO_sol_p")] = collect(HO_NLOLsol.p[:,1])
        record_results[("HO_internal_sol")] = internal_values 

        record_results[("HO_qerror")] = HO_qerror
        record_results[("HO_hams_err")] = relative_hams_err
        record_results[("HO_max_hams_err")] = maximum(relative_hams_err)

        save("TR_NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)_LobattoLegendre.jld2",record_results)

        ### Figures in the paper
        p = plot(layout=@layout([a; b; c]), label="", size=(700, 700), plot_title="HarmonicOscillator,h = $(int_step)")

        plot!(p[1], int_step/40:int_step/40:int_timespan, vcat(hcat(internal_values...)[2:end,:]...), label="S$(S)R$(R)Q$(Q)reluk=$(k_relu)", ylims=(-0.6, 0.6))
        plot!(p[1], int_step/40:int_step/40:int_timespan, collect(HO_pref.q[:, 1])[2:end], label="Analytic Solution", xaxis="time", yaxis="q₁")

        plot!(p[2], 0:int_step:int_timespan, collect(HO_NLOLsol.p[:, 1]), label="S$(S)R$(R)Q$(Q)reluk=$(k_relu)", ylims=(-0.6, 0.6))
        plot!(p[2], 0:int_step/40:int_timespan, collect(HO_pref.p[:, 1]), label="Analytic Solution", xaxis="time", yaxis="p₁")

        plot!(p[3], 0:int_step:int_timespan, relative_hams_err, label="S$(S)R$(R)Q$(Q)reluk=$(k_relu)", xaxis="time", yaxis="Relative Hamiltonian error")
        savefig(p, "TR_NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)_LobattoLegendre.pdf")
#     end
# end
