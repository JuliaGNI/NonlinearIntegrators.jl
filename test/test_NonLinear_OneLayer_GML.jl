using Pkg

# cd("IntegratorNN/GeometricIntegrators.jl")
# cd("..")
cd("IntegratorNN")

Pkg.activate(".")

using GeometricIntegrators
using NonlinearIntegrators
using QuadratureRules
using CompactBasisFunctions
using GeometricProblems
# using BenchmarkTools
# using Plots
# Set up the Harmonic Oscillator problem
int_step =0.5
int_timespan = 10.0
HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(tstep=int_step,tspan=(0,int_timespan))
# HO_pref = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(tspan = (0,int_timespan),tstep = int_step/40))
# HO_truth = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(tspan = (0,int_timespan),tstep = int_step))


QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
BGau4 = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QGau4))
# #set up the Coupled Harmonic Oscillator problem
# CHO = GeometricProblems.CoupledHarmonicOscillator.lodeproblem(tstep=0.5,tspan=(0,20))
# CHO_pref = integrate(CHO, CGVI(BGau4, QGau4))

# #set up the OuterSolarSystem
# OSS = GeometricProblems.OuterSolarSystem.lodeproblem(tstep=0.25,tspan=(0,2.5),n=3)
# OSS_pref = integrate(OSS, CGVI(BGau4, QGau4))



S = 4
relu3 = x->max(0,x) .^3
QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
OLnetwork = OneLayerNetwork_GML{Float64}(relu3,S)
NLOLCGVNI_Gml = NonLinear_OneLayer_GML(OLnetwork,QGau4,show_status = false,bias_interval = [-pi,pi],dict_amount = 400000)

#HarmonicOscillator
HO_NLOLsol,stages_values,state = integrate(HO_lode, NLOLCGVNI_Gml)
relative_maximum_error(HO_NLOLsol.q,HO_truth.q)
HO_NLOLsol.q

plot(collect(0.1:0.1:48), vec(transpose(stages_values[:, 2:end, 1]))[1:480], size=(700, 300))
plot!(collect(0.1:0.1:48), collect(HO_pref.q[:, 1])[2:481], label="Truth", linestyle=:dash, linecolor=:black)
scatter!(collect(1:4:50) .- 1, collect(HO_NLOLsol.q[:, 1]), label="Discrete solution")
    

plot(collect(1950.1:0.1:2000), vec(transpose(stages_values[:, 2:end, 1]))[end-500+1:end], size=(700, 300))
plot!(collect(1950.1:0.1:2000), collect(HO_pref.q[:, 1])[end-500+1:end], label="Truth", linestyle=:dash, linecolor=:black)
scatter!(collect(1952:4:2000), collect(HO_NLOLsol.q[:, 1])[end-12:end], label="Discrete solution")
relative_maximum_error(vec(transpose(stages_values[:, 2:end, 1])),collect(HO_pref.q[:,1])[2:end])

#DoublePendulum
DP_lode = GeometricProblems.DoublePendulum.lodeproblem(tstep=int_step,tspan=(0,int_timespan))
DP_NLOLsol = integrate(DP_lode, NLOLCGVNI_Gml)
DP_NLOLsol.q
# #CoupledHarmonicOscillator
# CHO_NLOLsol = integrate(CHO, NLOLCGVNI)
# relative_maximum_error(CHO_NLOLsol.q,CHO_pref.q)

# #OuterSolarSystem
# OSS_NLOLsol = integrate(OSS, NLOLCGVNI)
# relative_maximum_error(OSS_NLOLsol.q,OSS_pref.q)

