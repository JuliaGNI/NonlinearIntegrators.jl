using Pkg

# cd("IntegratorNN/GeometricIntegrators.jl")
# cd("..")
# cd("IntegratorNN")

Pkg.activate(".")

using GeometricIntegrators
using NonlinearIntegrators
using QuadratureRules
using CompactBasisFunctions
using GeometricProblems


# Set up the Harmonic Oscillator problem
int_step = 0.5
int_timespan = 5.0
HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(tspan = (0,int_timespan),tstep = int_step)
HO_pref = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(tspan = (0,int_timespan),tstep = int_step))



QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
BGau4 = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QGau4))
# #set up the Coupled Harmonic Oscillator problem
# CHO = GeometricProblems.CoupledHarmonicOscillator.lodeproblem(tstep=0.5,tspan=(0,20))
# CHO_pref = integrate(CHO, CGVI(BGau4, QGau4))

# #set up the OuterSolarSystem
# OSS = GeometricProblems.OuterSolarSystem.lodeproblem(tstep=0.25,tspan=(0,2.5),n=3)
# OSS_pref = integrate(OSS, CGVI(BGau4, QGau4))



S = 6
relu2 = x->max(0,x) .^3
QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
OLnetwork = OneLayerNetwork_GML{Float64}(relu2,S)
NLOLCGVNI_Gml = NonLinear_OneLayer_GML(OLnetwork,QGau4,show_status = true,bias_interval = [-pi,pi],dict_amount = 100000)

#HarmonicOscillator
HO_NLOLsol = integrate(HO_lode, NLOLCGVNI_Gml)
relative_maximum_error(HO_NLOLsol.q,HO_pref.q)
HO_NLOLsol.q
# DP_lode = GeometricProblems.DoublePendulum.lodeproblem(tstep=int_step,tspan=(0,int_timespan))
# DP_NLOLsol = integrate(DP_lode, NLOLCGVNI)

# #CoupledHarmonicOscillator
# CHO_NLOLsol = integrate(CHO, NLOLCGVNI)
# relative_maximum_error(CHO_NLOLsol.q,CHO_pref.q)

# #OuterSolarSystem
# OSS_NLOLsol = integrate(OSS, NLOLCGVNI)
# relative_maximum_error(OSS_NLOLsol.q,OSS_pref.q)

