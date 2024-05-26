using Pkg
# cd("IntegratorNN/GeometricIntegrators.jl")
cd("..")
cd("IntegratorNN")

Pkg.activate(".")

using GeometricIntegrators
using NonlinearIntegrators
using QuadratureRules
using CompactBasisFunctions
using GeometricProblems

#Set up the Harmonic Oscillator problem
int_step = 0.5
int_timespan = 1
HO_iode = GeometricProblems.HarmonicOscillator.iodeproblem(tspan = (0,int_timespan),tstep = int_step)
HO_pref = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(tspan = (0,int_timespan),tstep = int_step))

#set up the DoublePendulum
DP_lode = GeometricProblems.DoublePendulum.lodeproblem(tstep=int_step,tspan=(0,5*int_step))
DP_pref = integrate(DP_lode, Gauss(8)) 


# Set up the NonLinearOneLayerBasis
S = 8
W = ones(S,1)
bias = ones(S,1)
square(x) = x^2
OLnetwork = OneLayerNetwork(sin,S,W,bias)
QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
NLOLCGVNI = NonLinear_OneLayer_Lux(OLnetwork,QGau4,training_epochs=50000)

NLOLsol = integrate(HO_iode, NLOLCGVNI) 
relative_maximum_error(NLOLsol.q,HO_pref.q) 

DP_NLOLsol = integrate(DP_lode, NLOLCGVNI) 
relative_maximum_error(DP_NLOLsol.q,DP_pref.q) 

NonlinearIntegrators.draw_comparison(DP_lode,GeometricProblems.DoublePendulum.hamiltonian,
                ["CGVN","Gauss(8)",],
                DP_NLOLsol,DP_pref)

