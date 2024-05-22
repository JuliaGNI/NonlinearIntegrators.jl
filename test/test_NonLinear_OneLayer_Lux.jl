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


# Set up the Harmonic Oscillator problem
int_step = 0.1
int_timespan = 1
HO_iode = GeometricProblems.HarmonicOscillator.iodeproblem(tspan = (0,int_timespan),tstep = int_step)
HO_pref = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(tspan = (0,int_timespan),tstep = int_step))

# Set up the NonLinearOneLayerBasis
S = 10

W = ones(S,1)
bias = ones(S,1)
square(x) = x^2
OLnetwork = OneLayerNetwork(tanh,S,W,bias)
QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
NLOLCGVNI = NonLinear_OneLayer_Lux(OLnetwork,QGau4)
NLOLsol = integrate(HO_iode, NLOLCGVNI) 

relative_maximum_error(NLOLsol.q,HO_pref.q) 


#set up the DoublePendulum
int_step = 0.1
DP = GeometricProblems.DoublePendulum.lodeproblem(tstep=int_step,tspan=(0,100*int_step))
DP_NLOLsol = integrate(DP, NLOLCGVNI) 
DP_pref = integrate(DP, Gauss(8)) 
relative_maximum_error(DP_NLOLsol.q,DP_pref.q) 

using Plots
hams = [GeometricProblems.DoublePendulum.hamiltonian(0,q,p,DP.parameters) for (q,p) in zip(collect(DP_NLOLsol.q[:]),collect(DP_NLOLsol.p[:]))]
plot(hams)