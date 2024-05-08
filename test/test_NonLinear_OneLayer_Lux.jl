using Pkg
# cd("IntegratorNN/GeometricIntegrators.jl")
cd("..")
cd("IntegratorNN")

Pkg.activate(".")

using GeometricIntegrators
using NonlinearIntegrators
using QuadratureRules
using CompactBasisFunctions
using GeometricProblems.HarmonicOscillator


# Set up the Harmonic Oscillator problem
int_step = 0.1
int_timespan = 1
HO_iode = HarmonicOscillator.iodeproblem(tspan = (0,int_timespan),tstep = int_step)
HO_pref = HarmonicOscillator.exact_solution(HarmonicOscillator.podeproblem(tspan = (0,int_timespan),tstep = int_step))

# Set up the NonLinearOneLayerBasis
S = 4
W = ones(S,1)
bias = ones(S,1)
square(x) = x^2
OLnetwork = OneLayerNetwork(square,S,W,bias)
QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
NLOLCGVNI = NonLinear_OneLayer_Lux(OLnetwork,QGau4)
NLOLsol = integrate(HO_iode, NLOLCGVNI) 

relative_maximum_error(NLOLsol.q,HO_pref.q) 