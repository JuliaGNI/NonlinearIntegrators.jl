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
int_timespan = 5
HO_iode = HarmonicOscillator.iodeproblem(tspan = (0,int_timespan),tstep = int_step)
HO_pref = HarmonicOscillator.exact_solution(HarmonicOscillator.podeproblem(tspan = (0,int_timespan),tstep = int_step))

# Set up the NonLinearOneLayerBasis
S = 4
square(x) = x^2
OLnetwork = NonlinearIntegrators.OneLayerNetwork_Lux{Float64}(S,square)
QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
NLOLCGVNI = NonlinearIntegrators.NonLinear_OneLayer_Lux(OLnetwork,QGau4)
NLOLsol = integrate(HO_iode, NLOLCGVNI) 
relative_maximum_error(NLOLsol.q,HO_pref.q) 

DP_NLOLsol = integrate(DP_lode, NLOLCGVNI) 
relative_maximum_error(DP_NLOLsol.q,DP_pref.q) 

NonlinearIntegrators.draw_comparison(DP_lode,GeometricProblems.DoublePendulum.hamiltonian,
                ["CGVN","Gauss(8)",],
                DP_NLOLsol,DP_pref)

