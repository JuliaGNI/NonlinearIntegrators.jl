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
int_step = 1.0
int_timespan = 10.0
HO_iode = GeometricProblems.HarmonicOscillator.iodeproblem(tspan = (0,int_timespan),tstep = int_step)
HO_pref = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(tspan = (0,int_timespan),tstep = int_step))


QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
BGau4 = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QGau4))

# Set up the DoublePendulum problem
DP_lode = GeometricProblems.DoublePendulum.lodeproblem(tstep=int_step,tspan=(0,int_timespan))
DP_pref = integrate(DP_lode,Gauss(8))

# Set up the NonLinearOneLayerBasis
S = 10
square(x) = x^2
OLnetwork = NonlinearIntegrators.OneLayerNetwork_Lux{Float64}(S,cos)
NLOLCGVNI = NonlinearIntegrators.NonLinear_OneLayer_Lux(OLnetwork,QGau4)

NLOLsol = integrate(HO_iode, NLOLCGVNI) 
relative_maximum_error(NLOLsol.q,HO_pref.q) 

@time DP_NLOLsol = integrate(DP_lode, NLOLCGVNI) 
relative_maximum_error(DP_NLOLsol.q,DP_pref.q) 

NonlinearIntegrators.draw_comparison(DP_lode,GeometricProblems.DoublePendulum.hamiltonian,
                ["CGVN","Gauss(8)",],
                DP_NLOLsol,DP_pref)

