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
using Test
# Set up the Harmonic Oscillator problem
int_step = 1.0
int_timespan = 5

QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
BGau4 = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QGau4))

# Set up the DoublePendulum problem
DP_lode = GeometricProblems.DoublePendulum.lodeproblem(tstep=int_step,tspan=(0,int_timespan))
DP_pref = integrate(DP_lode,CGVI(BGau4, QGau4))
initial_hamiltonian = GeometricProblems.DoublePendulum.hamiltonian(0.0,DP_lode.ics.q,DP_lode.ics.p,DP_lode.parameters)
# Set up the NonLinearOneLayerBasis
S = 16
D = 2
square(x) = x^2
OLnetwork = NonlinearIntegrators.OneLayerVectorValueNet_GML{Float64}(S,cos,2)
NLOLCGVNI = NonlinearIntegrators.NonLinear_OneLayer_VectorValue_GML(OLnetwork,QGau4,GeometricProblems.DoublePendulum,
                                                problem_initial_hamitltonian =initial_hamiltonian, use_hamiltonian_loss=true,training_epochs=50000)

DP_NLOLsol = integrate(DP_lode, NLOLCGVNI) 
@test relative_maximum_error(DP_NLOLsol.q,DP_pref.q) < 0.00001
