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
int_timespan = 1.0

QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
BGau4 = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QGau4))

# Set up the DoublePendulum problem
DP_lode = GeometricProblems.DoublePendulum.lodeproblem(tstep=int_step,tspan=(0,int_timespan))
# DP_pref = integrate(DP_lode,CGVI(BGau4, QGau4))

# Set up the NonLinearOneLayerBasis
S = 4
D = 2
square(x) = x^2
OLnetwork = NonlinearIntegrators.OneLayerVectorValueNet_Lux{Float64}(S,tanh,2)
NLOLCGVNI = NonlinearIntegrators.NonLinear_OneLayer_VectorValue_Lux(OLnetwork,QGau4,training_epochs=50000)

DP_NLOLsol = integrate(DP_lode, NLOLCGVNI) 
relative_maximum_error(DP_NLOLsol.q,DP_pref.q) 