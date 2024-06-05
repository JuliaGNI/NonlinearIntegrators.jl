using Pkg
# cd("IntegratorNN/GeometricIntegrators.jl")
# cd("..")
cd("IntegratorNN")

Pkg.activate(".")

using GeometricIntegrators
using NonlinearIntegrators
using QuadratureRules
using CompactBasisFunctions
using GeometricProblems: HarmonicOscillator
using GeometricProblems
using Test

# Set up the Harmonic Oscillator problem
int_step = 0.6
int_timespan = 3.

#set up the DoublePendulum
DP_lode = GeometricProblems.DoublePendulum.lodeproblem(tstep=int_step,tspan=(0,int_timespan))
DP_pref = integrate(DP_lode,Gauss(8))


S₁ = 5
S = 5
square(x) = x^2
sigmoid(x) = 1 / (1 + exp(-x))

Densenetwork = DenseNet_Lux{Float64}(tanh,S₁,S)
QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
NL_DenseLux = NonLinear_DenseNet_Lux(Densenetwork,QGau4,training_epochs = 50000)
DP_Densesol = integrate(DP_lode, NL_DenseLux)
@test relative_maximum_error(DP_Densesol.q,DP_pref.q) < 0.001