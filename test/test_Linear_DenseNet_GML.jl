using Pkg

# cd("IntegratorNN/GeometricIntegrators.jl")
cd("..")
cd("IntegratorNN")

Pkg.activate(".")

using GeometricIntegrators
using NonlinearIntegrators
using QuadratureRules
using CompactBasisFunctions
using GeometricProblems: HarmonicOscillator
using GeometricProblems


# Set up the Harmonic Oscillator problem
int_step = 0.5
int_timespan = 5.

HO_iode = GeometricProblems.HarmonicOscillator.lodeproblem(tspan = (0,int_timespan),tstep = int_step)
HO_pref = HarmonicOscillator.exact_solution(HarmonicOscillator.podeproblem(tspan = (0,int_timespan),tstep = int_step))

#set up the Coupled Harmonic Oscillator problem
CHO = GeometricProblems.CoupledHarmonicOscillator.lodeproblem(tstep=int_step,tspan=(0,int_timespan))

QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
BGau4 = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QGau4))
CHO_pref = integrate(CHO, CGVI(BGau4, QGau4))

#set up the OuterSolarSystem
OSS = GeometricProblems.OuterSolarSystem.lodeproblem(tstep=int_step,tspan=(0,int_timespan),n=3)
OSS_pref = integrate(OSS, CGVI(BGau4, QGau4))




S₁ = 4
S = 2
square(x) = x^2
OLnetwork = DenseNet_GML{Float64}(tanh,S₁,S)
QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
NL_DenseGML = Linear_DenseNet_GML(OLnetwork,QGau4)


#HarmonicOscillator
HO_NLOLsol = integrate(HO_iode, NL_DenseGML)
relative_maximum_error(HO_NLOLsol.q,HO_pref.q)


#CoupledHarmonicOscillator
CHO_NLOLsol = integrate(CHO, NL_DenseGML)
relative_maximum_error(CHO_NLOLsol.q,CHO_pref.q)

#OuterSolarSystem
OSS_NLOLsol = integrate(OSS, NL_DenseGML)
relative_maximum_error(OSS_NLOLsol.q,OSS_pref.q)