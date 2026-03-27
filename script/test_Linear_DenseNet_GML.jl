using GeometricIntegrators
# using NonlinearIntegrators
using QuadratureRules
using CompactBasisFunctions
using GeometricProblems: HarmonicOscillator
using GeometricProblems
using GeometricIntegratorsBase

# Set up the Harmonic Oscillator problem
int_step = 0.1
int_timespan = 0.2

HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(timespan = (0,int_timespan),timestep = int_step)
HO_pref = HarmonicOscillator.exact_solution(HarmonicOscillator.podeproblem(timespan = (0,int_timespan),timestep = int_step))
R = 8
QGau = QuadratureRules.LobattoLegendreQuadrature(R)
BGau = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QGau))
cgvi_sol = integrate(HO_lode,CGVI(BGau, QGau))
# #set up the Coupled Harmonic Oscillator problem
# CHO = GeometricProblems.CoupledHarmonicOscillator.lodeproblem(timestep=int_step,timespan=(0,int_timespan))

# QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
# BGau4 = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QGau4))
# CHO_pref = integrate(CHO, CGVI(BGau4, QGau4))

# #set up the OuterSolarSystem
# OSS = GeometricProblems.OuterSolarSystem.lodeproblem(timestep=int_step,timespan=(0,int_timespan),n=3)
# OSS_pref = integrate(OSS, CGVI(BGau4, QGau4))




S₁ = 3
S = 5
square(x) = x^2
OLnetwork = DenseNet_GML{Float64}(tanh,S₁,S)
QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
L_DenseGML = Linear_DenseNet_GML(OLnetwork,QGau4,training_epochs = 1000)


#HarmonicOscillator
HO_NLOLsol = integrate(HO_lode, L_DenseGML)
@show relative_maximum_error(HO_NLOLsol.q,HO_pref.q)


# #CoupledHarmonicOscillator
# CHO_NLOLsol = integrate(CHO, L_DenseGML)
# relative_maximum_error(CHO_NLOLsol.q,CHO_pref.q)

# #OuterSolarSystem
# OSS_NLOLsol = integrate(OSS, L_DenseGML)
# relative_maximum_error(OSS_NLOLsol.q,OSS_pref.q)

