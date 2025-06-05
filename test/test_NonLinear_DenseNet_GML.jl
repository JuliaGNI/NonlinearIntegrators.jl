using GeometricIntegrators 
using NonlinearIntegrators
using QuadratureRules
using CompactBasisFunctions
using GeometricProblems: HarmonicOscillator
using GeometricProblems


# Set up the Harmonic Oscillator problem
int_step = 0.6
int_timespan = 60.

HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(tspan = (0,int_timespan),tstep = int_step)
HO_pref = HarmonicOscillator.exact_solution(HarmonicOscillator.podeproblem(tspan = (0,int_timespan),tstep = int_step))

#set up the Coupled Harmonic Oscillator problem
CHO = GeometricProblems.CoupledHarmonicOscillator.lodeproblem(tstep=int_step,tspan=(0,int_timespan))
QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
BGau4 = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QGau4))
CHO_pref = integrate(CHO, CGVI(BGau4, QGau4))

#set up the OuterSolarSystem
OSS = GeometricProblems.OuterSolarSystem.lodeproblem(tstep=int_step,tspan=(0,int_timespan),n=3)
OSS_pref = integrate(OSS, CGVI(BGau4, QGau4))


#set up the DoublePendulum
DP = GeometricProblems.DoublePendulum.lodeproblem(tstep=int_step,tspan=(0,int_timespan))
DP_pref_lode = GeometricProblems.DoublePendulum.lodeproblem(tstep=0.1,tspan=(0,int_timespan))
DP_pref = integrate(DP_pref_lode, CGVI(BGau4, QGau4))


S₁ = 6
S = 4
square(x) = x^2
sigmoid(x) = 1 / (1 + exp(-x))

Densenetwork = DenseNet_GML{Float64}(tanh,S₁,S)
QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
NL_DenseGML = NonLinear_DenseNet_GML(Densenetwork,QGau4,training_epochs =1 )


#HarmonicOscillator
HO_Dense_sol = integrate(HO_lode, NL_DenseGML)
relative_maximum_error(HO_Dense_sol.q,HO_pref.q)

#CoupledHarmonicOscillator
CHO_Densesol = integrate(CHO, NL_DenseGML)
relative_maximum_error(CHO_Densesol.q,CHO_pref.q)

#DoublePendulum
DP_Densesol = integrate(DP, NL_DenseGML)
relative_maximum_error(DP_Densesol.q,DP_pref.q)

plot(0:0.1:60,collect(DP_pref.q[:,1]))
plot!(0:0.6:60,collect(DP_Densesol.q[:,1]))
savefig("./Dpp2.png")



#OuterSolarSystem
OSS_Densesol = integrate(OSS, NL_DenseGML)
relative_maximum_error(OSS_Densesol.q,OSS_pref.q)