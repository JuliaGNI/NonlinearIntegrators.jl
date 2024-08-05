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
int_step = 0.0625
int_timespan = 100*int_step
# HO_iode = GeometricProblems.HarmonicOscillator.iodeproblem(tspan = (0,int_timespan),tstep = int_step)
# HO_pref = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(tspan = (0,int_timespan),tstep = int_step))
# initial_hamiltonian = GeometricProblems.HarmonicOscillator.hamiltonian(0.0,HO_iode.ics.q,HO_iode.ics.p,HO_iode.parameters)

R = 4
Q = 2*R
QGau4 = QuadratureRules.GaussLegendreQuadrature(R)
BGau4 = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QGau4))

# Set up the DoublePendulum problem
DP_lode = GeometricProblems.DoublePendulum.lodeproblem(tstep=int_step/20,tspan=(0,int_timespan))
@time DP_pref = integrate(DP_lode,Gauss(8))
# DP_CGVI = integrate(DP_lode,CGVI(BGau4,QGau4))
initial_hamiltonian = GeometricProblems.DoublePendulum.hamiltonian(0.0,DP_lode.ics.q,DP_lode.ics.p,DP_lode.parameters)

# Set up the NonLinearOneLayerBasis

S =6
square(x) = x^2
relu2(x) = max(0,x)^2
OLnetwork = NonlinearIntegrators.OneLayerNetwork_Lux{Float64}(S,tanh,2)
NLOLCGVNI = NonlinearIntegrators.NonLinear_OneLayer_Lux(OLnetwork,QGau4,GeometricProblems.HarmonicOscillator,
problem_initial_hamitltonian = initial_hamiltonian, use_hamiltonian_loss=false,show_status=true,initial_guess_methods = "OGA1d")#OGA1d
print(" R = $R h =$(int_step) S = $(S)\n")
@time NLOLsol = integrate(DP_lode, NLOLCGVNI) 
@show relative_maximum_error(NLOLsol.q,DP_pref.q) 

NLOLsol_hams = [GeometricProblems.DoublePendulum.hamiltonian(0,q,p,DP_lode.parameters) for (q,p) in zip(collect(NLOLsol.q[:]),collect(NLOLsol.p[:]))]
relative_hams_err = abs.((NLOLsol_hams .- initial_hamiltonian)/initial_hamiltonian)

@show maximum(relative_hams_err)
# @time DP_NLOLsol = integrate(DP_lode, NLOLCGVNI) 
# @show relative_maximum_error(DP_NLOLsol.q,DP_pref.q)
# @show relative_maximum_error(DP_CGVI.q,DP_pref.q)

draw_comparison_DP("Double Pendulum,h = $(int_step)","Truth",DP_pref,relative_hams_err,
["S$(S)R$(R)Q$(Q)tanh"],
NLOLsol,
h=int_step,plotrange = int_timespan,
save_path = "2507h$(int_step)_tanh_S$(S)R$(R)_DP_hamiltonian.pdf")

