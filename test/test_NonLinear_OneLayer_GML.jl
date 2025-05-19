using GeometricIntegrators
using NonlinearIntegrators
using QuadratureRules
using CompactBasisFunctions
using GeometricProblems
# using BenchmarkTools
using Plots

# Set up the Harmonic Oscillator problem
int_step =0.5
int_timespan = 10.0
HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(tstep=int_step,tspan=(0,int_timespan))
initial_hamiltonian = GeometricProblems.HarmonicOscillator.hamiltonian(0.0, HO_lode.ics.q, HO_lode.ics.p, HO_lode.parameters)

HO_pref = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(tstep=int_step,tspan=(0,int_timespan)))

R = 4
Q = 2 * R
QGau4 = QuadratureRules.GaussLegendreQuadrature(R)
BGau4 = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QGau4))


S = 4
relu3 = x->max(0,x) .^3
QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
OLnetwork = OneLayerNetwork_GML{Float64}(relu3,S)
NLOLCGVNI_Gml = NonLinear_OneLayer_GML(OLnetwork,QGau4,show_status = false,bias_interval = [-pi,pi],dict_amount = 400000)

#HarmonicOscillator
HO_NLOLsol,internal_values = integrate(HO_lode, NLOLCGVNI_Gml)
@show relative_maximum_error(HO_NLOLsol.q,HO_pref.q)

# figure for q
plot(int_step/40:int_step/40:int_timespan,vcat(hcat(internal_values...)[2:end,:]...))
plot!(int_step/40:int_step/40:int_timespan, collect(HO_pref.q[:, 1])[2:end], label="Truth", linestyle=:dash, linecolor=:black)
scatter!(collect(0:int_step:int_timespan), collect(HO_NLOLsol.q[:, 1]), label="Discrete solution")

#### Figures in the paper
hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_NLOLsol.q[:]), collect(HO_NLOLsol.p[:]))]
relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)

p = plot(layout=@layout([a; b; c]), label="", size=(700, 700), plot_title="HarmonicOscillator,h = $(int_step)")

plot!(p[1], int_step/40:int_step/40:int_timespan,vcat(hcat(internal_values...)[2:end,:]...), label="S$(S)R$(R)Q$(Q)relu3", ylims=(-0.6, 0.6))
plot!(p[1], int_step/40:int_step/40:int_timespan, collect(HO_pref.q[:, 1])[2:end], label="Analytic Solution", xaxis="time", yaxis="q₁")

plot!(p[2], 0:int_step:int_timespan,collect(HO_NLOLsol.p[:, 1]), label="S$(S)R$(R)Q$(Q)relu3", ylims=(-0.6, 0.6))
plot!(p[2], 0:int_step/40:int_timespan, collect(HO_pref.p[:, 1]), label="Analytic Solution", xaxis="time", yaxis="p₁")

plot!(p[3], 0:int_step:int_timespan, relative_hams_err, label="S$(S)R$(R)Q$(Q)relu3", xaxis="time", yaxis="Relative Hamiltonian error")


#DoublePendulum
S = 8
R = 8
QGau4 = QuadratureRules.GaussLegendreQuadrature(R)
OLnetwork = OneLayerNetwork_GML{Float64}(tanh,S)
NLOLCGVNI_Gml = NonLinear_OneLayer_GML(OLnetwork,QGau4,show_status = false,bias_interval = [-pi,pi],dict_amount = 400000)

int_step =1.0
int_timespan = 10.0

DP_params = (
    l₁ = 1.0,
    l₂ = 1.0,
    m₁ = 1.0,
    m₂ = 1.0,
    g = 1.0,
    )

# DP_ics = (t = 0.0, q = [0.7853981633974483, 1.5707963267948966], p = [0.2776801836348979, 0.39269908169872414], v = [0.0, 0.39269908169872414])

DP_lode = GeometricProblems.DoublePendulum.lodeproblem(tstep = int_step, tspan = (0,int_timespan), parameters = DP_params)
initial_hamiltonian = GeometricProblems.DoublePendulum.hamiltonian(0.0, DP_lode.ics.q, DP_lode.ics.p, DP_lode.parameters)

DP_NLOLsol,DP_internal = integrate(DP_lode, NLOLCGVNI_Gml)
DP_hams = [GeometricProblems.DoublePendulum.hamiltonian(0, q, p, DP_lode.parameters) for (q, p) in zip(collect(DP_NLOLsol.q[:]), collect(DP_NLOLsol.p[:]))]

DP_ref1 = integrate(DP_lode, Gauss(8))
@show relative_maximum_error(DP_NLOLsol.q,DP_ref1.q)


DP_hams = [GeometricProblems.DoublePendulum.hamiltonian(0, q, p, DP_lode.parameters) for (q, p) in zip(collect(DP_NLOLsol.q[:]), collect(DP_NLOLsol.p[:]))]
pref_lode = GeometricProblems.DoublePendulum.lodeproblem(tstep=int_step/40,tspan=(0,int_timespan))
DP_pref= integrate(pref_lode, Gauss(8))

DP_relative_hams_err = abs.((DP_hams .- initial_hamiltonian) / initial_hamiltonian)


DP_internal_q1 = Array{Vector}(undef,Int(int_timespan/int_step))
DP_internal_q2 = Array{Vector}(undef,Int(int_timespan/int_step))

for i in 1:Int(int_timespan/int_step)
    DP_internal_q1[i] = DP_internal[i][:,1]
    DP_internal_q2[i] = DP_internal[i][:,2]
end

# Figures for the paper
p = plot(layout=@layout([a b; c d; e]), label="", size=(700, 700), plot_title="S$(S)R$(R)Q$(Q)relu3")# d;e

plot!(p[1], int_step/40:int_step/40:int_timespan, vcat(hcat(DP_internal_q1...)[2:end,:]...), label="S$(S)R$(R)Q$(Q)relu3", xaxis="time", yaxis="q₁")
plot!(p[1], 0:int_step/40:int_timespan, collect(DP_pref.q[:, 1]), label="Reference Solution", ylims=(-2, 2))

plot!(p[2], int_step/40:int_step/40:int_timespan, vcat(hcat(DP_internal_q2...)[2:end,:]...), label="S$(S)R$(R)Q$(Q)relu3", xaxis="time", yaxis="q₂")
plot!(p[2], 0:int_step/40:int_timespan, collect(DP_pref.q[:, 2]), label="Reference Solution", ylims=(-2, 2))

plot!(p[3], 0:int_step:int_timespan, collect(DP_NLOLsol.p[:, 1]), label="S$(S)R$(R)Q$(Q)relu3", xaxis="time", yaxis="p₁")
plot!(p[3], 0:int_step/40:int_timespan, collect(DP_pref.p[:, 1]), label="Reference Solution", ylims=(-3, 3))

plot!(p[4], 0:int_step:int_timespan, collect(DP_NLOLsol.p[:, 2]), label="S$(S)R$(R)Q$(Q)relu3", xaxis="time", yaxis="p₂")
plot!(p[4], 0:int_step/40:int_timespan, collect(DP_pref.p[:, 2]), label="Reference Solution", ylims=(-3, 3))

plot!(p[5], 0:int_step:int_timespan, DP_relative_hams_err, label="S$(S)R$(R)Q$(Q)relu3", xaxis="time", yaxis="Relative Hamiltonian error")
