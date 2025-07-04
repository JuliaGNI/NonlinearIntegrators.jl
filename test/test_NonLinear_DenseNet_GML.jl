using GeometricIntegrators 
using NonlinearIntegrators
using QuadratureRules
using CompactBasisFunctions
using GeometricProblems: HarmonicOscillator
using GeometricProblems
using Plots


# Set up the Harmonic Oscillator problem
int_step = 0.6
int_timespan = 60.0

HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(tspan = (0,int_timespan),tstep = int_step)
HO_pref = HarmonicOscillator.exact_solution(HarmonicOscillator.podeproblem(tspan = (0,int_timespan),tstep = int_step))
initial_hamiltonian = GeometricProblems.HarmonicOscillator.hamiltonian(0.0, HO_lode.ics.q, HO_lode.ics.p, HO_lode.parameters)



S₁ = 5
S = 5
square(x) = x^2
sigmoid(x) = 1 / (1 + exp(-x))

Densenetwork = DenseNet_GML{Float64}(tanh,S₁,S)
QGau4 = QuadratureRules.GaussLegendreQuadrature(6)
NL_DenseGML = NonLinear_DenseNet_GML(Densenetwork,QGau4,training_epochs =50000 )

HO_Dense_sol = integrate(HO_lode, NL_DenseGML)
relative_maximum_error(HO_Dense_sol.q,HO_pref.q)

plot(HO_pref.q[:,1])
plot!(HO_Dense_sol.q[:,1], label="DenseNet_GML", linestyle=:dash, linecolor=:black)

hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_Dense_sol.q[:]), collect(HO_Dense_sol.p[:]))]
relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)

p = plot(layout=@layout([a; b; c]), label="", size=(700, 700), plot_title="HarmonicOscillator,h = $(int_step)")

plot!(p[1], 0:int_step:int_timespan, collect(HO_Dense_sol.q[:, 1]), label="NVI_Dense", ylims=(-0.6, 0.6))
plot!(p[1], 0:int_step:int_timespan, collect(HO_pref.q[:, 1]), label="Analytic Solution", xaxis="time", yaxis="q₁")

plot!(p[2], 0:int_step:int_timespan, collect(HO_Dense_sol.p[:, 1]), label="NVI_Dense", ylims=(-0.6, 0.6))
plot!(p[2], 0:int_step:int_timespan, collect(HO_pref.p[:, 1]), label="Analytic Solution", xaxis="time", yaxis="p₁")

plot!(p[3], 0:int_step:int_timespan, relative_hams_err, label="NVI_Dense", xaxis="time", yaxis="Relative Hamiltonian error")
savefig(p, "result_figures/NVI_DenseS₁$(S₁)_S$(S)_tanh_harmonic_oscillator.png")
