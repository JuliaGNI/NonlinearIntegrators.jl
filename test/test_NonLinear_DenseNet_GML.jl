using GeometricIntegrators 
using NonlinearIntegrators
using QuadratureRules
using CompactBasisFunctions
using GeometricProblems: HarmonicOscillator
using GeometricProblems
using Plots
using SimpleSolvers 
using JLD2

GeometricIntegrators.Integrators.default_linesearch(method::NonLinear_DenseNet_GML) =SimpleSolvers.Backtracking()
# GeometricIntegrators.Integrators.default_linesearch(method::PR_Integrator) =SimpleSolvers.Quadratic2()


# f_suctol = 2eps()
# f_abstol = 2eps()
# max_iterations = 10000
# h_step =1.0

f_suctol = eval(Meta.parse(ARGS[4]))
f_abstol = eval(Meta.parse(ARGS[3]))
max_iterations = parse(Int,ARGS[2])
h_step = parse(Float64,ARGS[1])

GeometricIntegrators.Integrators.default_options(method::NonLinear_DenseNet_GML) = (
    # f_abstol = 8eps(),
    # f_suctol = 2eps(),
    # f_abstol = parse(Float64,eval(ARGS[4])),
    f_suctol = f_suctol,
    f_abstol = f_abstol,
    max_iterations = max_iterations,
    linesearch=GeometricIntegrators.Integrators.default_linesearch(method),
)


# Set up the Harmonic Oscillator problem
int_step = h_step
int_timespan = 100.0

HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(timespan = (0,int_timespan),timestep = int_step)
HO_pref = HarmonicOscillator.exact_solution(HarmonicOscillator.podeproblem(timespan = (0,int_timespan),timestep = int_step))
HO_pref2 = HarmonicOscillator.exact_solution(HarmonicOscillator.podeproblem(timespan = (0,int_timespan),timestep = int_step/40))
initial_hamiltonian = GeometricProblems.HarmonicOscillator.hamiltonian(0.0, HO_lode.ics.q, HO_lode.ics.p, HO_lode.parameters)

S₁ = 5
S = 5
square(x) = x^2
sigmoid(x) = 1 / (1 + exp(-x))
relu3 = x -> max(0, x)^3
Densenetwork = DenseNet_GML{Float64}(tanh,S₁,S)
# R = 4
for R in [12,16,24]#
    Q = 2 * R
    record_results = Dict()

    QGau = QuadratureRules.GaussLegendreQuadrature(R)
    NL_DenseGML = NonLinear_DenseNet_GML(Densenetwork,QGau,training_epochs =50000 )

    HO_Dense_sol,internal_values = integrate(HO_lode, NL_DenseGML)
    HO_qerror = relative_maximum_error(HO_Dense_sol.q,HO_pref.q)

    hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_Dense_sol.q[:]), collect(HO_Dense_sol.p[:]))]
    relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)

    record_results[("HO_sol_q")] = collect(HO_Dense_sol.q[:,1])
    record_results[("HO_sol_p")] = collect(HO_Dense_sol.p[:,1])
    record_results[("HO_qerror")] = HO_qerror
    record_results[("HO_hams_err")] = relative_hams_err

    p = plot(layout=@layout([a; b; c]), label="", size=(700, 700), plot_title="HarmonicOscillator,h = $(int_step)")

    plot!(p[1], int_step/40:int_step/40:int_timespan, vcat(hcat(internal_values...)[2:end,:]...), label="R$(R)Q$(Q)tanh", ylims=(-0.6, 0.6))
    plot!(p[1], int_step/40:int_step/40:int_timespan, collect(HO_pref2.q[:, 1])[2:end], label="Analytic Solution", xaxis="time", yaxis="q₁")

    plot!(p[2], 0:int_step:int_timespan, collect(HO_Dense_sol.p[:, 1]), label="R$(R)Q$(Q)tanh", ylims=(-0.6, 0.6))
    plot!(p[2], 0:int_step:int_timespan, collect(HO_pref.p[:, 1]), label="Analytic Solution", xaxis="time", yaxis="p₁")

    plot!(p[3], 0:int_step:int_timespan, relative_hams_err, label="R$(R)Q$(Q)tanh", xaxis="time", yaxis="Relative Hamiltonian error")

    # filename2 = @sprintf(
    #     "parallel_result_figures/Backtracking2_R%d_h%.2f_iter%d_fabs%.2e_fsuc%.2e_TT%d.pdf",
        # R, h_step, max_iterations, f_abstol, f_suctol,TT)
    # savefig(p,"result_figures/NVI_DenseS₁$(S₁)_S$(S)_fabs$(f_abstol)_fsuc$(f_suctol)_iter$(iterations)_h$(int_step)_tspan$(int_timespan)_tanh_harmonic_oscillator.png")
    savefig(p,"parallel_result_figures/NVI_Densefabs$(f_abstol)_fsuc$(f_suctol)_iter$(max_iterations)_h$(h_step)_R$(R)tanh_harmonic_oscillator.pdf")

    save("parallel_result_figures/NVI_Densefabs$(f_abstol)_fsuc$(f_suctol)_iter$(max_iterations)_h$(h_step)_R$(R)tanh_harmonic_oscillator.jld2",record_results)
end