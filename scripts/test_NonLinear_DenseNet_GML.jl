using Distributed
addprocs(5)

@everywhere begin
    using GeometricIntegrators 
    using NonlinearIntegrators
    using QuadratureRules
    using CompactBasisFunctions
    using GeometricProblems: HarmonicOscillator
    using GeometricProblems
    using Plots
    using SimpleSolvers 
    using JLD2
    using CairoMakie
    using Base.Threads

    reg_factor = 1e-5
    GeometricIntegratorsBase.default_options(method::NonLinear_DenseNet_GML) = (
        max_iterations = 10000,
        regularization_factor = reg_factor,
        linesearch=GeometricIntegratorsBase.default_linesearch(method), 
        f_abstol = 2eps(),
        x_suctol = 2eps()
    )
    
    int_timespan = 100.0
    R = 8
    Q = 2 * R
    tick_size = 22
    label_size = 22

end 


# f_suctol = eval(Meta.parse(ARGS[4]))
# f_abstol = eval(Meta.parse(ARGS[3]))
# max_iterations = parse(Int,ARGS[2])
# int_step = parse(Float64,ARGS[1])

# Set up the Harmonic Oscillator problem

pmap([0.05,0.1,0.2,0.5,1.0]) do int_step
    @info "Testing NonLinear_DenseNet_GML with step size h = $int_step"
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

    QGau = QuadratureRules.GaussLegendreQuadrature(R)
    NL_DenseGML = NonLinear_DenseNet_GML(Densenetwork,QGau,training_epochs = 50000,initial_guess_method=TrainingMethod())

    HO_Dense_sol,internal_values = integrate(HO_lode, NL_DenseGML)
    HO_qerror = relative_maximum_error(HO_Dense_sol.q,HO_pref.q)

    hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_Dense_sol.q[:]), collect(HO_Dense_sol.p[:]))]
    relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)


    record_results = Dict()
    record_results[("HO_sol_q")] = collect(HO_Dense_sol.q[:,1])
    record_results[("HO_sol_p")] = collect(HO_Dense_sol.p[:,1])
    record_results[("HO_qerror")] = HO_qerror
    record_results[("HO_hams_err")] = relative_hams_err
    save("NVI_HO_Dense_h$(int_step)R$(R)tanh_reg_factor=$(reg_factor).jld2",record_results)

    fig = Figure(size = (1000, 650))
    # Label(fig[0, 1], "Step Size h = $h", fontsize = 28, tellwidth = false)
    
    sol_q = collect(HO_Dense_sol.q[:, 1])
    total_length = length(sol_q)
    half_length = Int((length(sol_q) -1 ) ÷ 2)

    ax = Axis(fig[1, 1], xlabel="Time", ylabel = "q(t)",
    xticks = ([0,half_length,total_length], ["0","500","1000"]),yticklabelsize=tick_size, xticklabelsize=tick_size,xlabelsize=label_size, ylabelsize=label_size)
    lines!(ax, sol_q, )

    sol_p = collect(HO_Dense_sol.p[:, 1])
    ax = Axis(fig[2, 1], xlabel="Time", ylabel = "p(t)",
    xticks = ([0,half_length,total_length], ["0","500","1000"]),yticklabelsize=tick_size, xticklabelsize=tick_size,xlabelsize=label_size, ylabelsize=label_size)
    lines!(ax, sol_p, )

    hams_err = relative_hams_err
    ax = Axis(fig[3, 1], xlabel="Time", ylabel = "Relative Hamiltonian Error",
    xticks = ([0,half_length,total_length], ["0","500","1000"]),yticklabelsize=tick_size, xticklabelsize=tick_size,xlabelsize=label_size, ylabelsize=label_size)
    lines!(ax, hams_err)
    save("NVI_HO_Dense_h$(int_step)R$(R)tanh_reg_factor=$(reg_factor).pdf", fig)


end