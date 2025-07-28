using Symbolics
using CompactBasisFunctions
using NonlinearIntegrators
using QuadratureRules

using GeometricProblems
using GeometricIntegrators
using SimpleSolvers
using CairoMakie
using Printf
using JLD2

# function PR_plot_1d(PR_sol, internal_sol, pref,relative_ham_err, h_step, TT, title_name)
#     p = plot(layout=@layout([a; b; c]), label="", size=(700, 700)) 

#     plot!(p[1],collect(0:h_step/40:TT),collect(pref.q[:,1]), label="Reference Solution",xaxis="time", yaxis="q₁")
#     plot!(p[1],h_step/40:h_step/40:TT, vcat(hcat(internal_sol...)[2:end,:]...),label = "VISE Solution")
#     scatter!(p[1],collect(0:h_step:TT),collect(PR_sol.q[:,1]), label="VISE Discrete Solution")

#     plot!(p[2],collect(0:h_step/40:TT),collect(pref.p[:,1]), label="Reference Solution",xaxis="time", yaxis="p₁")
#     scatter!(p[2],collect(0:h_step:TT),collect(PR_sol.p[:,1]), label="VISE Discrete Solution")

#     plot!(p[3], 0:h_step:TT, relative_ham_err, label = "VISE Solution", xaxis="time", yaxis="Relative Hamiltonian error")
#     savefig("result_figures/$(title_name).pdf")
# end

# function PR_plot_2d(PR_sol, internal_sol, pref,relative_ham_err, h_step, TT, title_name)
#     internal_q1 = Array{Vector}(undef,Int(TT/h_step))
#     internal_q2 = Array{Vector}(undef,Int(TT/h_step))

#     for i in 1:Int(TT/h_step)
#         internal_q1[i] = internal_sol[i][:,1]
#         internal_q2[i] = internal_sol[i][:,2]
#     end

#     # Figures for the paper
#     p = plot(layout=@layout([a b; c d; e]), label="", size=(700, 700))# d;e

#     plot!(p[1], h_step/40:h_step/40:TT, vcat(hcat(internal_q1...)[2:end,:]...), label="SINDy Solution", xaxis="time", yaxis="q₁")
#     plot!(p[1], 0:h_step/40:TT, collect(pref.q[:, 1]), label="Reference Solution")
#     scatter!(p[1],collect(0:h_step:TT),collect(PR_sol.q[:,1]), label="SINDy Discrete Solution")

#     plot!(p[2], h_step/40:h_step/40:TT, vcat(hcat(internal_q2...)[2:end,:]...), label="SINDy Solution", xaxis="time", yaxis="q₂")
#     plot!(p[2], 0:h_step/40:TT, collect(pref.q[:, 2]), label="Reference Solution")
#     scatter!(p[2],collect(0:h_step:TT),collect(PR_sol.q[:,2]), label="SINDy Discrete Solution")

#     plot!(p[3], 0:h_step:TT, collect(PR_sol.p[:, 1]), label="SINDy Solution", xaxis="time", yaxis="p₁")
#     plot!(p[3], 0:h_step/40:TT, collect(pref.p[:, 1]), label="Reference Solution")

#     plot!(p[4], 0:h_step:TT, collect(PR_sol.p[:, 2]), label="SINDy Solution", xaxis="time", yaxis="p₂")
#     plot!(p[4], 0:h_step/40:TT, collect(pref.p[:, 2]), label="Reference Solution")

#     plot!(p[5], 0:h_step:TT, relative_ham_err, label="SINDy Solution", xaxis="time", yaxis="Relative Hamiltonian error")
#     savefig(p, "result_figures/$(title_name).pdf")
# end

t0 = time()

# GeometricIntegrators.Integrators.default_linesearch(method::PR_Integrator) =SimpleSolvers.Backtracking()
GeometricIntegrators.Integrators.default_linesearch(method::PR_Integrator) =SimpleSolvers.Quadratic2()

GeometricIntegrators.Integrators.default_options(method::PR_Integrator) = (
    # f_abstol = 8eps(),
    # f_suctol = 2eps(),
    # f_abstol = parse(Float64,eval(ARGS[4])),
    f_suctol = eval(Meta.parse(ARGS[4])),
    f_abstol = eval(Meta.parse(ARGS[3])),
    max_iterations = parse(Int,ARGS[2]),
    linesearch=GeometricIntegrators.Integrators.default_linesearch(method),
)

# R = parse(Int,ARGS[1])

for R in [8,16,32] # 4,
    h_step = parse(Float64,ARGS[1])
    record_results = Dict()
    # Harmonic Oscillator
    begin 
        @variables W[1:3] ttt
        q_expr = W[1] *sin(W[2]* ttt + W[3])

        PRB = PR_Basis{Float64}([q_expr], [W], ttt,1)
        TT = 200.0
        # h_step = 5.0
        # h_step = ARGS[2]
        HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(timespan = (0,TT),timestep = h_step)

        initial_hamiltonian = GeometricProblems.HarmonicOscillator.hamiltonian(0.0, HO_lode.ics.q, HO_lode.ics.p, HO_lode.parameters)
        HO_truth = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(timespan = (0,TT),timestep = h_step))
        HO_plot = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(timespan = (0,TT),timestep = h_step/40))

        # R = 32
        QGau4 = QuadratureRules.GaussLegendreQuadrature(R)
        PR_Int = PR_Integrator(PRB, QGau4,[[-0.5000433352162222,0.705350078478666,-1.5678140333370576]]) # Pass the init W into the integrator instead of basis                                               
        # PR_Int = PR_Integrator(PRB, QGau4,[[-0.500,sqrt(0.5),-pi/2]])                           

        println("Start to run Harmonic Oscillator Problem with PR_Integrator! R = $(R), h = $(h_step)")
        t1 = time()
        PR_sol,internal_sol,x_list = integrate(HO_lode, PR_Int)
        t2 = time()
        hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(PR_sol.q[:]), collect(PR_sol.p[:]))]
        relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)
        println("\n NISE:")
        @show relative_maximum_error(PR_sol.q,HO_truth.q)
        @show maximum(relative_hams_err)

        # pref = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(timespan = (0,TT),timestep = h_step/40))
        # PR_plot_1d(PR_sol, internal_sol, HO_plot, relative_hams_err, h_step, TT, "HarmonicOscillator,h$(h_step)_T$(TT)_R$(R)")
        # println("Finish integrating Harmonic Oscillator Problem with PR_Integrator!, Figure Saved!")
        
        HO_imp_sol = integrate(HO_lode, ImplicitMidpoint())
        HO_imp_ham = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_imp_sol.q[:]), collect(HO_imp_sol.p[:]))]
        HO_relative_imp_ham_err = abs.((HO_imp_ham .- initial_hamiltonian) / initial_hamiltonian)
        println("\n ImplicitMidpoint:")
        @show relative_maximum_error(HO_imp_sol.q,HO_truth.q)
        @show maximum(HO_relative_imp_ham_err)


        QGau = GaussLegendreQuadrature(R)
        BGau = Lagrange(QuadratureRules.nodes(QGau))
        HO_cgvi_sol = integrate(HO_lode, CGVI(BGau, QGau)) 
        HO_cgvi_ham = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_cgvi_sol.q[:]), collect(HO_cgvi_sol.p[:]))]
        HO_relative_cgvi_ham_err = abs.((HO_cgvi_ham .- initial_hamiltonian) / initial_hamiltonian)

        println("\n CGVI R = $(R):")
        @show relative_maximum_error(HO_cgvi_sol.q,HO_truth.q)
        @show maximum(HO_relative_cgvi_ham_err)

        QGau3 = QuadratureRules.GaussLegendreQuadrature(3)
        BGau3 = Lagrange(QuadratureRules.nodes(QGau3))
        HO_cgvi_sol3 = integrate(HO_lode, CGVI(BGau3, QGau3))
        HO_cgvi_ham3 = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_cgvi_sol3.q[:]), collect(HO_cgvi_sol3.p[:]))]
        HO_relative_cgvi_ham_err3 = abs.((HO_cgvi_ham3 .- initial_hamiltonian) / initial_hamiltonian)
        println("\n CGVI with 3 nodes:")
        @show relative_maximum_error(HO_cgvi_sol3.q,HO_truth.q)
        @show maximum(HO_relative_cgvi_ham_err3)
        t3 = time()
        println("Time taken for VISE: $(t2 - t1) seconds")
        println("Time taken for Reference: $(t3 - t2) seconds")
        # # try to find a error on the same scale with NISE
        # balance_h = 0.1
        # HO_lode2 = GeometricProblems.HarmonicOscillator.lodeproblem(timespan = (0,TT),timestep = balance_h)
        # imp_sol2 = integrate(HO_lode2, ImplicitMidpoint())
        # HO_truth2 = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(timespan = (0,TT),timestep = balance_h))
        # imp_ham2 = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode2.parameters) for (q, p) in zip(collect(imp_sol2.q[:]), collect(imp_sol2.p[:]))]
        # relative_imp_ham_err2 = abs.((imp_ham2 .- initial_hamiltonian) / initial_hamiltonian)
        # println("\n ImplicitMidpoint with balance_h:")
        # @show relative_maximum_error(imp_sol2.q,HO_truth2.q)
        # @show maximum(relative_imp_ham_err2)

        # balance_h = 15
        # HO_lode2 = GeometricProblems.HarmonicOscillator.lodeproblem(timespan = (0,TT),timestep = balance_h)
        # HO_truth2 = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(timespan = (0,TT),timestep = balance_h))

        # QGau = GaussLegendreQuadrature(8)
        # BGau = Lagrange(QuadratureRules.nodes(QGau))
        # cgvi_sol2 = integrate(HO_lode2, CGVI(BGau, QGau)) 
        # cgvi_ham2 = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(cgvi_sol2.q[:]), collect(cgvi_sol2.p[:]))]
        # relative_cgvi_ham_err2 = abs.((cgvi_ham2 .- initial_hamiltonian) / initial_hamiltonian)
        # println("\n CGVI with balance_h:")
        # @show relative_maximum_error(cgvi_sol2.q,HO_truth2.q)
        # @show maximum(relative_cgvi_ham_err2)

        record_results[("HO_PR_sol_q")] = collect(PR_sol.q[:,1])
        record_results[("HO_PR_sol_p")] = collect(PR_sol.p[:,1])
        record_results[("HO_internal_sol")] = internal_sol
        record_results[("HO_x_list")] = x_list
        record_results[("HO_qerror")] = relative_maximum_error(PR_sol.q,HO_truth.q)
        record_results[("HO_hams_err")] = maximum(relative_hams_err)

        record_results[("HO_imp_hams_err")] = maximum(HO_relative_imp_ham_err)
        record_results[("HO_imp_qerror")] = relative_maximum_error(HO_imp_sol.q,HO_truth.q)

        record_results[("HO_cgvi_hams_err")] = maximum(HO_relative_cgvi_ham_err)
        record_results[("HO_cgvi_qerror")] = relative_maximum_error(HO_cgvi_sol.q,HO_truth.q)

        record_results[("HO_cgvi_hams_err3")] = maximum(HO_relative_cgvi_ham_err3)
        record_results[("HO_cgvi_qerror3")] = relative_maximum_error(HO_cgvi_sol3.q,HO_truth.q)

    end

    #### Pendulum
    begin
    #     TT = 150.0
    #     h_step = 1.0
        pendulum_lode = GeometricProblems.Pendulum.lodeproblem(timespan = (0,TT),timestep = h_step)
        pd_ref_sol = integrate(pendulum_lode, Gauss(8))
        initial_hamiltonian = GeometricProblems.Pendulum.hamiltonian(0.0, pendulum_lode.ics.q, pendulum_lode.ics.p, pendulum_lode.parameters)
        @show initial_hamiltonian

        # f_try(x₁) = cos((x₁ * 0.45644) - 1.1466) * 1.1931
        @variables W[1:3] ttt
        q_expr = W[1] *cos(W[2]* ttt + W[3])
        PRB = PR_Basis{Float64}([q_expr], [W], ttt,1)
        # R = 4
        println("Start to run Pendulum Problem with PR_Integrator! h = $(h_step), R= $(R)")

        QGau = QuadratureRules.GaussLegendreQuadrature(R)
        PR_Int = PR_Integrator(PRB, QGau,[[1.1931,0.45644,-1.1466]])

        pendulum_PR_sol,pendulum_internal_sol,pendulum_x_list = integrate(pendulum_lode, PR_Int)
        @show relative_maximum_error(pendulum_PR_sol.q,pd_ref_sol.q)

        pendulum_hams = [GeometricProblems.Pendulum.hamiltonian(0.0, q, p, pendulum_lode.parameters) for (q, p) in zip(collect(pendulum_PR_sol.q[:]), collect(pendulum_PR_sol.p[:]))]
        pendulum_relative_hams_err = abs.((pendulum_hams .- initial_hamiltonian) / initial_hamiltonian)
        @show maximum(pendulum_relative_hams_err)

        pendulum_plot = GeometricProblems.Pendulum.lodeproblem(timespan = (0,TT),timestep = h_step/40)
        pendulum_sol_plot = integrate(pendulum_plot, Gauss(8))
        # PR_plot_1d(PR_sol, internal_sol, sol_plot, relative_hams_err, h_step, TT, "Pendulum,h$(h_step)_T$(TT)_R$(R)")
        # println("Finish integrating Pendulum Problem with PR_Integrator!, Figure Saved!")

        pd_imp_sol = integrate(pendulum_lode, ImplicitMidpoint())
        pd_imp_ham = [GeometricProblems.Pendulum.hamiltonian(0, q, p, pendulum_lode.parameters) for (q, p) in zip(collect(pd_imp_sol.q[:]), collect(pd_imp_sol.p[:]))]
        pd_relative_imp_ham_err = abs.((pd_imp_ham .- initial_hamiltonian) / initial_hamiltonian)
        println("\n ImplicitMidpoint:")
        @show relative_maximum_error(pd_imp_sol.q,pd_ref_sol.q)
        @show maximum(pd_relative_imp_ham_err)

        QGau = GaussLegendreQuadrature(R)
        BGau = Lagrange(QuadratureRules.nodes(QGau))
        pd_cgvi_sol = integrate(pendulum_lode, CGVI(BGau, QGau))
        pd_cgvi_ham = [GeometricProblems.Pendulum.hamiltonian(0, q, p, pendulum_lode.parameters) for (q, p) in zip(collect(pd_cgvi_sol.q[:]), collect(pd_cgvi_sol.p[:]))]
        pd_relative_cgvi_ham_err = abs.((pd_cgvi_ham .- initial_hamiltonian) / initial_hamiltonian)
        println("\n CGVI:")
        @show relative_maximum_error(pd_cgvi_sol.q,pd_ref_sol.q)
        @show maximum(pd_relative_cgvi_ham_err)
        #Plotting the results

        QGau3 = QuadratureRules.GaussLegendreQuadrature(3)
        BGau3 = Lagrange(QuadratureRules.nodes(QGau3))
        pd_cgvi_sol3 = integrate(pendulum_lode, CGVI(BGau3, QGau3))
        pd_cgvi_ham3 = [GeometricProblems.Pendulum.hamiltonian(0, q, p, pendulum_lode.parameters) for (q, p) in zip(collect(pd_cgvi_sol3.q[:]), collect(pd_cgvi_sol3.p[:]))]
        pd_relative_cgvi_ham_err3 = abs.((pd_cgvi_ham3 .- initial_hamiltonian) / initial_hamiltonian)
        println("\n CGVI with 3 nodes:")
        @show relative_maximum_error(pd_cgvi_sol3.q,pd_ref_sol.q)
        @show maximum(pd_relative_cgvi_ham_err3)

        record_results[("Pendulum_PR_sol_q")] = collect(pendulum_PR_sol.q[:,1])
        record_results[("Pendulum_PR_sol_p")] = collect(pendulum_PR_sol.p[:,1])
        record_results[("Pendulum_internal_sol")] = pendulum_internal_sol
        record_results[("Pendulum_x_list")] = pendulum_x_list
        record_results[("Pendulum_qerror")] = relative_maximum_error(pendulum_PR_sol.q,pd_ref_sol.q)
        record_results[("Pendulum_hams_err")] = maximum(pendulum_relative_hams_err) 

        record_results[("Pendulum_imp_hams_err")] = maximum(pd_relative_imp_ham_err)
        record_results[("Pendulum_imp_qerror")] = relative_maximum_error(pd_imp_sol.q,pd_ref_sol.q)

        record_results[("Pendulum_cgvi_hams_err")] = maximum(pd_relative_cgvi_ham_err)
        record_results[("Pendulum_cgvi_qerror")] = relative_maximum_error(pd_cgvi_sol.q,pd_ref_sol.q)

        record_results[("Pendulum_cgvi_hams_err3")] = maximum(pd_relative_cgvi_ham_err3)
        record_results[("Pendulum_cgvi_qerror3")] = relative_maximum_error(pd_cgvi_sol3.q,pd_ref_sol.q)


    end



    ### Perturbed Pendulum
    begin
        println("Start to run Perturbed Pendulum Problem with PR_Integrator!")

        # TT = 150.0
        # h_step = 5.0
        lode = GeometricProblems.PerturbedPendulum.lodeproblem(timespan = (0,TT),timestep = h_step)
        pp_ref_sol = integrate(lode, Gauss(8))
        pp_initial_hamiltonian = GeometricProblems.PerturbedPendulum.hamiltonian(0.0, lode.ics.q, lode.ics.p, lode.parameters)
        @show pp_initial_hamiltonian

        # R = 16
        QGau = QuadratureRules.GaussLegendreQuadrature(R)

        @variables W[1:3] ttt
        q_expr = W[1] *cos(W[2]* ttt + W[3])
        PRB = PR_Basis{Float64}([q_expr], [W], ttt,1)
        PR_Int = PR_Integrator(PRB, QGau,[[-0.51941,-0.47405,2.8713]])
        println("Start to run Perturbed Pendulum Problem with PR_Integrator! R = $(R), h = $(h_step)")
        pp_PR_sol,pp_internal_sol,pp_x_list = integrate(lode, PR_Int)
        @show relative_maximum_error(PR_sol.q,pp_ref_sol.q)

        pp_hams = [GeometricProblems.PerturbedPendulum.hamiltonian(0.0, q, p, lode.parameters) for (q, p) in zip(collect(pp_PR_sol.q[:]), collect(pp_PR_sol.p[:]))]
        pp_relative_hams_err = abs.((pp_hams .- pp_initial_hamiltonian) / pp_initial_hamiltonian)
        @show maximum(pp_relative_hams_err)

        pp_imp_sol = integrate(lode, ImplicitMidpoint())
        pp_imp_ham = [GeometricProblems.PerturbedPendulum.hamiltonian(0, q, p, lode.parameters) for (q, p) in zip(collect(pp_imp_sol.q[:]), collect(pp_imp_sol.p[:]))]
        pp_relative_imp_ham_err = abs.((pp_imp_ham .- pp_initial_hamiltonian) / pp_initial_hamiltonian)
        println("\n ImplicitMidpoint:")
        @show relative_maximum_error(pp_imp_sol.q,pp_ref_sol.q)
        @show maximum(pp_relative_imp_ham_err)

        QGau = GaussLegendreQuadrature(R)
        BGau = Lagrange(QuadratureRules.nodes(QGau))
        pp_cgvi_sol = integrate(lode, CGVI(BGau, QGau))
        pp_cgvi_ham = [GeometricProblems.PerturbedPendulum.hamiltonian(0, q, p, lode.parameters) for (q, p) in zip(collect(pp_cgvi_sol.q[:]), collect(pp_cgvi_sol.p[:]))]
        pp_relative_cgvi_ham_err = abs.((pp_cgvi_ham .- pp_initial_hamiltonian) / pp_initial_hamiltonian)
        println("\n CGVI:")
        @show relative_maximum_error(pp_cgvi_sol.q,pp_ref_sol.q)
        @show maximum(pp_relative_cgvi_ham_err)

        pp_lode_plot = GeometricProblems.PerturbedPendulum.lodeproblem(timespan = (0,TT),timestep = h_step/40)
        pp_sol_plot = integrate(pp_lode_plot, Gauss(8))
        # PR_plot_1d(PR_sol, internal_sol, sol_plot, relative_hams_err, h_step, TT, "Perturbed_Pendulum,h$(h_step)_T$(TT)_R$(R)")
        # println("Finish integrating Perturbed Pendulum Problem with PR_Integrator!, Figure Saved!")

        QGau3 = QuadratureRules.GaussLegendreQuadrature(3)
        BGau3 = Lagrange(QuadratureRules.nodes(QGau3))
        pp_cgvi_sol3 = integrate(lode, CGVI(BGau3, QGau3))
        pp_cgvi_ham3 = [GeometricProblems.PerturbedPendulum.hamiltonian(0, q, p, lode.parameters) for (q, p) in zip(collect(pp_cgvi_sol3.q[:]), collect(pp_cgvi_sol3.p[:]))]
        pp_relative_cgvi_ham_err3 = abs.((pp_cgvi_ham3 .- pp_initial_hamiltonian) / pp_initial_hamiltonian)
        println("\n CGVI with 3 nodes:")
        @show relative_maximum_error(pp_cgvi_sol3.q,pp_ref_sol.q)
        @show maximum(pp_relative_cgvi_ham_err3)

        record_results[("PerturbedPendulum_PR_sol_q")] = collect(pp_PR_sol.q[:,1])
        record_results[("PerturbedPendulum_PR_sol_p")] = collect(pp_PR_sol.p[:,1])

        record_results[("PerturbedPendulum_internal_sol")] = pp_internal_sol
        record_results[("PerturbedPendulum_x_list")] = pp_x_list
        record_results[("PerturbedPendulum_qerror")] = relative_maximum_error(pp_PR_sol.q,pp_ref_sol.q)
        record_results[("PerturbedPendulum_hams_err")] = maximum(pp_relative_hams_err)

        record_results[("PerturbedPendulum_imp_hams_err")] = maximum(pp_relative_imp_ham_err)
        record_results[("PerturbedPendulum_imp_qerror")] = relative_maximum_error(pp_imp_sol.q,pp_ref_sol.q)    

        record_results[("PerturbedPendulum_cgvi_hams_err")] = maximum(pp_relative_cgvi_ham_err)
        record_results[("PerturbedPendulum_cgvi_qerror")] = relative_maximum_error(pp_cgvi_sol.q,pp_ref_sol.q)

        record_results[("PerturbedPendulum_cgvi_hams_err3")] = maximum(pp_relative_cgvi_ham_err3)
        record_results[("PerturbedPendulum_cgvi_qerror3")] = relative_maximum_error(pp_cgvi_sol3.q,pp_ref_sol.q)

    end


    #### Henon Heiles
    begin     
        println("Start to run Henon Heiles Problem with PR_Integrator!")

        # TT = 100.0
        # h_step = 1.0
        HHlode = GeometricProblems.HenonHeilesPotential.lodeproblem([0.1,0.1],[0.1,0.1],timespan = (0,TT),timestep = h_step)
        HH_ref_sol = integrate(HHlode, Gauss(8))

        HH_initial_hamiltonian = GeometricProblems.HenonHeilesPotential.hamiltonian(0.0, HHlode.ics.q, HHlode.ics.p, HHlode.parameters)
        @show HH_initial_hamiltonian

        # R = 8
        QGau = QuadratureRules.GaussLegendreQuadrature(R)

        @variables W1[1:4] ttt
        #(0.14831 * cos(-0.64812 + x₁)) - 0.018712
        q₁_expr = W1[1] *cos(W1[2]* ttt + W1[3]) + W1[4]

        @variables W2[1:4]
        # 0.14298 * cos(- 0.97215 * x₁+ 0.7615)))-0.0013983
        q₂_expr = W2[1] *cos(W2[2]* ttt + W2[3]) + W2[4]

        PRB = PR_Basis{Float64}([q₁_expr,q₂_expr], [W1,W2], ttt,2)

        PR_Int = PR_Integrator(PRB, QGau,[[0.14831,1.0,-0.64812,- 0.018712],[0.14298,- 0.97215,0.7615,-0.0013983]]) # Pass the init W into the integrator instead of basis                                               
        HH_PR_sol,HH_internal_sol,HH_x_list = integrate(HHlode, PR_Int)
        @show relative_maximum_error(HH_PR_sol.q,HH_ref_sol.q)

        HH_hams = [GeometricProblems.HenonHeilesPotential.hamiltonian(0.0, q, p, HHlode.parameters) for (q, p) in zip(collect(HH_PR_sol.q[:]), collect(HH_PR_sol.p[:]))]
        HH_relative_hams_err = abs.((HH_hams .- HH_initial_hamiltonian) / HH_initial_hamiltonian)
        @show maximum(HH_relative_hams_err)

        HHlode_plot = GeometricProblems.HenonHeilesPotential.lodeproblem([0.1,0.1],[0.1,0.1],timespan = (0,TT),timestep = h_step/40)
        HH_sol_plot = integrate(HHlode_plot, Gauss(8))

        # PR_plot_2d(PR_sol, internal_sol, sol_plot,relative_hams_err, h_step, TT, "HenonHeiles_Potential,h$(h_step)_T$(TT)_R$(R)")
        # println("Finish integrating HenonHeiles Potential Problem with PR_Integrator!, Figure Saved!")

        HH_imp_sol = integrate(HHlode, ImplicitMidpoint())
        HH_imp_ham = [GeometricProblems.HenonHeilesPotential.hamiltonian(0, q, p, HHlode.parameters) for (q, p) in zip(collect(HH_imp_sol.q[:]), collect(HH_imp_sol.p[:]))]
        HH_relative_imp_ham_err = abs.((HH_imp_ham .- HH_initial_hamiltonian) / HH_initial_hamiltonian)
        println("\n ImplicitMidpoint:")
        @show relative_maximum_error(HH_imp_sol.q,HH_ref_sol.q)
        @show maximum(HH_relative_imp_ham_err)

        QGau = GaussLegendreQuadrature(R)
        BGau = Lagrange(QuadratureRules.nodes(QGau))
        HH_cgvi_sol = integrate(HHlode, CGVI(BGau, QGau))
        HH_cgvi_ham = [GeometricProblems.HenonHeilesPotential.hamiltonian(0, q, p, HHlode.parameters) for (q, p) in zip(collect(HH_cgvi_sol.q[:]), collect(HH_cgvi_sol.p[:]))]
        HH_relative_cgvi_ham_err = abs.((HH_cgvi_ham .- HH_initial_hamiltonian) / HH_initial_hamiltonian)
        println("\n CGVI:")
        @show relative_maximum_error(HH_cgvi_sol.q,HH_ref_sol.q)
        @show maximum(HH_relative_cgvi_ham_err)

        QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
        BGau4 = Lagrange(QuadratureRules.nodes(QGau4))
        HH_cgvi_sol4 = integrate(HHlode, CGVI(BGau4, QGau4))
        HH_cgvi_ham4 = [GeometricProblems.HenonHeilesPotential.hamiltonian(0, q, p, HHlode.parameters) for (q, p) in zip(collect(HH_cgvi_sol4.q[:]), collect(HH_cgvi_sol4.p[:]))]
        HH_relative_cgvi_ham_err4 = abs.((HH_cgvi_ham4 .- HH_initial_hamiltonian) / HH_initial_hamiltonian)
        println("\n CGVI with 4 nodes:")
        @show relative_maximum_error(HH_cgvi_sol4.q,HH_ref_sol.q)
        @show maximum(HH_relative_cgvi_ham_err4)

        # record_results[("HenonHeiles_PR_sol")] = HH_PR_sol
        record_results[("HenonHeiles_PR_sol_q1")] = collect(HH_PR_sol.q[:,1])
        record_results[("HenonHeiles_PR_sol_q2")] = collect(HH_PR_sol.q[:,2])
        record_results[("HenonHeiles_PR_sol_p1")] = collect(HH_PR_sol.p[:,1])
        record_results[("HenonHeiles_PR_sol_p2")] = collect(HH_PR_sol.p[:,2])

        record_results[("HenonHeiles_internal_sol")] = HH_internal_sol
        record_results[("HenonHeiles_x_list")] = HH_x_list
        record_results[("HenonHeiles_qerror")] = relative_maximum_error(HH_PR_sol.q,HH_ref_sol.q)
        record_results[("HenonHeiles_hams_err")] = maximum(HH_relative_hams_err)

        record_results[("HenonHeiles_imp_hams_err")] = maximum(HH_relative_imp_ham_err)
        record_results[("HenonHeiles_imp_qerror")] = relative_maximum_error(HH_imp_sol.q,HH_ref_sol.q)  

        record_results[("HenonHeiles_cgvi_hams_err")] = maximum(HH_relative_cgvi_ham_err)
        record_results[("HenonHeiles_cgvi_qerror")] = relative_maximum_error(HH_cgvi_sol.q,HH_ref_sol.q)

        record_results[("HenonHeiles_cgvi_hams_err4")] = maximum(HH_relative_cgvi_ham_err4)
        record_results[("HenonHeiles_cgvi_qerror4")] = relative_maximum_error(HH_cgvi_sol4.q,HH_ref_sol.q)

    end

    f_suctol = eval(Meta.parse(ARGS[4]))
    f_abstol = eval(Meta.parse(ARGS[3]))
    max_iterations = parse(Int,ARGS[2])
    h_step = parse(Float64,ARGS[1])


    filename2 = @sprintf(
        "parallel_result_figures/Quadratic_R%d_h%.2f_iter%d_fabs%.2e_fsuc%.2e_TT%d.jld2",
        R, h_step, max_iterations, f_abstol, f_suctol,TT)
    save(filename2,record_results)

    # # Plotting the results
    begin
        fig = Figure(size = (2200,3800),linewidth = 3,markersize = 13)
        label_size = 30
        tick_size = 25
        label_font_size = 35
        # legend_fig = Legend(fig[1,1:3],)
        q1_axis = Axis(fig[1, 1],xlabel = "Time", ylabel = "q₁",xlabelsize = label_size, ylabelsize = label_size,yticklabelsize = tick_size,xticklabelsize = tick_size)
        q2_axis = Axis(fig[1, 2],xlabel = "Time", ylabel = "p₁", xlabelsize = label_size, ylabelsize = label_size,yticklabelsize = tick_size,xticklabelsize = tick_size)
        ham_axis = Axis(fig[1, 3],xlabel = "Time", ylabel = "Relative Hamiltonian Error", xlabelsize = label_size, ylabelsize = label_size,yticklabelsize = tick_size,xticklabelsize = tick_size)
        linewidth = 3
        # h_step = 5.0
        # TT = 150.0
        t_dense = collect(0:h_step/40:TT)
        t_vise_dense = h_step/40:h_step/40:TT
        t_coarse = collect(0:h_step:TT)

        # Plot 1: q₁ over time
        lines!(q1_axis, t_dense, collect(HO_plot.q[:,1]), label="Analytical Solution",color = :black,linestyle = :dash,linewidth = linewidth)
        lines!(q1_axis, t_vise_dense, vcat(hcat(internal_sol...)[2:end,:]...), label="VISE Continuous Solution", color = :orange)
        scatter!(q1_axis,t_coarse, collect(HO_imp_sol.q[:, 1]), label="Implicit Midpoint Solution",color = :red)
        scatter!(q1_axis, t_coarse, collect(HO_cgvi_sol.q[:,1]), label="Galerkin Integrator Solution",color = :green)
        scatter!(q1_axis,t_coarse, collect(PR_sol.q[:,1]), label="VISE Discrete Solution",color = :blue)
        vlines!(q1_axis,[30.0],linestyle=:dashdot,color = :purple,label = "Training Region")


        # Plot 2: p₁ over time
        lines!(q2_axis, t_dense, collect(HO_plot.p[:,1]), label="Analytical Solution" ,color = :black,linestyle = :dash,linewidth = linewidth)
        scatter!(q2_axis, t_coarse, collect(HO_cgvi_sol.p[:,1]), label="Galerkin Integrator Solution ",color = :green)
        scatter!(q2_axis, t_coarse, collect(HO_imp_sol.p[:,1]), label="Implicit Midpoint Solution ",color = :red)
        scatter!(q2_axis, t_coarse, collect(PR_sol.p[:,1]), label="VISE Discrete Solution ",color = :blue)

        # Plot 3: Relative Hamiltonian error
        lines!(ham_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ",color =:blue)
        lines!(ham_axis, t_coarse, HO_relative_imp_ham_err, label="Implicit Midpoint Solution ",color = :red)
        lines!(ham_axis, t_coarse, HO_relative_cgvi_ham_err, label="Galerkin Integrator Solution ",color = :green)
        scatter!(ham_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ",color =:blue)
        scatter!(ham_axis, t_coarse, HO_relative_imp_ham_err, label="Implicit Midpoint Solution ",color = :red)
        scatter!(ham_axis, t_coarse, HO_relative_cgvi_ham_err, label="Galerkin Integrator Solution ",color = :green)

        Label(fig[1:1, 0], "Harmonic Oscillator", rotation = pi/2,
            fontsize = label_font_size,tellheight = false)

        # Legend(fig[2, 1:3], q1_axis,orientation = :horizontal,labelsize = 30, framevisible = false,nbanks = 2)    
        # save("result_figures/HO.pdf", fig)
        # fig
        
        # legend_fig = Legend(fig[1,1:3],)
        q1_axis = Axis(fig[2, 1],xlabel = "Time", ylabel = "q₁",xlabelsize = label_size, ylabelsize = label_size,yticklabelsize = tick_size,xticklabelsize = tick_size)
        q2_axis = Axis(fig[2, 2],xlabel = "Time", ylabel = "p₁", xlabelsize = label_size, ylabelsize = label_size,yticklabelsize = tick_size,xticklabelsize = tick_size)
        ham_axis = Axis(fig[2, 3],xlabel = "Time", ylabel = "Relative Hamiltonian Error", xlabelsize = label_size, ylabelsize = label_size,yticklabelsize = tick_size,xticklabelsize = tick_size)

        # h_step = 1.0
        t_dense = collect(0:h_step/40:TT)
        t_vise_dense = h_step/40:h_step/40:TT
        t_coarse = collect(0:h_step:TT)

        # Plot 1: q₁ over time
        lines!(q1_axis, t_coarse, collect(pd_ref_sol.q[:,1]), label="Reference Solution ",color = :black,linestyle = :dash,linewidth = linewidth)
        lines!(q1_axis, t_vise_dense, vcat(hcat(pendulum_internal_sol...)[2:end,:]...), label="VISE Continuous Solution", color = :orange)
        scatter!(q1_axis,t_coarse, collect(pd_imp_sol.q[:, 1]), label="Implicit Midpoint Solution",color = :red)
        scatter!(q1_axis, t_coarse, collect(pd_cgvi_sol.q[:,1]), label="Galerkin Integrator Solution",color = :green)
        scatter!(q1_axis,t_coarse, collect(pendulum_PR_sol.q[:,1]), label="VISE Discrete Solution",color = :blue)
        vlines!(q1_axis,[100.0],linestyle=:dashdot,color = :purple,label = "Training Region")

        # Plot 2: p₁ over time
        lines!(q2_axis, t_coarse, collect(pd_ref_sol.p[:,1]), label="Reference Solution" ,color = :black,linestyle = :dash,linewidth = linewidth)
        scatter!(q2_axis, t_coarse, collect(pd_cgvi_sol.p[:,1]), label="Galerkin Integrator Solution ",color = :green)
        scatter!(q2_axis, t_coarse, collect(pd_imp_sol.p[:,1]), label="Implicit Midpoint Solution ",color = :red)
        scatter!(q2_axis, t_coarse, collect(pendulum_PR_sol.p[:,1]), label="VISE Discrete Solution ",color = :blue)

        # Plot 3: Relative Hamiltonian error
        lines!(ham_axis, t_coarse, pendulum_relative_hams_err, label="VISE Discrete Solution ",color =:blue)
        lines!(ham_axis, t_coarse, pd_relative_imp_ham_err, label="Implicit Midpoint Solution ",color = :red)
        lines!(ham_axis, t_coarse, pd_relative_cgvi_ham_err, label="Galerkin Integrator Solution ",color = :green)
        scatter!(ham_axis, t_coarse, pendulum_relative_hams_err, label="VISE Discrete Solution ",color =:blue)
        scatter!(ham_axis, t_coarse, pd_relative_imp_ham_err, label="Implicit Midpoint Solution ",color = :red)
        scatter!(ham_axis, t_coarse, pd_relative_cgvi_ham_err, label="Galerkin Integrator Solution ",color = :green)
        
        Label(fig[2, 0], "Pendulum", rotation = pi/2,
            fontsize = label_font_size,tellheight = false)
        # Legend(fig[3, 1:3], q1_axis,orientation = :horizontal,labelsize = 30, 
        #     framevisible = false,nbanks = 2)    
        # save("result_figures/hopd.pdf", fig)
        # fig


        # legend_fig = Legend(fig[1,1:3],)
        q1_axis = Axis(fig[3, 1],xlabel = "Time", ylabel = "q₁",xlabelsize = label_size, ylabelsize = label_size,yticklabelsize = tick_size,xticklabelsize = tick_size)
        q2_axis = Axis(fig[3, 2],xlabel = "Time", ylabel = "p₁", xlabelsize = label_size, ylabelsize = label_size,yticklabelsize = tick_size,xticklabelsize = tick_size)
        ham_axis = Axis(fig[3, 3],xlabel = "Time", ylabel = "Relative Hamiltonian Error", xlabelsize = label_size, ylabelsize = label_size,yticklabelsize = tick_size,xticklabelsize = tick_size)

        # TT = 150.0
        # h_step = 5.0
        t_dense = collect(0:h_step/40:TT)
        t_vise_dense = h_step/40:h_step/40:TT
        t_coarse = collect(0:h_step:TT)

        # Plot 1: q₁ over time
        lines!(q1_axis, t_dense, collect(pp_sol_plot.q[:,1]), label="Reference Solution ",color = :black,linestyle = :dash,linewidth = linewidth)
        lines!(q1_axis, t_vise_dense, vcat(hcat(pp_internal_sol...)[2:end,:]...), label="VISE Continuous Solution", color = :orange)
        scatter!(q1_axis,t_coarse, collect(pp_imp_sol.q[:, 1]), label="Implicit Midpoint Solution",color = :red)
        scatter!(q1_axis, t_coarse, collect(pp_cgvi_sol.q[:,1]), label="Galerkin Integrator Solution",color = :green)
        scatter!(q1_axis,t_coarse, collect(pp_PR_sol.q[:,1]), label="VISE Discrete Solution",color = :blue)
        vlines!(q1_axis,[100.0],linestyle=:dashdot,color = :purple,label = "Training Region")

        # Plot 2: p₁ over time
        lines!(q2_axis, t_dense, collect(pp_sol_plot.p[:,1]), label="Reference Solution" ,color = :black,linestyle = :dash,linewidth = linewidth)
        scatter!(q2_axis, t_coarse, collect(pp_cgvi_sol.p[:,1]), label="Galerkin Integrator Solution ",color = :green)
        scatter!(q2_axis, t_coarse, collect(pp_imp_sol.p[:,1]), label="Implicit Midpoint Solution ",color = :red)
        scatter!(q2_axis, t_coarse, collect(pp_PR_sol.p[:,1]), label="VISE Discrete Solution ",color = :blue)

        # Plot 3: Relative Hamiltonian error
        lines!(ham_axis, t_coarse, pp_relative_hams_err, label="VISE Discrete Solution ",color =:blue)
        lines!(ham_axis, t_coarse, pp_relative_imp_ham_err, label="Implicit Midpoint Solution ",color = :red)
        lines!(ham_axis, t_coarse, pp_relative_cgvi_ham_err, label="Galerkin Integrator Solution ",color = :green)
        scatter!(ham_axis, t_coarse, pp_relative_hams_err, label="VISE Discrete Solution ",color =:blue)
        scatter!(ham_axis, t_coarse, pp_relative_imp_ham_err, label="Implicit Midpoint Solution ",color = :red)
        scatter!(ham_axis, t_coarse, pp_relative_cgvi_ham_err, label="Galerkin Integrator Solution ",color = :green)
        Label(fig[3, 0], "Perturbed Pendulum", rotation = pi/2,
            fontsize = label_font_size,tellheight = false)

        # HenonHeiles_Potential 
        # legend_fig = Legend(fig[1,1:3],)
        q1_axis = Axis(fig[4, 1],xlabel = "Time", ylabel = "q₁",xlabelsize = label_size, ylabelsize = label_size,yticklabelsize = tick_size,xticklabelsize = tick_size,limits = (nothing, (-0.15, 0.15)))
        p1_axis = Axis(fig[4, 2],xlabel = "Time", ylabel = "p₁", xlabelsize = label_size, ylabelsize = label_size,yticklabelsize = tick_size,xticklabelsize = tick_size,limits = (nothing, (-0.15, 0.15)))
        q2_axis = Axis(fig[5, 1],xlabel = "Time", ylabel = "q₂",xlabelsize = label_size, ylabelsize = label_size,yticklabelsize = tick_size,xticklabelsize = tick_size,limits = (nothing, (-0.25, 0.25)))
        p2_axis = Axis(fig[5, 2],xlabel = "Time", ylabel = "p₂", xlabelsize = label_size, ylabelsize = label_size,yticklabelsize = tick_size,xticklabelsize = tick_size,limits = (nothing, (-0.25, 0.25)))
        ham_axis = Axis(fig[5, 3],xlabel = "Time", ylabel = "Relative Hamiltonian Error", xlabelsize = label_size, ylabelsize = label_size,yticklabelsize = tick_size,xticklabelsize = tick_size,limits = (nothing, (0.0, 1.0)))

        # h_step = 1.0
        # TT = 100.0
        t_dense = collect(0:h_step/40:TT)
        t_vise_dense = h_step/40:h_step/40:TT
        t_coarse = collect(0:h_step:TT)

        internal_q1 = Array{Vector}(undef,Int(TT/h_step))
        internal_q2 = Array{Vector}(undef,Int(TT/h_step))

        for i in 1:Int(TT/h_step)
            internal_q1[i] = HH_internal_sol[i][:,1]
            internal_q2[i] = HH_internal_sol[i][:,2]
        end

        # Plot 1: q₁ over time
        lines!(q1_axis, t_dense, collect(HH_sol_plot.q[:,1]), label="Reference Solution ",color = :black,linestyle = :dash,linewidth = linewidth)
        lines!(q1_axis, t_vise_dense, vcat(hcat(internal_q1...)[2:end,:]...), label="VISE Continuous Solution", color = :orange)
        scatter!(q1_axis,t_coarse, collect(HH_imp_sol.q[:, 1]), label="Implicit Midpoint Solution",color = :red)
        scatter!(q1_axis, t_coarse, collect(HH_cgvi_sol.q[:,1]), label="Galerkin Integrator Solution",color = :green)
        scatter!(q1_axis,t_coarse, collect(HH_PR_sol.q[:,1]), label="VISE Discrete Solution",color = :blue)
        vlines!(q1_axis,[10.0],linestyle=:dashdot,color = :purple,label = "Training Region")

        # Plot 2: p₁ over time
        lines!(p1_axis, t_dense, collect(HH_sol_plot.p[:,1]), label="Reference Solution" ,color = :black,linestyle = :dash,linewidth = linewidth)
        scatter!(p1_axis, t_coarse, collect(HH_cgvi_sol.p[:,1]), label="Galerkin Integrator Solution ",color = :green)
        scatter!(p1_axis, t_coarse, collect(HH_imp_sol.p[:,1]), label="Implicit Midpoint Solution ",color = :red)
        scatter!(p1_axis, t_coarse, collect(HH_PR_sol.p[:,1]), label="VISE Discrete Solution ",color = :blue)


        # Plot 1: q2 over time
        lines!(q2_axis, t_dense, collect(HH_sol_plot.q[:,2]), label="Reference Solution ",color = :black,linestyle = :dash,linewidth = linewidth)
        lines!(q2_axis, t_vise_dense, vcat(hcat(internal_q2...)[2:end,:]...), label="VISE Continuous Solution", color = :orange)
        scatter!(q2_axis,t_coarse, collect(HH_imp_sol.q[:, 2]), label="Implicit Midpoint Solution",color = :red)
        scatter!(q2_axis, t_coarse, collect(HH_cgvi_sol.q[:,2]), label="Galerkin Integrator Solution",color = :green)
        scatter!(q2_axis,t_coarse, collect(HH_PR_sol.q[:,2]), label="VISE Discrete Solution",color = :blue)
        vlines!(q2_axis,[10.0],linestyle=:dashdot,color = :purple,label = "Training Region")

        # Plot 2: p2 over time
        lines!(p2_axis, t_dense, collect(HH_sol_plot.p[:,2]), label="Reference Solution" ,color = :black,linestyle = :dash,linewidth = linewidth)
        scatter!(p2_axis, t_coarse, collect(HH_cgvi_sol.p[:,2]), label="Galerkin Integrator Solution ",color = :green)
        scatter!(p2_axis, t_coarse, collect(HH_imp_sol.p[:,2]), label="Implicit Midpoint Solution ",color = :red)
        scatter!(p2_axis, t_coarse, collect(HH_PR_sol.p[:,2]), label="VISE Discrete Solution ",color = :blue)


        # Plot 3: Relative Hamiltonian error
        lines!(ham_axis, t_coarse, HH_relative_hams_err, label="VISE Discrete Solution ",color =:blue)
        lines!(ham_axis, t_coarse, HH_relative_imp_ham_err, label="Implicit Midpoint Solution ",color = :red)
        lines!(ham_axis, t_coarse, HH_relative_cgvi_ham_err, label="Galerkin Integrator Solution ",color = :green)
        scatter!(ham_axis, t_coarse, HH_relative_hams_err, label="VISE Discrete Solution ",color =:blue)
        scatter!(ham_axis, t_coarse, HH_relative_imp_ham_err, label="Implicit Midpoint Solution ",color = :red)
        scatter!(ham_axis, t_coarse, HH_relative_cgvi_ham_err, label="Galerkin Integrator Solution ",color = :green)
        Label(fig[4:5, 0], "Hénon-Heiles Potential", rotation = pi/2,
            fontsize = label_font_size,tellheight = false)

        Legend(fig[6, 1:3], q1_axis,orientation = :horizontal,labelsize = label_font_size, 
            framevisible = false,nbanks = 2)    
        # save("result_figures/full.pdf", fig)
        
        # fig

        filename = @sprintf(
        "parallel_result_figures/Quadratic_R%d_h%.2f_iter%d_fabs%.2e_fsuc%.2e_TT%d.pdf",
        R, h_step, max_iterations, f_abstol, f_suctol,TT)
        save(filename, fig)

    end

    t5 = time()
    println("Total time taken: ", t5 - t0, " seconds")
    println("All results saved to ", filename)
end
# begin 
#     TT = 100.0
#     h_step = 0.3
#     pendulum_lode = GeometricProblems.Pendulum.lodeproblem(timespan = (0,TT),timestep = h_step)
#     pd_ref_sol = integrate(pendulum_lode, Gauss(8))

#     TT = 100.0
#     h_step = 0.1
#     pendulum_lode = GeometricProblems.Pendulum.lodeproblem(timespan = (0,TT),timestep = h_step)
#     continuous_sol = integrate(pendulum_lode, Gauss(8))

#     pd_ref_fig = Figure(size = (800,1200), linewidth = 2, markersize = 8)
#     q_axis = Axis(pd_ref_fig[1,1],xlabel = "t", ylabel = "q",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
#     lin1= lines!(q_axis, collect(0:0.1:TT), collect(continuous_sol.q[:,1]), label = "Reference Solution", linewidth = 2,color = :red)
#     sca1 = scatter!(q_axis,collect(0:0.3:100),collect(pd_ref_sol.q[:,1]),label = "Observation Points")
#     Label(pd_ref_fig[1,0],"Pendulum", rotation = pi/2,fontsize = 30,tellheight = false)

#     lp_axis = Axis(pd_ref_fig[2,1],xlabel = "x", ylabel = "u",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
#     input = randn(100) .-2
#     labels = exact_u.(0,input)
#     scatter!(lp_axis,input,labels,label = "Observation Points")
#     lin2 = lines!(lp_axis,collect(-5:0.01:1),exact_u.(0,collect(-5:0.01:1)),label = "Initial Condition",linewidth = 2,color = :green)
#     Label(pd_ref_fig[2,0],"Linear Transport Equation", rotation = pi/2,fontsize = 30,tellheight = false)

#     Legend(pd_ref_fig[3, 1], [lin1,lin2,sca1],["Pendulum Reference Solution",
#             "Linear Transport Equation Initial Condition",
#             "Observation Points"],
#     orientation = :horizontal,labelsize = 30, 
#             framevisible = false,nbanks = 3)  
#     save("result_figures/pd_lp_illu.svg", pd_ref_fig)
#     pd_ref_fig

# end

# begin
#     function pendulum_p_prediction(t,params)
#         #q_expr = W[1] *cos(W[2]* ttt + W[3])
#         - params[1] * params[2]* sin(params[2] * t + params[3]) 
#     end

#     function pendulum_q_prediction(t,params)
#         params[1] * cos(params[2] * t + params[3])   
#     end


#     ill_fig2 = Figure(size = (800,1200), linewidth = 2, markersize = 8)
#     pq_axis = Axis(ill_fig2[1,1],xlabel = "q(t)", ylabel = "p(t)",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)

#     h = 0.05
#     t1 = collect(0:h:1)
#     q1 = [pendulum_q_prediction(ti, x_list[1]) for ti in t1]
#     p1 = [pendulum_p_prediction(ti, x_list[1]) for ti in t1]
#     lin1 = lines!(pq_axis, q1, p1, label="VISE Solution",)

#     t2 = collect(1:h:2)
#     q2 = [pendulum_q_prediction(ti, x_list[2]) for ti in t2]
#     p2 = [pendulum_p_prediction(ti, x_list[2]) for ti in t2]
#     lin2 = lines!(pq_axis, q2, p2, label="VISE Solution")

#     t3 = collect(2:h:3)
#     q3 = [pendulum_q_prediction(ti, x_list[3]) for ti in t3]
#     p3 = [pendulum_p_prediction(ti, x_list[3]) for ti in t3]
#     lin3 = lines!(pq_axis, q3, p3, label="VISE Solution")

#     t4 = collect(3:h:4)
#     q4 = [pendulum_q_prediction(ti, x_list[4]) for ti in t4]
#     p4 = [pendulum_p_prediction(ti, x_list[4]) for ti in t4]
#     lin4 = lines!(pq_axis, q4, p4, label="VISE Solution")

#     t5 = collect(4:h:5)
#     q5 = [pendulum_q_prediction(ti, x_list[5]) for ti in t5]
#     p5 = [pendulum_p_prediction(ti, x_list[5]) for ti in t5]
#     lin5 = lines!(pq_axis, q5, p5, label="VISE Solution")

#     t6 = collect(5:h:6)
#     q6 = [pendulum_q_prediction(ti, x_list[6]) for ti in t6]
#     p6 = [pendulum_p_prediction(ti, x_list[6]) for ti in t6]
#     lin6 = lines!(pq_axis, q6, p6, label="VISE Solution")

#     t7 = collect(6:h:7)
#     q7 = [pendulum_q_prediction(ti, x_list[7]) for ti in t7]
#     p7 = [pendulum_p_prediction(ti, x_list[7]) for ti in t7]
#     lin7 = lines!(pq_axis, q7, p7, label="VISE Solution")   

#     t8 = collect(7:h:8)
#     q8 = [pendulum_q_prediction(ti, x_list[8]) for ti in t8]
#     p8 = [pendulum_p_prediction(ti, x_list[8]) for ti in t8]
#     lin8 = lines!(pq_axis, q8, p8, label="VISE Solution")

#     t9 = collect(8:h:9)
#     q9 = [pendulum_q_prediction(ti, x_list[9]) for ti in t9]
#     p9 = [pendulum_p_prediction(ti, x_list[9]) for ti in t9]
#     lin9 = lines!(pq_axis, q9, p9, label="VISE Solution")

#     t10 = collect(9:h:10)
#     q10 = [pendulum_q_prediction(ti, x_list[10]) for ti in t10]
#     p10 = [pendulum_p_prediction(ti, x_list[10]) for ti in t10]
#     lin10 = lines!(pq_axis, q10, p10, label="VISE Solution")

#     t11 = collect(10:h:11)
#     q11 = [pendulum_q_prediction(ti, x_list[11]) for ti in t11]
#     p11 = [pendulum_p_prediction(ti, x_list[11]) for ti in t11]
#     lin11 = lines!(pq_axis, q11, p11, label="VISE Solution")

#     t12 = collect(11:h:12)
#     q12 = [pendulum_q_prediction(ti, x_list[12]) for ti in t12]
#     p12 = [pendulum_p_prediction(ti, x_list[12]) for ti in t12]
#     lin12 = lines!(pq_axis, q12, p12, label="VISE Solution")

#     t13 = collect(12:0.05:13)
#     q13 = [pendulum_q_prediction(ti, x_list[13]) for ti in t13]
#     p13 = [pendulum_p_prediction(ti, x_list[13]) for ti in t13]
#     lin13 = lines!(pq_axis, q13, p13, label="VISE Solution")

#     t14 = collect(13:0.05:14)
#     q14 = [pendulum_q_prediction(ti, x_list[14]) for ti in t14]
#     p14 = [pendulum_p_prediction(ti, x_list[14]) for ti in t14]
#     lin14 = lines!(pq_axis, q14, p14, label="VISE Solution")

#     t15 = collect(14:0.05:15)
#     q15 = [pendulum_q_prediction(ti, x_list[15]) for ti in t15]
#     p15 = [pendulum_p_prediction(ti, x_list[15]) for ti in t15]
#     lin15 = lines!(pq_axis, q15, p15, label="VISE Solution")

#     line_ref = lines!(pq_axis,collect(pd_ref_sol.q[:,1]) ,collect(pd_ref_sol.p[:,1]), label="Pendulum Reference Solution", linewidth = 1, color = :blue,linestyle = :dash)
#     Label(ill_fig2[1,0],"Pendulum", rotation = pi/2,fontsize = 30,tellheight = false)


#     border_x = collect(-5:0.01:1)
#     t1 = collect(0:0.01:0.1)
#     t2 = collect(0.1:0.01:0.2)
#     t3 = collect(0.2:0.01:0.3)
#     t4 = collect(0.3:0.01:0.4)  
#     t5 = collect(0.4:0.01:0.5)

#     lt_params1= [1.9998958731356202,0.3130332665012085,3.8434752175319824,-0.5000260330714674,0.009429492502348818,]
#     lt_params2= [  1.999895874334041,
#     0.31303326677611865,
#     3.8434752174883293,
#     -0.5000260327718281,
#     0.009429492503866939,]

#     lt_params3= [  1.9998958742619493,
#     0.3130332669098647,
#     3.843475217415565,
#     -0.5000260327898925,
#     0.009429492506388108,]

#     lt_params4= [   1.9998958745081066,
#     0.3130332670424184,
#     3.843475217368593,
#     -0.5000260327283398,
#     0.009429492508022639,]

#     lt_params5= [  1.999895874539698,
#     0.3130332670668317,
#     3.8434752173588,
#     -0.5000260327204497,
#     0.00942949250836107]

#     function lt_u_SindySol(x,t,p)
#         exp((p[1] * (x - 0.2*t) + p[2])*(x - 0.2 * t + p[3])*p[4]) * p[5]
#     end

#     u_sol_ls1 =zeros(length(border_x), length(t1));
#     for (i, xx) in enumerate(border_x)
#         for (j, tt) in enumerate(t1)
#             u_sol_ls1[i,j] = lt_u_SindySol(xx, tt, lt_params1)
#         end
#     end 

#     u_sol_ls2 =zeros(length(border_x), length(t2));
#     for (i, xx) in enumerate(border_x)
#         for (j, tt) in enumerate(t2)
#             u_sol_ls2[i,j] = lt_u_SindySol(xx, tt, lt_params2)
#         end
#     end

#     u_sol_ls3 =zeros(length(border_x), length(t3));
#     for (i, xx) in enumerate(border_x)
#         for (j, tt) in enumerate(t3)
#             u_sol_ls3[i,j] = lt_u_SindySol(xx, tt, lt_params3)
#         end
#     end

#     u_sol_ls4 =zeros(length(border_x), length(t4));
#     for (i, xx) in enumerate(border_x)
#         for (j, tt) in enumerate(t4)
#             u_sol_ls4[i,j] = lt_u_SindySol(xx, tt, lt_params4)
#         end
#     end

#     u_sol_ls5 =zeros(length(border_x), length(t5));
#     for (i, xx) in enumerate(border_x)
#         for (j, tt) in enumerate(t5)
#             u_sol_ls5[i,j] = lt_u_SindySol(xx, tt, lt_params5)
#         end
#     end

#     ux_axis = Axis(ill_fig2[2,1],xlabel = "x", ylabel = "t",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
#     hm = heatmap!(ux_axis, border_x, t1, u_sol_ls1)
#     heatmap!(ux_axis, border_x, t2 .+ 0.01, u_sol_ls2)
#     heatmap!(ux_axis, border_x, t3, u_sol_ls3)
#     heatmap!(ux_axis, border_x, t4, u_sol_ls4)
#     heatmap!(ux_axis, border_x, t5, u_sol_ls5)
#     hl1 = hlines!(ux_axis, [0.1], linestyle = :dash, label = "Discrete Time Step", linewidth = 2, color = :white)
#     hlines!(ux_axis, [0.2], linestyle = :dash, label = "Discrete Time Step", linewidth = 2, color = :white)
#     hlines!(ux_axis, [0.3], linestyle = :dash, label = "Discrete Time Step", linewidth = 2, color = :white)
#     hlines!(ux_axis, [0.4], linestyle = :dash, label = "Discrete Time Step", linewidth = 2, color = :white)

#     Label(ill_fig2[2,0],"Linear Transport Equation", rotation = pi/2,fontsize = 30,tellheight = false)
#     Colorbar(ill_fig2[2, 2], hm)


#     Legend(ill_fig2[3, 1], [line_ref,lin1,hl1],["Pendulum Reference Solution",
#             "Pendulum VISE Solution",
#             "Discrete Time Step"],
#     orientation = :horizontal,labelsize = 30, 
#             framevisible = false,nbanks = 3)  

#     save("result_figures/ill_pendulum_lp_right.pdf", ill_fig2)
# end