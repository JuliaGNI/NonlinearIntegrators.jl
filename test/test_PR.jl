using Symbolics
using CompactBasisFunctions
using NonlinearIntegrators
using QuadratureRules

using GeometricProblems
using GeometricIntegrators
using SimpleSolvers
# using Plots
using CairoMakie


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


# Harmonic Oscillator
# begin 
#     @variables W[1:3] ttt
#     q_expr = W[1] *sin(W[2]* ttt + W[3])

#     PRB = PR_Basis{Float64}([q_expr], [W], ttt,1)
#     TT = 150.0
#     h_step = 5.0
#     HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(tspan = (0,TT),tstep = h_step)

#     initial_hamiltonian = GeometricProblems.HarmonicOscillator.hamiltonian(0.0, HO_lode.ics.q, HO_lode.ics.p, HO_lode.parameters)
#     HO_truth = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(tspan = (0,TT),tstep = h_step))
#     HO_plot = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(tspan = (0,TT),tstep = h_step/40))

#     R = 32
#     QGau4 = QuadratureRules.GaussLegendreQuadrature(R)
#     PR_Int = PR_Integrator(PRB, QGau4,[[-0.5000433352162222,0.705350078478666,-1.5678140333370576]]) # Pass the init W into the integrator instead of basis                                               
#     # PR_Int = PR_Integrator(PRB, QGau4,[[-0.500,sqrt(0.5),-pi/2]])                           

#     println("Start to run Harmonic Oscillator Problem with PR_Integrator! R = $(R), h = $(h_step)")
#     PR_sol,internal_sol,x_list = integrate(HO_lode, PR_Int)

#     hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(PR_sol.q[:]), collect(PR_sol.p[:]))]
#     relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)
#     println("\n NISE:")
#     @show relative_maximum_error(PR_sol.q,HO_truth.q)
#     @show maximum(relative_hams_err)

#     pref = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(tspan = (0,TT),tstep = h_step/40))
#     # PR_plot_1d(PR_sol, internal_sol, HO_plot, relative_hams_err, h_step, TT, "HarmonicOscillator,h$(h_step)_T$(TT)_R$(R)")
#     # println("Finish integrating Harmonic Oscillator Problem with PR_Integrator!, Figure Saved!")
    
#     HO_imp_sol = integrate(HO_lode, ImplicitMidpoint())
#     HO_imp_ham = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_imp_sol.q[:]), collect(HO_imp_sol.p[:]))]
#     HO_relative_imp_ham_err = abs.((HO_imp_ham .- initial_hamiltonian) / initial_hamiltonian)
#     println("\n ImplicitMidpoint:")
#     @show relative_maximum_error(HO_imp_sol.q,HO_truth.q)
#     @show maximum(HO_relative_imp_ham_err)


#     QGau = GaussLegendreQuadrature(R)
#     BGau = Lagrange(QuadratureRules.nodes(QGau))
#     HO_cgvi_sol = integrate(HO_lode, CGVI(BGau, QGau)) 
#     HO_cgvi_ham = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_cgvi_sol.q[:]), collect(HO_cgvi_sol.p[:]))]
#     HO_relative_cgvi_ham_err = abs.((HO_cgvi_ham .- initial_hamiltonian) / initial_hamiltonian)

#     println("\n CGVI R = $(R):")
#     @show relative_maximum_error(HO_cgvi_sol.q,HO_truth.q)
#     @show maximum(HO_relative_cgvi_ham_err)

#     QGau3 = QuadratureRules.GaussLegendreQuadrature(3)
#     BGau3 = Lagrange(QuadratureRules.nodes(QGau3))
#     HO_cgvi_sol3 = integrate(HO_lode, CGVI(BGau3, QGau3))
#     HO_cgvi_ham3 = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_cgvi_sol3.q[:]), collect(HO_cgvi_sol3.p[:]))]
#     HO_relative_cgvi_ham_err3 = abs.((HO_cgvi_ham3 .- initial_hamiltonian) / initial_hamiltonian)
#     println("\n CGVI with 3 nodes:")
#     @show relative_maximum_error(HO_cgvi_sol3.q,HO_truth.q)
#     @show maximum(HO_relative_cgvi_ham_err3)

#     # # try to find a error on the same scale with NISE
#     # balance_h = 0.1
#     # HO_lode2 = GeometricProblems.HarmonicOscillator.lodeproblem(tspan = (0,TT),tstep = balance_h)
#     # imp_sol2 = integrate(HO_lode2, ImplicitMidpoint())
#     # HO_truth2 = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(tspan = (0,TT),tstep = balance_h))
#     # imp_ham2 = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode2.parameters) for (q, p) in zip(collect(imp_sol2.q[:]), collect(imp_sol2.p[:]))]
#     # relative_imp_ham_err2 = abs.((imp_ham2 .- initial_hamiltonian) / initial_hamiltonian)
#     # println("\n ImplicitMidpoint with balance_h:")
#     # @show relative_maximum_error(imp_sol2.q,HO_truth2.q)
#     # @show maximum(relative_imp_ham_err2)

#     # balance_h = 15
#     # HO_lode2 = GeometricProblems.HarmonicOscillator.lodeproblem(tspan = (0,TT),tstep = balance_h)
#     # HO_truth2 = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(tspan = (0,TT),tstep = balance_h))

#     # QGau = GaussLegendreQuadrature(8)
#     # BGau = Lagrange(QuadratureRules.nodes(QGau))
#     # cgvi_sol2 = integrate(HO_lode2, CGVI(BGau, QGau)) 
#     # cgvi_ham2 = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(cgvi_sol2.q[:]), collect(cgvi_sol2.p[:]))]
#     # relative_cgvi_ham_err2 = abs.((cgvi_ham2 .- initial_hamiltonian) / initial_hamiltonian)
#     # println("\n CGVI with balance_h:")
#     # @show relative_maximum_error(cgvi_sol2.q,HO_truth2.q)
#     # @show maximum(relative_cgvi_ham_err2)

# end


# #### Pendulum
# begin
    TT = 100.0
    h_step = 0.1
    pendulum_lode = GeometricProblems.Pendulum.lodeproblem(tspan = (0,TT),tstep = h_step)
    pd_ref_sol = integrate(pendulum_lode, Gauss(8))
#     initial_hamiltonian = GeometricProblems.Pendulum.hamiltonian(0.0, pendulum_lode.ics.q, pendulum_lode.ics.p, pendulum_lode.parameters)
#     @show initial_hamiltonian

#     # f_try(x₁) = cos((x₁ * 0.45644) - 1.1466) * 1.1931
#     @variables W[1:3] ttt
#     q_expr = W[1] *cos(W[2]* ttt + W[3])
#     PRB = PR_Basis{Float64}([q_expr], [W], ttt,1)
#     R = 4
#     println("Start to run Pendulum Problem with PR_Integrator! h = $(h_step), R= $(R)")

#     QGau = QuadratureRules.GaussLegendreQuadrature(R)
#     PR_Int = PR_Integrator(PRB, QGau,[[1.1931,0.45644,-1.1466]])

#     pendulum_PR_sol,pendulum_internal_sol,x_list = integrate(pendulum_lode, PR_Int)
#     @show relative_maximum_error(pendulum_PR_sol.q,pd_ref_sol.q)

#     pendulum_hams = [GeometricProblems.Pendulum.hamiltonian(0.0, q, p, pendulum_lode.parameters) for (q, p) in zip(collect(pendulum_PR_sol.q[:]), collect(pendulum_PR_sol.p[:]))]
#     pendulum_relative_hams_err = abs.((pendulum_hams .- initial_hamiltonian) / initial_hamiltonian)
#     @show maximum(pendulum_relative_hams_err)

#     pendulum_plot = GeometricProblems.Pendulum.lodeproblem(tspan = (0,TT),tstep = h_step/40)
#     pendulum_sol_plot = integrate(pendulum_plot, Gauss(8))
#     # PR_plot_1d(PR_sol, internal_sol, sol_plot, relative_hams_err, h_step, TT, "Pendulum,h$(h_step)_T$(TT)_R$(R)")
#     # println("Finish integrating Pendulum Problem with PR_Integrator!, Figure Saved!")

#     pd_imp_sol = integrate(pendulum_lode, ImplicitMidpoint())
#     pd_imp_ham = [GeometricProblems.Pendulum.hamiltonian(0, q, p, pendulum_lode.parameters) for (q, p) in zip(collect(pd_imp_sol.q[:]), collect(pd_imp_sol.p[:]))]
#     pd_relative_imp_ham_err = abs.((pd_imp_ham .- initial_hamiltonian) / initial_hamiltonian)
#     println("\n ImplicitMidpoint:")
#     @show relative_maximum_error(pd_imp_sol.q,pd_ref_sol.q)
#     @show maximum(pd_relative_imp_ham_err)

#     QGau = GaussLegendreQuadrature(R)
#     BGau = Lagrange(QuadratureRules.nodes(QGau))
#     pd_cgvi_sol = integrate(pendulum_lode, CGVI(BGau, QGau))
#     pd_cgvi_ham = [GeometricProblems.Pendulum.hamiltonian(0, q, p, pendulum_lode.parameters) for (q, p) in zip(collect(pd_cgvi_sol.q[:]), collect(pd_cgvi_sol.p[:]))]
#     pd_relative_cgvi_ham_err = abs.((pd_cgvi_ham .- initial_hamiltonian) / initial_hamiltonian)
#     println("\n CGVI:")
#     @show relative_maximum_error(pd_cgvi_sol.q,pd_ref_sol.q)
#     @show maximum(pd_relative_cgvi_ham_err)
#     #Plotting the results

#     QGau3 = QuadratureRules.GaussLegendreQuadrature(3)
#     BGau3 = Lagrange(QuadratureRules.nodes(QGau3))
#     pd_cgvi_sol3 = integrate(pendulum_lode, CGVI(BGau3, QGau3))
#     pd_cgvi_ham3 = [GeometricProblems.Pendulum.hamiltonian(0, q, p, pendulum_lode.parameters) for (q, p) in zip(collect(pd_cgvi_sol3.q[:]), collect(pd_cgvi_sol3.p[:]))]
#     pd_relative_cgvi_ham_err3 = abs.((pd_cgvi_ham3 .- initial_hamiltonian) / initial_hamiltonian)
#     println("\n CGVI with 3 nodes:")
#     @show relative_maximum_error(pd_cgvi_sol3.q,pd_ref_sol.q)
#     @show maximum(pd_relative_cgvi_ham_err3)
# end



#### Perturbed Pendulum
# begin
#     println("Start to run Perturbed Pendulum Problem with PR_Integrator!")

#     TT = 150.0
#     h_step = 5.0
#     lode = GeometricProblems.PerturbedPendulum.lodeproblem(tspan = (0,TT),tstep = h_step)
#     pp_ref_sol = integrate(lode, Gauss(8))
#     pp_initial_hamiltonian = GeometricProblems.PerturbedPendulum.hamiltonian(0.0, lode.ics.q, lode.ics.p, lode.parameters)
#     @show pp_initial_hamiltonian

#     R = 16
#     QGau = QuadratureRules.GaussLegendreQuadrature(R)

#     @variables W[1:3] ttt
#     q_expr = W[1] *cos(W[2]* ttt + W[3])
#     PRB = PR_Basis{Float64}([q_expr], [W], ttt,1)
#     PR_Int = PR_Integrator(PRB, QGau,[[-0.51941,-0.47405,2.8713]])
#     println("Start to run Perturbed Pendulum Problem with PR_Integrator! R = $(R), h = $(h_step)")
#     pp_PR_sol,pp_internal_sol,x_list = integrate(lode, PR_Int)
#     @show relative_maximum_error(PR_sol.q,pp_ref_sol.q)

#     pp_hams = [GeometricProblems.PerturbedPendulum.hamiltonian(0.0, q, p, lode.parameters) for (q, p) in zip(collect(pp_PR_sol.q[:]), collect(pp_PR_sol.p[:]))]
#     pp_relative_hams_err = abs.((pp_hams .- pp_initial_hamiltonian) / pp_initial_hamiltonian)
#     @show maximum(pp_relative_hams_err)

#     pp_imp_sol = integrate(lode, ImplicitMidpoint())
#     pp_imp_ham = [GeometricProblems.PerturbedPendulum.hamiltonian(0, q, p, lode.parameters) for (q, p) in zip(collect(pp_imp_sol.q[:]), collect(pp_imp_sol.p[:]))]
#     pp_relative_imp_ham_err = abs.((pp_imp_ham .- pp_initial_hamiltonian) / pp_initial_hamiltonian)
#     println("\n ImplicitMidpoint:")
#     @show relative_maximum_error(pp_imp_sol.q,pp_ref_sol.q)
#     @show maximum(pp_relative_imp_ham_err)

#     QGau = GaussLegendreQuadrature(R)
#     BGau = Lagrange(QuadratureRules.nodes(QGau))
#     pp_cgvi_sol = integrate(lode, CGVI(BGau, QGau))
#     pp_cgvi_ham = [GeometricProblems.PerturbedPendulum.hamiltonian(0, q, p, lode.parameters) for (q, p) in zip(collect(pp_cgvi_sol.q[:]), collect(pp_cgvi_sol.p[:]))]
#     pp_relative_cgvi_ham_err = abs.((pp_cgvi_ham .- pp_initial_hamiltonian) / pp_initial_hamiltonian)
#     println("\n CGVI:")
#     @show relative_maximum_error(pp_cgvi_sol.q,pp_ref_sol.q)
#     @show maximum(pp_relative_cgvi_ham_err)

#     pp_lode_plot = GeometricProblems.PerturbedPendulum.lodeproblem(tspan = (0,TT),tstep = h_step/40)
#     pp_sol_plot = integrate(pp_lode_plot, Gauss(8))
#     # PR_plot_1d(PR_sol, internal_sol, sol_plot, relative_hams_err, h_step, TT, "Perturbed_Pendulum,h$(h_step)_T$(TT)_R$(R)")
#     # println("Finish integrating Perturbed Pendulum Problem with PR_Integrator!, Figure Saved!")

#     QGau3 = QuadratureRules.GaussLegendreQuadrature(3)
#     BGau3 = Lagrange(QuadratureRules.nodes(QGau3))
#     pp_cgvi_sol3 = integrate(lode, CGVI(BGau3, QGau3))
#     pp_cgvi_ham3 = [GeometricProblems.PerturbedPendulum.hamiltonian(0, q, p, lode.parameters) for (q, p) in zip(collect(pp_cgvi_sol3.q[:]), collect(pp_cgvi_sol3.p[:]))]
#     pp_relative_cgvi_ham_err3 = abs.((pp_cgvi_ham3 .- pp_initial_hamiltonian) / pp_initial_hamiltonian)
#     println("\n CGVI with 3 nodes:")
#     @show relative_maximum_error(pp_cgvi_sol3.q,pp_ref_sol.q)
#     @show maximum(pp_relative_cgvi_ham_err3)
# end


# #### Henon Heiles
# begin     
#     println("Start to run Henon Heiles Problem with PR_Integrator!")

#     TT = 100.0
#     h_step = 1.0
#     HHlode = GeometricProblems.HenonHeilesPotential.lodeproblem([0.1,0.1],[0.1,0.1],tspan = (0,TT),tstep = h_step)
#     HH_ref_sol = integrate(HHlode, Gauss(8))

#     HH_initial_hamiltonian = GeometricProblems.HenonHeilesPotential.hamiltonian(0.0, HHlode.ics.q, HHlode.ics.p, HHlode.parameters)
#     @show HH_initial_hamiltonian

#     R = 8
#     QGau = QuadratureRules.GaussLegendreQuadrature(R)

#     @variables W1[1:4] ttt
#     #(0.14831 * cos(-0.64812 + x₁)) - 0.018712
#     q₁_expr = W1[1] *cos(W1[2]* ttt + W1[3]) + W1[4]

#     @variables W2[1:4]
#     # 0.14298 * cos(- 0.97215 * x₁+ 0.7615)))-0.0013983
#     q₂_expr = W2[1] *cos(W2[2]* ttt + W2[3]) + W2[4]

#     PRB = PR_Basis{Float64}([q₁_expr,q₂_expr], [W1,W2], ttt,2)

#     PR_Int = PR_Integrator(PRB, QGau,[[0.14831,1.0,-0.64812,- 0.018712],[0.14298,- 0.97215,0.7615,-0.0013983]]) # Pass the init W into the integrator instead of basis                                               
#     HH_PR_sol,HH_internal_sol,x_list = integrate(HHlode, PR_Int)
#     @show relative_maximum_error(HH_PR_sol.q,HH_ref_sol.q)

#     HH_hams = [GeometricProblems.HenonHeilesPotential.hamiltonian(0.0, q, p, HHlode.parameters) for (q, p) in zip(collect(HH_PR_sol.q[:]), collect(HH_PR_sol.p[:]))]
#     HH_relative_hams_err = abs.((HH_hams .- HH_initial_hamiltonian) / HH_initial_hamiltonian)
#     @show maximum(HH_relative_hams_err)

#     HHlode_plot = GeometricProblems.HenonHeilesPotential.lodeproblem([0.1,0.1],[0.1,0.1],tspan = (0,TT),tstep = h_step/40)
#     HH_sol_plot = integrate(HHlode_plot, Gauss(8))

#     # PR_plot_2d(PR_sol, internal_sol, sol_plot,relative_hams_err, h_step, TT, "HenonHeiles_Potential,h$(h_step)_T$(TT)_R$(R)")
#     # println("Finish integrating HenonHeiles Potential Problem with PR_Integrator!, Figure Saved!")

#     HH_imp_sol = integrate(HHlode, ImplicitMidpoint())
#     HH_imp_ham = [GeometricProblems.HenonHeilesPotential.hamiltonian(0, q, p, HHlode.parameters) for (q, p) in zip(collect(HH_imp_sol.q[:]), collect(HH_imp_sol.p[:]))]
#     HH_relative_imp_ham_err = abs.((HH_imp_ham .- HH_initial_hamiltonian) / HH_initial_hamiltonian)
#     println("\n ImplicitMidpoint:")
#     @show relative_maximum_error(HH_imp_sol.q,HH_ref_sol.q)
#     @show maximum(HH_relative_imp_ham_err)

#     QGau = GaussLegendreQuadrature(R)
#     BGau = Lagrange(QuadratureRules.nodes(QGau))
#     HH_cgvi_sol = integrate(HHlode, CGVI(BGau, QGau))
#     HH_cgvi_ham = [GeometricProblems.HenonHeilesPotential.hamiltonian(0, q, p, HHlode.parameters) for (q, p) in zip(collect(HH_cgvi_sol.q[:]), collect(HH_cgvi_sol.p[:]))]
#     HH_relative_cgvi_ham_err = abs.((HH_cgvi_ham .- HH_initial_hamiltonian) / HH_initial_hamiltonian)
#     println("\n CGVI:")
#     @show relative_maximum_error(HH_cgvi_sol.q,HH_ref_sol.q)
#     @show maximum(HH_relative_cgvi_ham_err)

#     QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
#     BGau4 = Lagrange(QuadratureRules.nodes(QGau4))
#     HH_cgvi_sol4 = integrate(HHlode, CGVI(BGau4, QGau4))
#     HH_cgvi_ham4 = [GeometricProblems.HenonHeilesPotential.hamiltonian(0, q, p, HHlode.parameters) for (q, p) in zip(collect(HH_cgvi_sol4.q[:]), collect(HH_cgvi_sol4.p[:]))]
#     HH_relative_cgvi_ham_err4 = abs.((HH_cgvi_ham4 .- HH_initial_hamiltonian) / HH_initial_hamiltonian)
#     println("\n CGVI with 4 nodes:")
#     @show relative_maximum_error(HH_cgvi_sol4.q,HH_ref_sol.q)
#     @show maximum(HH_relative_cgvi_ham_err4)

# end


# # Plotting the results
# begin
#     fig = Figure(size = (2200,3800),linewidth = 2,markersize = 13)
#     # legend_fig = Legend(fig[1,1:3],)
#     q1_axis = Axis(fig[1, 1],xlabel = "Time", ylabel = "q₁",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
#     q2_axis = Axis(fig[1, 2],xlabel = "Time", ylabel = "p₁", xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
#     ham_axis = Axis(fig[1, 3],xlabel = "Time", ylabel = "Relative Hamiltonian Error", xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
#     linewidth = 3
#     h_step = 5.0
#     TT = 150.0
#     t_dense = collect(0:h_step/40:TT)
#     t_vise_dense = h_step/40:h_step/40:TT
#     t_coarse = collect(0:h_step:TT)

#     # Plot 1: q₁ over time
#     lines!(q1_axis, t_dense, collect(HO_plot.q[:,1]), label="Analytical Solution",color = :black,linestyle = :dash,linewidth = linewidth)
#     lines!(q1_axis, t_vise_dense, vcat(hcat(internal_sol...)[2:end,:]...), label="VISE Continuous Solution", color = :orange)
#     scatter!(q1_axis,t_coarse, collect(HO_imp_sol.q[:, 1]), label="Implicit Midpoint Solution",color = :red)
#     scatter!(q1_axis, t_coarse, collect(HO_cgvi_sol.q[:,1]), label="Galerkin Integrator Solution",color = :green)
#     scatter!(q1_axis,t_coarse, collect(PR_sol.q[:,1]), label="VISE Discrete Solution",color = :blue)
#     vlines!(q1_axis,[30.0],linestyle=:dashdot,color = :purple,label = "Training Region")


#     # Plot 2: p₁ over time
#     lines!(q2_axis, t_dense, collect(HO_plot.p[:,1]), label="Analytical Solution" ,color = :black,linestyle = :dash,linewidth = linewidth)
#     scatter!(q2_axis, t_coarse, collect(HO_cgvi_sol.p[:,1]), label="Galerkin Integrator Solution ",color = :green)
#     scatter!(q2_axis, t_coarse, collect(HO_imp_sol.p[:,1]), label="Implicit Midpoint Solution ",color = :red)
#     scatter!(q2_axis, t_coarse, collect(PR_sol.p[:,1]), label="VISE Discrete Solution ",color = :blue)

#     # Plot 3: Relative Hamiltonian error
#     lines!(ham_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ",color =:blue)
#     lines!(ham_axis, t_coarse, HO_relative_imp_ham_err, label="Implicit Midpoint Solution ",color = :red)
#     lines!(ham_axis, t_coarse, HO_relative_cgvi_ham_err, label="Galerkin Integrator Solution ",color = :green)
#     scatter!(ham_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ",color =:blue)
#     scatter!(ham_axis, t_coarse, HO_relative_imp_ham_err, label="Implicit Midpoint Solution ",color = :red)
#     scatter!(ham_axis, t_coarse, HO_relative_cgvi_ham_err, label="Galerkin Integrator Solution ",color = :green)

#     Label(fig[1:1, 0], "Harmonic Oscillator", rotation = pi/2,
#         fontsize = 30,tellheight = false)

#     # Legend(fig[2, 1:3], q1_axis,orientation = :horizontal,labelsize = 30, framevisible = false,nbanks = 2)    
#     # save("result_figures/HO.pdf", fig)
#     # fig
    
#     # legend_fig = Legend(fig[1,1:3],)
#     q1_axis = Axis(fig[2, 1],xlabel = "Time", ylabel = "q₁",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
#     q2_axis = Axis(fig[2, 2],xlabel = "Time", ylabel = "p₁", xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
#     ham_axis = Axis(fig[2, 3],xlabel = "Time", ylabel = "Relative Hamiltonian Error", xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)

#     h_step = 1.0
#     t_dense = collect(0:h_step/40:TT)
#     t_vise_dense = h_step/40:h_step/40:TT
#     t_coarse = collect(0:h_step:TT)

#     # Plot 1: q₁ over time
#     lines!(q1_axis, t_coarse, collect(pd_ref_sol.q[:,1]), label="Reference Solution ",color = :black,linestyle = :dash,linewidth = linewidth)
#     lines!(q1_axis, t_vise_dense, vcat(hcat(pendulum_internal_sol...)[2:end,:]...), label="VISE Continuous Solution", color = :orange)
#     scatter!(q1_axis,t_coarse, collect(pd_imp_sol.q[:, 1]), label="Implicit Midpoint Solution",color = :red)
#     scatter!(q1_axis, t_coarse, collect(pd_cgvi_sol.q[:,1]), label="Galerkin Integrator Solution",color = :green)
#     scatter!(q1_axis,t_coarse, collect(pendulum_PR_sol.q[:,1]), label="VISE Discrete Solution",color = :blue)
#     vlines!(q1_axis,[100.0],linestyle=:dashdot,color = :purple,label = "Training Region")

#     # Plot 2: p₁ over time
#     lines!(q2_axis, t_coarse, collect(pd_ref_sol.p[:,1]), label="Reference Solution" ,color = :black,linestyle = :dash,linewidth = linewidth)
#     scatter!(q2_axis, t_coarse, collect(pd_cgvi_sol.p[:,1]), label="Galerkin Integrator Solution ",color = :green)
#     scatter!(q2_axis, t_coarse, collect(pd_imp_sol.p[:,1]), label="Implicit Midpoint Solution ",color = :red)
#     scatter!(q2_axis, t_coarse, collect(pendulum_PR_sol.p[:,1]), label="VISE Discrete Solution ",color = :blue)

#     # Plot 3: Relative Hamiltonian error
#     lines!(ham_axis, t_coarse, pendulum_relative_hams_err, label="VISE Discrete Solution ",color =:blue)
#     lines!(ham_axis, t_coarse, pd_relative_imp_ham_err, label="Implicit Midpoint Solution ",color = :red)
#     lines!(ham_axis, t_coarse, pd_relative_cgvi_ham_err, label="Galerkin Integrator Solution ",color = :green)
#     scatter!(ham_axis, t_coarse, pendulum_relative_hams_err, label="VISE Discrete Solution ",color =:blue)
#     scatter!(ham_axis, t_coarse, pd_relative_imp_ham_err, label="Implicit Midpoint Solution ",color = :red)
#     scatter!(ham_axis, t_coarse, pd_relative_cgvi_ham_err, label="Galerkin Integrator Solution ",color = :green)
    
#     Label(fig[2, 0], "Pendulum", rotation = pi/2,
#         fontsize = 30,tellheight = false)
#     # Legend(fig[3, 1:3], q1_axis,orientation = :horizontal,labelsize = 30, 
#     #     framevisible = false,nbanks = 2)    
#     # save("result_figures/hopd.pdf", fig)
#     # fig


#     # legend_fig = Legend(fig[1,1:3],)
#     q1_axis = Axis(fig[3, 1],xlabel = "Time", ylabel = "q₁",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
#     q2_axis = Axis(fig[3, 2],xlabel = "Time", ylabel = "p₁", xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
#     ham_axis = Axis(fig[3, 3],xlabel = "Time", ylabel = "Relative Hamiltonian Error", xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)

#     TT = 150.0
#     h_step = 5.0
#     t_dense = collect(0:h_step/40:TT)
#     t_vise_dense = h_step/40:h_step/40:TT
#     t_coarse = collect(0:h_step:TT)

#     # Plot 1: q₁ over time
#     lines!(q1_axis, t_dense, collect(pp_sol_plot.q[:,1]), label="Reference Solution ",color = :black,linestyle = :dash,linewidth = linewidth)
#     lines!(q1_axis, t_vise_dense, vcat(hcat(pp_internal_sol...)[2:end,:]...), label="VISE Continuous Solution", color = :orange)
#     scatter!(q1_axis,t_coarse, collect(pp_imp_sol.q[:, 1]), label="Implicit Midpoint Solution",color = :red)
#     scatter!(q1_axis, t_coarse, collect(pp_cgvi_sol.q[:,1]), label="Galerkin Integrator Solution",color = :green)
#     scatter!(q1_axis,t_coarse, collect(pp_PR_sol.q[:,1]), label="VISE Discrete Solution",color = :blue)
#     vlines!(q1_axis,[100.0],linestyle=:dashdot,color = :purple,label = "Training Region")

#     # Plot 2: p₁ over time
#     lines!(q2_axis, t_dense, collect(pp_sol_plot.p[:,1]), label="Reference Solution" ,color = :black,linestyle = :dash,linewidth = linewidth)
#     scatter!(q2_axis, t_coarse, collect(pp_cgvi_sol.p[:,1]), label="Galerkin Integrator Solution ",color = :green)
#     scatter!(q2_axis, t_coarse, collect(pp_imp_sol.p[:,1]), label="Implicit Midpoint Solution ",color = :red)
#     scatter!(q2_axis, t_coarse, collect(pp_PR_sol.p[:,1]), label="VISE Discrete Solution ",color = :blue)

#     # Plot 3: Relative Hamiltonian error
#     lines!(ham_axis, t_coarse, pp_relative_hams_err, label="VISE Discrete Solution ",color =:blue)
#     lines!(ham_axis, t_coarse, pp_relative_imp_ham_err, label="Implicit Midpoint Solution ",color = :red)
#     lines!(ham_axis, t_coarse, pp_relative_cgvi_ham_err, label="Galerkin Integrator Solution ",color = :green)
#     scatter!(ham_axis, t_coarse, pp_relative_hams_err, label="VISE Discrete Solution ",color =:blue)
#     scatter!(ham_axis, t_coarse, pp_relative_imp_ham_err, label="Implicit Midpoint Solution ",color = :red)
#     scatter!(ham_axis, t_coarse, pp_relative_cgvi_ham_err, label="Galerkin Integrator Solution ",color = :green)
#     Label(fig[3, 0], "Perturbed Pendulum", rotation = pi/2,
#         fontsize = 30,tellheight = false)

#     # HenonHeiles_Potential 
#     # legend_fig = Legend(fig[1,1:3],)
#     q1_axis = Axis(fig[4, 1],xlabel = "Time", ylabel = "q₁",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
#     p1_axis = Axis(fig[4, 2],xlabel = "Time", ylabel = "p₁", xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
#     q2_axis = Axis(fig[5, 1],xlabel = "Time", ylabel = "q₂",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
#     p2_axis = Axis(fig[5, 2],xlabel = "Time", ylabel = "p₂", xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
#     ham_axis = Axis(fig[5, 3],xlabel = "Time", ylabel = "Relative Hamiltonian Error", xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)

#     h_step = 1.0
#     TT = 100.0
#     t_dense = collect(0:h_step/40:TT)
#     t_vise_dense = h_step/40:h_step/40:TT
#     t_coarse = collect(0:h_step:TT)

#     internal_q1 = Array{Vector}(undef,Int(TT/h_step))
#     internal_q2 = Array{Vector}(undef,Int(TT/h_step))

#     for i in 1:Int(TT/h_step)
#         internal_q1[i] = HH_internal_sol[i][:,1]
#         internal_q2[i] = HH_internal_sol[i][:,2]
#     end

#     # Plot 1: q₁ over time
#     lines!(q1_axis, t_dense, collect(HH_sol_plot.q[:,1]), label="Reference Solution ",color = :black,linestyle = :dash,linewidth = linewidth)
#     lines!(q1_axis, t_vise_dense, vcat(hcat(internal_q1...)[2:end,:]...), label="VISE Continuous Solution", color = :orange)
#     scatter!(q1_axis,t_coarse, collect(HH_imp_sol.q[:, 1]), label="Implicit Midpoint Solution",color = :red)
#     scatter!(q1_axis, t_coarse, collect(HH_cgvi_sol.q[:,1]), label="Galerkin Integrator Solution",color = :green)
#     scatter!(q1_axis,t_coarse, collect(HH_PR_sol.q[:,1]), label="VISE Discrete Solution",color = :blue)
#     vlines!(q1_axis,[10.0],linestyle=:dashdot,color = :purple,label = "Training Region")

#     # Plot 2: p₁ over time
#     lines!(p1_axis, t_dense, collect(HH_sol_plot.p[:,1]), label="Reference Solution" ,color = :black,linestyle = :dash,linewidth = linewidth)
#     scatter!(p1_axis, t_coarse, collect(HH_cgvi_sol.p[:,1]), label="Galerkin Integrator Solution ",color = :green)
#     scatter!(p1_axis, t_coarse, collect(HH_imp_sol.p[:,1]), label="Implicit Midpoint Solution ",color = :red)
#     scatter!(p1_axis, t_coarse, collect(HH_PR_sol.p[:,1]), label="VISE Discrete Solution ",color = :blue)


#     # Plot 1: q2 over time
#     lines!(q2_axis, t_dense, collect(HH_sol_plot.q[:,2]), label="Reference Solution ",color = :black,linestyle = :dash,linewidth = linewidth)
#     lines!(q2_axis, t_vise_dense, vcat(hcat(internal_q2...)[2:end,:]...), label="VISE Continuous Solution", color = :orange)
#     scatter!(q2_axis,t_coarse, collect(HH_imp_sol.q[:, 2]), label="Implicit Midpoint Solution",color = :red)
#     scatter!(q2_axis, t_coarse, collect(HH_cgvi_sol.q[:,2]), label="Galerkin Integrator Solution",color = :green)
#     scatter!(q2_axis,t_coarse, collect(HH_PR_sol.q[:,2]), label="VISE Discrete Solution",color = :blue)
#     vlines!(q2_axis,[10.0],linestyle=:dashdot,color = :purple,label = "Training Region")

#     # Plot 2: p2 over time
#     lines!(p2_axis, t_dense, collect(HH_sol_plot.p[:,2]), label="Reference Solution" ,color = :black,linestyle = :dash,linewidth = linewidth)
#     scatter!(p2_axis, t_coarse, collect(HH_cgvi_sol.p[:,2]), label="Galerkin Integrator Solution ",color = :green)
#     scatter!(p2_axis, t_coarse, collect(HH_imp_sol.p[:,2]), label="Implicit Midpoint Solution ",color = :red)
#     scatter!(p2_axis, t_coarse, collect(HH_PR_sol.p[:,2]), label="VISE Discrete Solution ",color = :blue)


#     # Plot 3: Relative Hamiltonian error
#     lines!(ham_axis, t_coarse, HH_relative_hams_err, label="VISE Discrete Solution ",color =:blue)
#     lines!(ham_axis, t_coarse, HH_relative_imp_ham_err, label="Implicit Midpoint Solution ",color = :red)
#     lines!(ham_axis, t_coarse, HH_relative_cgvi_ham_err, label="Galerkin Integrator Solution ",color = :green)
#     scatter!(ham_axis, t_coarse, HH_relative_hams_err, label="VISE Discrete Solution ",color =:blue)
#     scatter!(ham_axis, t_coarse, HH_relative_imp_ham_err, label="Implicit Midpoint Solution ",color = :red)
#     scatter!(ham_axis, t_coarse, HH_relative_cgvi_ham_err, label="Galerkin Integrator Solution ",color = :green)
#     Label(fig[4:5, 0], "Hénon-Heiles Potential", rotation = pi/2,
#         fontsize = 30,tellheight = false)


#     Legend(fig[6, 1:3], q1_axis,orientation = :horizontal,labelsize = 30, 
#         framevisible = false,nbanks = 2)    
#     save("result_figures/full.pdf", fig)
#     fig


# end


TT = 100.0
h_step = 0.3
pendulum_lode = GeometricProblems.Pendulum.lodeproblem(tspan = (0,TT),tstep = h_step)
pd_ref_sol = integrate(pendulum_lode, Gauss(8))

TT = 100.0
h_step = 0.1
pendulum_lode = GeometricProblems.Pendulum.lodeproblem(tspan = (0,TT),tstep = h_step)
continuous_sol = integrate(pendulum_lode, Gauss(8))

pd_ref_fig = Figure(size = (800,1200), linewidth = 2, markersize = 8)
q_axis = Axis(pd_ref_fig[1,1],xlabel = "t", ylabel = "q",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
lin1= lines!(q_axis, collect(0:0.1:TT), collect(continuous_sol.q[:,1]), label = "Reference Solution", linewidth = 2,color = :red)
sca1 = scatter!(q_axis,collect(0:0.3:100),collect(pd_ref_sol.q[:,1]),label = "Observation Points")
Label(pd_ref_fig[1,0],"Pendulum", rotation = pi/2,fontsize = 30,tellheight = false)

lp_axis = Axis(pd_ref_fig[2,1],xlabel = "x", ylabel = "u",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
input = randn(100) .-2
labels = exact_u.(0,input)
scatter!(lp_axis,input,labels,label = "Observation Points")
lin2 = lines!(lp_axis,collect(-5:0.01:1),exact_u.(0,collect(-5:0.01:1)),label = "Initial Condition",linewidth = 2,color = :green)
Label(pd_ref_fig[2,0],"Linear Transport Equation", rotation = pi/2,fontsize = 30,tellheight = false)

Legend(pd_ref_fig[3, 1], [lin1,lin2,sca1],["Pendulum Reference Solution",
        "Linear Transport Equation Initial Condition",
        "Observation Points"],
orientation = :horizontal,labelsize = 30, 
        framevisible = false,nbanks = 3)  
save("result_figures/pd_lp_illu.svg", pd_ref_fig)
pd_ref_fig