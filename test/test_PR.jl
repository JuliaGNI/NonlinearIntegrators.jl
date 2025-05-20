using Symbolics
using CompactBasisFunctions
using NonlinearIntegrators
using QuadratureRules

using GeometricProblems
using GeometricIntegrators
using Plots



function PR_plot_1d(PR_sol, internal_sol, pref,relative_ham_err, h_step, TT, title_name)
    p = plot(layout=@layout([a; b; c]), label="", size=(700, 700), plot_title=title_name) 

    plot!(p[1],collect(0:h_step/40:TT),collect(pref.q[:,1]), label="Reference Solution",xaxis="time", yaxis="q₁")
    plot!(p[1],h_step/40:h_step/40:TT, vcat(hcat(internal_sol...)[2:end,:]...),label = "SINDy Solution")
    scatter!(p[1],collect(0:h_step:TT),collect(PR_sol.q[:,1]), label="SINDy Discrete Solution")

    plot!(p[2],collect(0:h_step/40:TT),collect(pref.p[:,1]), label="Reference Solution",xaxis="time", yaxis="p₁")
    scatter!(p[2],collect(0:h_step:TT),collect(PR_sol.p[:,1]), label="SINDy Discrete Solution")

    plot!(p[3], 0:h_step:TT, relative_ham_err, label = "SINDy Solution", xaxis="time", yaxis="Relative Hamiltonian error")
    savefig("result_figures/$(title_name).pdf")
end

function PR_plot_2d(PR_sol, internal_sol, pref,relative_ham_err, h_step, TT, title_name)
    internal_q1 = Array{Vector}(undef,Int(TT/h_step))
    internal_q2 = Array{Vector}(undef,Int(TT/h_step))

    for i in 1:Int(TT/h_step)
        internal_q1[i] = internal_sol[i][:,1]
        internal_q2[i] = internal_sol[i][:,2]
    end

    # Figures for the paper
    p = plot(layout=@layout([a b; c d; e]), label="", size=(700, 700), plot_title=title_name)# d;e

    plot!(p[1], h_step/40:h_step/40:TT, vcat(hcat(internal_q1...)[2:end,:]...), label="SINDy Solution", xaxis="time", yaxis="q₁")
    plot!(p[1], 0:h_step/40:TT, collect(pref.q[:, 1]), label="Reference Solution")
    scatter!(p[1],collect(0:h_step:TT),collect(PR_sol.q[:,1]), label="SINDy Discrete Solution")

    plot!(p[2], h_step/40:h_step/40:TT, vcat(hcat(internal_q2...)[2:end,:]...), label="SINDy Solution", xaxis="time", yaxis="q₂")
    plot!(p[2], 0:h_step/40:TT, collect(pref.q[:, 2]), label="Reference Solution")
    scatter!(p[2],collect(0:h_step:TT),collect(PR_sol.q[:,2]), label="SINDy Discrete Solution")

    plot!(p[3], 0:h_step:TT, collect(PR_sol.p[:, 1]), label="SINDy Solution", xaxis="time", yaxis="p₁")
    plot!(p[3], 0:h_step/40:TT, collect(pref.p[:, 1]), label="Reference Solution")

    plot!(p[4], 0:h_step:TT, collect(PR_sol.p[:, 2]), label="SINDy Solution", xaxis="time", yaxis="p₂")
    plot!(p[4], 0:h_step/40:TT, collect(pref.p[:, 2]), label="Reference Solution")

    plot!(p[5], 0:h_step:TT, relative_ham_err, label="SINDy Solution", xaxis="time", yaxis="Relative Hamiltonian error")
    savefig(p, "result_figures/$(title_name).pdf")
end


# Harmonic Oscillator
begin 
    println("Start to run Harmonic Oscillator Problem with PR_Integrator!")
    @variables W[1:3] ttt
    q_expr = W[1] *sin(W[2]* ttt + W[3])

    PRB = PR_Basis{Float64}([q_expr], [W], ttt,1)
    TT = 150.0
    h_step = 5.0
    HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(tspan = (0,TT),tstep = h_step)
    initial_hamiltonian = GeometricProblems.HarmonicOscillator.hamiltonian(0.0, HO_lode.ics.q, HO_lode.ics.p, HO_lode.parameters)
    HO_truth = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(tspan = (0,TT),tstep = h_step))

    R = 32
    QGau4 = QuadratureRules.GaussLegendreQuadrature(R)
    PR_Int = PR_Integrator(PRB, QGau4,[[-0.5000433352162222,0.705350078478666,-1.5678140333370576]]) # Pass the init W into the integrator instead of basis                                               
    # PR_Int = PR_Integrator(PRB, QGau4,[[-0.500,sqrt(0.5),-pi/2]])                           

    PR_sol,internal_sol,x_list = integrate(HO_lode, PR_Int)
    @show relative_maximum_error(PR_sol.q,HO_truth.q)

    hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(PR_sol.q[:]), collect(PR_sol.p[:]))]
    relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)

    HO_plot = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(tspan = (0,TT),tstep = h_step/40))
    PR_plot_1d(PR_sol, internal_sol, HO_plot, relative_hams_err, h_step, TT, "HarmonicOscillator,h$(h_step)_T$(TT)_R$(R)")
    println("Finish integrating Harmonic Oscillator Problem with PR_Integrator!, Figure Saved!")

end


#### Pendulum
begin
    println("Start to run Pendulum Problem with PR_Integrator!")

    TT = 150.0
    h_step = 1.0
    pendulum_lode = GeometricProblems.Pendulum.lodeproblem(tspan = (0,TT),tstep = h_step)
    ref_sol = integrate(pendulum_lode, Gauss(8))
    initial_hamiltonian = GeometricProblems.Pendulum.hamiltonian(0.0, pendulum_lode.ics.q, pendulum_lode.ics.p, pendulum_lode.parameters)
    @show initial_hamiltonian

    # f_try(x₁) = cos((x₁ * 0.45644) - 1.1466) * 1.1931
    @variables W[1:3] ttt
    q_expr = W[1] *cos(W[2]* ttt + W[3])
    PRB = PR_Basis{Float64}([q_expr], [W], ttt,1)
    R = 8
    QGau = QuadratureRules.GaussLegendreQuadrature(R)
    PR_Int = PR_Integrator(PRB, QGau,[[1.1931,0.45644,-1.1466]])

    PR_sol,internal_sol,x_list = integrate(pendulum_lode, PR_Int)
    @show relative_maximum_error(PR_sol.q,ref_sol.q)

    hams = [GeometricProblems.Pendulum.hamiltonian(0.0, q, p, pendulum_lode.parameters) for (q, p) in zip(collect(PR_sol.q[:]), collect(PR_sol.p[:]))]
    relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)

    pendulum_plot = GeometricProblems.Pendulum.lodeproblem(tspan = (0,TT),tstep = h_step/40)
    sol_plot = integrate(pendulum_plot, Gauss(8))
    PR_plot_1d(PR_sol, internal_sol, sol_plot, relative_hams_err, h_step, TT, "Pendulum,h$(h_step)_T$(TT)_R$(R)")
    println("Finish integrating Pendulum Problem with PR_Integrator!, Figure Saved!")

end


#### Perturbed Pendulum
begin
    println("Start to run Perturbed Pendulum Problem with PR_Integrator!")

    TT = 150.0
    h_step = 5.0
    lode = GeometricProblems.PerturbedPendulum.lodeproblem(tspan = (0,TT),tstep = h_step)
    ref_sol = integrate(lode, Gauss(8))
    initial_hamiltonian = GeometricProblems.PerturbedPendulum.hamiltonian(0.0, lode.ics.q, lode.ics.p, lode.parameters)
    @show initial_hamiltonian

    R = 16
    QGau = QuadratureRules.GaussLegendreQuadrature(R)

    @variables W[1:3] ttt
    q_expr = W[1] *cos(W[2]* ttt + W[3])
    PRB = PR_Basis{Float64}([q_expr], [W], ttt,1)
    PR_Int = PR_Integrator(PRB, QGau,[[-0.51941,-0.47405,2.8713]])

    PR_sol,internal_sol,x_list = integrate(lode, PR_Int)
    @show relative_maximum_error(PR_sol.q,ref_sol.q)

    hams = [GeometricProblems.PerturbedPendulum.hamiltonian(0.0, q, p, lode.parameters) for (q, p) in zip(collect(PR_sol.q[:]), collect(PR_sol.p[:]))]
    relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)

    lode_plot = GeometricProblems.PerturbedPendulum.lodeproblem(tspan = (0,TT),tstep = h_step/40)
    sol_plot = integrate(lode_plot, Gauss(8))
    PR_plot_1d(PR_sol, internal_sol, sol_plot, relative_hams_err, h_step, TT, "Perturbed_Pendulum,h$(h_step)_T$(TT)_R$(R)")
    println("Finish integrating Perturbed Pendulum Problem with PR_Integrator!, Figure Saved!")
end


#### Henon Heiles
begin     
    println("Start to run Henon Heiles Problem with PR_Integrator!")

    TT = 150.0
    h_step = 1.0
    HHlode = GeometricProblems.HenonHeilesPotential.lodeproblem([0.1,0.1],[0.1,0.1],tspan = (0,TT),tstep = h_step)
    ref_sol = integrate(HHlode, Gauss(8))

    initial_hamiltonian = GeometricProblems.HenonHeilesPotential.hamiltonian(0.0, HHlode.ics.q, HHlode.ics.p, HHlode.parameters)
    @show initial_hamiltonian

    R = 8
    QGau = QuadratureRules.GaussLegendreQuadrature(R)

    @variables W1[1:4] ttt
    #(0.14831 * cos(-0.64812 + x₁)) - 0.018712
    q₁_expr = W1[1] *cos(W1[2]* ttt + W1[3]) + W1[4]

    @variables W2[1:4]
    # 0.14298 * cos(- 0.97215 * x₁+ 0.7615)))-0.0013983
    q₂_expr = W2[1] *cos(W2[2]* ttt + W2[3]) + W2[4]

    PRB = PR_Basis{Float64}([q₁_expr,q₂_expr], [W1,W2], ttt,2)

    PR_Int = PR_Integrator(PRB, QGau,[[0.14831,1.0,-0.64812,- 0.018712],[0.14298,- 0.97215,0.7615,-0.0013983]]) # Pass the init W into the integrator instead of basis                                               
    PR_sol,internal_sol,x_list = integrate(HHlode, PR_Int)
    @show relative_maximum_error(PR_sol.q,ref_sol.q)

    hams = [GeometricProblems.HenonHeilesPotential.hamiltonian(0.0, q, p, HHlode.parameters) for (q, p) in zip(collect(PR_sol.q[:]), collect(PR_sol.p[:]))]
    relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)

    HHlode_plot = GeometricProblems.HenonHeilesPotential.lodeproblem([0.1,0.1],[0.1,0.1],tspan = (0,TT),tstep = h_step/40)
    sol_plot = integrate(HHlode_plot, Gauss(8))

    PR_plot_2d(PR_sol, internal_sol, sol_plot,relative_hams_err, h_step, TT, "HenonHeiles_Potential,h$(h_step)_T$(TT)_R$(R)")
    println("Finish integrating HenonHeiles Potential Problem with PR_Integrator!, Figure Saved!")

end

