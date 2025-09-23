using CairoMakie
using JLD2

using GeometricIntegrators
using GeometricProblems
using QuadratureRules
using CompactBasisFunctions

HarmonicOscillator_h1 = load("/Users/zeyuanli/Desktop/untitled folder 2/HO/Backtracking2_R8_h1.00_iter1000_fabs4.44e-16_fsuc4.44e-16_TT200.jld2")
HarmonicOscillator_h2 = load("/Users/zeyuanli/Desktop/untitled folder 2/HO/Backtracking2_R16_h2.00_iter1000_fabs1.78e-15_fsuc1.78e-15_TT200.jld2")
HarmonicOscillator_h5 = load("/Users/zeyuanli/Desktop/untitled folder 2/HO/Backtracking2_R8_h5.00_iter1000_fabs1.78e-15_fsuc1.78e-15_TT200.jld2")

fig = Figure(linewidth=2, markersize=8, size=(1200, 1200))#size = (2200,3800)
label_size = 20
tick_size = 15
label_font_size = 20
linewidth = 3
TT = 200

initial_expression(t) = -0.5000433352162222 *sin(0.705350078478666t-1.5678140333370576)
initial_p(t) = -0.5000433352162222*0.705350078478666*cos(0.705350078478666t-1.5678140333370576)


# max_ham_err = maximum(relative_hams_err_init_expr) # 0.004789170283308719
# min_ham_err = minimum(relative_hams_err_init_expr) # 1.290448580215653e-6

begin # h = 1

    h_step = 1.0
    HO_plot = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(timespan=(0, TT), timestep=h_step / 40))
    HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(timespan=(0, TT), timestep=h_step)
    HO_imp_sol = integrate(HO_lode, ImplicitMidpoint())

    QGau = GaussLegendreQuadrature(8)
    BGau = Lagrange(QuadratureRules.nodes(QGau))
    HO_cgvi_sol = integrate(HO_lode, CGVI(BGau, QGau))

    t_dense = collect(0:h_step/40:TT)
    t_vise_dense = h_step/40:h_step/40:TT
    t_coarse = collect(0:h_step:TT)

    init_q_list = [initial_expression(t) for t in t_coarse]
    init_p_list = [initial_p(t) for t in t_coarse]
    ham_init_expr = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(init_q_list, init_p_list)]
    relative_hams_err_init_expr = abs.((ham_init_expr .- initial_hamiltonian) / initial_hamiltonian)# Plot 3: Relative Hamiltonian error

    # legend_fig = Legend(fig[1,1:3],)
    q1_axis = Axis(fig[1, 1], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)
    p1_axis = Axis(fig[1, 2], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)
    ham_axis = Axis(fig[1, 3], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)
    lines!(q1_axis, t_dense, collect(HO_plot.q[:, 1]), label="Analytical Solution", color=:black, linestyle=:dash, linewidth=linewidth)

    HO_h1_internal_sol = HarmonicOscillator_h1["HO_internal_sol"]
    lines!(q1_axis, t_vise_dense, vcat(hcat(HO_h1_internal_sol...)[2:end, :]...), label="VISE Continuous Solution", color=:orange)
    scatter!(q1_axis, t_coarse, collect(HO_imp_sol.q[:, 1]), label="Implicit Midpoint Solution", color=:red)
    scatter!(q1_axis, t_coarse, collect(HO_cgvi_sol.q[:, 1]), label="Galerkin Integrator Solution", color=:green)
    # scatter!(q1_axis, t_coarse, init_q_list, label="Initial Expression", color=:pink)
    PR_sol_q = HarmonicOscillator_h1["HO_PR_sol_q"]
    scatter!(q1_axis, t_coarse, PR_sol_q, label="VISE Discrete Solution", color=:blue)
    vlines!(q1_axis, [30.0], linestyle=:dashdot, color=:purple, label="Training Region")
    r = rect(Vec(0,0), 30.0, 1)
    poly!(q1_axis,r, color = (:gray, 0.1), strokewidth=0)

    PR_sol_p = HarmonicOscillator_h1["HO_PR_sol_p"]
    lines!(p1_axis, t_dense, collect(HO_plot.p[:, 1]), label="Analytical Solution", color=:black, linestyle=:dash, linewidth=linewidth)
    scatter!(p1_axis, t_coarse, collect(HO_cgvi_sol.p[:, 1]), label="Galerkin Integrator Solution ", color=:green)
    # scatter!(p1_axis, t_coarse, init_p_list, label="Initial Expression", color=:pink)
    scatter!(p1_axis, t_coarse, collect(HO_imp_sol.p[:, 1]), label="Implicit Midpoint Solution ", color=:red)
    scatter!(p1_axis, t_coarse, PR_sol_p, label="VISE Discrete Solution ", color=:blue)

    hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(PR_sol_q, PR_sol_p)]
    initial_hamiltonian = hams[1]
    relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)# Plot 3: Relative Hamiltonian error

    HO_imp_ham = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_imp_sol.q[:]), collect(HO_imp_sol.p[:]))]
    HO_relative_imp_ham_err = abs.((HO_imp_ham .- initial_hamiltonian) / initial_hamiltonian)

    HO_cgvi_ham = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_cgvi_sol.q[:]), collect(HO_cgvi_sol.p[:]))]
    HO_relative_cgvi_ham_err = abs.((HO_cgvi_ham .- initial_hamiltonian) / initial_hamiltonian)

    lines!(ham_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ", color=:blue)
    lines!(ham_axis, t_coarse, HO_relative_imp_ham_err, label="Implicit Midpoint Solution ", color=:red)
    lines!(ham_axis, t_coarse, HO_relative_cgvi_ham_err, label="Galerkin Integrator Solution ", color=:green)
    # lines!(ham_axis, t_coarse, relative_hams_err_init_expr, label="Initial Expression", color=:pink)
    scatter!(ham_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ", color=:blue)
    scatter!(ham_axis, t_coarse, HO_relative_imp_ham_err, label="Implicit Midpoint Solution ", color=:red)
    scatter!(ham_axis, t_coarse, HO_relative_cgvi_ham_err, label="Galerkin Integrator Solution ", color=:green)
    # scatter!(ham_axis, t_coarse, relative_hams_err_init_expr, label="Initial Expression", color=:pink)
    Label(fig[1, 0], "h = 1.0", rotation=pi / 2, fontsize=label_font_size, tellheight=false)

end

begin # h = 2
    h_step = 2.0
    HO_plot = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(timespan=(0, TT), timestep=h_step / 40))
    HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(timespan=(0, TT), timestep=h_step)
    HO_imp_sol = integrate(HO_lode, ImplicitMidpoint())

    QGau = GaussLegendreQuadrature(16)
    BGau = Lagrange(QuadratureRules.nodes(QGau))
    HO_cgvi_sol = integrate(HO_lode, CGVI(BGau, QGau))

    t_dense = collect(0:h_step/40:TT)
    t_vise_dense = h_step/40:h_step/40:TT
    t_coarse = collect(0:h_step:TT)
    init_q_list = [initial_expression(t) for t in t_coarse]
    init_p_list = [initial_p(t) for t in t_coarse]
    ham_init_expr = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(init_q_list, init_p_list)]
    relative_hams_err_init_expr = abs.((ham_init_expr .- initial_hamiltonian) / initial_hamiltonian)# Plot 3: Relative Hamiltonian error
    # legend_fig = Legend(fig[1,1:3],)
    q2_axis = Axis(fig[2, 1], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)
    p2_axis = Axis(fig[2, 2], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)
    ham2_axis = Axis(fig[2, 3], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)
    lines!(q2_axis, t_dense, collect(HO_plot.q[:, 1]), label="Analytical Solution", color=:black, linestyle=:dash, linewidth=linewidth)

    HO_h2_internal_sol = HarmonicOscillator_h2["HO_internal_sol"]
    lines!(q2_axis, t_vise_dense, vcat(hcat(HO_h2_internal_sol...)[2:end, :]...), label="VISE Continuous Solution", color=:orange)
    scatter!(q2_axis, t_coarse, collect(HO_imp_sol.q[:, 1]), label="Implicit Midpoint Solution", color=:red)
    scatter!(q2_axis, t_coarse, collect(HO_cgvi_sol.q[:, 1]), label="Galerkin Integrator Solution", color=:green)
    # scatter!(q2_axis, t_coarse, init_q_list, label="Initial Expression", color=:pink)
    PR_sol_q = HarmonicOscillator_h2["HO_PR_sol_q"]
    scatter!(q2_axis, t_coarse, PR_sol_q, label="VISE Discrete Solution", color=:blue)
    vlines!(q2_axis, [30.0], linestyle=:dashdot, color=:purple, label="Training Region")


    PR_sol_p = HarmonicOscillator_h2["HO_PR_sol_p"]
    lines!(p2_axis, t_dense, collect(HO_plot.p[:, 1]), label="Analytical Solution", color=:black, linestyle=:dash, linewidth=linewidth)
    scatter!(p2_axis, t_coarse, collect(HO_cgvi_sol.p[:, 1]), label="Galerkin Integrator Solution ", color=:green)
    # scatter!(p2_axis, t_coarse, init_p_list, label="Initial Expression", color=:pink)
    scatter!(p2_axis, t_coarse, collect(HO_imp_sol.p[:, 1]), label="Implicit Midpoint Solution ", color=:red)
    scatter!(p2_axis, t_coarse, PR_sol_p, label="VISE Discrete Solution ", color=:blue)

    hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(PR_sol_q, PR_sol_p)]
    initial_hamiltonian = hams[1]
    relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)# Plot 3: Relative Hamiltonian error

    HO_imp_ham = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_imp_sol.q[:]), collect(HO_imp_sol.p[:]))]
    HO_relative_imp_ham_err = abs.((HO_imp_ham .- initial_hamiltonian) / initial_hamiltonian)

    HO_cgvi_ham = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_cgvi_sol.q[:]), collect(HO_cgvi_sol.p[:]))]
    HO_relative_cgvi_ham_err = abs.((HO_cgvi_ham .- initial_hamiltonian) / initial_hamiltonian)

    lines!(ham2_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ", color=:blue)
    lines!(ham2_axis, t_coarse, HO_relative_imp_ham_err, label="Implicit Midpoint Solution ", color=:red)
    lines!(ham2_axis, t_coarse, HO_relative_cgvi_ham_err, label="Galerkin Integrator Solution ", color=:green)
    # lines!(ham2_axis, t_coarse, relative_hams_err_init_expr, label="Initial Expression", color=:pink)
    scatter!(ham2_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ", color=:blue)
    scatter!(ham2_axis, t_coarse, HO_relative_imp_ham_err, label="Implicit Midpoint Solution ", color=:red)
    scatter!(ham2_axis, t_coarse, HO_relative_cgvi_ham_err, label="Galerkin Integrator Solution ", color=:green)
    # scatter!(ham2_axis, t_coarse, relative_hams_err_init_expr, label="Initial Expression", color=:pink)
    Label(fig[2, 0], "h = 2.0", rotation=pi / 2, fontsize=label_font_size, tellheight=false)

end

begin # h = 5.0
    h_step = 5.0
    HO_plot = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(timespan=(0, TT), timestep=h_step / 40))
    HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(timespan=(0, TT), timestep=h_step)
    HO_imp_sol = integrate(HO_lode, ImplicitMidpoint())

    QGau = GaussLegendreQuadrature(8)
    BGau = Lagrange(QuadratureRules.nodes(QGau))
    HO_cgvi_sol = integrate(HO_lode, CGVI(BGau, QGau))

    t_dense = collect(0:h_step/40:TT)
    t_vise_dense = h_step/40:h_step/40:TT
    t_coarse = collect(0:h_step:TT)
    init_q_list = [initial_expression(t) for t in t_coarse]
    init_p_list = [initial_p(t) for t in t_coarse]
    ham_init_expr = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(init_q_list, init_p_list)]
    relative_hams_err_init_expr = abs.((ham_init_expr .- initial_hamiltonian) / initial_hamiltonian)# Plot 3: Relative Hamiltonian error
    # legend_fig = Legend(fig[1,1:3],)
    q5_axis = Axis(fig[3, 1], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)
    p5_axis = Axis(fig[3, 2], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)
    ham5_axis = Axis(fig[3, 3], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)
    lines!(q5_axis, t_dense, collect(HO_plot.q[:, 1]), label="Analytical Solution", color=:black, linestyle=:dash, linewidth=linewidth)

    HO_h5_internal_sol = HarmonicOscillator_h5["HO_internal_sol"]
    lines!(q5_axis, t_vise_dense, vcat(hcat(HO_h5_internal_sol...)[2:end, :]...), label="VISE Continuous Solution", color=:orange)
    scatter!(q5_axis, t_coarse, collect(HO_imp_sol.q[:, 1]), label="Implicit Midpoint Solution", color=:red)
    scatter!(q5_axis, t_coarse, collect(HO_cgvi_sol.q[:, 1]), label="Galerkin Integrator Solution", color=:green)
    # scatter!(q5_axis, t_coarse, init_q_list, label="Initial Expression", color=:pink)
    PR_sol_q = HarmonicOscillator_h5["HO_PR_sol_q"]
    scatter!(q5_axis, t_coarse, PR_sol_q, label="VISE Discrete Solution", color=:blue)
    vlines!(q5_axis, [30.0], linestyle=:dashdot, color=:purple, label="Training Region")


    PR_sol_p = HarmonicOscillator_h5["HO_PR_sol_p"]
    lines!(p5_axis, t_dense, collect(HO_plot.p[:, 1]), label="Analytical Solution", color=:black, linestyle=:dash, linewidth=linewidth)
    scatter!(p5_axis, t_coarse, collect(HO_cgvi_sol.p[:, 1]), label="Galerkin Integrator Solution ", color=:green)
    # scatter!(p5_axis, t_coarse, init_p_list, label="Initial Expression", color=:pink)
    scatter!(p5_axis, t_coarse, collect(HO_imp_sol.p[:, 1]), label="Implicit Midpoint Solution ", color=:red)
    scatter!(p5_axis, t_coarse, PR_sol_p, label="VISE Discrete Solution ", color=:blue)

    hams = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(PR_sol_q, PR_sol_p)]
    initial_hamiltonian = hams[1]
    relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)# Plot 3: Relative Hamiltonian error

    HO_imp_ham = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_imp_sol.q[:]), collect(HO_imp_sol.p[:]))]
    HO_relative_imp_ham_err = abs.((HO_imp_ham .- initial_hamiltonian) / initial_hamiltonian)

    HO_cgvi_ham = [GeometricProblems.HarmonicOscillator.hamiltonian(0, q, p, HO_lode.parameters) for (q, p) in zip(collect(HO_cgvi_sol.q[:]), collect(HO_cgvi_sol.p[:]))]
    HO_relative_cgvi_ham_err = abs.((HO_cgvi_ham .- initial_hamiltonian) / initial_hamiltonian)

    lines!(ham5_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ", color=:blue)
    lines!(ham5_axis, t_coarse, HO_relative_imp_ham_err, label="Implicit Midpoint Solution ", color=:red)
    lines!(ham5_axis, t_coarse, HO_relative_cgvi_ham_err, label="Galerkin Integrator Solution ", color=:green)
    # lines!(ham5_axis, t_coarse, relative_hams_err_init_expr, label="Initial Expression", color=:pink)
    scatter!(ham5_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ", color=:blue)
    scatter!(ham5_axis, t_coarse, HO_relative_imp_ham_err, label="Implicit Midpoint Solution ", color=:red)
    scatter!(ham5_axis, t_coarse, HO_relative_cgvi_ham_err, label="Galerkin Integrator Solution ", color=:green)
    # scatter!(ham5_axis, t_coarse, relative_hams_err_init_expr, label="Initial Expression", color=:pink)
    Label(fig[3, 0], "h = 5.0", rotation=pi / 2, fontsize=label_font_size, tellheight=false)
end

Label(fig[0, 1], "q₁", fontsize=label_font_size, tellwidth=false)
Label(fig[0, 2], "p₁", fontsize=label_font_size, tellwidth=false)
Label(fig[0, 3], "Relative Hamiltonian Error", fontsize=label_font_size, tellwidth=false)

Label(fig[4, 1], "time", fontsize=label_font_size, tellwidth=false)
Label(fig[4, 2], "time", fontsize=label_font_size, tellwidth=false)
Label(fig[4, 3], "time", fontsize=label_font_size, tellwidth=false)

Legend(fig[5, 1:3], q1_axis, orientation=:horizontal, labelsize=label_font_size,
    framevisible=false, nbanks=2)

save("result_figures/HOh125_without_initial_expression.pdf", fig)



PPD_h1 = load("/Users/zeyuanli/Desktop/untitled folder 2/ppd/Backtracking2_R8_h1.00_iter1000_fabs1.78e-15_fsuc4.44e-16_TT200.jld2")
PPD_h2 = load("/Users/zeyuanli/Desktop/untitled folder 2/ppd/Backtracking2_R16_h2.00_iter1000_fabs1.78e-15_fsuc1.78e-15_TT200.jld2")
PPD_h5 = load("/Users/zeyuanli/Desktop/untitled folder 2/ppd/Backtracking2_R16_h5.00_iter1000_fabs4.44e-16_fsuc4.44e-16_TT200.jld2")


function PDD_init_q(t;params = [-0.51941, -0.47405, 2.8713])
    params[1] * cos(params[2] * t + params[3])   
end

function PDD_init_p(t,q;params = [-0.51941, -0.47405, 2.8713])
    - params[1] * params[2]* sin(params[2] * t + params[3]) + q*(0.3*0.5*sin(2pi/3))
end

PPD_fig = Figure(linewidth=2, markersize=8, size=(1200, 1200))#size = (2200,3800)
begin # h= 1.0
    h_step = 1.0
    PPD_ref = GeometricProblems.PerturbedPendulum.lodeproblem(timespan=(0, TT), timestep=h_step / 40)
    PPD_plot = integrate(PPD_ref, Gauss(8))
    PPD_lode = GeometricProblems.PerturbedPendulum.lodeproblem(timespan=(0, TT), timestep=h_step)
    PPD_imp_sol = integrate(PPD_lode, ImplicitMidpoint())

    QGau = GaussLegendreQuadrature(8)
    BGau = Lagrange(QuadratureRules.nodes(QGau))
    PPD_cgvi_sol = integrate(PPD_lode, CGVI(BGau, QGau))

    t_dense = collect(0:h_step/40:TT)
    t_vise_dense = h_step/40:h_step/40:TT
    t_coarse = collect(0:h_step:TT)

    init_expr_q_list = [PDD_init_q(t) for t in t_coarse]
    init_expr_p_list = [PDD_init_p(ti,qi) for (ti,qi) in zip(t_coarse, init_expr_q_list)]
    init_expr_hams = [GeometricProblems.PerturbedPendulum.hamiltonian(0, q, p, PPD_lode.parameters) for (q, p) in zip(init_expr_q_list, init_expr_p_list)]
    init_expr_relative_hams_err = abs.((init_expr_hams .- initial_hamiltonian) / initial_hamiltonian)# Plot 3: Relative Hamiltonian error   

    q1_axis = Axis(PPD_fig[1, 1], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)
    p1_axis = Axis(PPD_fig[1, 2], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)
    ham_axis = Axis(PPD_fig[1, 3], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)
    lines!(q1_axis, t_dense, collect(PPD_plot.q[:, 1]), label="Reference Solution", color=:black, linestyle=:dash, linewidth=linewidth)

    PPD_h1_internal_sol = PPD_h1["PerturbedPendulum_internal_sol"]
    lines!(q1_axis, t_vise_dense, vcat(hcat(PPD_h1_internal_sol...)[2:end, :]...), label="VISE Continuous Solution", color=:orange)
    scatter!(q1_axis, t_coarse, collect(PPD_imp_sol.q[:, 1]), label="Implicit Midpoint Solution", color=:red)
    scatter!(q1_axis, t_coarse, collect(PPD_cgvi_sol.q[:, 1]), label="Galerkin Integrator Solution", color=:green)
    scatter!(q1_axis, t_coarse, init_expr_q_list, label="Initial Expression", color=:pink)
    PR_sol_q_PPD_h1 = PPD_h1["PerturbedPendulum_PR_sol_q"]
    scatter!(q1_axis, t_coarse, PR_sol_q_PPD_h1, label="VISE Discrete Solution", color=:blue)
    vlines!(q1_axis, [30.0], linestyle=:dashdot, color=:purple, label="Training Region")

    PR_sol_p_PPD_h1 = PPD_h1["PerturbedPendulum_PR_sol_p"]
    lines!(p1_axis, t_dense, collect(PPD_plot.p[:, 1]), label="Reference Solution", color=:black, linestyle=:dash, linewidth=linewidth)
    scatter!(p1_axis, t_coarse, collect(PPD_cgvi_sol.p[:, 1]), label="Galerkin Integrator Solution ", color=:green)
    scatter!(p1_axis, t_coarse, collect(PPD_imp_sol.p[:, 1]), label="Implicit Midpoint Solution ", color=:red)
    scatter!(p1_axis, t_coarse, init_expr_p_list, label="Initial Expression", color=:pink)

    scatter!(p1_axis, t_coarse, PR_sol_p_PPD_h1, label="VISE Discrete Solution ", color=:blue)

    hams = [GeometricProblems.PerturbedPendulum.hamiltonian(0, q, p, PPD_lode.parameters) for (q, p) in zip(PR_sol_q_PPD_h1, PR_sol_p_PPD_h1)]
    initial_hamiltonian = hams[1]
    relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)# Plot 3: Relative Hamiltonian error   
    PPD_imp_ham = [GeometricProblems.PerturbedPendulum.hamiltonian(0, q, p, PPD_lode.parameters) for (q, p) in zip(collect(PPD_imp_sol.q[:]), collect(PPD_imp_sol.p[:]))]
    PPD_relative_imp_ham_err = abs.((PPD_imp_ham .- initial_hamiltonian) / initial_hamiltonian)
    PPD_cgvi_ham = [GeometricProblems.PerturbedPendulum.hamiltonian(0, q, p, PPD_lode.parameters) for (q, p) in zip(collect(PPD_cgvi_sol.q[:]), collect(PPD_cgvi_sol.p[:]))]
    PPD_relative_cgvi_ham_err = abs.((PPD_cgvi_ham .- initial_hamiltonian) / initial_hamiltonian)

    lines!(ham_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ", color=:blue)
    lines!(ham_axis, t_coarse, PPD_relative_imp_ham_err, label="Implicit Midpoint Solution ", color=:red)
    lines!(ham_axis, t_coarse, PPD_relative_cgvi_ham_err, label="Galerkin Integrator Solution ", color=:green)
    lines!(ham_axis, t_coarse, init_expr_relative_hams_err, label="Initial Expression", color=:pink)
    scatter!(ham_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ", color=:blue)
    scatter!(ham_axis, t_coarse, PPD_relative_imp_ham_err, label="Implicit Midpoint Solution ", color=:red)
    scatter!(ham_axis, t_coarse, PPD_relative_cgvi_ham_err, label="Galerkin Integrator Solution ", color=:green)
    scatter!(ham_axis, t_coarse, init_expr_relative_hams_err, label="Initial Expression", color=:pink)
    Label(PPD_fig[1, 0], "h = 1.0", rotation=pi / 2, fontsize=label_font_size, tellheight=false)
end

begin # h = 2.0
    h_step = 2.0
    PPD_ref = GeometricProblems.PerturbedPendulum.lodeproblem(timespan=(0, TT), timestep=h_step / 40)
    PPD_plot = integrate(PPD_ref, Gauss(8))
    PPD_lode = GeometricProblems.PerturbedPendulum.lodeproblem(timespan=(0, TT), timestep=h_step)
    PPD_imp_sol = integrate(PPD_lode, ImplicitMidpoint())

    t_dense = collect(0:h_step/40:TT)
    t_vise_dense = h_step/40:h_step/40:TT
    t_coarse = collect(0:h_step:TT)
    init_expr_q_list = [PDD_init_q(t) for t in t_coarse]
    init_expr_p_list = [PDD_init_p(ti,qi) for (ti,qi) in zip(t_coarse, init_expr_q_list)]

    init_expr_hams = [GeometricProblems.PerturbedPendulum.hamiltonian(0, q, p, PPD_lode.parameters) for (q, p) in zip(init_expr_q_list, init_expr_p_list)]
    init_expr_relative_hams_err = abs.((init_expr_hams .- initial_hamiltonian) / initial_hamiltonian)# Plot 3: Relative Hamiltonian error   

    QGau = GaussLegendreQuadrature(16)
    BGau = Lagrange(QuadratureRules.nodes(QGau))
    PPD_cgvi_sol = integrate(PPD_lode, CGVI(BGau, QGau))



    q2_axis = Axis(PPD_fig[2, 1], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)
    p2_axis = Axis(PPD_fig[2, 2], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)
    ham2_axis = Axis(PPD_fig[2, 3], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)
    lines!(q2_axis, t_dense, collect(PPD_plot.q[:, 1]), label="Reference Solution", color=:black, linestyle=:dash, linewidth=linewidth)

    PPD_h2_internal_sol = PPD_h2["PerturbedPendulum_internal_sol"]
    lines!(q2_axis, t_vise_dense, vcat(hcat(PPD_h2_internal_sol...)[2:end, :]...), label="VISE Continuous Solution", color=:orange)
    scatter!(q2_axis, t_coarse, collect(PPD_imp_sol.q[:, 1]), label="Implicit Midpoint Solution", color=:red)
    scatter!(q2_axis, t_coarse, collect(PPD_cgvi_sol.q[:, 1]), label="Galerkin Integrator Solution", color=:green)
    scatter!(q2_axis, t_coarse, init_expr_q_list, label="Initial Expression", color=:pink)

    PR_sol_q_PPD_h2 = PPD_h2["PerturbedPendulum_PR_sol_q"]
    scatter!(q2_axis, t_coarse, PR_sol_q_PPD_h2, label="VISE Discrete Solution", color=:blue)
    vlines!(q2_axis, [30.0], linestyle=:dashdot, color=:purple, label="Training Region")

    PR_sol_p_PPD_h2 = PPD_h2["PerturbedPendulum_PR_sol_p"]
    lines!(p2_axis, t_dense, collect(PPD_plot.p[:, 1]), label="Reference Solution", color=:black, linestyle=:dash, linewidth=linewidth)
    scatter!(p2_axis, t_coarse, collect(PPD_cgvi_sol.p[:, 1]), label="Galerkin Integrator Solution ", color=:green)
    scatter!(p2_axis, t_coarse, init_expr_p_list, label="Initial Expression", color=:pink)

    scatter!(p2_axis, t_coarse, collect(PPD_imp_sol.p[:, 1]), label="Implicit Midpoint Solution ", color=:red)
    scatter!(p2_axis, t_coarse, PR_sol_p_PPD_h2, label="VISE Discrete Solution ", color=:blue)

    hams = [GeometricProblems.PerturbedPendulum.hamiltonian(0, q, p, PPD_lode.parameters) for (q, p) in zip(PR_sol_q_PPD_h2, PR_sol_p_PPD_h2)]
    initial_hamiltonian = hams[1]
    relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)# Plot 3: Relative Hamiltonian error   
    PPD_imp_ham = [GeometricProblems.PerturbedPendulum.hamiltonian(0, q, p, PPD_lode.parameters) for (q, p) in zip(collect(PPD_imp_sol.q[:]), collect(PPD_imp_sol.p[:]))]
    PPD_relative_imp_ham_err = abs.((PPD_imp_ham .- initial_hamiltonian) / initial_hamiltonian)

    PPD_cgvi_ham = [GeometricProblems.PerturbedPendulum.hamiltonian(0, q, p, PPD_lode.parameters) for (q, p) in zip(collect(PPD_cgvi_sol.q[:]), collect(PPD_cgvi_sol.p[:]))]
    PPD_relative_cgvi_ham_err = abs.((PPD_cgvi_ham .- initial_hamiltonian) / initial_hamiltonian)
    lines!(ham2_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ", color=:blue)
    lines!(ham2_axis, t_coarse, PPD_relative_imp_ham_err, label="Implicit Midpoint Solution ", color=:red)
    lines!(ham2_axis, t_coarse, PPD_relative_cgvi_ham_err, label="Galerkin Integrator Solution ", color=:green)
    lines!(ham2_axis, t_coarse, init_expr_relative_hams_err, label="Initial Expression", color=:pink)

    scatter!(ham2_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ", color=:blue)
    scatter!(ham2_axis, t_coarse, PPD_relative_imp_ham_err, label="Implicit Midpoint Solution ", color=:red)
    scatter!(ham2_axis, t_coarse, PPD_relative_cgvi_ham_err, label="Galerkin Integrator Solution ", color=:green)
    scatter!(ham2_axis, t_coarse, init_expr_relative_hams_err, label="Initial Expression", color=:pink)
    Label(PPD_fig[2, 0], "h = 2.0", rotation=pi / 2, fontsize=label_font_size, tellheight=false)
end

begin # h = 5.0
    h_step = 5.0
    PPD_ref = GeometricProblems.PerturbedPendulum.lodeproblem(timespan=(0, TT), timestep=h_step / 40)
    PPD_plot = integrate(PPD_ref, Gauss(8))
    PPD_lode = GeometricProblems.PerturbedPendulum.lodeproblem(timespan=(0, TT), timestep=h_step)
    PPD_imp_sol = integrate(PPD_lode, ImplicitMidpoint())

    QGau = GaussLegendreQuadrature(16)
    BGau = Lagrange(QuadratureRules.nodes(QGau))
    PPD_cgvi_sol = integrate(PPD_lode, CGVI(BGau, QGau))

    t_dense = collect(0:h_step/40:TT)
    t_vise_dense = h_step/40:h_step/40:TT
    t_coarse = collect(0:h_step:TT)
    init_expr_q_list = [PDD_init_q(t) for t in t_coarse]
    init_expr_p_list = [PDD_init_p(ti,qi) for (ti,qi) in zip(t_coarse, init_expr_q_list)]

    init_expr_hams = [GeometricProblems.PerturbedPendulum.hamiltonian(0, q, p, PPD_lode.parameters) for (q, p) in zip(init_expr_q_list, init_expr_p_list)]
    init_expr_relative_hams_err = abs.((init_expr_hams .- initial_hamiltonian) / initial_hamiltonian)# Plot 3: Relative Hamiltonian error   

    q5_axis = Axis(PPD_fig[3, 1], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)
    p5_axis = Axis(PPD_fig[3, 2], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)
    ham5_axis = Axis(PPD_fig[3, 3], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)
    lines!(q5_axis, t_dense, collect(PPD_plot.q[:, 1]), label="Reference Solution", color=:black, linestyle=:dash, linewidth=linewidth)

    PPD_h5_internal_sol = PPD_h5["PerturbedPendulum_internal_sol"]
    lines!(q5_axis, t_vise_dense, vcat(hcat(PPD_h5_internal_sol...)[2:end, :]...), label="VISE Continuous Solution", color=:orange)
    scatter!(q5_axis, t_coarse, collect(PPD_imp_sol.q[:, 1]), label="Implicit Midpoint Solution", color=:red)
    scatter!(q5_axis, t_coarse, collect(PPD_cgvi_sol.q[:, 1]), label="Galerkin Integrator Solution", color=:green)
    scatter!(q5_axis, t_coarse, init_expr_q_list, label="Initial Expression", color=:pink)
    PR_sol_q_PPD_h5 = PPD_h5["PerturbedPendulum_PR_sol_q"]
    scatter!(q5_axis, t_coarse, PR_sol_q_PPD_h5, label="VISE Discrete Solution", color=:blue)
    vlines!(q5_axis, [100.0], linestyle=:dashdot, color=:purple, label="Training Region")

    PR_sol_p_PPD_h5 = PPD_h5["PerturbedPendulum_PR_sol_p"]
    lines!(p5_axis, t_dense, collect(PPD_plot.p[:, 1]), label="Reference Solution", color=:black, linestyle=:dash, linewidth=linewidth)
    scatter!(p5_axis, t_coarse, collect(PPD_cgvi_sol.p[:, 1]), label="Galerkin Integrator Solution ", color=:green)
    scatter!(p5_axis, t_coarse, init_expr_p_list, label="Initial Expression", color=:pink)
    scatter!(p5_axis, t_coarse, collect(PPD_imp_sol.p[:, 1]), label="Implicit Midpoint Solution ", color=:red)
    scatter!(p5_axis, t_coarse, PR_sol_p_PPD_h5, label="VISE Discrete Solution ", color=:blue)

    hams = [GeometricProblems.PerturbedPendulum.hamiltonian(0, q, p, PPD_lode.parameters) for (q, p) in zip(PR_sol_q_PPD_h5, PR_sol_p_PPD_h5)]
    initial_hamiltonian = hams[1]
    relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)# Plot 3: Relative Hamiltonian error   
    PPD_imp_ham = [GeometricProblems.PerturbedPendulum.hamiltonian(0, q, p, PPD_lode.parameters) for (q, p) in zip(collect(PPD_imp_sol.q[:]), collect(PPD_imp_sol.p[:]))]
    PPD_relative_imp_ham_err = abs.((PPD_imp_ham .- initial_hamiltonian) / initial_hamiltonian)

    PPD_cgvi_ham = [GeometricProblems.PerturbedPendulum.hamiltonian(0, q, p, PPD_lode.parameters) for (q, p) in zip(collect(PPD_cgvi_sol.q[:]), collect(PPD_cgvi_sol.p[:]))]
    PPD_relative_cgvi_ham_err = abs.((PPD_cgvi_ham .- initial_hamiltonian) / initial_hamiltonian)
    lines!(ham5_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ", color=:blue)
    lines!(ham5_axis, t_coarse, PPD_relative_imp_ham_err, label="Implicit Midpoint Solution ", color=:red)
    lines!(ham5_axis, t_coarse, PPD_relative_cgvi_ham_err, label="Galerkin Integrator Solution ", color=:green)
    lines!(ham5_axis, t_coarse, init_expr_relative_hams_err, label="Initial Expression", color=:pink)
    scatter!(ham5_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ", color=:blue)
    scatter!(ham5_axis, t_coarse, PPD_relative_imp_ham_err, label="Implicit Midpoint Solution ", color=:red)
    scatter!(ham5_axis, t_coarse, PPD_relative_cgvi_ham_err, label="Galerkin Integrator Solution ", color=:green)
    scatter!(ham5_axis, t_coarse, init_expr_relative_hams_err, label="Initial Expression", color=:pink)
    Label(PPD_fig[3, 0], "h = 5.0", rotation=pi / 2, fontsize=label_font_size, tellheight=false)
end

Label(PPD_fig[0, 1], "q₁", fontsize=label_font_size, tellwidth=false)
Label(PPD_fig[0, 2], "p₁", fontsize=label_font_size, tellwidth=false)
Label(PPD_fig[0, 3], "Relative Hamiltonian Error", fontsize=label_font_size, tellwidth=false)

Label(PPD_fig[4, 1], "time", fontsize=label_font_size, tellwidth=false)
Label(PPD_fig[4, 2], "time", fontsize=label_font_size, tellwidth=false)
Label(PPD_fig[4, 3], "time", fontsize=label_font_size, tellwidth=false)

Legend(PPD_fig[5, 1:3], q1_axis, orientation=:horizontal, labelsize=label_font_size,
    framevisible=false, nbanks=2)

save("result_figures/PPDh125_with_initial_expression.pdf", PPD_fig)

max_relative_hams_err = maximum(init_expr_relative_hams_err) # 0.0438997042568547
min_relative_hams_err = minimum(init_expr_relative_hams_err) # 0.0001801883588217746


HHP_h1 = load("/Users/zeyuanli/Desktop/untitled folder 2/Backtracking2_R16_h1.00_iter1000_fabs1.78e-15_fsuc1.78e-15_TT200.jld2")
HHP_h2 = load("/Users/zeyuanli/Desktop/untitled folder 2/Backtracking2_R16_h2.00_iter1000_fabs4.44e-16_fsuc1.78e-15_TT200.jld2")
HHP_h5 = load("/Users/zeyuanli/Desktop/untitled folder 2/Backtracking2_R16_h5.00_iter1000_fabs4.44e-16_fsuc1.78e-15_TT200.jld2")

function HHP_hamiltonian(q1,q2,p1,p2)
    λ = 1.0
    0.5 * (p1^2 + p2^2) + 0.5 * (q1^2 + q2^2) + λ * (q1^2 * q2 - q2^3 / 3) 
end

q₁_expr(t) = 0.14831 * cos(-0.64812 + t) - 0.018712
q₂_expr(t) = 0.14298 * cos(- 0.97215 * t+ 0.7615)-0.0013983
p₁_expr(t) = - 0.14831 * sin(-0.64812 + t)
p₂_expr(t) = 0.97215 * 0.14298 * sin(- 0.97215 * t+ 0.7615)


HHP_fig = Figure(linewidth=2, markersize=8, size=(1400, 1000))#size = (2200,3800)
begin # h= 1.0
    TT = 60.0
    h_step = 1.0
    HHP_ref = GeometricProblems.HenonHeilesPotential.lodeproblem([0.1,0.1],[0.1,0.1],timespan = (0,TT),timestep = h_step/40)
    HHP_plot = integrate(HHP_ref, Gauss(8))
    HHlode = GeometricProblems.HenonHeilesPotential.lodeproblem([0.1,0.1],[0.1,0.1],timespan = (0,TT),timestep = h_step)
    HHP_imp_sol = integrate(HHlode, ImplicitMidpoint())

    QGau = GaussLegendreQuadrature(16)
    BGau = Lagrange(QuadratureRules.nodes(QGau))
    HHP_cgvi_sol = integrate(HHlode, CGVI(BGau, QGau))

    t_dense = collect(0:h_step/40:TT)
    t_vise_dense = h_step/40:h_step/40:TT
    t_coarse = collect(0:h_step:TT)

    q1_axis = Axis(HHP_fig[1, 1], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size,limits = (nothing, (-0.2, 0.2)))
    q2_axis = Axis(HHP_fig[1, 2], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size,limits = (nothing, (-0.2, 0.2)))
    p1_axis = Axis(HHP_fig[1, 3], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size,limits = (nothing, (-0.2, 0.2)))
    p2_axis = Axis(HHP_fig[1, 4], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size,limits = (nothing, (-0.2, 0.2)))
    ham_axis = Axis(HHP_fig[1, 5], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)

    HHP_h1_internal_sol = HHP_h1["HenonHeiles_internal_sol"]
    internal_q1 = Array{Vector}(undef,Int(TT/h_step))
    internal_q2 = Array{Vector}(undef,Int(TT/h_step))
    for i in 1:Int(TT/h_step)
        internal_q1[i] = HHP_h1_internal_sol[i][:,1]
        internal_q2[i] = HHP_h1_internal_sol[i][:,2]
    end

    lines!(q1_axis, t_dense, collect(HHP_plot.q[:, 1]), label="Reference Solution", color=:black, linestyle=:dash, linewidth=linewidth)
    lines!(q1_axis, t_vise_dense, vcat(hcat(internal_q1...)[2:end, :]...), label="VISE Continuous Solution", color=:orange)
    scatter!(q1_axis, t_coarse, collect(HHP_imp_sol.q[:, 1]), label="Implicit Midpoint Solution", color=:red)
    scatter!(q1_axis, t_coarse, collect(HHP_cgvi_sol.q[:, 1]), label="Galerkin Integrator Solution", color=:green)
    init_expr_q1_list = [q₁_expr(t) for t in 0:h_step:TT]
    scatter!(q1_axis, t_coarse, init_expr_q1_list, label="Initial Expression", color=:pink)
    PR_sol_q1 = HHP_h1["HenonHeiles_PR_sol_q1"][1:Int(TT/h_step)+1]
    scatter!(q1_axis, t_coarse, PR_sol_q1, label="VISE Discrete Solution", color=:blue)
    vlines!(q1_axis, [10.0], linestyle=:dashdot, color=:purple, label="Training Region")

    lines!(q2_axis, t_dense, collect(HHP_plot.q[:, 2]), label="Reference Solution", color=:black, linestyle=:dash, linewidth=linewidth)
    lines!(q2_axis, t_vise_dense, vcat(hcat(internal_q2...)[2:end, :]...), label="VISE Continuous Solution", color=:orange)
    scatter!(q2_axis, t_coarse, collect(HHP_cgvi_sol.q[:, 2]), label="Galerkin Integrator Solution ", color=:green)
    scatter!(q2_axis, t_coarse, collect(HHP_imp_sol.q[:, 2]), label="Implicit Midpoint Solution ", color=:red)
    init_expr_q2_list = [q₂_expr(t) for t in 0:h_step:TT]
    scatter!(q2_axis, t_coarse, init_expr_q2_list, label="Initial Expression", color=:pink)
    PR_sol_q2 = HHP_h1["HenonHeiles_PR_sol_q2"][1:Int(TT/h_step)+1]
    scatter!(q2_axis, t_coarse, PR_sol_q2, label="VISE Discrete Solution ", color=:blue)
    vlines!(q2_axis, [10.0], linestyle=:dashdot, color=:purple, label="Training Region")

    lines!(p1_axis, t_dense, collect(HHP_plot.p[:, 1]), label="Reference Solution", color=:black, linestyle=:dash, linewidth=linewidth)
    scatter!(p1_axis, t_coarse, collect(HHP_cgvi_sol.p[:, 1]), label="Galerkin Integrator Solution ", color=:green)
    scatter!(p1_axis, t_coarse, collect(HHP_imp_sol.p[:, 1]), label="Implicit Midpoint Solution ", color=:red)
    PR_sol_p1 = HHP_h1["HenonHeiles_PR_sol_p1"][1:Int(TT/h_step)+1]
    scatter!(p1_axis, t_coarse, PR_sol_p1, label="VISE Discrete Solution ", color=:blue)

    lines!(p2_axis, t_dense, collect(HHP_plot.p[:, 2]), label="Reference Solution", color=:black, linestyle=:dash, linewidth=linewidth)
    scatter!(p2_axis, t_coarse, collect(HHP_cgvi_sol.p[:, 2]), label="Galerkin Integrator Solution ", color=:green)
    scatter!(p2_axis, t_coarse, collect(HHP_imp_sol.p[:, 2]), label="Implicit Midpoint Solution ", color=:red)
    PR_sol_p2 = HHP_h1["HenonHeiles_PR_sol_p2"][1:Int(TT/h_step)+1]
    scatter!(p2_axis, t_coarse, PR_sol_p2, label="VISE Discrete Solution ", color=:blue)


    ref_hams = HHP_hamiltonian.(collect(HHP_plot.q[:, 1]), collect(HHP_plot.q[:, 2]), collect(HHP_plot.p[:, 1]), collect(HHP_plot.p[:, 2]))
    initial_hamiltonian = ref_hams[1]

    init_expr_hams = [HHP_hamiltonian(q₁_expr(t), q₂_expr(t), p₁_expr(t), p₂_expr(t)) for t in 0:h_step:TT]
    init_expr_relative_hams_err = abs.((init_expr_hams .- initial_hamiltonian) / initial_hamiltonian)

    hams = [HHP_hamiltonian(q1, q2, p1, p2) for (q1, q2, p1, p2) in zip(PR_sol_q1, PR_sol_q2, PR_sol_p1, PR_sol_p2)]
    relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)# Plot 3: Relative Hamiltonian error   
    HHP_imp_ham = [HHP_hamiltonian(q1, q2, p1, p2) for (q1, q2, p1, p2) in zip(collect(HHP_imp_sol.q[:, 1]), collect(HHP_imp_sol.q[:, 2]), collect(HHP_imp_sol.p[:, 1]), collect(HHP_imp_sol.p[:, 2]))]
    HHP_relative_imp_ham_err = abs.((HHP_imp_ham .- initial_hamiltonian) / initial_hamiltonian)
    HHP_cgvi_ham = [HHP_hamiltonian(q1, q2, p1, p2) for (q1, q2, p1, p2) in zip(collect(HHP_cgvi_sol.q[:, 1]), collect(HHP_cgvi_sol.q[:, 2]), collect(HHP_cgvi_sol.p[:, 1]), collect(HHP_cgvi_sol.p[:, 2]))]
    HHP_relative_cgvi_ham_err = abs.((HHP_cgvi_ham .- initial_hamiltonian) / initial_hamiltonian)

    lines!(ham_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ", color=:blue)
    lines!(ham_axis, t_coarse, HHP_relative_imp_ham_err, label="Implicit Midpoint Solution ", color=:red)
    lines!(ham_axis, t_coarse, HHP_relative_cgvi_ham_err, label="Galerkin Integrator Solution ", color=:green)
    lines!(ham_axis, t_coarse, init_expr_relative_hams_err, label="Initial Expression", color=:pink)
    scatter!(ham_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ", color=:blue)
    scatter!(ham_axis, t_coarse, HHP_relative_imp_ham_err, label="Implicit Midpoint Solution ", color=:red)
    scatter!(ham_axis, t_coarse, HHP_relative_cgvi_ham_err, label="Galerkin Integrator Solution ", color=:green)
    scatter!(ham_axis, t_coarse, init_expr_relative_hams_err, label="Initial Expression", color=:pink)

    Label(HHP_fig[1, 0], "h = 1.0", rotation=pi / 2, fontsize=label_font_size, tellheight=false)
end

begin # h = 2.0
    h_step = 2.0
    HHP_ref = GeometricProblems.HenonHeilesPotential.lodeproblem([0.1,0.1],[0.1,0.1],timespan = (0,TT),timestep = h_step/40)
    HHP_plot = integrate(HHP_ref, Gauss(8))
    HHlode = GeometricProblems.HenonHeilesPotential.lodeproblem([0.1,0.1],[0.1,0.1],timespan = (0,TT),timestep = h_step)
    HHP_imp_sol = integrate(HHlode, ImplicitMidpoint())

    QGau = GaussLegendreQuadrature(16)
    BGau = Lagrange(QuadratureRules.nodes(QGau))
    HHP_cgvi_sol = integrate(HHlode, CGVI(BGau, QGau))

    t_dense = collect(0:h_step/40:TT)
    t_vise_dense = h_step/40:h_step/40:TT
    t_coarse = collect(0:h_step:TT)

    q1_axis = Axis(HHP_fig[2, 1], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size,limits = (nothing, (-0.2, 0.2)))
    q2_axis = Axis(HHP_fig[2, 2], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size,limits = (nothing, (-0.2, 0.2)))
    p1_axis = Axis(HHP_fig[2, 3], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size,limits = (nothing, (-0.2, 0.2)))
    p2_axis = Axis(HHP_fig[2, 4], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size,limits = (nothing, (-0.2, 0.2)))
    ham_axis = Axis(HHP_fig[2, 5], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size)

    HHP_h2_internal_sol = HHP_h2["HenonHeiles_internal_sol"]
    internal_q1 = Array{Vector}(undef,Int(TT/h_step))
    internal_q2 = Array{Vector}(undef,Int(TT/h_step))
    for i in 1:Int(TT/h_step)
        internal_q1[i] = HHP_h2_internal_sol[i][:,1]
        internal_q2[i] = HHP_h2_internal_sol[i][:,2]
    end 

    lines!(q1_axis, t_dense, collect(HHP_plot.q[:, 1]), label="Reference Solution", color=:black, linestyle=:dash, linewidth=linewidth)
    lines!(q1_axis, t_vise_dense, vcat(hcat(internal_q1...)[2:end, :]...), label="VISE Continuous Solution", color=:orange)
    scatter!(q1_axis, t_coarse, collect(HHP_imp_sol.q[:, 1]), label="Implicit Midpoint Solution", color=:red)
    scatter!(q1_axis, t_coarse, collect(HHP_cgvi_sol.q[:, 1]), label="Galerkin Integrator Solution", color=:green)
    init_expr_q1_list = [q₁_expr(t) for t in 0:h_step:TT]
    scatter!(q1_axis, t_coarse, init_expr_q1_list, label="Initial Expression", color=:pink)
    PR_sol_q1 = HHP_h2["HenonHeiles_PR_sol_q1"][1:Int(TT/h_step)+1]
    scatter!(q1_axis, t_coarse, PR_sol_q1, label="VISE Discrete Solution", color=:blue)
    vlines!(q1_axis, [10.0], linestyle=:dashdot, color=:purple, label="Training Region")    
    lines!(q2_axis, t_dense, collect(HHP_plot.q[:, 2]), label="Reference Solution", color=:black, linestyle=:dash, linewidth=linewidth)

    lines!(q2_axis, t_vise_dense, vcat(hcat(internal_q2...)[2:end, :]...), label="VISE Continuous Solution", color=:orange)
    scatter!(q2_axis, t_coarse, collect(HHP_cgvi_sol.q[:, 2]), label="Galerkin Integrator Solution ", color=:green)
    scatter!(q2_axis, t_coarse, collect(HHP_imp_sol.q[:, 2]), label="Implicit Midpoint Solution ", color=:red)
    init_expr_q2_list = [q₂_expr(t) for t in 0:h_step:TT]
    scatter!(q2_axis, t_coarse, init_expr_q2_list, label="Initial Expression", color=:pink)
    PR_sol_q2 = HHP_h2["HenonHeiles_PR_sol_q2"][1:Int(TT/h_step)+1]
    scatter!(q2_axis, t_coarse, PR_sol_q2, label="VISE Discrete Solution ", color=:blue)
    vlines!(q2_axis, [10.0], linestyle=:dashdot, color=:purple, label="Training Region")

    lines!(p1_axis, t_dense, collect(HHP_plot.p[:, 1]), label="Reference Solution", color=:black, linestyle=:dash, linewidth=linewidth)
    scatter!(p1_axis, t_coarse, collect(HHP_cgvi_sol.p[:, 1]), label="Galerkin Integrator Solution ", color=:green)
    scatter!(p1_axis, t_coarse, collect(HHP_imp_sol.p[:, 1]), label="Implicit Midpoint Solution", color=:red)
    PR_sol_p1 = HHP_h2["HenonHeiles_PR_sol_p1"][1:Int(TT/h_step)+1]
    scatter!(p1_axis, t_coarse, PR_sol_p1, label="VISE Discrete Solution", color=:blue)

    lines!(p2_axis, t_dense, collect(HHP_plot.p[:, 2]), label="Reference Solution", color=:black, linestyle=:dash, linewidth=linewidth)
    scatter!(p2_axis, t_coarse, collect(HHP_cgvi_sol.p[:, 2]), label="Galerkin Integrator Solution ", color=:green)
    scatter!(p2_axis, t_coarse, collect(HHP_imp_sol.p[:, 2]), label="Implicit Midpoint Solution ", color=:red)
    PR_sol_p2 = HHP_h2["HenonHeiles_PR_sol_p2"][1:Int(TT/h_step)+1]
    scatter!(p2_axis, t_coarse, PR_sol_p2, label="VISE Discrete Solution ", color=:blue)

    ref_hams = HHP_hamiltonian.(collect(HHP_plot.q[:, 1]), collect(HHP_plot.q[:, 2]), collect(HHP_plot.p[:, 1]), collect(HHP_plot.p[:, 2]))
    initial_hamiltonian = ref_hams[1]   
    init_expr_hams = [HHP_hamiltonian(q₁_expr(t), q₂_expr(t), p₁_expr(t), p₂_expr(t)) for t in 0:h_step:TT]
    init_expr_relative_hams_err = abs.((init_expr_hams .- initial_hamiltonian) / initial_hamiltonian)
    hams = [HHP_hamiltonian(q1, q2, p1, p2) for (q1, q2, p1, p2) in zip(PR_sol_q1, PR_sol_q2, PR_sol_p1, PR_sol_p2)]
    relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)# Plot 3: Relative Hamiltonian error
    HHP_imp_ham = [HHP_hamiltonian(q1, q2, p1, p2) for (q1, q2, p1, p2) in zip(collect(HHP_imp_sol.q[:, 1]), collect(HHP_imp_sol.q[:, 2]), collect(HHP_imp_sol.p[:, 1]), collect(HHP_imp_sol.p[:, 2]))]
    HHP_relative_imp_ham_err = abs.((HHP_imp_ham .- initial_hamiltonian) / initial_hamiltonian)
    HHP_cgvi_ham = [HHP_hamiltonian(q1, q2, p1, p2) for (q1, q2, p1, p2) in zip(collect(HHP_cgvi_sol.q[:, 1]), collect(HHP_cgvi_sol.q[:, 2]), collect(HHP_cgvi_sol.p[:, 1]), collect(HHP_cgvi_sol.p[:, 2]))]
    HHP_relative_cgvi_ham_err = abs.((HHP_cgvi_ham .- initial_hamiltonian) / initial_hamiltonian)   

    lines!(ham_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ", color=:blue)
    lines!(ham_axis, t_coarse, HHP_relative_imp_ham_err, label="Implicit Midpoint Solution ", color=:red)
    lines!(ham_axis, t_coarse, HHP_relative_cgvi_ham_err, label="Galerkin Integrator Solution ", color=:green)
    lines!(ham_axis, t_coarse, init_expr_relative_hams_err, label="Initial Expression", color=:pink)
    scatter!(ham_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ", color=:blue)
    scatter!(ham_axis, t_coarse, HHP_relative_imp_ham_err, label="Implicit Midpoint Solution ", color=:red)
    scatter!(ham_axis, t_coarse, HHP_relative_cgvi_ham_err, label="Galerkin Integrator Solution ", color=:green)
    scatter!(ham_axis, t_coarse, init_expr_relative_hams_err, label="Initial Expression", color=:pink)

    Label(HHP_fig[2, 0], "h = 2.0", rotation=pi / 2, fontsize=label_font_size, tellheight=false)
end

begin # h = 5.0
    h_step = 5.0
    HHP_ref = GeometricProblems.HenonHeilesPotential.lodeproblem([0.1,0.1],[0.1,0.1],timespan = (0,TT),timestep = h_step/40)
    HHP_plot = integrate(HHP_ref, Gauss(8))
    HHlode = GeometricProblems.HenonHeilesPotential.lodeproblem([0.1,0.1],[0.1,0.1],timespan = (0,TT),timestep = h_step)
    HHP_imp_sol = integrate(HHlode, ImplicitMidpoint())

    QGau = GaussLegendreQuadrature(16)
    BGau = Lagrange(QuadratureRules.nodes(QGau))
    HHP_cgvi_sol = integrate(HHlode, CGVI(BGau, QGau))

    t_dense = collect(0:h_step/40:TT)
    t_vise_dense = h_step/40:h_step/40:TT
    t_coarse = collect(0:h_step:TT)

    q1_axis = Axis(HHP_fig[3, 1], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size,limits = (nothing, (-0.2, 0.2)))
    q2_axis = Axis(HHP_fig[3, 2], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size,limits = (nothing, (-0.2, 0.2)))
    p1_axis = Axis(HHP_fig[3, 3], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size,limits = (nothing, (-0.2, 0.2)))
    p2_axis = Axis(HHP_fig[3, 4], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size,limits = (nothing, (-0.2, 0.2)))
    ham_axis = Axis(HHP_fig[3, 5], xlabelsize=label_size, ylabelsize=label_size, yticklabelsize=tick_size, xticklabelsize=tick_size,limits = (nothing, (-0.05, 1.0)))

    HHP_h5_internal_sol = HHP_h5["HenonHeiles_internal_sol"]
    internal_q1 = Array{Vector}(undef,Int(TT/h_step))
    internal_q2 = Array{Vector}(undef,Int(TT/h_step))
    for i in 1:Int(TT/h_step)
        internal_q1[i] = HHP_h5_internal_sol[i][:,1]
        internal_q2[i] = HHP_h5_internal_sol[i][:,2]
    end 

    lines!(q1_axis, t_dense, collect(HHP_plot.q[:, 1]), label="Reference Solution", color=:black, linestyle=:dash, linewidth=linewidth)
    lines!(q1_axis, t_vise_dense, vcat(hcat(internal_q1...)[2:end, :]...), label="VISE Continuous Solution", color=:orange)
    scatter!(q1_axis, t_coarse, collect(HHP_imp_sol.q[:, 1]), label="Implicit Midpoint Solution", color=:red)
    scatter!(q1_axis, t_coarse, collect(HHP_cgvi_sol.q[:, 1]), label="Galerkin Integrator Solution", color=:green)
        init_expr_q1_list = [q₁_expr(t) for t in 0:h_step:TT]
    scatter!(q1_axis, t_coarse, init_expr_q1_list, label="Initial Expression", color=:pink)
    PR_sol_q1 = HHP_h5["HenonHeiles_PR_sol_q1"][1:Int(TT/h_step)+1]
    scatter!(q1_axis, t_coarse, PR_sol_q1, label="VISE Discrete Solution", color=:blue)
    vlines!(q1_axis, [10.0], linestyle=:dashdot, color=:purple, label="Training Region")    
    lines!(q2_axis, t_dense, collect(HHP_plot.q[:, 2]), label="Reference Solution", color=:black, linestyle=:dash, linewidth=linewidth)
    lines!(q2_axis, t_vise_dense, vcat(hcat(internal_q2...)[2:end, :]...), label="VISE Continuous Solution", color=:orange) 

    scatter!(q2_axis, t_coarse, collect(HHP_cgvi_sol.q[:, 2]), label="Galerkin Integrator Solution ", color=:green)
    scatter!(q2_axis, t_coarse, collect(HHP_imp_sol.q[:, 2]), label="Implicit Midpoint Solution ", color=:red)
        init_expr_q2_list = [q₂_expr(t) for t in 0:h_step:TT]
    scatter!(q2_axis, t_coarse, init_expr_q2_list, label="Initial Expression", color=:pink)
    PR_sol_q2 = HHP_h5["HenonHeiles_PR_sol_q2"][1:Int(TT/h_step)+1]
    scatter!(q2_axis, t_coarse, PR_sol_q2, label="VISE Discrete Solution ", color=:blue)
    vlines!(q2_axis, [10.0], linestyle=:dashdot, color=:purple, label="Training Region")    

    lines!(p1_axis, t_dense, collect(HHP_plot.p[:, 1]), label="Reference Solution", color=:black, linestyle=:dash, linewidth=linewidth)
    scatter!(p1_axis, t_coarse, collect(HHP_cgvi_sol.p[:, 1]), label="Galerkin Integrator Solution ", color=:green)
    scatter!(p1_axis, t_coarse, collect(HHP_imp_sol.p[:, 1]), label="Implicit Midpoint Solution ", color=:red)
    PR_sol_p1 = HHP_h5["HenonHeiles_PR_sol_p1"][1:Int(TT/h_step)+1]
    scatter!(p1_axis, t_coarse, PR_sol_p1, label="VISE Discrete Solution ", color=:blue)    

    lines!(p2_axis, t_dense, collect(HHP_plot.p[:, 2]), label="Reference Solution", color=:black, linestyle=:dash, linewidth=linewidth)
    scatter!(p2_axis, t_coarse, collect(HHP_cgvi_sol.p[:, 2]), label="Galerkin Integrator Solution ", color=:green)
    scatter!(p2_axis, t_coarse, collect(HHP_imp_sol.p[:, 2]), label="Implicit Midpoint Solution ", color=:red)
    PR_sol_p2 = HHP_h5["HenonHeiles_PR_sol_p2"][1:Int(TT/h_step)+1]
    scatter!(p2_axis, t_coarse, PR_sol_p2, label="VISE Discrete Solution ", color=:blue)

    ref_hams = HHP_hamiltonian.(collect(HHP_plot.q[:, 1]), collect(HHP_plot.q[:, 2]), collect(HHP_plot.p[:, 1]), collect(HHP_plot.p[:, 2]))
    initial_hamiltonian = ref_hams[1]

    init_expr_hams = [HHP_hamiltonian(q₁_expr(t), q₂_expr(t), p₁_expr(t), p₂_expr(t)) for t in 0:h_step:TT]
    init_expr_relative_hams_err = abs.((init_expr_hams .- initial_hamiltonian) / initial_hamiltonian)

    hams = [HHP_hamiltonian(q1, q2, p1, p2) for (q1, q2, p1, p2) in zip(PR_sol_q1, PR_sol_q2, PR_sol_p1, PR_sol_p2)]
    relative_hams_err = abs.((hams .- initial_hamiltonian) / initial_hamiltonian)# Plot 3: Relative Hamiltonian error
    HHP_imp_ham = [HHP_hamiltonian(q1, q2, p1, p2) for (q1, q2, p1, p2) in zip(collect(HHP_imp_sol.q[:, 1]), collect(HHP_imp_sol.q[:, 2]), collect(HHP_imp_sol.p[:, 1]), collect(HHP_imp_sol.p[:, 2]))]
    HHP_relative_imp_ham_err = abs.((HHP_imp_ham .- initial_hamiltonian) / initial_hamiltonian)
    HHP_cgvi_ham = [HHP_hamiltonian(q1, q2, p1, p2) for (q1, q2, p1, p2) in zip(collect(HHP_cgvi_sol.q[:, 1]), collect(HHP_cgvi_sol.q[:, 2]), collect(HHP_cgvi_sol.p[:, 1]), collect(HHP_cgvi_sol.p[:, 2]))]

    HHP_relative_cgvi_ham_err = abs.((HHP_cgvi_ham .- initial_hamiltonian) / initial_hamiltonian)
    lines!(ham_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ", color=:blue)
    lines!(ham_axis, t_coarse, HHP_relative_imp_ham_err, label="Implicit Midpoint Solution ", color=:red)
    lines!(ham_axis, t_coarse, HHP_relative_cgvi_ham_err, label="Galerkin Integrator Solution ", color=:green)
    lines!(ham_axis, t_coarse, init_expr_relative_hams_err, label="Initial Expression", color=:pink)
    scatter!(ham_axis, t_coarse, relative_hams_err, label="VISE Discrete Solution ", color=:blue)
    scatter!(ham_axis, t_coarse, HHP_relative_imp_ham_err, label="Implicit Midpoint Solution ", color=:red)
    scatter!(ham_axis, t_coarse, HHP_relative_cgvi_ham_err, label="Galerkin Integrator Solution ", color=:green)
    scatter!(ham_axis, t_coarse, init_expr_relative_hams_err, label="Initial Expression", color=:pink)
    Label(HHP_fig[3, 0], "h = 5.0", rotation=pi / 2, fontsize=label_font_size, tellheight=false)
end


Label(HHP_fig[0, 1], "q₁", fontsize=label_font_size, tellwidth=false)
Label(HHP_fig[0, 2], "q₂", fontsize=label_font_size, tellwidth=false)
Label(HHP_fig[0, 3], "p₁", fontsize=label_font_size, tellwidth=false)
Label(HHP_fig[0, 4], "p₂", fontsize=label_font_size, tellwidth=false)
Label(HHP_fig[0, 5], "Relative Hamiltonian Error", fontsize=label_font_size, tellwidth=false)

Label(HHP_fig[4, 1], "time", fontsize=label_font_size, tellwidth=false)
Label(HHP_fig[4, 2], "time", fontsize=label_font_size, tellwidth=false)
Label(HHP_fig[4, 3], "time", fontsize=label_font_size, tellwidth=false)
Label(HHP_fig[4, 4], "time", fontsize=label_font_size, tellwidth=false)
Label(HHP_fig[4, 5], "time", fontsize=label_font_size, tellwidth=false)

Legend(HHP_fig[5, 1:5], q1_axis, orientation=:horizontal, labelsize=label_font_size,
    framevisible=false, nbanks=2)

save("result_figures/HHP_h125_with_initial_expression.pdf", HHP_fig)



h_step = 0.1
TT = 100.0

PPD_lode = GeometricProblems.PerturbedPendulum.lodeproblem(timespan=(0, TT), timestep=h_step)
PPD_ref_sol = integrate(PPD_lode,  Gauss(8))

ft_size = 40
lb_size = 50
first_page_figure = Figure(linewidth=2, markersize=10, size=(1000, 1000))
q1_axis = Axis(first_page_figure[1, 1], xlabelsize=60, ylabelsize=60, yticklabelsize=30, xticklabelsize=30,)
lines!(q1_axis, collect(PPD_ref_sol.t), collect(PPD_ref_sol.q[:, 1]), label="Reference Trajectory", linewidth=linewidth)
scatter!(q1_axis, collect(PPD_ref_sol.t), collect(PPD_ref_sol.q[:, 1]), label="Training Samples", color=:blue,)
Label(first_page_figure[0, 1], "Perturbed Pendulum", fontsize=ft_size, tellwidth=false)
Label(first_page_figure[1, 0], "q₁", fontsize=ft_size, tellheight=false)
Label(first_page_figure[2, 1], "time", fontsize=ft_size, tellwidth=false)
Legend(first_page_figure[3, 1], q1_axis, orientation=:horizontal, labelsize=lb_size,
    framevisible=false, nbanks=2)
first_page_figure
save("result_figures/PPD_first_page.pdf", first_page_figure)



x_list = PPD_h5["PerturbedPendulum_x_list"]
function pendulum_q_prediction(t,params)
    params[1] * cos(params[2] * t + params[3])   
end

function pendulum_p_prediction(t,params,q)
    - params[1] * params[2]* sin(params[2] * t + params[3]) + q*(0.3*0.5*sin(2pi/3))
end

PPD_lode = GeometricProblems.PerturbedPendulum.lodeproblem(timespan=(0, TT), timestep=h_step)
PPD_Gau_sol = integrate(PPD_lode, Gauss(8))

PPD_phase_figure = Figure(linewidth=5, markersize=20, size=(1000, 1000))
qp_axis = Axis(PPD_phase_figure[1, 1], xlabelsize=30, ylabelsize=30, yticklabelsize=30, xticklabelsize=30,)


h = 0.05
i = 1
t1 = collect(5.0* (i-1):h:5.0*i)
q1 = [pendulum_q_prediction(ti, x_list[i]) for ti in t1]
p1 = [pendulum_p_prediction(ti, x_list[i],qi) for (ti,qi) in zip(t1, q1)]
line1 =  lines!(qp_axis, q1, p1, label="VISE Continuous Solution")

for i in 2:length(x_list)
    t1 = collect(5.0* (i-1):h:5.0*i)
    q1 = [pendulum_q_prediction(ti, x_list[i]) for ti in t1]
    p1 = [pendulum_p_prediction(ti, x_list[i],qi) for (ti,qi) in zip(t1, q1)]
    lines!(qp_axis, q1, p1, label="VISE Continuous Solution")
end
sca1 = scatter!(qp_axis, PR_sol_q_PPD_h5, PR_sol_p_PPD_h5, label="VISE Discrete Solution", color=:blue)
sca2 = scatter!(qp_axis, collect(PPD_Gau_sol.q[:, 1]), collect(PPD_Gau_sol.p[:, 1]), label="Reference Solution", color=:red,marker =:dtriangle)

Label(PPD_phase_figure[0, 1], "Perturbed Pendulum Phase Space", fontsize=ft_size, tellwidth=false)
Label(PPD_phase_figure[2, 1], "q₁", fontsize=ft_size, tellwidth=false)
Label(PPD_phase_figure[1, 0], "p₁", fontsize=ft_size, tellheight=false)

Legend(PPD_phase_figure[3, 1], [sca1,sca2,line1],["VISE Discrete Solution",
        "Reference Discrete Solution","VISE Continuous Solution"],
orientation = :horizontal,labelsize = lb_size, 
        framevisible = false,nbanks = 3)  

save("result_figures/PPD_phase_space.pdf", PPD_phase_figure)


record_results = load("/Users/zeyuanli/Desktop/untitled folder 2/Backtracking2_R16_h5.00_iter1000_fabs4.44e-16_fsuc1.78e-15_TT200.jld2")

@show record_results[("HenonHeiles_qerror")] 
@show record_results[("HenonHeiles_hams_err")] 

@show record_results[("HenonHeiles_imp_qerror")]
@show record_results[("HenonHeiles_imp_hams_err")]

@show record_results[("HenonHeiles_cgvi_qerror")]
@show record_results[("HenonHeiles_cgvi_hams_err")] 

@show record_results[("HenonHeiles_cgvi_qerror4")]
@show record_results[("HenonHeiles_cgvi_hams_err4")]
