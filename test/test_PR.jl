cd("IntegratorNN")
# cd("..")
using Pkg
Pkg.activate(".")
using Symbolics
using CompactBasisFunctions
using NonlinearIntegrators
using QuadratureRules

using GeometricProblems
using GeometricIntegrators

@variables W[1:3] ttt
q_expr = W[1] *sin(W[2]* ttt + W[3])

PRB = PR_Basis{Float64}([q_expr], [W], ttt,1)
 #TODO: what if the expression is different for different dimensions?
TT = 300.0
h_step = 5.0
HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(tspan = (0,TT),tstep = h_step)
HO_pref = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(tspan = (0,TT),tstep = h_step/50))

QGau4 = QuadratureRules.GaussLegendreQuadrature(8)
PR_Int = PR_Integrator(PRB, QGau4,[[-0.5000433352162222,0.705350078478666,-1.5678140333370576]]) # Pass the init W into the integrator instead of basis                                               
# PR_Int = PR_Integrator(PRB, QGau4,[[-0.500,sqrt(0.5),-pi/2]]) # Pass the init W into the integrator instead of basis                                               

PR_sol = integrate(HO_lode, PR_Int)

using Plots
gr()
plot(collect(0:h_step/50:TT),collect(HO_pref.q[:,1]))
scatter!(collect(0:h_step:TT),collect(PR_sol.q[:,1]))


#### Pendulum
# f_try(x₁) = cos((x₁ * 0.45644) - 1.1466) * 1.1931

@variables W[1:3] ttt
q_expr = W[1] *cos(W[2]* ttt + W[3])
PRB = PR_Basis{Float64}([q_expr], [W], ttt,1)
TT = 50.0
h_step = 5.0
pendulum_lode = GeometricProblems.Pendulum.lodeproblem(tspan = (0,TT),tstep = h_step)
QGau4 = QuadratureRules.GaussLegendreQuadrature(8)

PR_Int = PR_Integrator(PRB, QGau4,[[1.1931,0.45644,-1.1466]]) # Pass the init W into the integrator instead of basis                                               
PR_sol = integrate(pendulum_lode, PR_Int)


pendulum_lode_ref = GeometricProblems.Pendulum.lodeproblem(tspan = (0,TT),tstep = h_step/50)
ref_sol = integrate(pendulum_lode_ref, Gauss(2))

using Plots
plot(collect(0:h_step/50:TT),collect(ref_sol[1].q[:,1]))
scatter!(collect(0:h_step:TT),collect(PR_sol[1].q[:,1]))

plot(collect(0:0.1:200),collect(ref_sol[1].q[:,1]))
# f_try(x) = cos(-0.80705 - (x * -0.96711)) * 0.72078
f_try(x) = cos((x * -0.45644) + 1.1467) / 0.8384
plot(collect(0:0.1:200),f_try.(collect(0:0.1:200)), label="Predicted")



#### Perturbed Pendulum
@variables W[1:3] ttt
q_expr = W[1] *cos(W[2]* ttt + W[3])
PRB = PR_Basis{Float64}([q_expr], [W], ttt,1)
TT = 100.0
h_step = 10.0
perpendulum_lode = GeometricProblems.PerturbedPendulum.lodeproblem(tspan = (0,TT),tstep = h_step)
QGau4 = QuadratureRules.GaussLegendreQuadrature(8)

PR_Int = PR_Integrator(PRB, QGau4,[[-0.51941,-0.47405,2.8713]]) # Pass the init W into the integrator instead of basis                                               
PR_sol = integrate(perpendulum_lode, PR_Int)


ref_lode = GeometricProblems.PerturbedPendulum.lodeproblem(tspan = (0,100))
ref_sol = integrate(ref_lode, ImplicitMidpoint())
plot(collect(0:0.1:TT),collect(ref_sol[1].q[:,1]))
scatter!(collect(0:h_step:TT),collect(PR_sol[1].q[:,1]))



#### Henon Heiles
TT = 100.0
h_step = 2.0
HHlode = GeometricProblems.HenonHeilesPotential.lodeproblem([0.1,0.1],[0.1,0.1],tspan = (0,TT),tstep = h_step)

@variables W1[1:4] ttt
#(0.14831 * cos(-0.64812 + x₁)) - 0.018712
q₁_expr = W1[1] *cos(W1[2]* ttt + W1[3]) + W1[4]

@variables W2[1:4]
# 0.14298 * cos(- 0.97215 * x₁+ 0.7615)))-0.0013983
q₂_expr = W2[1] *cos(W2[2]* ttt + W2[3]) + W2[4]

PRB = PR_Basis{Float64}([q₁_expr,q₂_expr], [W1,W2], ttt,2)
QGau8 = QuadratureRules.GaussLegendreQuadrature(8)
PR_Int = PR_Integrator(PRB, QGau8,[[0.14831,1.0,-0.64812,- 0.018712],[0.14298,- 0.97215,0.7615,-0.0013983]]) # Pass the init W into the integrator instead of basis                                               
PR_sol = integrate(HHlode, PR_Int)


HHlode_ref = GeometricProblems.HenonHeilesPotential.lodeproblem([0.1,0.1],[0.1,0.1],tspan = (0,TT),tstep = h_step/50)
ref_sol = integrate(HHlode_ref, ImplicitMidpoint())

using Plots
plot(collect(0:h_step/50:TT),collect(ref_sol[1].q[:,1]))
scatter!(collect(0:h_step:TT),collect(PR_sol[1].q[:,1]))
savefig("HHh=2.png")

sol_mat = hcat(PR_sol[2][:,2:end,1]'...)[1,:]
plot(collect(h_step/40:h_step/40:TT),sol_mat,label = "Continuous Solution")
scatter!(collect(0:h_step:TT),collect(PR_sol[1].q[:,1]),label = "Discrete Solution")
plot!(collect(0:h_step/50:TT),collect(ref_sol[1].q[:,1]), label = "Reference")



