using Symbolics
using CompactBasisFunctions
using NonlinearIntegrators
using QuadratureRules

using GeometricProblems
using GeometricIntegrators
using SimpleSolvers

GeometricIntegrators.Integrators.default_linesearch(method::PR_Integrator) =SimpleSolvers.Backtracking()

GeometricIntegrators.Integrators.default_options(method::PR_Integrator) = (
    # f_abstol = 8eps(),
    # f_suctol = 2eps(),
    # f_abstol = parse(Float64,eval(ARGS[4])),
    f_suctol = 2eps(),
    f_abstol = 2eps(),
    max_iterations = 10000,
    linesearch=GeometricIntegrators.Integrators.default_linesearch(method),
)

@variables W[1:3] ttt
q_expr = W[1] *sin(W[2]* ttt + W[3])

PRB = PR_Basis{Float64}([q_expr], [W], ttt,1)
TT = 200.0
h_step = 5.0
# h_step = ARGS[2]
HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(timespan = (0,TT),timestep = h_step)

initial_hamiltonian = GeometricProblems.HarmonicOscillator.hamiltonian(0.0, HO_lode.ics.q, HO_lode.ics.p, HO_lode.parameters)
HO_truth = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(timespan = (0,TT),timestep = h_step))
HO_plot = GeometricProblems.HarmonicOscillator.exact_solution(GeometricProblems.HarmonicOscillator.podeproblem(timespan = (0,TT),timestep = h_step/40))

R = 32
QGau4 = QuadratureRules.GaussLegendreQuadrature(R)
PR_Int = PR_Integrator(PRB, QGau4,[[-0.5000433352162222,0.705350078478666,-1.5678140333370576]]) # Pass the init W into the integrator instead of basis                                               
# PR_Int = PR_Integrator(PRB, QGau4,[[-0.500,sqrt(0.5),-pi/2]])                           

println("Start to run Harmonic Oscillator Problem with PR_Integrator! R = $(R), h = $(h_step)")
t1 = time()
PR_sol,internal_sol,x_list = integrate(HO_lode, PR_Int)

@show relative_maximum_error(PR_sol.q,HO_truth.q)



TT = 200.0
h_step = 0.5
HHlode = GeometricProblems.HenonHeilesPotential.lodeproblem([0.1,0.1],[0.1,0.1],timespan = (0,TT),timestep = h_step)
HH_ref_sol = integrate(HHlode, Gauss(8))

HH_initial_hamiltonian = GeometricProblems.HenonHeilesPotential.hamiltonian(0.0, HHlode.ics.q, HHlode.ics.p, HHlode.parameters)
@show HH_initial_hamiltonian

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
HH_PR_sol,HH_internal_sol,HH_x_list = integrate(HHlode, PR_Int)
@show relative_maximum_error(HH_PR_sol.q,HH_ref_sol.q)

