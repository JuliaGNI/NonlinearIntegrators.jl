using GeometricIntegrators
using CairoMakie
using QuadratureRules
using CompactBasisFunctions
using GeometricProblems
using NonlinearIntegrators

# Set up the Harmonic Oscillator problem
int_step = 1.0
int_timespan = 200.0
HO_lode = GeometricProblems.HarmonicOscillator.lodeproblem(timestep=int_step,timespan=(0,int_timespan))
initial_hamiltonian = GeometricProblems.HarmonicOscillator.hamiltonian(0.0, HO_lode.ics.q, HO_lode.ics.p, HO_lode.parameters)

# Set up the Quadrature Rules 
R = 8
QLob = QuadratureRules.LobattoLegendreQuadrature(R)

# Generate the basis functions on the quadrature nodes
BLob = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QLob))

# mix quadrature and basis functions to form the CGVI methods
sol_QLob_BLob = integrate(HO_lode,CGVI_standard(BLob, QLob))



