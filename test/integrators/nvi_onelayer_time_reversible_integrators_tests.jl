using GeometricIntegratorsBase
using NonlinearIntegrators
using QuadratureRules
using GeometricProblems.HarmonicOscillator
using GeometricSolutions:relative_maximum_error
using Test
using Logging

# enable all debug logs for this test file
# ENV["JULIA_DEBUG"] = "all"

GeometricIntegratorsBase.default_options(method::Time_reversible_OneLayer) = (
    max_iterations = 10000,
    regularization_factor = 1e-5,
    linesearch=GeometricIntegratorsBase.default_linesearch(method), 
)



#HarmonicOscillator Problem
HO_lode = lodeproblem()
HO_ref = exact_solution(podeproblem())

# Define the quadrature rule and the one-layer network
R = 8
QGau = QuadratureRules.GaussLegendreQuadrature(R)

k_relu = 3
S = 4
relu = x->max(0.0,x) ^ k_relu
OLnetwork = OneLayerNetwork_GML{Float64}(relu,S)

# Set up the integrator and solve the problem
@info "Testing NonLinear_OneLayer_GML with OGA1d initial guess method"
NVI_time_reversible_int = Time_reversible_OneLayer(OLnetwork, QGau, show_status = false, bias_interval = [-pi,pi], dict_amount = 400000)
HO_NLOLsol,internal_values = integrate(HO_lode, NVI_time_reversible_int)

@test relative_maximum_error(HO_NLOLsol.q,HO_ref.q) < 1e-12
