using GeometricIntegratorsBase
using NonlinearIntegrators
using QuadratureRules
using GeometricProblems.HarmonicOscillator
using GeometricSolutions:relative_maximum_error
using Test
using Logging

# enable all debug logs for this test file
# ENV["JULIA_DEBUG"] = "all"

GeometricIntegratorsBase.default_options(method::NonLinear_OneLayer_GML) = (
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
NLOLCGVNI_Gml_OGA = NonLinear_OneLayer_GML(OLnetwork, QGau,  bias_interval = [-pi,pi], dict_amount = 400000)
HO_NLOLsol_OGA = integrate(HO_lode, NLOLCGVNI_Gml_OGA)
@test relative_maximum_error(HO_NLOLsol_OGA.sol.q,HO_ref.q) < 1e-12

@info "Testing NonLinear_OneLayer_GML with Training initial guess method"
NLOLCGVNI_Gml_Training= NonLinear_OneLayer_GML(OLnetwork, QGau, bias_interval = [-pi,pi], dict_amount = 400000, initial_guess_method=TrainingMethod())
HO_NLOLsol_Training = integrate(HO_lode, NLOLCGVNI_Gml_Training)

@info "The following test is unstable and may fail due to the randomness in the training process. As a result, there is a Random.seed() inside the initial_params! function."
@test relative_maximum_error(HO_NLOLsol_Training.sol.q,HO_ref.q) < 1e-2