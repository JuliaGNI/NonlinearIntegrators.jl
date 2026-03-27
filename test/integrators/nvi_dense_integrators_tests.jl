using GeometricIntegrators 
using NonlinearIntegrators
using QuadratureRules
using CompactBasisFunctions
using GeometricProblems.HarmonicOscillator
using Test
using Logging

# enable all debug logs for this test file
# ENV["JULIA_DEBUG"] = "all"

#HarmonicOscillator Problem
HO_lode = lodeproblem()
HO_ref = exact_solution(podeproblem())

# Define the network structure and the quadrature rule
S₁ = 5
S = 5
Densenetwork = DenseNet_GML{Float64}(tanh,S₁,S)


R = 8 
QGau = QuadratureRules.GaussLegendreQuadrature(R)

@info "Testing NonLinear_DenseNet_GML with Training initial guess method"
NL_DenseGML_training = NonLinear_DenseNet_GML(Densenetwork,QGau,training_epochs = 50000,initial_guess_method=TrainingMethod())
HO_Dense_sol_training,internal_values = integrate(HO_lode, NL_DenseGML_training)

@info "The following test is unstable and may fail due to the randomness in the training process. As a result, there is a Random.seed() inside the initial_params! function."
@test relative_maximum_error(HO_Dense_sol_training.q,HO_ref.q) < 1e-2

@info "Testing NonLinear_DenseNet_GML with LSGD initial guess method"
NL_DenseGML_LSGD = NonLinear_DenseNet_GML(Densenetwork,QGau,training_epochs = 50000,initial_guess_method=LSGD())
HO_Dense_sol_LSGD,internal_values = integrate(HO_lode, NL_DenseGML_LSGD)

@info "The following test is unstable and may fail due to the randomness in the training process. As a result, there is a Random.seed() inside the initial_params! function."
@test relative_maximum_error(HO_Dense_sol_LSGD.q,HO_ref.q) < 1e-2