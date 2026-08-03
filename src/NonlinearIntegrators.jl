module NonlinearIntegrators

using GeometricIntegrators
using GeometricIntegrators.Integrators: create_internal_stage_vector
using GeometricIntegratorsBase
import GeometricIntegratorsBase: default_solver, default_options, initsolver, CacheDict, Cache, cache, CacheType, solutionstep, reset!, default_iguess, iguess
import GeometricIntegratorsBase: problem, method, parameters, SolverMethod, history, solver, residual!, copy_internal_variables!, internal, current, update!, solverstate
import GeometricIntegratorsBase: compute_vectorfields!, _extrapolate!, internal_variables, nlsolution, integrate!, IODEIntegratorCache, LODEMethod
import GeometricBase: datatype, timetype, ntime
import GeometricBase: initialtime, finaltime, timespan, timestep, periodicity, NullPeriodicity
using GeometricSolutions: relative_maximum_error

using QuadratureRules
using CompactBasisFunctions
using Zygote
using Random
using Optimisers
using Statistics
using Base
using StaticArrays
using SimpleSolvers: Newton, Options, NonlinearSolver, solve!, DogLeg
import GeometricMachineLearning
using SymbolicNeuralNetworks
using AbstractNeuralNetworks
using LinearAlgebra
using ForwardDiff


include("methods.jl")
export OneLayerMethod, DenseNetMethod, NetworkIntegratorMethod
export IntegratorExtrapolation
export InitialParametersMethod, TrainingMethod, OGA1d, OGA1d_Legacy, LSGD

include("network_integrators/utilities.jl")

include("network_integrators/NetworkIntegratorCore.jl")

include("network_basis/NetworkBasisCore.jl")

include("network_basis/NetworkBasis.jl")
export NetworkBasis, DenseNetBasis, OneLayerNetBasis

include("network_basis/DenseNet_GML.jl")
export DenseNet_GML

include("network_basis/OneLayerNetwork_GML.jl")
export OneLayerNetwork_GML

include("network_integrators/NonLinear_OneLayer_GML.jl")
include("network_integrators/NonLinear_DenseNet_GML.jl")
export NonLinear_OneLayer_GML, NonLinear_DenseNet_GML

include("network_integrators/Hardcode_int.jl")
export Hardcode_int

include("network_integrators/Time_reversible_OneLayer.jl")
include("network_integrators/Time_reversible_Hardcode_int.jl")
export Time_Reversible_Hardcode
export Time_reversible_OneLayer

# Sindy models
using Symbolics
include("SINDy_methods/PR_Int.jl")
include("SINDy_methods/PR_basis.jl")
export PR_Integrator, PR_Basis

# CGVI Standard
include("CGVI_standard/CGVI_standard.jl")
export CGVI_standard
end
