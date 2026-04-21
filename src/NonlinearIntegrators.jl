module NonlinearIntegrators

using GeometricIntegrators
using GeometricIntegrators.Integrators: create_internal_stage_vector
using GeometricIntegratorsBase
import GeometricIntegratorsBase: default_solver, default_options, initsolver, CacheDict, Cache, cache, CacheType, solutionstep, reset!, default_iguess, iguess
import GeometricIntegratorsBase: problem, method, parameters, SolverMethod, history, solver, residual!, copy_internal_variables!, internal, current, update!, solverstate
import GeometricIntegratorsBase: _state, _vectorfield, compute_vectorfields!, _extrapolate!, internal_variables, nlsolution, integrate!, IODEIntegratorCache, LODEMethod
import GeometricBase: datatype, timetype, ntime
import GeometricBase: initialtime, finaltime, timespan, timestep, periodicity, NullPeriodicity
using GeometricSolutions: relative_maximum_error

using NonlinearIntegrators
using QuadratureRules
using CompactBasisFunctions
using Zygote
using Random
using Optimisers
using Statistics
using Base
using StaticArrays
using SimpleSolvers: NewtonMethod, Options, NonlinearSolver, Newton, solve!, DogLeg
import GeometricMachineLearning
using SymbolicNeuralNetworks
using AbstractNeuralNetworks
using LinearAlgebra
using BSplineKit
using ForwardDiff
using Infiltrator


include("methods.jl")
export OneLayerMethod, DenseNetMethod, NetworkIntegratorMethod
export IntegratorExtrapolation
export InitialParametersMethod, TrainingMethod, OGA1d, LSGD

include("network_integrators/utilities.jl")

include("network_basis/NetworkBasis.jl")
export NetworkBasis, DenseNetBasis, OneLayerNetBasis

include("network_basis/DenseNet_GML.jl")
# include("network_basis/DenseNet_Lux.jl")
export DenseNet_GML
# export DenseNet_Lux

include("network_basis/OneLayerNetwork_GML.jl")
# include("network_basis/OneLayerNetwork_Lux.jl")
# include("network_basis/OneLayerNetwork.jl")
export OneLayerNetwork_GML
# export OneLayerNetwork_Lux,OneLayerNetwork

# include("network_basis/OneLayerVectorValueNet_Lux.jl")
# include("network_basis/OneLayerVectorValueNet_GML.jl")
# export OneLayerVectorValueNet_Lux,OneLayerVectorValueNet_GML

include("network_integrators/NonLinear_OneLayer_GML.jl")
# include("network_integrators/NonLinear_OneLayer_Lux.jl")
include("network_integrators/NonLinear_DenseNet_GML.jl")
# include("network_integrators/NonLinear_DenseNet_Lux.jl")
# include("network_integrators/Linear_DenseNet_GML.jl")
export NonLinear_OneLayer_GML, NonLinear_DenseNet_GML
# export NonLinear_OneLayer_Lux, NonLinear_DenseNet_Lux, Linear_DenseNet_GML

include("network_integrators/Hardcode_int.jl")
export Hardcode_int


include("network_integrators/Time_reversible_OneLayer.jl")
include("network_integrators/Time_reversible_Hardcode_int.jl")
export Time_Reversible_Hardcode
export Time_reversible_OneLayer

# include("network_integrators/NonLinear_OneLayer_VectorValue_Lux.jl")
# include("network_integrators/NonLinear_OneLayer_VectorValue_GML.jl")
# export NonLinear_OneLayer_VectorValue_Lux, NonLinear_OneLayer_VectorValue_GML

# BSpline
# include("BSpline/BSplineBasis.jl")
# include("BSpline/CGVI_SplineBasis.jl")
# export BSplineDirichlet, CGVI_BSpline

# Nonlinear BSpline
# include("BSpline/NL_BSplineBasis.jl")
# include("BSpline/NL_Spline_CGVI.jl")
# export Nonlinear_BSpline_Basis, Nonlinear_BSpline_Integrator

# Sindy models
using Symbolics
include("SINDy_methods/PR_Int.jl")
include("SINDy_methods/PR_basis.jl")
export PR_Integrator, PR_Basis

# CGVI Standard
include("CGVI_standard/CGVI_standard.jl")
export CGVI_standard
end
