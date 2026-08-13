module NonlinearIntegrators

using GeometricIntegrators
using GeometricIntegrators.Integrators: create_internal_stage_vector
using GeometricIntegratorsBase
import GeometricIntegratorsBase: default_solver, default_options, initsolver, CacheDict, Cache, cache, CacheType, solutionstep, reset!, default_iguess, iguess
import GeometricIntegratorsBase: problem, method, parameters, SolverMethod, history, solver, residual!, copy_internal_variables!, internal, current, update!, solverstate
import GeometricIntegratorsBase: compute_vectorfields!, _extrapolate!, internal_variables, nlsolution, integrate!, IODEIntegratorCache, LODEMethod
import GeometricBase: datatype, timetype, ntime
import GeometricBase: initialtime, finaltime, timespan, timestep, periodicity, NullPeriodicity
using GeometricSolutions: GeometricSolution

using QuadratureRules
using CompactBasisFunctions
# `basis`, `nbasis` and `nnodes` are extended for the bases *and* for the methods, from
# several files. Import them so that a bare definition anywhere extends the one generic
# function per name rather than silently creating a shadowing NonlinearIntegrators.nbasis —
# which is what turned every internal `nbasis(method(int))` call site into a MethodError.
# `basis` and `nnodes` are declared method-free in GeometricBase and extended by both
# CompactBasisFunctions and QuadratureRules; `nbasis` belongs to CompactBasisFunctions.
import GeometricBase: basis, nnodes
import CompactBasisFunctions: nbasis
using Zygote
using Random
using Optimisers
using Statistics
using StaticArrays
using SimpleSolvers: Newton, solve!
# `import`, not `using`: GeometricOptimizers re-exports SimpleSolvers' `solve!` and `Newton` (the
# same generics, which it `import`s), so qualifying its names keeps each call site explicit about
# which package it means.
import GeometricOptimizers
using SymbolicNeuralNetworks
using AbstractNeuralNetworks
using LinearAlgebra
using ForwardDiff


include("methods.jl")
export OneLayerMethod, DenseNetMethod, NetworkIntegratorMethod
export IntegratorExtrapolation
export InitialParametersMethod, TrainingMethod, LSGD

include("network_integrators/utilities.jl")

# Orthogonal Greedy Algorithm. The core (dictionaries, selection rules, fits and the
# greedy loop) is integrator-agnostic and comes first; the per-integrator adapters have
# to come after the integrator definitions, since they dispatch on them.
include("oga/numerics.jl")
include("oga/fits.jl")
include("oga/selection.jl")
include("oga/dictionaries.jl")
include("oga/types.jl")
include("oga/greedy.jl")
export OGA, OGA1d, OGA1dNormalized, OGA1dStable, OGA2d, OGASphere, OGA1dNormalEquations
export OGADictionary, BiasGrid1d, WeightBiasGrid2d, AngularGrid, Refined
export OGASelection, RawProjection, NormalizedProjection, OrthogonalProjection
export OGAFit, WeightedQR, IncrementalQR, PivotedQR, TruncatedSVD, NormalEquationsFit
export OGASymmetry, NoSymmetry, MirrorPairs, SharedMirrorPairs
export oga_fit, OGAResult, oga_label

# Fields, accessors, traits, cache supertype and the shared step/solve machinery that all
# five network integrators have in common. Comes after `oga/types.jl`, whose `OGA1d()` is
# the default `initial_guess_method` in the core constructor.
include("network_integrators/NetworkIntegratorCore.jl")
export NetworkIntegratorCore, NetworkIntegratorCache

include("network_basis/NetworkBasisCore.jl")
export NetworkBasisCore

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

# The OGA seeds for the four integrators above, plus the original-paper reference.
include("oga/adapters.jl")
include("oga/normal_equations.jl")

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
