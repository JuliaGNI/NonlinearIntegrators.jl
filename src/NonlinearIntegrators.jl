module NonlinearIntegrators

using GeometricEquations
using GeometricIntegratorsBase
import GeometricIntegratorsBase: default_solver, default_options, initsolver, CacheDict, Cache, cache, CacheType, solutionstep, reset!, default_iguess, iguess
import GeometricIntegratorsBase: problem, method, parameters, SolverMethod, history, solver, residual!, copy_internal_variables!, internal, current, update!, solverstate
import GeometricIntegratorsBase: compute_vectorfields!, _extrapolate!, internal_variables, nlsolution, integrate!, IODEIntegratorCache, LODEMethod
import GeometricBase: datatype, timetype, ntime
import GeometricBase: initialtime, finaltime, timespan, timestep, periodicity, NullPeriodicity
using GeometricSolutions: GeometricSolution, timesteps

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
# No `using Zygote`: the one call site (the velocity of the hand-written ansatz in
# `ShallowNetAutodiff`/`ShallowNetAutodiffReversible`) was reverse-mode differentiating a
# scalar ℝ→ℝ function and has been switched to the `ForwardDiff.derivative` that was already
# defined alongside it. Zygote is still reachable transitively through the optimizer stack;
# it is just no longer a direct dependency of this package.
using Random
using Statistics
using StaticArrays
using SimpleSolvers: Newton, solve_with_status!
# `import`, not `using`: `GeometricOptimizers.Newton` is a *different type* from the `Newton` on the
# line above — an `OptimizerMethod` of its own, against SimpleSolvers' `NonlinearSolverMethod` — and
# it is exported. A blanket `using` would therefore put a second, unrelated `Newton` in scope beside
# the one `default_solver(::VISE) = Newton()` means; the explicit `using SimpleSolvers: Newton`
# above would still win the unqualified name, but that is a precedence rule to rely on rather than a
# design. `import` introduces no name at all, so the question does not arise and every optimizer
# call site has to say `GeometricOptimizers.`.
#
# The other two shared names are not clashes: `update!` is GeometricBase's generic in all three
# packages, and `solve!` is SimpleSolvers' own, which GeometricOptimizers `import`s rather than
# defines. This module no longer takes `solve!` from SimpleSolvers — the nonlinear solves go through
# `solve_with_status!` — but qualifying the one remaining call keeps it reading as the optimizer's,
# next to the rest of the seeding code.
import GeometricOptimizers
using SymbolicNeuralNetworks
using AbstractNeuralNetworks
# `AbstractNeuralNetworks` 0.7 no longer exports `NeuralNetworkParameters`: the parameter container
# moved out to the package of that name, where the type is called `NetworkParameters`, and the alias
# left behind is deliberately unexported so that every user of it says where it came from.
import AbstractNeuralNetworks: NeuralNetworkParameters
using LinearAlgebra
using ForwardDiff


include("methods.jl")
export ShallowNetMethod, DenseNetMethod, NetworkIntegratorMethod
export IntegratorExtrapolation
export InitialParametersMethod, TrainingMethod, LSGD

include("nvi/utilities.jl")

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
include("nvi/network_integrator_core.jl")
export NetworkIntegratorCore, NetworkIntegratorCache

include("nvi/network_basis_core.jl")
export NetworkBasisCore

include("nvi/network_basis.jl")
export NetworkBasis, AbstractDenseNetBasis, AbstractShallowNetBasis
export has_symbolic_derivatives

include("nvi/densenet_basis.jl")
export DenseNetBasis

include("nvi/shallownet_basis.jl")
export ShallowNetBasis

# The Lux and vector-valued network variants, and the BSpline bases and integrators, are
# retired: they live under `obsolete/` and are not part of the package.

include("nvi/shallownet.jl")
include("nvi/densenet.jl")
export ShallowNet, DenseNet

include("nvi/shallownet_autodiff.jl")
export ShallowNetAutodiff

include("nvi/shallownet_reversible.jl")
include("nvi/shallownet_autodiff_reversible.jl")
export ShallowNetReversible, ShallowNetAutodiffReversible

# The OGA seeds for the four integrators above, plus the original-paper reference.
include("oga/adapters.jl")
include("oga/normal_equations.jl")

# Variational integrators with a symbolic expression as the ansatz.
using Symbolics
include("vise/vise.jl")
include("vise/vise_basis.jl")
export VISE, VISEBasis

# The linear reference integrator this package used to carry, `CGVINodal` — continuous
# Galerkin on a nodal basis — now lives in GeometricIntegrators alongside `CGVI`, which is where
# a linear variational integrator belongs (JuliaGNI/GeometricIntegrators.jl#219). Reach it as
# `GeometricIntegrators.CGVINodal`, from v0.18.3 on.
end
