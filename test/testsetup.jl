# Shared setup and helpers for the NonlinearIntegrators test suite.
#
# The suite is parametrized over `TEST_TYPES` so that every integrator and basis
# is exercised at more than one floating point precision. The central invariant
# is *no silent upcast*: a run started at `Float32` must stay at `Float32` all the
# way to the final state (see `assert_no_upcast`). Float16 is intentionally
# excluded here — the OGA/Newton path stays ill-conditioned at half precision;
# that case is covered by a dedicated regression test in test/integration.

using Test
using NonlinearIntegrators
using QuadratureRules
using CompactBasisFunctions
using GeometricIntegratorsBase
using GeometricProblems.HarmonicOscillator
using GeometricSolutions: relative_maximum_error
using LinearAlgebra: SingularException
using Symbolics: @variables

const TEST_TYPES = (Float64, Float32)

# Type-generic ReLU^k activation: `max(zero(x), x)^k`, never `max(0.0, x)`, so the
# network is evaluated at the working precision rather than silently upcasting.
relu_k(k::Int = 3) = x -> max(zero(x), x)^k

# ---- basis / quadrature builders -------------------------------------------

build_onelayer_basis(::Type{T}; S = 4, k = 3) where {T} =
    OneLayerNetwork_GML{T}(relu_k(k), S)

build_densenet_basis(::Type{T}; S₁ = 3, S = 3) where {T} =
    DenseNet_GML{T}(tanh, S₁, S)

function build_pr_basis(::Type{T}) where {T}
    @variables tvar
    @variables Wv[1:3]
    q_expr = Wv[1] * cos(Wv[2] * tvar + Wv[3])
    PR_Basis{T}([q_expr], [Wv], tvar, 1)
end

gauss(::Type{T}, R = 8) where {T} = QuadratureRules.GaussLegendreQuadrature(T, R)
lobatto(::Type{T}, R = 4) where {T} = QuadratureRules.LobattoLegendreQuadrature(T, R)

# ---- problem builder --------------------------------------------------------

# Minimal Harmonic Oscillator LODE problem at precision `T`. A short time span and
# a single/couple of steps keep the network solves fast; smoke/unit tests only
# check type propagation, not long-time accuracy.
function ho_problem(::Type{T}; timespan = (T(0.0), T(0.2)), timestep = T(0.1)) where {T}
    params = HarmonicOscillator.default_parameters(T)
    HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)], T;
        timespan = timespan, timestep = timestep, parameters = params)
end

# ---- assertions -------------------------------------------------------------

# The single "no silent upcast" gate. `q` is a solution state variable (indexable
# per time step); its final entry must retain the working element type `T`.
assert_no_upcast(q, ::Type{T}) where {T} = @test eltype(q[end]) == T

# Run a thunk that performs an integration which may legitimately fail to converge
# at reduced precision (near-singular Newton system). Returns the result, or
# `nothing` if the solve was singular. Any other error propagates.
function try_integrate(f)
    try
        return f()
    catch e
        e isa SingularException && return nothing
        rethrow()
    end
end
