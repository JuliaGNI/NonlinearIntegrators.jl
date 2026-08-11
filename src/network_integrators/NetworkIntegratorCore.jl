struct NetworkIntegratorCore{T, NNODES, basisType <: Basis{T},
                              ET <: Extrapolation,
                              IPMT <: InitialParametersMethod}
    basis                     :: basisType
    quadrature                :: QuadratureRule{T, NNODES}
    b                         :: SVector{NNODES, T}
    c                         :: SVector{NNODES, T}
    extrapolation_substep      :: Int
    network_inputs            :: Matrix{T}
    initial_trajectory_method :: ET
    initial_guess_method      :: IPMT
    training_epochs           :: Int
    show_status               :: Bool
    record_grid_points        :: Int

    function NetworkIntegratorCore(
        basis :: Basis{T}, quadrature :: QuadratureRule{T};
        extrapolation_substep      :: Int  = 10,
        training_epochs           :: Int  = 50000,
        show_status               :: Bool = true,
        initial_trajectory_method :: ET   = IntegratorExtrapolation(),
        initial_guess_method      :: IPMT = OGA1d(),
        record_grid_points        :: Int  = 41,
    ) where {T, ET <: Extrapolation, IPMT <: InitialParametersMethod}
        NNODES = QuadratureRules.nnodes(quadrature)
        b = SVector{NNODES, T}(QuadratureRules.weights(quadrature))
        c = SVector{NNODES, T}(QuadratureRules.nodes(quadrature))
        network_inputs = reshape(
            collect(zero(T):one(T)/extrapolation_substep:one(T)), 1, extrapolation_substep + 1)
        new{T, NNODES, typeof(basis), ET, IPMT}(
            basis, quadrature, b, c, extrapolation_substep, network_inputs,
            initial_trajectory_method, initial_guess_method, training_epochs, show_status,
            record_grid_points)
    end
end

# Forward NetworkIntegratorCore fields so method.basis, method.extrapolation_substep, etc. keep
# working. The `hasfield` test is a compile-time constant, so it costs nothing; without it a
# future subtype that does not aggregate a `common` would fail on *every* property read,
# including the ones it defines itself.
@inline function Base.getproperty(m::NetworkIntegratorMethod, s::Symbol)
    if hasfield(typeof(m), :common) &&
       s in (:basis, :quadrature, :b, :c, :extrapolation_substep,
             :network_inputs, :initial_trajectory_method, :initial_guess_method,
             :training_epochs, :show_status, :record_grid_points)
        return getfield(getfield(m, :common), s)
    end
    return getfield(m, s)
end

# Without this, `hasproperty(m, :basis)` is false even though `m.basis` works, and REPL
# tab-completion lists only the concrete struct's own fields.
Base.propertynames(m::NetworkIntegratorMethod, private::Bool = false) =
    (fieldnames(typeof(m))..., fieldnames(NetworkIntegratorCore)...)

# Shared accessor functions
basis(m::NetworkIntegratorMethod)  = m.basis
nbasis(m::NetworkIntegratorMethod) = m.basis.S
quadrature(m::NetworkIntegratorMethod)                   = m.quadrature
nnodes(m::NetworkIntegratorMethod)                       = QuadratureRules.nnodes(m.quadrature)
activation(m::NetworkIntegratorMethod)                   = m.basis.activation
extrapolation_substep(m::NetworkIntegratorMethod)         = m.common.extrapolation_substep
training_epochs(m::NetworkIntegratorMethod)              = m.common.training_epochs
show_status(m::NetworkIntegratorMethod)                  = m.common.show_status
initial_trajectory_method(m::NetworkIntegratorMethod)    = m.common.initial_trajectory_method

# Shared trait functions 
GeometricIntegratorsBase.isexplicit(::Union{NetworkIntegratorMethod, Type{<:NetworkIntegratorMethod}}) = false
GeometricIntegratorsBase.isimplicit(::Union{NetworkIntegratorMethod, Type{<:NetworkIntegratorMethod}}) = true
GeometricIntegratorsBase.issymmetric(::Union{NetworkIntegratorMethod, Type{<:NetworkIntegratorMethod}}) = missing
# issymmetric = true is overridden in Time_reversible_OneLayer.jl and Time_reversible_Hardcode_int.jl
# `missing`, not `true`: the continuous-Galerkin construction is symplectic for a linear basis,
# but nothing here establishes it for a network ansatz whose parameters are re-fitted every step.
# Claiming symplecticity would let downstream code select these methods on a property they have
# not been shown to have. Override per integrator once there is a proof or a measurement.
GeometricIntegratorsBase.issymplectic(::Union{NetworkIntegratorMethod, Type{<:NetworkIntegratorMethod}}) = missing

default_solver(::NetworkIntegratorMethod) = Newton()
default_iguess_integrator(::NetworkIntegratorMethod) = ImplicitMidpoint()

# `iguess` and `initial_trajectory_method` are two different vocabularies, and conflating them
# was a bug. `initial_trajectory_method` is *ours*: it picks which `initial_trajectory!` runs.
# `iguess` is the *framework's*: it is the extrapolation `GeometricIntegratorsBase.solutionstep!`
# applies, and it only has methods for the framework's own types (`NoInitialGuess`,
# `NoExtrapolation`, `EulerExtrapolation`, `HermiteExtrapolation`, `MidpointExtrapolation`).
#
# `Hardcode_int`, `Time_Reversible_Hardcode` and `Time_reversible_OneLayer` used to override
# `default_iguess` with this package's `IntegratorExtrapolation`, which is in neither list — so
# requesting `initial_trajectory_method = HermiteExtrapolation()` raised `MethodError` out of
# `solutionstep!` on the first step, at every element type.
#
# The default is therefore left to the framework (`NoInitialGuess()`). Note what that means:
# `solutionstep!(sol, history, problem, ::NoInitialGuess)` returns `sol` unchanged, so the
# Hermite `initial_trajectory!` methods extrapolate nothing unless the caller also passes
# `initialguess = HermiteExtrapolation()` to `GeometricIntegrator`. That is how the benchmark
# harness drives its `hermite` configuration, and it is the supported way to get a real
# Hermite warm start; `initial_trajectory_method` alone only selects the code path.
default_iguess(::NetworkIntegratorMethod) = GeometricIntegratorsBase.NoInitialGuess()

# Shared abstract cache type — concrete caches subtype this instead of IODEIntegratorCache directly.
abstract type NetworkIntegratorCache{ST} <: IODEIntegratorCache{ST} end

GeometricIntegrators.Integrators.nlsolution(cache::NetworkIntegratorCache) = cache.x

function GeometricIntegrators.Integrators.reset!(cache::NetworkIntegratorCache, _, q, p)
    copyto!(cache.q̄, q)
    copyto!(cache.p̄, p)
end

# Unified initial_guess! for all NetworkIntegratorMethod subtypes.
function GeometricIntegrators.Integrators.initial_guess!(
        sol, history, params, int::GeometricIntegrator{<:NetworkIntegratorMethod})
    initial_trajectory!(sol, history, params, int, method(int).initial_trajectory_method)
    @debug "network inputs" method(int).network_inputs
    @debug "network labels" cache(int).network_labels
    initial_params!(int, method(int).initial_guess_method, sol)
end

# Default IntegratorExtrapolation initial_trajectory!.
function initial_trajectory!(
        sol, history, params, int::GeometricIntegrator{<:NetworkIntegratorMethod},
        ::IntegratorExtrapolation)
    local N = extrapolation_substep(method(int))
    local D = length(cache(int).q̃)
    local h = timestep(int)
    local S = nbasis(method(int))
    local x = nlsolution(int)

    tem_ode = similar(int.problem, [zero(h), h], h / N,
        (q=StateVariable(sol.q[:]), p=StateVariable(sol.p[:])))
    tem_sol = integrate(tem_ode, default_iguess_integrator(method(int)))

    for k in 1:D
        cache(int).network_labels[:, k] = tem_sol.q[:, k]
        cache(int).q̃[k] = tem_sol.q[:, k][end]
        cache(int).p̃[k] = tem_sol.p[:, k][end]
        x[D*S + k] = cache(int).p̃[k]
    end
end

# Unified HermiteExtrapolation initial_trajectory! (OneLayer template).
# Uses solutionstep! to populate network_labels, then seeds p̃ into x[D*S+k].
# Hardcode types keep their own override (they seed x directly instead of labels).
# DenseNet keeps its own override (uses initialguess! API).
function initial_trajectory!(
        sol, history, params, int::GeometricIntegrator{<:NetworkIntegratorMethod},
        ::HermiteExtrapolation)
    local N = extrapolation_substep(method(int))
    local D = length(cache(int).q̃)
    local h = timestep(int)
    local S = nbasis(method(int))
    local x = nlsolution(int)
    local network_inputs = method(int).network_inputs

    for i in 1:(N+1)
        soltmp = (
            t = sol.t + (network_inputs[i] - 1) * h,
            q = cache(int).q̃,
            p = cache(int).p̃,
            q̇ = cache(int).ṽ,
            ṗ = cache(int).f̃,
        )
        solutionstep!(soltmp, history, problem(int), iguess(int))
        for k in 1:D
            cache(int).network_labels[i, k] = cache(int).q̃[k]
        end
    end
    soltmp = (
        t = sol.t,
        q = cache(int).q̃,
        p = cache(int).p̃,
        q̇ = cache(int).ṽ,
        ṗ = cache(int).f̃,
    )
    solutionstep!(soltmp, history, problem(int), iguess(int))
    for k in 1:D
        x[D*S + k] = cache(int).p̃[k]
    end
end

# "No initial guess": rather than extrapolating a trajectory, use the previous
# solution as a constant seed. Every stage label is set to the previous qₙ (so the
# subsequent OGA/parameter fit targets a flat trajectory) and the momentum degree of
# freedom is seeded with the previous pₙ. This is the cheapest possible warm start and
# is useful as a baseline against the midpoint/Hermite extrapolations.
function initial_trajectory!(
        sol, history, params, int::GeometricIntegrator{<:NetworkIntegratorMethod},
        ::NoExtrapolation)
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local x = nlsolution(int)
    for k in 1:D
        cache(int).network_labels[:, k] .= sol.q[k]
        cache(int).q̃[k] = sol.q[k]
        cache(int).p̃[k] = sol.p[k]
        x[D*S + k] = sol.p[k]
    end
end

function GeometricIntegrators.Integrators.residual!(
        b::AbstractVector{ST}, x::AbstractVector{ST}, sol, params,
        int::GeometricIntegrator{<:NetworkIntegratorMethod}) where {ST}
    @assert axes(x) == axes(b)
    GeometricIntegrators.Integrators.components!(x, sol, params, int)
    GeometricIntegrators.Integrators.residual!(b, sol, params, int)
end

# Default DT-form update!: copy q̃/p̃ from cache into solution.
# Hardcode_int and Time_Reversible_Hardcode override this with physics-specific momentum.
function GeometricIntegrators.Integrators.update!(
        sol, params, int::GeometricIntegrator{<:NetworkIntegratorMethod}, DT)
    sol.q .= cache(int, DT).q̃
    sol.p .= cache(int, DT).p̃
end

# x-form update!: run components! then delegate to DT-form update!.
# Identical across all NetworkIntegratorMethod subtypes.
function GeometricIntegrators.Integrators.update!(
        sol, params, x::AbstractVector{DT},
        int::GeometricIntegrator{<:NetworkIntegratorMethod}) where {DT}
    GeometricIntegrators.Integrators.components!(x, sol, params, int)
    GeometricIntegrators.Integrators.update!(sol, params, int, DT)
end

# integrate_step!: Newton solve → record finer solution → final update.
# record_finer_solution! runs before update! so that sol.q still holds q_n
# (the start of the step) when the trajectory is recorded.
function GeometricIntegrators.Integrators.integrate_step!(
        sol, history, params,
        int::GeometricIntegrator{<:NetworkIntegratorMethod, <:AbstractProblemIODE})
    solve!(nlsolution(int), solver(int), solverstate(int), (sol, params, int))
    record_finer_solution!(sol, int)
    GeometricIntegrators.Integrators.update!(sol, params, nlsolution(int), int)
end

function GeometricIntegrators.Integrators.integrate!(
        sol::GeometricSolution,
        int::GeometricIntegrator{<:NetworkIntegratorMethod},
        n₁::Int, n₂::Int)
    @assert n₁ ≥ 1
    @assert n₂ ≥ n₁
    @assert n₂ ≤ ntime(sol)

    solstep = solutionstep(int, sol[n₁-1])
    internal_values = Vector{Matrix}(undef, n₂ - n₁ + 1)

    for n in n₁:n₂
        @debug "integrate! step" n
        reset!(solstep, timesteps(sol)[n])
        integrate!(solstep, int)
        copy!(sol, current(solstep), n)

        havenan = false
        for s in current(solstep)
            havenan = havenan || any(isnan, s)
        end
        if havenan
            @warn "Solver encountered NaNs in solution at timestep n=$(n)."
            # break
        end

        if hasproperty(cache(int), :stage_values)
            # Offset by n₁: the vector is sized n₂-n₁+1, so indexing by `n` would leave the
            # first n₁-1 slots `#undef` and run off the end for any restart with n₁ > 1.
            internal_values[n-n₁+1] = deepcopy(cache(int).stage_values)
        end
    end

    return sol, internal_values
end
