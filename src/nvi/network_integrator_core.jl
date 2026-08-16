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
        show_status               :: Bool = false,
        initial_trajectory_method :: ET   = IntegratorExtrapolation(),
        initial_guess_method      :: IPMT = OGA1d(),
        record_grid_points        :: Int  = 41,
    ) where {T, ET <: Extrapolation, IPMT <: InitialParametersMethod}
        NNODES = nnodes(quadrature)
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
nnodes(m::NetworkIntegratorMethod)                       = nnodes(m.quadrature)
activation(m::NetworkIntegratorMethod)                   = m.basis.activation
extrapolation_substep(m::NetworkIntegratorMethod)         = m.common.extrapolation_substep
training_epochs(m::NetworkIntegratorMethod)              = m.common.training_epochs
show_status(m::NetworkIntegratorMethod)                  = m.common.show_status
initial_trajectory_method(m::NetworkIntegratorMethod)    = m.common.initial_trajectory_method

# Shared trait functions 
GeometricIntegratorsBase.isexplicit(::Union{NetworkIntegratorMethod, Type{<:NetworkIntegratorMethod}}) = false
GeometricIntegratorsBase.isimplicit(::Union{NetworkIntegratorMethod, Type{<:NetworkIntegratorMethod}}) = true
GeometricIntegratorsBase.issymmetric(::Union{NetworkIntegratorMethod, Type{<:NetworkIntegratorMethod}}) = missing
# issymmetric = true is overridden in shallownet_reversible.jl and shallownet_autodiff_reversible.jl
# `missing`, not `true`: the continuous-Galerkin construction is symplectic for a linear basis,
# but nothing here establishes it for a network ansatz whose parameters are re-fitted every step.
# Claiming symplecticity would let downstream code select these methods on a property they have
# not been shown to have. Override per integrator once there is a proof or a measurement.
GeometricIntegratorsBase.issymplectic(::Union{NetworkIntegratorMethod, Type{<:NetworkIntegratorMethod}}) = missing

default_solver(::NetworkIntegratorMethod) = Newton()
# `initial_trajectory!` below integrates a LODE sub-problem and reads both `q` and `p` back out, so
# this needs the `IODEProblem`/`LODEProblem` methods of `ImplicitMidpoint` rather than an
# ODE-only or `q`-only implicit midpoint.
default_iguess_integrator(::NetworkIntegratorMethod) = GeometricIntegratorsBase.ImplicitMidpoint()

# `iguess` and `initial_trajectory_method` are two different vocabularies, and conflating them
# was a bug. `initial_trajectory_method` is *ours*: it picks which `initial_trajectory!` runs.
# `iguess` is the *framework's*: it is the extrapolation `GeometricIntegratorsBase.solutionstep!`
# applies, and it only has methods for the framework's own types (`NoInitialGuess`,
# `NoExtrapolation`, `EulerExtrapolation`, `HermiteExtrapolation`, `MidpointExtrapolation`).
#
# `ShallowNetAutodiff`, `ShallowNetAutodiffReversible` and `ShallowNetReversible` used to override
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

# ---- the two shallow-net caches ---------------------------------------------
#
# Four caches used to be declared, one per shallow-net integrator, and within each pair they
# were byte-identical apart from the struct name and *one* line — the length of the unknown
# vector `x`:
#
#   ShallowNet                   D*(3S+1)   ShallowNetReversible          D*(2S+1)
#   ShallowNetAutodiff           D*(3S+1)   ShallowNetAutodiffReversible  D*(2S+1)
#
# (The reversible variants store only the `S/2` independent hidden parameters, their mirror
# images being determined.) That is ~86 lines duplicated twice. The count is a constructor
# argument here, `nx` — the number of unknowns *per dimension* — so each integrator states its
# own layout in one place, next to the `components!` that reads it.
#
# The pair split is real, not cosmetic: the symbolic integrators evaluate `basis.dqdθ`/`dvdθ`
# and need `r₀`/`r₁`/`m`/`a` plus the four `dqd*`/`dvd*` node arrays, while the autodiff ones
# differentiate a flat parameter vector and need `ps_vec` and the two gradient buffers instead.

"""
    SymbolicShallowNetCache{ST}(ics, nx, S, R, N; record_grid_points = 41)

Cache for the shallow-net integrators that evaluate the *symbolically compiled* derivatives of
their basis: [`ShallowNet`](@ref) and [`ShallowNetReversible`](@ref).

`nx` is the number of unknowns per dimension.
"""
struct SymbolicShallowNetCache{ST} <: NetworkIntegratorCache{ST}
    x::Vector{ST}

    q̄::Vector{ST}
    p̄::Vector{ST}

    q̃::Vector{ST}
    p̃::Vector{ST}
    ṽ::Vector{ST}
    f̃::Vector{ST}

    X::Vector{Vector{ST}}
    Q::Vector{Vector{ST}}
    P::Vector{Vector{ST}}
    V::Vector{Vector{ST}}
    F::Vector{Vector{ST}}

    ps::Vector{@NamedTuple{L1::@NamedTuple{W::Matrix{ST}, b::Vector{ST}},L2::@NamedTuple{W::Matrix{ST}}}}

    # One-element input buffer for the network / derivative kernels, which take a *vector* of
    # evaluation points. Every call site used to write the literal `[quad_nodes[j]]`,
    # `[zero(ST)]` or `[one(ST)]`, allocating a fresh vector per quadrature node per dimension
    # on every Newton residual and every Jacobian column.
    tbuf::Vector{ST}

    r₀::Matrix{ST}
    r₁::Matrix{ST}
    m::Array{ST,3}
    a::Array{ST,3}

    dqdWc::Array{ST,3}
    dqdbc::Array{ST,3}
    dvdWc::Array{ST,3}
    dvdbc::Array{ST,3}

    dqdWr₁::Matrix{ST}
    dqdWr₀::Matrix{ST}
    dqdbr₁::Matrix{ST}
    dqdbr₀::Matrix{ST}

    stage_values::Matrix{ST}
    network_labels::Matrix{ST}

    function SymbolicShallowNetCache{ST}(ics, nx::Int, S::Int, R::Int, N::Int;
                                         record_grid_points::Int = 41) where {ST}
        D = length(vec(ics.q))

        x = zeros(ST, D * nx)

        q̄ = zeros(ST, D)
        p̄ = zeros(ST, D)

        q̃ = zeros(ST, D)
        p̃ = zeros(ST, D)
        ṽ = zeros(ST, D)
        f̃ = zeros(ST, D)

        X = create_internal_stage_vector(ST, D, S)
        Q = create_internal_stage_vector(ST, D, R)
        P = create_internal_stage_vector(ST, D, R)
        V = create_internal_stage_vector(ST, D, R)
        F = create_internal_stage_vector(ST, D, R)

        ps = [(L1=(W=zeros(ST, S, 1), b=zeros(ST, S)), L2=(W=zeros(ST, 1, S),)) for _ in 1:D]

        tbuf = zeros(ST, 1)

        r₀ = zeros(ST, S, D)
        r₁ = zeros(ST, S, D)
        m = zeros(ST, R, S, D)
        a = zeros(ST, R, S, D)

        dqdWc = zeros(ST, R, S, D)
        dqdbc = zeros(ST, R, S, D)
        dvdWc = zeros(ST, R, S, D)
        dvdbc = zeros(ST, R, S, D)

        dqdWr₁ = zeros(ST, S, D)
        dqdWr₀ = zeros(ST, S, D)
        dqdbr₁ = zeros(ST, S, D)
        dqdbr₀ = zeros(ST, S, D)

        stage_values = zeros(ST, record_grid_points, D)
        network_labels = zeros(ST, N + 1, D)

        new(x, q̄, p̄, q̃, p̃, ṽ, f̃, X, Q, P, V, F, ps, tbuf, r₀, r₁, m, a,
            dqdWc, dqdbc, dvdWc, dvdbc, dqdWr₁, dqdWr₀, dqdbr₁, dqdbr₀,
            stage_values, network_labels)
    end
end

"""
    AutodiffShallowNetCache{ST}(ics, nx, S, R, N; record_grid_points = 41)

Cache for the shallow-net integrators that differentiate a hand-written ansatz with
`ForwardDiff`: [`ShallowNetAutodiff`](@ref) and [`ShallowNetAutodiffReversible`](@ref).

`nx` is the number of unknowns per dimension.
"""
struct AutodiffShallowNetCache{ST} <: NetworkIntegratorCache{ST}
    x::Vector{ST}

    q̄::Vector{ST}
    p̄::Vector{ST}

    q̃::Vector{ST}
    p̃::Vector{ST}
    ṽ::Vector{ST}
    f̃::Vector{ST}

    # No `X` here, unlike `SymbolicShallowNetCache`. The symbolic pair writes the last-layer
    # weights into `X` and reads them back in `residual!` against `m`/`a`/`r₀`/`r₁`; the
    # autodiff pair goes through `ps_vec` instead and never touches it, so the field was `D`
    # vectors of length `S` allocated per cache and read nowhere — the same thing `s̃` was.
    Q::Vector{Vector{ST}}
    P::Vector{Vector{ST}}
    V::Vector{Vector{ST}}
    F::Vector{Vector{ST}}

    ps::Vector{@NamedTuple{L1::@NamedTuple{W::Matrix{ST}, b::Vector{ST}},L2::@NamedTuple{W::Matrix{ST}}}}

    # Flat [W2 | W1 | b1] view of one dimension's parameters, which is the layout the
    # hand-written ansatz and its ForwardDiff gradients take, plus the two gradient buffers the
    # in-place `∂…!` entry points write into. `components!` used to allocate all three per
    # quadrature node per dimension.
    ps_vec::Vector{ST}
    g_buf::Vector{ST}
    gv_buf::Vector{ST}

    dqdW2c::Array{ST,3}
    dvdW2c::Array{ST,3}
    dqdW1c::Array{ST,3}
    dvdW1c::Array{ST,3}
    dqdbc::Array{ST,3}
    dvdbc::Array{ST,3}

    stage_values::Matrix{ST}
    network_labels::Matrix{ST}

    function AutodiffShallowNetCache{ST}(ics, nx::Int, S::Int, R::Int, N::Int;
                                         record_grid_points::Int = 41) where {ST}
        D = length(vec(ics.q))

        x = zeros(ST, D * nx)

        q̄ = zeros(ST, D)
        p̄ = zeros(ST, D)

        q̃ = zeros(ST, D)
        p̃ = zeros(ST, D)
        ṽ = zeros(ST, D)
        f̃ = zeros(ST, D)

        Q = create_internal_stage_vector(ST, D, R)
        P = create_internal_stage_vector(ST, D, R)
        V = create_internal_stage_vector(ST, D, R)
        F = create_internal_stage_vector(ST, D, R)

        ps = [(L1=(W=zeros(ST, S, 1), b=zeros(ST, S)), L2=(W=zeros(ST, 1, S),)) for _ in 1:D]

        ps_vec = zeros(ST, 3S)
        g_buf  = zeros(ST, 3S)
        gv_buf = zeros(ST, 3S)

        dqdW2c = zeros(ST, R, S, D)
        dvdW2c = zeros(ST, R, S, D)
        dqdW1c = zeros(ST, R, S, D)
        dvdW1c = zeros(ST, R, S, D)
        dqdbc = zeros(ST, R, S, D)
        dvdbc = zeros(ST, R, S, D)

        stage_values = zeros(ST, record_grid_points, D)
        network_labels = zeros(ST, N + 1, D)

        new(x, q̄, p̄, q̃, p̃, ṽ, f̃, Q, P, V, F, ps, ps_vec, g_buf, gv_buf,
            dqdW2c, dvdW2c, dqdW1c, dvdW1c, dqdbc, dvdbc,
            stage_values, network_labels)
    end
end

GeometricIntegratorsBase.nlsolution(cache::NetworkIntegratorCache) = cache.x

function GeometricIntegratorsBase.reset!(cache::NetworkIntegratorCache, _, q, p)
    copyto!(cache.q̄, q)
    copyto!(cache.p̄, p)
end

# Unified initial_guess! for all NetworkIntegratorMethod subtypes.
function GeometricIntegratorsBase.initial_guess!(
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

# Unified HermiteExtrapolation initial_trajectory! (shallow-net template).
# Uses solutionstep! to populate network_labels, then seeds p̃ into x[D*S+k].
# The autodiff types keep their own override (they seed x directly instead of labels).
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

function GeometricIntegratorsBase.residual!(
        b::AbstractVector{ST}, x::AbstractVector{ST}, sol, params,
        int::GeometricIntegrator{<:NetworkIntegratorMethod}) where {ST}
    @assert axes(x) == axes(b)
    GeometricIntegratorsBase.components!(x, sol, params, int)
    GeometricIntegratorsBase.residual!(b, sol, params, int)
end

# `update_solution!` is this package's own function, not an extension of the framework's
# `update!`. Splitting the update into "run components!, then write the cache into `sol`" is a
# local convention — `GeometricIntegratorsBase` has no such second form — and defining it as a
# fourth `update!` method meant a signature of the shape `(Any, Any, GeometricIntegrator, Any)`,
# which is ambiguous against the framework's own `(Any, Any, Any, GeometricIntegrator)`
# (`explicit_euler.jl`). Five of the eight ambiguities Aqua reported were exactly that pair —
# one per DT-form definition: this default plus the four overrides in `ShallowNetAutodiff`,
# `ShallowNetAutodiffReversible`, `CGVINodal` and `VISE` — for no benefit, since nothing
# dispatches on this generically. Giving it its own name removes all five.
#
# The trailing argument is also `::Type{DT}` now rather than an untyped `DT`. It always was a
# type — the x-form below passes the element type of `x` — so this documents it and lets it be
# used as a type parameter directly.
#
# Default: copy q̃/p̃ from the cache into the solution. `ShallowNetAutodiff` and
# `ShallowNetAutodiffReversible` override it, recomputing `p` from the quadrature.
function update_solution!(
        sol, params, int::GeometricIntegrator{<:NetworkIntegratorMethod}, ::Type{DT}) where {DT}
    sol.q .= cache(int, DT).q̃
    sol.p .= cache(int, DT).p̃
end

# The framework extension point: run components!, then delegate to `update_solution!`.
# Identical across all NetworkIntegratorMethod subtypes.
function GeometricIntegratorsBase.update!(
        sol, params, x::AbstractVector{DT},
        int::GeometricIntegrator{<:NetworkIntegratorMethod}) where {DT}
    GeometricIntegratorsBase.components!(x, sol, params, int)
    update_solution!(sol, params, int, DT)
end

# integrate_step!: Newton solve → record finer solution → final update.
# record_finer_solution! runs before update! so that sol.q still holds q_n
# (the start of the step) when the trajectory is recorded.
function GeometricIntegratorsBase.integrate_step!(
        sol, history, params,
        int::GeometricIntegrator{<:NetworkIntegratorMethod, <:AbstractProblemIODE})
    solverstatus = solve_with_status!(nlsolution(int), solver(int), solverstate(int), (sol, params, int))
    check_solver_status(solverstatus, int)
    record_finer_solution!(sol, int)
    GeometricIntegratorsBase.update!(sol, params, nlsolution(int), int)
end

function GeometricIntegratorsBase.integrate!(
        sol::GeometricSolution,
        int::GeometricIntegrator{<:NetworkIntegratorMethod},
        n₁::Int, n₂::Int)
    @assert n₁ ≥ 1
    @assert n₂ ≥ n₁
    @assert n₂ ≤ ntime(sol)

    solstep = solutionstep(int, sol[n₁-1])
    internal_values = Vector{typeof(cache(int).stage_values)}(undef, n₂ - n₁ + 1)

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

        # `hasfield(typeof(...))`, not `hasproperty(...)`: this is a compile-time constant, so
        # the branch folds away instead of being re-tested every step.
        if hasfield(typeof(cache(int)), :stage_values)
            # Offset by n₁: the vector is sized n₂-n₁+1, so indexing by `n` would leave the
            # first n₁-1 slots `#undef` and run off the end for any restart with n₁ > 1.
            # `copy`, not `deepcopy`: `stage_values` is a plain `Matrix{ST}` of floats.
            internal_values[n-n₁+1] = copy(cache(int).stage_values)
        end
    end

    return sol, internal_values
end
