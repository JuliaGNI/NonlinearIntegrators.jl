"""
    VISE(basis::VISEBasis{T}, quadrature, init_w; extrapolation_substep = 10,
         record_grid_points = 41)

Variational integrator on a **symbolic** ansatz: one closed-form expression per degree of
freedom, whose free weights are solved for at every step.

Where the network integrators fit a shallow or dense network, `VISE` takes an ansatz the
caller writes down — `W₁·cos(W₂·t + W₃)`, say — and lets [`VISEBasis`](@ref) differentiate it
symbolically with respect to `t` and to each weight. The discrete Euler–Lagrange equations are
then solved for the weights directly. An ansatz that spans the exact solution therefore
reproduces it to Newton's residual floor rather than to a discretisation order.

# Arguments

  - `basis`: a [`VISEBasis`](@ref) carrying the compiled ansatz and its derivatives.
  - `quadrature`: the quadrature rule for the action integral.
  - `init_w`: one initial weight vector per degree of freedom. Also the fallback restart point:
    `initial_guess!` reuses the previous step's weights unless they have drifted from `init_w`
    by more than 1 in norm.

# Keywords

  - `extrapolation_substep = 10`: sub-steps of the warm-start extrapolation.
  - `record_grid_points = 41`: rows of the per-step `stage_values` record — the continuous
    solution *between* two discrete steps, returned as the second element of `integrate`'s
    tuple. Same keyword as on the network integrators.

# Note

Unlike the network integrators, `integrate` returns a **three**-element tuple here:
`(sol, internal_values, each_step_solution)`, the last being the converged weight vector of
every step.

`quadrature` is a type parameter rather than an untyped field: it was the one untyped field of
this struct, and `method.quadrature.nodes` is read at the top of `components!`, so `quad_nodes`
came back `Any` and poisoned every expression it appeared in — including the arguments of the
compiled basis functions.
"""
struct VISE{T,NNODES,basisType<:Basis{T},quadType} <: LODEMethod
    basis::basisType
    quadrature::quadType

    b::SVector{NNODES,T}
    c::SVector{NNODES,T}

    init_w::Vector{Vector{T}}
    extrapolation_substep::Int
    # Rows of the `stage_values` recording grid, as on the five network integrators. It was a
    # hard-coded 41 in two places that had to agree — the buffer in `VISECache` and the loop in
    # `record_finer_solution!` — with no way to change either.
    record_grid_points::Int

    function VISE(basis::Basis{T}, quadrature, init_w::Vector{Vector{T}};
        extrapolation_substep::Int=10, record_grid_points::Int=41) where {T}
        quad_weights = quadrature.weights
        quad_nodes = quadrature.nodes
        NNODES = nnodes(quadrature)
        new{T,NNODES,typeof(basis),typeof(quadrature)}(basis, quadrature, quad_weights, quad_nodes,
            init_w, extrapolation_substep, record_grid_points)
    end
end

basis(method::VISE) = method.basis
quadrature(method::VISE) = method.quadrature
nnodes(method::VISE) = nnodes(method.quadrature)

# Qualified with `GeometricIntegratorsBase.`, which these four were not.
#
# None of `isexplicit`, `isimplicit`, `issymmetric`, `issymplectic` is imported into this module
# (see the import list at the top of `src/NonlinearIntegrators.jl`), so a bare definition here
# did not extend the framework's generic — it created a *new*, shadowing
# `NonlinearIntegrators.isexplicit` that nothing outside this package ever calls. The framework
# therefore answered `isexplicit(::VISE) === missing` and `isimplicit(::VISE) === missing`,
# i.e. "unknown", where the intent was `false` and `true`. Any downstream code selecting an
# integrator on those properties saw the wrong answer. The network integrators got this right
# (`network_integrator_core.jl` qualifies all four); VISE was the copy that did not.
GeometricIntegratorsBase.isexplicit(::Union{VISE,Type{<:VISE}}) = false
GeometricIntegratorsBase.isimplicit(::Union{VISE,Type{<:VISE}}) = true
GeometricIntegratorsBase.issymmetric(::Union{VISE,Type{<:VISE}}) = missing
GeometricIntegratorsBase.issymplectic(::Union{VISE,Type{<:VISE}}) = missing

default_solver(::VISE) = Newton()
extrapolation_substep(method::VISE) = method.extrapolation_substep
default_iguess_integrator(::VISE) = GeometricIntegratorsBase.ImplicitMidpoint()

struct VISECache{ST,R} <: IODEIntegratorCache{ST}
    x::Vector{ST}
    int_x::Vector{ST}

    q̄::Vector{ST}
    p̄::Vector{ST}

    q̃::Vector{ST}
    p̃::Vector{ST}
    ṽ::Vector{ST}
    f̃::Vector{ST}

    # X::Vector{Vector{ST}}
    Q::Vector{Vector{ST}}
    P::Vector{Vector{ST}}
    V::Vector{Vector{ST}}
    F::Vector{Vector{ST}}

    # All eight of these were untyped (`::Any`), and their *contents* were `Vector{Any}` as
    # well, because the builders started from `mat = []`.
    dqdWc::Vector{Matrix{ST}}
    dvdWc::Vector{Matrix{ST}}

    dqdWr₁::Vector{Vector{ST}}
    dqdWr₀::Vector{Vector{ST}}
    dvdWr₁::Vector{Vector{ST}}
    dvdWr₀::Vector{Vector{ST}}
    tem_W::Vector{Vector{ST}}

    stage_values::Matrix{ST}

    function VISECache{ST,R}(W_sizes, ics; record_grid_points::Int = 41) where {ST,R}
        D = length(vec(ics.q))
        S = sum(W_sizes)
        x = zeros(ST, sum(S) + D)
        int_x = zeros(ST, S)

        q̄ = zeros(ST, D)
        p̄ = zeros(ST, D)

        # create temporary vectors
        q̃ = zeros(ST, D)
        p̃ = zeros(ST, D)
        ṽ = zeros(ST, D)
        f̃ = zeros(ST, D)

        # create internal stage vectors
        # X = create_internal_stage_vector(ST, D, S)
        Q = create_internal_stage_vector(ST, D, R)
        P = create_internal_stage_vector(ST, D, R)
        V = create_internal_stage_vector(ST, D, R)
        F = create_internal_stage_vector(ST, D, R)

        # dqdWc = zeros(ST, R, S, D)
        # dvdWc = zeros(ST, R, S, D)

        # dqdWr₁ = zeros(ST, S, D)
        # dqdWr₀ = zeros(ST, S, D)
        # dvdWr₁ = zeros(ST, S, D)
        # dvdWr₀ = zeros(ST, S, D)
        # tem_W = zeros(ST, D, S)

        dqdWc = create_quadrature_points_derivative_vector(ST, R, D, W_sizes)
        dvdWc = create_quadrature_points_derivative_vector(ST, R, D, W_sizes)

        dqdWr₁ = create_boundary_derivative_vector(ST, D, W_sizes)
        dqdWr₀ = create_boundary_derivative_vector(ST, D, W_sizes)
        dvdWr₁ = create_boundary_derivative_vector(ST, D, W_sizes)
        dvdWr₀ = create_boundary_derivative_vector(ST, D, W_sizes)

        tem_W = create_boundary_derivative_vector(ST, D, W_sizes)

        stage_values = zeros(ST, record_grid_points, D)

        new{ST,R}(x, int_x, q̄, p̄, q̃, p̃, ṽ, f̃, Q, P, V, F, dqdWc, dvdWc, dqdWr₁, dqdWr₀, dvdWr₁, dvdWr₀, tem_W, stage_values)
    end
end

GeometricIntegratorsBase.nlsolution(cache::VISECache) = cache.x


function GeometricIntegratorsBase.Cache{ST}(problem::AbstractProblemIODE, method::VISE; kwargs...) where {ST}
    VISECache{ST,nnodes(method)}(method.basis.W_sizes, initial_conditions(problem);
        record_grid_points = method.record_grid_points, kwargs...)
end

@inline GeometricIntegratorsBase.CacheType(ST, problem::AbstractProblemIODE, method::VISE) = VISECache{ST,nnodes(method)}

function GeometricIntegratorsBase.internal_variables(method::VISE, problem::AbstractProblemIODE)
    # intermidiate_x = [zeros(Int, length(x)) for x in method(int).init_w]
    S = sum(method.basis.W_sizes)

    # `datatype(problem)`, not the `zeros(S)` default: this used to hand back a `Float64`
    # buffer whatever precision the run was started at.
    intermidiate_x = zeros(datatype(problem), S)
    (int_x=intermidiate_x,)
end

function GeometricIntegratorsBase.reset!(cache::VISECache, t, q, p)
    copyto!(cache.q̄, q)
    copyto!(cache.p̄, p)
end

function GeometricIntegratorsBase.initial_guess!(sol, history, params, int::GeometricIntegrator{<:VISE})
    local S = sum(method(int).basis.W_sizes)
    local D = length(cache(int).q̃)
    local x = nlsolution(int)
    local integrator = default_iguess_integrator(method(int))
    local h = timestep(int)
    local problem = int.problem
    # `w₀` is built once: the old form spelled `vcat(method(int).init_w...)` twice, once in the
    # condition and once in the branch, and both are splatted allocations on every step.
    local w₀ = reduce(vcat, method(int).init_w)
    # `sol.t == h` was exact floating-point equality against accumulated time, standing in for
    # "this is the first step". Compare against the problem's own initial time with a tolerance.
    local isfirststep = sol.t ≤ initialtime(problem) + h * (1 + sqrt(eps(typeof(h))))
    if isfirststep || LinearAlgebra.norm(cache(int).int_x .- w₀) > 1.0
        copyto!(view(x, 1:S), w₀)
    else
        copyto!(view(x, 1:S), cache(int).int_x)
    end
    # `@debug`, not `println`: this ran unconditionally on every time step (VISE has no
    # `show_status` field to gate it) and the interpolation allocated a copy of `x[1:S]`.
    @debug "VISE initial guess" t = sol.t x_init = view(x, 1:S)

    tem_ode = similar(problem, [zero(h), h], h / 100, (q=StateVariable(sol.q[:]), p=StateVariable(sol.p[:])))
    tem_sol = integrate(tem_ode, integrator)

    for k in 1:D
        cache(int).p̃[k] = tem_sol.p[:, k][end]
        x[S+k] = cache(int).p̃[k]
    end

end

function GeometricIntegratorsBase.components!(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:VISE}) where {ST}
    local D = length(cache(int).q̃)
    local S = sum(method(int).basis.W_sizes)
    local W_sizes = method(int).basis.W_sizes
    local C = cache(int, ST)

    local quad_nodes = int.method.quadrature.nodes

    local q = cache(int, ST).q̃
    local p = cache(int, ST).p̃
    local Q = cache(int, ST).Q
    local V = cache(int, ST).V
    local P = cache(int, ST).P
    local F = cache(int, ST).F

    local tem_W = cache(int, ST).tem_W
    local dqdWc = cache(int, ST).dqdWc
    local dvdWc = cache(int, ST).dvdWc
    local dqdWr₁ = cache(int, ST).dqdWr₁
    local dqdWr₀ = cache(int, ST).dqdWr₀

    local DVDW = method(int).basis.dvdW
    local DQDW = method(int).basis.dqdW
    local q_expr = method(int).basis.q_expr
    local v_expr = method(int).basis.v_expr


    # for i in eachindex(X)
    #     for k in eachindex(X[i])
    #         tem_W[k,i] = x[D*(i-1)+k]
    #     end
    # end
    # copy x to X.
    start_idx = 1
    for (d, W_size) in enumerate(W_sizes)
        copyto!(tem_W[d], view(x, start_idx:start_idx+W_size-1))
        start_idx += W_size
    end

    # copy x to p # momenta
    for k in eachindex(p)
        p[k] = x[S+k]
    end

    # compute the derivatives of the coefficients on the quadrature nodes and at the boundaries
    for d in 1:D
        for j in eachindex(quad_nodes)
            for p in eachindex(tem_W[d])
                dqdWc[d][j, p] = DQDW[d][p](tem_W[d], sol.t - timestep(int) + quad_nodes[j] * timestep(int))
                dvdWc[d][j, p] = DVDW[d][p](tem_W[d], sol.t - timestep(int) + quad_nodes[j] * timestep(int))
            end
        end
        # `tem_W[d]`, not `tem_W[d][:]`: the compiled function only reads its argument, so
        # the `[:]` copy of the whole weight vector was pure waste — and it was made once per
        # quadrature node per dimension per residual evaluation everywhere it appeared below.
        # A loop replaces `map`, which allocated a fresh vector to immediately copy out of.
        for p in eachindex(DQDW[d])
            dqdWr₀[d][p] = DQDW[d][p](tem_W[d], sol.t - timestep(int))
            dqdWr₁[d][p] = DQDW[d][p](tem_W[d], sol.t)
        end
    end

    # compute Q : q at quadrature points
    for i in eachindex(Q)
        for d in eachindex(Q[i])
            Q[i][d] = q_expr[d](tem_W[d], sol.t - timestep(int) + quad_nodes[i] * timestep(int))
        end
    end

    # compute q[t_{n+1}]
    for d in eachindex(q)
        q[d] = q_expr[d](tem_W[d], sol.t)
    end

    # compute V volicity at quadrature points
    for i in eachindex(V)
        for d in eachindex(V[i])
            V[i][d] = v_expr[d](tem_W[d], sol.t - timestep(int) + quad_nodes[i] * timestep(int))
            # V[i][d] = V[i][d] / timestep(int) #TODO:??? why divide by timestep
        end
    end

    # compute P=ϑ(Q,V) and F=f(Q,V)
    for i in eachindex(Q, V, P, F)
        equations(int).ϑ(P[i], sol.t - timestep(int) + quad_nodes[i] * timestep(int), Q[i], V[i], params)
        equations(int).f(F[i], sol.t - timestep(int) + quad_nodes[i] * timestep(int), Q[i], V[i], params)
    end
end


function GeometricIntegratorsBase.residual!(b::Vector{ST}, sol, params, int::GeometricIntegrator{<:VISE}) where {ST}
    local D = length(cache(int).q̃)
    local S = sum(method(int).basis.W_sizes)
    local W_sizes = method(int).basis.W_sizes

    local q̄ = sol.q
    local p̄ = sol.p
    local p̃ = cache(int, ST).p̃
    local P = cache(int, ST).P
    local F = cache(int, ST).F

    local tem_W = cache(int, ST).tem_W
    local dqdWc = cache(int, ST).dqdWc
    local dvdWc = cache(int, ST).dvdWc
    local dqdWr₁ = cache(int, ST).dqdWr₁
    local dqdWr₀ = cache(int, ST).dqdWr₀
    local q_expr = method(int).basis.q_expr

    # compute b = - [(P-AF)], the residual in actual action, vatiation with respect to Q_{n,i}
    current_idx = 1
    for k in 1:D
        for i in 1:W_sizes[k]
            z = zero(ST)
            for j in eachindex(P, F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdWc[k][j, i]
                z += timestep(int) * method(int).b[j] * P[j][k] * dvdWc[k][j, i]
            end
            b[current_idx] = (dqdWr₁[k][i] * p̃[k] - dqdWr₀[k][i] * p̄[k]) - z
            current_idx += 1
        end
    end

    @assert current_idx == S + 1 "Wrong indexing in residual computation"

    # the continue constraint from hamilton pontryagin principle
    for k in eachindex(q̄)
        b[S+k] = q̄[k] - q_expr[k](tem_W[k][:], sol.t - timestep(int))
    end
end

# Compute stages of Variational Partitioned Runge-Kutta methods.
function GeometricIntegratorsBase.residual!(b::AbstractVector{ST}, x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:VISE}) where {ST}
    # check that x and b are compatible
    @assert axes(x) == axes(b)

    # compute stages from nonlinear solver solution x
    GeometricIntegratorsBase.components!(x, sol, params, int)

    # compute residual vector
    GeometricIntegratorsBase.residual!(b, sol, params, int)
end


function update_solution!(sol, params, int::GeometricIntegrator{<:VISE}, ::Type{DT}) where {DT}
    sol.q .= cache(int, DT).q̃
    sol.p .= cache(int, DT).p̃

    local S = sum(method(int).basis.W_sizes)
    cache(int, DT).int_x .= nlsolution(int)[1:S]
    # sol.internal.int_x .= nlsolution(int)[1:S]
end

function GeometricIntegratorsBase.update!(sol, params, x::AbstractVector{DT}, int::GeometricIntegrator{<:VISE}) where {DT}
    # compute vector field at internal stages
    GeometricIntegratorsBase.components!(x, sol, params, int)

    # compute final update
    update_solution!(sol, params, int, DT)
end


function GeometricIntegratorsBase.integrate_step!(sol, history, params, int::GeometricIntegrator{<:VISE,<:AbstractProblemIODE})
    # Call the nonlinear solver and act on the outcome it reports. Argument order is
    # (x, solver, state, args), as in `cgvi/cgvi.jl` and the network integrators' shared
    # `integrate_step!`. It read `solve!(solver, x, args)` here, which matches no
    # `SimpleSolvers.solve!` method — a `MethodError` on the first step. It went unnoticed
    # because `test/unit/vise_unit.jl` had been written but was not `include`d by `runtests.jl`;
    # it is now, so the same slip would fail the suite.
    solverstatus = solve_with_status!(nlsolution(int), solver(int), solverstate(int), (sol, params, int))
    check_solver_status(solverstatus, int)

    # compute final update
    GeometricIntegratorsBase.update!(sol, params, nlsolution(int), int)
    @debug "VISE solution after solving" nlsolution(int)

    record_finer_solution!(sol, int)
end


function record_finer_solution!(sol, int::GeometricIntegrator{<:VISE})
    local x = nlsolution(int)
    local stage_values = cache(int).stage_values
    local q_expr = method(int).basis.q_expr
    local D = length(cache(int).q̃)
    local tem_W = cache(int).tem_W
    local W_sizes = method(int).basis.W_sizes

    # `record_grid_points` rows, built at the working element type — the same two lines the
    # network integrators use. This was `reshape(collect(0:1/40:1), 1, 41)`: a fixed 41 that had
    # to agree by hand with the `stage_values` buffer, and `Float64` whatever `ST` was.
    local N_plot = method(int).record_grid_points
    local T = eltype(x)
    network_inputs = reshape(collect(range(zero(T), one(T), N_plot)), 1, N_plot)

    # `copyto!`, not `tem_W[d] = …`: assigning the slice would rebind the cache slot to a fresh
    # vector rather than fill the buffer that is already there.
    start_idx = 1
    for (d, W_size) in enumerate(W_sizes)
        copyto!(tem_W[d], view(x, start_idx:start_idx+W_size-1))
        start_idx += W_size
    end

    for d in 1:D
        for i in eachindex(network_inputs)
            stage_values[i, d] = q_expr[d](tem_W[d], sol.t - timestep(int) + network_inputs[i] * timestep(int))
        end
    end

end


# `::Type{ST}`, not `ST::Type`, and a comprehension rather than `push!` onto `[]`.
#
# `ST::Type` passes the element type as a *value*, so the function is not specialised on it and
# `zeros(ST, …)` cannot be inferred. Starting from `mat = []` then made the result a
# `Vector{Any}` whose elements were `Any` too — so `dqdWc[d][j, p] = …` in `components!` was
# three dynamic operations deep, `R × W_size × D` times per residual evaluation.
create_quadrature_points_derivative_vector(::Type{ST}, R::Int, D::Int, W_sizes::Vector{Int}) where {ST} =
    [zeros(ST, R, W_sizes[d]) for d in 1:D]

create_boundary_derivative_vector(::Type{ST}, D::Int, W_sizes::Vector{Int}) where {ST} =
    [zeros(ST, W_sizes[d]) for d in 1:D]


function GeometricIntegratorsBase.integrate!(sol::GeometricSolution, int::GeometricIntegrator{<:VISE}, n₁::Int, n₂::Int)
    # check time steps range for consistency
    @assert n₁ ≥ 1
    @assert n₂ ≥ n₁
    @assert n₂ ≤ ntime(sol)

    # copy initial condition from solution to solutionstep and initialize
    solstep = solutionstep(int, sol[n₁-1])
    # Concrete element types, and offset by n₁: these are sized n₂-n₁+1, so indexing by `n`
    # would leave the first n₁-1 slots `#undef` and run off the end for any restart with
    # n₁ > 1. This is the same fix `network_integrator_core.jl` already carries; the VISE
    # copy of the loop had been left behind.
    internal_values = Vector{typeof(cache(int).stage_values)}(undef, n₂ - n₁ + 1)
    each_step_solution = Vector{typeof(nlsolution(int))}(undef, n₂ - n₁ + 1)
    # loop over time steps
    for n in n₁:n₂
        # integrate one step and copy solution from cache to solution
        reset!(solstep, timesteps(sol)[n])
        integrate!(solstep, int)
        copy!(sol, current(solstep), n)

        internal_values[n-n₁+1] = copy(cache(int).stage_values)
        each_step_solution[n-n₁+1] = copy(nlsolution(int))
    end

    return sol, internal_values, each_step_solution
end
