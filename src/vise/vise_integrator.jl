struct VISE{T,NNODES,basisType<:Basis{T}} <: LODEMethod
    basis::basisType
    quadrature

    b::SVector{NNODES,T}
    c::SVector{NNODES,T}

    init_w::Vector{Vector{T}}
    extrapolation_substep::Int

    function VISE(basis::Basis{T}, quadrature, init_w::Vector{Vector{T}};
        extrapolation_substep::Int=10) where {T}
        quad_weights = quadrature.weights
        quad_nodes = quadrature.nodes
        NNODES = nnodes(quadrature)
        new{T,NNODES,typeof(basis)}(basis, quadrature, quad_weights, quad_nodes, init_w, extrapolation_substep)
    end
end

basis(method::VISE) = method.basis
quadrature(method::VISE) = method.quadrature
nnodes(method::VISE) = nnodes(method.quadrature)

isexplicit(::Union{VISE,Type{<:VISE}}) = false
isimplicit(::Union{VISE,Type{<:VISE}}) = true
issymmetric(::Union{VISE,Type{<:VISE}}) = missing
issymplectic(::Union{VISE,Type{<:VISE}}) = missing

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
    s̃::Vector{ST}

    # X::Vector{Vector{ST}}
    Q::Vector{Vector{ST}}
    P::Vector{Vector{ST}}
    V::Vector{Vector{ST}}
    F::Vector{Vector{ST}}

    dqdWc
    dvdWc

    dqdWr₁
    dqdWr₀
    dvdWr₁
    dvdWr₀
    tem_W

    stage_values

    function VISECache{ST,R}(W_sizes, ics) where {ST,R}
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
        s̃ = zeros(ST, D)

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

        stage_values = zeros(ST, 41, D)

        new{ST,R}(x, int_x, q̄, p̄, q̃, p̃, ṽ, f̃, s̃, Q, P, V, F, dqdWc, dvdWc, dqdWr₁, dqdWr₀, dvdWr₁, dvdWr₀, tem_W, stage_values)
    end
end

GeometricIntegratorsBase.nlsolution(cache::VISECache) = cache.x


function GeometricIntegratorsBase.Cache{ST}(problem::AbstractProblemIODE, method::VISE; kwargs...) where {ST}
    VISECache{ST,nnodes(method)}(method.basis.W_sizes, initial_conditions(problem); kwargs...)
end

@inline GeometricIntegratorsBase.CacheType(ST, problem::AbstractProblemIODE, method::VISE) = VISECache{ST,nnodes(method)}

function GeometricIntegratorsBase.internal_variables(method::VISE, problem::AbstractProblemIODE)
    # intermidiate_x = [zeros(Int, length(x)) for x in method(int).init_w]
    S = sum(method.basis.W_sizes)

    intermidiate_x = zeros(S)
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
    if sol.t == h || LinearAlgebra.norm(cache(int).int_x .- vcat(method(int).init_w...)) > 1.0
        x[1:S] = vcat(method(int).init_w...)
    else
        x[1:S] = cache(int).int_x
    end
    println("current time: $(sol.t), initial guess x: $(x[1:S])")

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
        tem_W[d][:] = x[start_idx:start_idx+W_size-1]
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
        dqdWr₀[d][:] = map(f -> f(tem_W[d][:], sol.t - timestep(int)), DQDW[d])
        dqdWr₁[d][:] = map(f -> f(tem_W[d][:], sol.t), DQDW[d])
    end

    # compute Q : q at quadrature points
    for i in eachindex(Q)
        for d in eachindex(Q[i])
            Q[i][d] = q_expr[d](tem_W[d][:], sol.t - timestep(int) + quad_nodes[i] * timestep(int))
        end
    end

    # compute q[t_{n+1}]
    for d in eachindex(q)
        q[d] = q_expr[d](tem_W[d][:], sol.t)
    end

    # compute V volicity at quadrature points
    for i in eachindex(V)
        for d in eachindex(V[i])
            V[i][d] = v_expr[d](tem_W[d][:], sol.t - timestep(int) + quad_nodes[i] * timestep(int))
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


function GeometricIntegratorsBase.update!(sol, params, int::GeometricIntegrator{<:VISE}, DT)
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
    GeometricIntegratorsBase.update!(sol, params, int, DT)
end


function GeometricIntegratorsBase.integrate_step!(sol, history, params, int::GeometricIntegrator{<:VISE,<:AbstractProblemIODE})
    # call nonlinear solver
    # solve!(nlsolution(int), (b,x) -> GeometricIntegratorsBase.residual!(b, x, sol, params, int), solver(int))+
    # Argument order is (x, solver, args), as in `CGVINodal.jl` and the network
    # integrators' shared `integrate_step!`. It read `solve!(solver, x, args)` here, which
    # matches no `SimpleSolvers.solve!` method — a `MethodError` on the first step. It went
    # unnoticed because `test/unit/vise_integrator_unit.jl` was written but never included.
    solve!(nlsolution(int), solver(int), (sol, params, int))

    # print solver status
    # print_solver_status(int.solver.status, int.solver.params)

    # check if solution contains NaNs or error bounds are violated
    # check_solver_status(int.solver.status, int.solver.params)

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

    network_inputs = reshape(collect(0:1/40:1), 1, 41)

    start_idx = 1
    for (d, W_size) in enumerate(W_sizes)
        tem_W[d][:] = x[start_idx:start_idx+W_size-1]
        start_idx += W_size
    end

    # for i in 1:S
    #     for k in 1:D
    #         tem_W[k,i] = x[D*(i-1)+k]
    #     end
    # end
    for d in 1:D
        for i in eachindex(network_inputs)
            stage_values[i, d] = q_expr[d](tem_W[d][:], sol.t - timestep(int) + network_inputs[i] * timestep(int))
        end
    end

end


function create_quadrature_points_derivative_vector(ST::Type, R::Int, D::Int, W_sizes::Vector{Int})
    mat = []
    for d in 1:D
        push!(mat, zeros(ST, R, W_sizes[d]))
    end
    return mat
end

function create_boundary_derivative_vector(ST::Type, D::Int, W_sizes::Vector{Int})
    mat = []
    for d in 1:D
        push!(mat, zeros(ST, W_sizes[d]))
    end
    return mat
end


function GeometricIntegratorsBase.integrate!(sol::GeometricSolution, int::GeometricIntegrator{<:VISE}, n₁::Int, n₂::Int)
    # check time steps range for consistency
    @assert n₁ ≥ 1
    @assert n₂ ≥ n₁
    @assert n₂ ≤ ntime(sol)

    # copy initial condition from solution to solutionstep and initialize
    solstep = solutionstep(int, sol[n₁-1])
    internal_values = Vector{Matrix}(undef, n₂ - n₁ + 1)
    each_step_solution = Vector{Vector}(undef, n₂ - n₁ + 1)
    # loop over time steps
    for n in n₁:n₂
        # integrate one step and copy solution from cache to solution
        reset!(solstep, timesteps(sol)[n])
        integrate!(solstep, int)
        copy!(sol, current(solstep), n)

        internal_values[n] = deepcopy(cache(int).stage_values)
        each_step_solution[n] = deepcopy(nlsolution(int))
    end

    return sol, internal_values, each_step_solution
end
