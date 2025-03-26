struct PR_Integrator{T,NNODES,basisType <: Basis{T}} <: LODEMethod
    basis::basisType
    quadrature::QuadratureRule{T,NNODES}

    b::SVector{NNODES,T}
    c::SVector{NNODES,T}

    init_w::Vector{Vector{T}}
    nstages::Int

    function PR_Integrator(basis::Basis{T}, quadrature::QuadratureRule{T}, init_w::Vector{Vector{T}};
        nstages::Int=10) where T
        quad_weights = QuadratureRules.weights(quadrature)
        quad_nodes = QuadratureRules.nodes(quadrature)
        NNODES = QuadratureRules.nnodes(quadrature)
        new{T,NNODES,typeof(basis)}(basis, quadrature, quad_weights, quad_nodes, init_w, nstages)
    end
end 

CompactBasisFunctions.basis(method::PR_Integrator) = method.basis
quadrature(method::PR_Integrator) = method.quadrature
nnodes(method::PR_Integrator) = QuadratureRules.nnodes(method.quadrature)

isexplicit(::Union{PR_Integrator,Type{<:PR_Integrator}}) = false
isimplicit(::Union{PR_Integrator,Type{<:PR_Integrator}}) = true
issymmetric(::Union{PR_Integrator,Type{<:PR_Integrator}}) = missing
issymplectic(::Union{PR_Integrator,Type{<:PR_Integrator}}) = missing

default_solver(::PR_Integrator) = Newton()
nstages(method::PR_Integrator) = method.nstages
default_iguess_integrator(::PR_Integrator) = ImplicitMidpoint()

struct PR_IntegratorCache{ST,D,R} <: IODEIntegratorCache{ST,D}
    x::Vector{ST}

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

    function PR_IntegratorCache{ST,D,R}(W_sizes) where {ST,D,R}
        x = zeros(ST, sum(W_sizes) + D) 

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

        new{ST,D,R}(x, q̄, p̄, q̃, p̃, ṽ, f̃, s̃, Q, P, V, F,dqdWc, dvdWc, dqdWr₁, dqdWr₀, dvdWr₁, dvdWr₀, tem_W,stage_values)
    end
end

GeometricIntegrators.Integrators.nlsolution(cache::PR_IntegratorCache) = cache.x


function GeometricIntegrators.Integrators.Cache{ST}(problem::AbstractProblemIODE, method::PR_Integrator; kwargs...) where {ST}
    PR_IntegratorCache{ST,ndims(problem),nnodes(method)}(method.basis.W_sizes; kwargs...)
end

@inline GeometricIntegrators.Integrators.CacheType(ST, problem::AbstractProblemIODE, method::PR_Integrator) = PR_IntegratorCache{ST,ndims(problem),nnodes(method)}

@inline function Base.getindex(c::PR_IntegratorCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end

function GeometricIntegrators.Integrators.reset!(cache::PR_IntegratorCache, t, q, p)
    copyto!(cache.q̄, q)
    copyto!(cache.p̄, p)
end

function GeometricIntegrators.Integrators.initial_guess!(sol, history, params, int::GeometricIntegrator{<:PR_Integrator})
    local S = sum(method(int).basis.W_sizes)
    local D = ndims(int)
    local x = nlsolution(int)
    local integrator = default_iguess_integrator(method(int))
    local h = int.problem.tstep
    local problem = int.problem

    x[1:S] = vcat(method(int).init_w...)

    tem_ode = similar(problem, [0.0, h], h / 100, (q=StateVariable(sol.q[:]), p=StateVariable(sol.p[:])))
    tem_sol = integrate(tem_ode, integrator)

    for k in 1:D
        cache(int).p̃[k] = tem_sol[1].s.p[:, k][end]
        x[S+k] = cache(int).p̃[k]
    end

end

function GeometricIntegrators.Integrators.components!(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:PR_Integrator}) where {ST}
    local D = ndims(int)
    local S = sum(method(int).basis.W_sizes)
    local W_sizes = method(int).basis.W_sizes
    local C = cache(int, ST)

    local quad_nodes = QuadratureRules.nodes(int.method.quadrature)

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
    for (d,W_size) in enumerate(W_sizes)
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
            dqdWc[d][j, :] = map(f -> f(tem_W[d][:], sol.t - timestep(int) + quad_nodes[j] * timestep(int)), DQDW[d])
            dvdWc[d][j, :] = map(f -> f(tem_W[d][:], sol.t - timestep(int) + quad_nodes[j] * timestep(int)), DVDW[d])
        end
        dqdWr₀[d][:] = map(f -> f(tem_W[d][:], sol.t - timestep(int)), DQDW[d])
        dqdWr₁[d][:] = map(f -> f(tem_W[d][:], sol.t), DQDW[d])
    end

    # compute Q : q at quaadurature points
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


function GeometricIntegrators.Integrators.residual!(b::Vector{ST}, sol, params, int::GeometricIntegrator{<:PR_Integrator}) where {ST}
    local D = ndims(int)
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
function GeometricIntegrators.Integrators.residual!(b::AbstractVector{ST}, x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:PR_Integrator}) where {ST}
    # check that x and b are compatible
    @assert axes(x) == axes(b)

    # compute stages from nonlinear solver solution x
    GeometricIntegrators.Integrators.components!(x, sol, params, int)

    # compute residual vector
    GeometricIntegrators.Integrators.residual!(b, sol, params, int)
end


function GeometricIntegrators.Integrators.update!(sol, params, int::GeometricIntegrator{<:PR_Integrator}, DT)
    sol.q .= cache(int, DT).q̃
    sol.p .= cache(int, DT).p̃
end

function GeometricIntegrators.Integrators.update!(sol, params, x::AbstractVector{DT}, int::GeometricIntegrator{<:PR_Integrator}) where {DT}
    # compute vector field at internal stages
    GeometricIntegrators.Integrators.components!(x, sol, params, int)

    # compute final update
    GeometricIntegrators.Integrators.update!(sol, params, int, DT)
end


function GeometricIntegrators.Integrators.integrate_step!(sol, history, params, int::GeometricIntegrator{<:PR_Integrator, <:AbstractProblemIODE})
    # call nonlinear solver
    solve!(nlsolution(int), (b,x) -> GeometricIntegrators.Integrators.residual!(b, x, sol, params, int), solver(int))

    # print solver status
    # print_solver_status(int.solver.status, int.solver.params)

    # check if solution contains NaNs or error bounds are violated
    # check_solver_status(int.solver.status, int.solver.params)

    # compute final update
    GeometricIntegrators.Integrators.update!(sol, params, nlsolution(int), int)
    
    print("\n x: ",nlsolution(int))

    stages_compute!(sol, int)

end


function stages_compute!(sol, int::GeometricIntegrator{<:PR_Integrator})
    local x = nlsolution(int)
    local stage_values = cache(int).stage_values
    local q_expr = method(int).basis.q_expr
    local D = ndims(int)
    local tem_W = cache(int).tem_W
    local W_sizes = method(int).basis.W_sizes

    network_inputs = reshape(collect(0:1/40:1),1,41)

    start_idx = 1
    for (d,W_size) in enumerate(W_sizes)
        tem_W[d][:] = x[start_idx:start_idx+W_size-1]
        start_idx += W_size
    end

    # for i in 1:S
    #     for k in 1:D
    #         tem_W[k,i] = x[D*(i-1)+k]
    #     end
    # end
    print("\n x: ",x)
    for d in 1:D
        for i in eachindex(network_inputs)
            stage_values[i,d] = q_expr[d](tem_W[d][:], sol.t - timestep(int) + network_inputs[i] * timestep(int))
        end
    end

end


function create_quadrature_points_derivative_vector(ST::Type, R::Int, D::Int,W_sizes::Vector{Int})
    mat = []
    for d in 1:D
        push!(mat, zeros(ST, R, W_sizes[d]))
    end
    return mat
end

function create_boundary_derivative_vector(ST::Type, D::Int,W_sizes::Vector{Int})
    mat = []
    for d in 1:D
        push!(mat, zeros(ST, W_sizes[d]))
    end
    return mat
end
