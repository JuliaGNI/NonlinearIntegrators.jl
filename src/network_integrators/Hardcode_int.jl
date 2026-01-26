struct Hardcode_int{T,NBASIS,NNODES,basisType<:Basis{T},ET<:IntegratorExtrapolation,IPMT<:InitialParametersMethod} <: OneLayerMethod
    basis::basisType
    quadrature::QuadratureRule{T,NNODES}

    b::SVector{NNODES,T}
    c::SVector{NNODES,T}

    nstages::Int
    show_status::Bool
    network_inputs::Matrix{T}

    initial_trajectory::ET
    initial_guess_method::IPMT

    training_epochs::Int
    use_hamiltonian_loss::Bool

    bias_interval::Vector{T}
    dict_amount::Int
    function Hardcode_int(basis::Basis{T}, quadrature::QuadratureRule{T};
        nstages::Int=10, show_status::Bool=true, training_epochs::Int=50000, use_hamiltonian_loss::Bool=true,
        initial_trajectory::ET=IntegratorExtrapolation(),
        initial_guess_method::IPMT=OGA1d(),
        bias_interval=[-pi, pi], dict_amount=50000) where {T,ET,IPMT}
        # get number of quadrature nodes and number of basis functions
        # initial_trajectory_list = subtypes(Extrapolation)
        # @assert initial_trajectory in initial_trajectory_list "initial_trajectory should be one of $(initial_trajectory_list)"

        # initial_guess_methods_list = subtypes(InitialParametersMethod)
        # @assert initial_guess_method in initial_guess_methods_list "initial_guess_methods should be one of $(initial_guess_methods_list)"

        NNODES = QuadratureRules.nnodes(quadrature)
        NBASIS = basis.S

        # get quadrature nodes and weights
        quad_weights = QuadratureRules.weights(quadrature)
        quad_nodes = QuadratureRules.nodes(quadrature)

        network_inputs = reshape(collect(0:1/nstages:1), 1, nstages + 1)
        new{T,NBASIS,NNODES,typeof(basis),ET,IPMT}(basis, quadrature, quad_weights, quad_nodes, nstages, show_status, network_inputs, initial_trajectory, initial_guess_method,
            training_epochs, use_hamiltonian_loss, bias_interval, dict_amount)
    end
end
nbasis(method::Hardcode_int) = method.basis.S
CompactBasisFunctions.basis(method::Hardcode_int) = method.basis
quadrature(method::Hardcode_int) = method.quadrature
CompactBasisFunctions.nbasis(method::Hardcode_int) = method.basis.S
nnodes(method::Hardcode_int) = QuadratureRules.nnodes(method.quadrature)
activation(method::Hardcode_int) = method.basis.activation
nstages(method::Hardcode_int) = method.nstages
show_status(method::Hardcode_int) = method.show_status
training_epochs(method::Hardcode_int) = method.training_epochs

isexplicit(::Union{Hardcode_int,Type{<:Hardcode_int}}) = false
isimplicit(::Union{Hardcode_int,Type{<:Hardcode_int}}) = true
issymmetric(::Union{Hardcode_int,Type{<:Hardcode_int}}) = missing
issymplectic(::Union{Hardcode_int,Type{<:Hardcode_int}}) = missing

default_solver(::Hardcode_int) = NewtonMethod()
default_iguess(::Hardcode_int) = IntegratorExtrapolation()#CoupledHarmonicOscillator
default_iparams(::Hardcode_int) = OGA1d()
# default_iguess_integrator(::Hardcode_int) =  CGVI(Lagrange(QuadratureRules.nodes(QuadratureRules.GaussLegendreQuadrature(4))),QuadratureRules.GaussLegendreQuadrature(4))

default_iguess_integrator(::Hardcode_int) = ImplicitMidpoint()

struct Hardcode_intCache{ST,D,S,R,N} <: IODEIntegratorCache{ST,D}
    x::Vector{ST}

    q̄::Vector{ST}
    p̄::Vector{ST}

    q̃::Vector{ST}
    p̃::Vector{ST}
    ṽ::Vector{ST}
    f̃::Vector{ST}
    s̃::Vector{ST}

    X::Vector{Vector{ST}}
    Q::Vector{Vector{ST}}
    P::Vector{Vector{ST}}
    V::Vector{Vector{ST}}
    F::Vector{Vector{ST}}

    ps::Vector{@NamedTuple{L1::@NamedTuple{W::Matrix{ST}, b::Vector{ST}},L2::@NamedTuple{W::Matrix{ST}}}}

    dqdW2c::Array{ST,3}
    dvdW2c::Array{ST,3}
    dqdW1c::Array{ST,3}
    dvdW1c::Array{ST,3}
    dqdbc::Array{ST,3}
    dvdbc::Array{ST,3}

    dqdW2r₁::Matrix{ST}
    dqdW2r₀::Matrix{ST}
    dqdW1r₁::Matrix{ST}
    dqdW1r₀::Matrix{ST}
    dqdbr₁::Matrix{ST}
    dqdbr₀::Matrix{ST}

    current_step::Vector{ST}
    stage_values::Matrix{ST}
    network_labels::Matrix{ST}

    function Hardcode_intCache{ST,D,S,R,N}() where {ST,D,S,R,N}
        x = zeros(ST, D * (1 + 3 * S)) # Last layer Weight S (no bias for now) + q + hidden layer W S/2 + hidden layer bias S/2

        q̄ = zeros(ST, D)
        p̄ = zeros(ST, D)

        # create temporary vectors
        q̃ = zeros(ST, D)
        p̃ = zeros(ST, D)
        ṽ = zeros(ST, D)
        f̃ = zeros(ST, D)
        s̃ = zeros(ST, D)

        # create internal stage vectors
        X = create_internal_stage_vector(ST, D, S)
        Q = create_internal_stage_vector(ST, D, R)
        P = create_internal_stage_vector(ST, D, R)
        V = create_internal_stage_vector(ST, D, R)
        F = create_internal_stage_vector(ST, D, R)

        # create parameter vectors
        ps = [(L1=(W=zeros(ST, S, 1), b=zeros(ST, S)), L2=(W=zeros(ST, 1, S),)) for k in 1:D]

        dqdW2c = zeros(ST, R, S, D)
        dvdW2c = zeros(ST, R, S, D)
        dqdW1c = zeros(ST, R, S, D)
        dvdW1c = zeros(ST, R, S, D)
        dqdbc = zeros(ST, R, S, D)
        dvdbc = zeros(ST, R, S, D)

        dqdW2r₁ = zeros(ST, S, D)
        dqdW2r₀ = zeros(ST, S, D)
        dqdW1r₁ = zeros(ST, S, D)
        dqdW1r₀ = zeros(ST, S, D)
        dqdbr₁ = zeros(ST, S, D)
        dqdbr₀ = zeros(ST, S, D)

        current_step = zeros(ST, 1)
        stage_values = zeros(ST, 41, D)
        network_labels = zeros(ST, N + 1, D)

        new(x, q̄, p̄, q̃, p̃, ṽ, f̃, s̃, X, Q, P, V, F, ps, 
            dqdW2c, dvdW2c, dqdW1c, dvdW1c, dqdbc, dvdbc,
            dqdW2r₁, dqdW2r₀, dqdW1r₁, dqdW1r₀, dqdbr₁, dqdbr₀,
            current_step, stage_values, network_labels)
    end
end

GeometricIntegrators.Integrators.nlsolution(cache::Hardcode_intCache) = cache.x

function GeometricIntegrators.Integrators.Cache{ST}(problem::AbstractProblemIODE, method::Hardcode_int; kwargs...) where {ST}
    Hardcode_intCache{ST,ndims(problem),nbasis(method),nnodes(method),nstages(method)}(; kwargs...)
end

@inline GeometricIntegrators.Integrators.CacheType(ST, problem::AbstractProblemIODE, method::Hardcode_int) = Hardcode_intCache{ST,ndims(problem),nbasis(method),nnodes(method),nstages(method)}

@inline function Base.getindex(c::Hardcode_intCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end

function GeometricIntegrators.Integrators.reset!(cache::Hardcode_intCache, t, q, p)
    copyto!(cache.q̄, q)
    copyto!(cache.p̄, p)
end

function GeometricIntegrators.Integrators.initial_guess!(sol, history, params, int::GeometricIntegrator{<:Hardcode_int})
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local show_status = method(int).show_status
    local current_step = cache(int).current_step
    local initial_trajectory = method(int).initial_trajectory
    local initial_guess_method = method(int).initial_guess_method

    show_status ? print("\n current time step: $current_step") : nothing
    current_step[1] += 1

    initial_trajectory!(sol, history, params, int, initial_trajectory)

    if show_status
        print("\n network inputs \n")
        print(network_inputs)

        print("\n network labels from initial guess methods \n")
        print(network_labels')
    end

    initial_params!(int, initial_guess_method,sol)
end

function initial_trajectory!(sol, history, params, int::GeometricIntegrator{<:Hardcode_int}, initial_trajectory::HermiteExtrapolation)
    local D = ndims(int)
    local S = nbasis(method(int))
    local x = nlsolution(int)

    # TODO: here we should not initialise with the solution q but with the degree of freedom x,
    # obtained e.g. from an L2 projection of q onto the basis

    for i in eachindex(network_inputs)
        soltmp = (
            t=sol.t + network_inputs[i] * timestep(int),
            q=cache(int).q̃,
            p=cache(int).p̃,
            v=cache(int).ṽ,
            f=cache(int).f̃,
        )
        solutionstep!(soltmp, history, problem(int), iguess(int))

        for k in 1:D
            x[D*(i-1)+k] = cache(int).q̃[k]
        end
    end

    soltmp = (
        t=sol.t,
        q=cache(int).q̃,
        p=cache(int).p̃,
        v=cache(int).ṽ,
        f=cache(int).f̃,
    )
    solutionstep!(soltmp, history, problem(int), iguess(int))

    for k in 1:D
        x[D*S+k] = cache(int).p̃[k]
    end
end

function initial_trajectory!(sol, history, params, int::GeometricIntegrator{<:Hardcode_int}, initial_trajectory::IntegratorExtrapolation)
    local network_labels = cache(int).network_labels
    local integrator = default_iguess_integrator(method(int))
    local h = int.problem.timestep
    local nstages = method(int).nstages
    local D = ndims(int)
    local problem = int.problem
    local S = nbasis(method(int))
    local x = nlsolution(int)

    tem_ode = similar(problem, [0.0, h], h / nstages, (q=StateVariable(sol.q[:]), p=StateVariable(sol.p[:])))
    tem_sol = integrate(tem_ode, integrator)

    for k in 1:D
        network_labels[:, k] = tem_sol.q[:, k]#[1].s
        cache(int).q̃[k] = tem_sol.q[:, k][end]
        cache(int).p̃[k] = tem_sol.p[:, k][end]
        x[D*S+k] = cache(int).q̃[k]
    end
end

function apply_NN(t, ps, S, activation)
    W2 = ps[1:S]
    W1 = ps[S+1:2S]
    b1 = ps[2S+1:3S]

    z1 = W1 .* t .+ b1
    a1 = activation.(z1)
    z2 = sum(W2 .* a1)
    return z2
end

function NN_anstaz(ps, S::Int, activation, t, q̄, q)
    # q_h(t) = (1-t)q_n + t*q_{n+1} + t(1-t)NN(t)
    return (1.0 - t) * q̄ + t * q + t * (1.0 - t) * apply_NN(t, ps, S, activation)
end

VNN_anstaz_zygote(ps, S, activation, t, q̄, q) = Zygote.gradient(tt -> NN_anstaz(ps, S, activation, tt, q̄, q),t)[1]

VNN_anstaz(ps, S, activation, t, q̄, q) = ForwardDiff.derivative(tt -> NN_anstaz(ps, S, activation, tt, q̄, q), t)
∂NN_anstaz_∂params(ps, S, activation, t, q̄, q) = ForwardDiff.gradient(p -> NN_anstaz(p, S, activation, t, q̄, q), ps)
∂VNN_anstaz_∂params(ps, S, activation, t, q̄, q) = ForwardDiff.gradient(p -> VNN_anstaz(p, S, activation, t, q̄, q), ps)

∂NN_anstaz_∂q̄(ps,S,activation,t,q̄,q) = 1.0 .- t
∂NN_anstaz_∂q(ps,S,activation,t,q̄,q) = t

∂VNN_anstaz_∂q̄(ps,S,activation,t,q̄,q)= -1.0
∂VNN_anstaz_∂q(ps,S,activation,t,q̄,q) = 1.0

function initial_params!(int::GeometricIntegrator{<:Hardcode_int}, InitialParams::OGA1d, sol)
    local S = nbasis(method(int))
    local D = ndims(int)
    local quad_nodes = method(int).network_inputs # 假设是 1x(nstages+1) 的行向量
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local network_labels = cache(int).network_labels' # (D x nstages+1)
    local activation = method(int).basis.activation
    local x = nlsolution(int)
    local show_status = method(int).show_status
    local nstages = method(int).nstages
    local bias_interval = method(int).bias_interval
    local dict_amount = method(int).dict_amount
    local q_start = sol.q # 起点 q_n

    # 1. 准备积分权重和 Ansatz 因子
    local t_vec = quad_nodes[:] # 转为向量
    local quad_weights = simpson_quadrature(nstages)
    local t_factor = t_vec .* (1 .- t_vec) # 核心：Ansatz 中的 t(1-t)

    # 2. 构造“带权”基函数字典 (Weighted Dictionary)
    # 每一行代表一个基函数: g_i(t) = t(1-t) * σ(wt + b)
    B = bias_interval[1]:(bias_interval[2]-bias_interval[1])/dict_amount:bias_interval[2]
    # 允许正负权重以覆盖所有可能的非线性形状
    w_vals = [-1.0, 1.0]
    A_dict = hcat(repeat(w_vals, inner=length(B)), repeat(collect(B), outer=length(w_vals)))
    
    gx_quad = zeros(size(A_dict, 1), length(t_vec))
    for i in 1:size(A_dict, 1)
        w, b = A_dict[i, 1], A_dict[i, 2]
        # 基础神经元输出再乘以 t(1-t)
        gx_quad[i, :] = t_factor .* activation.(w .* t_vec .+ b)
    end

    # 归一化字典，提高搜索稳定性
    dict_norms = sqrt.( (gx_quad .^ 2) * quad_weights )
    dict_norms[dict_norms .< 1e-12] .= 1.0
    gx_quad_normed = gx_quad ./ dict_norms

    for d in 1:D
        # 3. 计算拟合目标：目标轨迹减去线性部分
        # 因为 Ansatz 是 q_h = linear + t(1-t)NN，所以 NN 要拟合的部分是 (q_label - linear)
        q_end_guess = network_labels[d, end] # 使用初值积分器给出的终点估计
        f_target = [network_labels[d, j] - ((1-t_vec[j])*q_start[d] + t_vec[j]*q_end_guess) for j in eachindex(t_vec)]
        
        f_res = copy(f_target)
        W1 = zeros(S)
        b1 = zeros(S)
        selected_g = zeros(S, length(t_vec))
        selected_indices = Int[]

        for k = 1:S
            # 4. 贪婪选择：寻找与当前残差方向最一致的神经元
            projections = gx_quad_normed * (f_res .* quad_weights)
            
            for idx in selected_indices
                projections[idx] = 0.0 # 避免重复选择
            end
            
            best_idx = argmax(abs.(projections))
            push!(selected_indices, best_idx)

            W1[k] = A_dict[best_idx, 1]
            b1[k] = A_dict[best_idx, 2]
            selected_g[k, :] = gx_quad[best_idx, :]

            # 5. 正交投影求解输出权重 W2
            # 这里的 Phi 已经包含了 t(1-t) 因子
            Phi = selected_g[1:k, :]
            Gk = Phi * (Phi .* quad_weights')'
            rhs = Phi * (f_target .* quad_weights)
            
            W2_k = (Gk + 1e-12*I) \ rhs

            # 更新残差
            f_res = f_target - (W2_k' * Phi)[:]

            # 写入临时参数结构
            ps[d][1].W[:] .= W1
            ps[d][1].b[:] .= b1
            ps[d][2].W[1:k] .= W2_k
        end
        
        # show_status ? println("Finish OGA for dimension $d, residual MSE: $(sum(f_res.^2))") : nothing
        println("Finish OGA for dimension $d, residual MSE: $(sum(f_res.^2))")
    end


    for k in 1:D
        for i in 1:S
            x[D*(i-1)+k] = ps[k][2].W[i]
        end
        
        x[D*S+k] = cache(int).q̃[k]
        
        for i in 1:S
            x[D*(S+1)+D*(i-1)+k] = ps[k][1].W[i]
            x[D*(S+1+S)+D*(i-1)+k] = ps[k][1].b[i]
        end
    end

    show_status ? println("Initial guess x for Newton solver updated via OGA.") : nothing
end

function GeometricIntegrators.Integrators.components!(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:Hardcode_int}) where {ST}
    local D = ndims(int)
    local S = nbasis(method(int))
    local C = cache(int, ST)

    local quad_nodes = QuadratureRules.nodes(int.method.quadrature)
    local q̄ = sol.q    
    
    local q = cache(int, ST).q̃
    local p = cache(int, ST).p̃
    local Q = cache(int, ST).Q
    local V = cache(int, ST).V
    local P = cache(int, ST).P
    local F = cache(int, ST).F
    local X = cache(int, ST).X

    local NN = method(int).basis.NN
    local ps = cache(int, ST).ps

    local dqdW2c = cache(int, ST).dqdW2c
    local dvdW2c = cache(int, ST).dvdW2c
    local dqdW1c = cache(int, ST).dqdW1c
    local dvdW1c = cache(int, ST).dvdW1c
    local dqdbc = cache(int, ST).dqdbc
    local dvdbc = cache(int, ST).dvdbc

    local dqdW2r₁ = cache(int, ST).dqdW2r₁
    local dqdW2r₀ = cache(int, ST).dqdW2r₀
    local dqdW1r₁ = cache(int, ST).dqdW1r₁
    local dqdW1r₀ = cache(int, ST).dqdW1r₀
    local dqdbr₁ = cache(int, ST).dqdbr₁
    local dqdbr₀ = cache(int, ST).dqdbr₀

    local activation = method(int).basis.activation

    # copy x to q 
    for k in eachindex(q)
        q[k] = x[D*S+k]
    end
        
    for k in 1:D
        for i in 1:S
            ps[k][2].W[i] = x[D*(i-1)+k]
            ps[k][1].W[i] = x[D*(S+1)+D*(i-1)+k]
            ps[k][1].b[i] = x[D*(S+1+S)+D*(i-1)+k]
        end
    end

    ps_vec = zeros(ST, 3S)
    # compute the derivatives of the coefficients on the quadrature nodes and at the boundaries
    for d in 1:D
        ps_vec[1:S] = ps[d][2].W[:]
        ps_vec[S+1:2S] = ps[d][1].W[:]
        ps_vec[2S+1:3S] = ps[d][1].b[:]

        for j in eachindex(quad_nodes)
            # @infiltrate
            g = ∂NN_anstaz_∂params(ps_vec,S,activation,quad_nodes[j],q̄[d],cache(int).q̃[d])
            dqdW2c[j, :, d] = g[1:S]
            dqdW1c[j, :, d] = g[S+1:2S]
            dqdbc[j, :, d] = g[2S+1:3S]

            gv = ∂VNN_anstaz_∂params(ps_vec,S,activation,quad_nodes[j],q̄[d],cache(int).q̃[d])
            dvdW1c[j, :, d] = gv[S+1:2S] 
            dvdbc[j, :, d] = gv[2S+1:3S] 
            dvdW2c[j, :, d] = gv[1:S]
        end

        g0 = ∂NN_anstaz_∂params(ps_vec,S,activation,0.0,q̄[d],cache(int).q̃[d])
        dqdW1r₀[:, d] = g0[S+1:2S]
        dqdbr₀[:, d] = g0[2S+1:3S]
        dqdW2r₀[:, d] = g0[1:S]

        g1 = ∂NN_anstaz_∂params(ps_vec,S,activation,1.0,q̄[d],cache(int).q̃[d])
        dqdW1r₁[:, d] = g1[S+1:2S]
        dqdbr₁[:, d] = g1[2S+1:3S]
        dqdW2r₁[:, d] = g1[1:S]
    end

    # compute Q : q at quaadurature points
    for d in 1:D
        ps_vec = zeros(ST, 3S)
        ps_vec[1:S] = ps[d][2].W[:]
        ps_vec[S+1:2S] = ps[d][1].W[:]
        ps_vec[2S+1:3S] = ps[d][1].b[:]
        for i in eachindex(quad_nodes)
            Q[i][d] = NN_anstaz(ps_vec, S, activation, quad_nodes[i], q̄[d], q[d])
        end
    end

    # compute V volicity at quadrature points
    for d in 1:D
        ps_vec = zeros(ST, 3S)
        ps_vec[1:S] = ps[d][2].W[:]
        ps_vec[S+1:2S] = ps[d][1].W[:]
        ps_vec[2S+1:3S] = ps[d][1].b[:]
        for i in eachindex(quad_nodes)
            V[i][d] = VNN_anstaz_zygote(ps_vec,S,activation,quad_nodes[i],q̄[d],q[d]) / timestep(int)
        end
    end

    # compute P=ϑ(Q,V) and F=f(Q,V)
    for i in eachindex(C.Q, C.V, C.P, C.F)
        tᵢ = sol.t + timestep(int) * (method(int).c[i] - 1)
        equations(int).ϑ(C.P[i], tᵢ, C.Q[i], C.V[i], params)
        equations(int).f(C.F[i], tᵢ, C.Q[i], C.V[i], params)
    end
end


function GeometricIntegrators.Integrators.residual!(b::Vector{ST}, sol, params, int::GeometricIntegrator{<:Hardcode_int}) where {ST}
    local D = ndims(int)
    local S = nbasis(method(int))
    local q̄ = sol.q
    local p̄ = sol.p
    local p̃ = cache(int, ST).p̃
    local P = cache(int, ST).P
    local F = cache(int, ST).F
    local X = cache(int, ST).X


    local dqdW2c = cache(int, ST).dqdW2c
    local dvdW2c = cache(int, ST).dvdW2c
    local dqdW1c = cache(int, ST).dqdW1c
    local dvdW1c = cache(int, ST).dvdW1c
    local dqdbc = cache(int, ST).dqdbc
    local dvdbc = cache(int, ST).dvdbc
    local quad_nodes = QuadratureRules.nodes(int.method.quadrature)

    local show_status = method(int).show_status
    
    # compute b = - [(P-AF)], the residual in actual action, vatiation with respect to Q_{n,i}
    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdW2c[j, i, k]
                z += method(int).b[j] * P[j][k] * dvdW2c[j, i, k]
            end
            b[D*(i-1)+k] =  z
        end
    end

    for k in eachindex(p̄)
        z = zero(ST)
        for j in eachindex(P, F)
            z += timestep(int) * method(int).b[j] * F[j][k] * (1-quad_nodes[j])
            z += method(int).b[j] * P[j][k] * (-1.0)
        end
        b[D*S+k] = p̄[k] + z
    end

    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdW1c[j, i, k]
                z += method(int).b[j] * P[j][k] * dvdW1c[j, i, k]
            end
            b[D*(S+1)+D*(i-1)+k] =  z
        end
    end

    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdbc[j, i, k]
                z += method(int).b[j] * P[j][k] * dvdbc[j, i, k]
            end
            b[D*(S+1+S)+D*(i-1)+k] = z
        end
    end
    # @infiltrate

    show_status ? println(" Residual vector b: \n", b) : nothing
    show_status ? println(" Norm of Residual vector b: ", norm(b)) : nothing
end

# Compute stages of Variational Partitioned Runge-Kutta methods.
function GeometricIntegrators.Integrators.residual!(b::AbstractVector{ST}, x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:Hardcode_int}) where {ST}
    # check that x and b are compatible
    @assert axes(x) == axes(b)

    # compute stages from nonlinear solver solution x
    GeometricIntegrators.Integrators.components!(x, sol, params, int)

    # compute residual vector
    GeometricIntegrators.Integrators.residual!(b, sol, params, int)
end


function GeometricIntegrators.Integrators.update!(sol, params, int::GeometricIntegrator{<:Hardcode_int}, DT)
    local D = ndims(int)
    local quad_nodes = QuadratureRules.nodes(int.method.quadrature)
    local P = cache(int).P
    local F = cache(int).F

    sol.q .= cache(int, DT).q̃

    for k in 1:D
        z = zero(1)
        for j in eachindex(P, F)
            # dQ/dq_{n+1} = τ, dV/dq_{n+1} = 1/h
            z += timestep(int) * method(int).b[j] * F[j][k] * (quad_nodes[j])
            z += method(int).b[j] * P[j][k]
        end
        sol.p[k] = z
    end
    # sol.p .= cache(int, DT).p̃
end

function GeometricIntegrators.Integrators.update!(sol, params, x::AbstractVector{DT}, int::GeometricIntegrator{<:Hardcode_int}) where {DT}
    # compute vector field at internal stages
    GeometricIntegrators.Integrators.components!(x, sol, params, int)

    # compute final update
    GeometricIntegrators.Integrators.update!(sol, params, int, DT)
end

function GeometricIntegrators.Integrators.integrate_step!(sol, history, params, int::GeometricIntegrator{<:Hardcode_int,<:AbstractProblemIODE})
    # call nonlinear solver
    # solve!(nlsolution(int), (b, x) -> GeometricIntegrators.Integrators.residual!(b, x, sol, params, int), solver(int))
    solve!(nlsolution(int),solver(int),  (sol, params, int))

    # print solver status
    # print_solver_status(int.solver.status, int.solver.params)

    # check if solution contains NaNs or error bounds are violated
    # check_solver_status(int.solver.status, int.solver.params)

    #compute the trajectory after solving by newton method
    stages_compute!(sol, int)

    # compute final update
    GeometricIntegrators.Integrators.update!(sol, params, nlsolution(int), int)

end

function stages_compute!(sol, int::GeometricIntegrator{<:Hardcode_int})
    local x = nlsolution(int)
    local stage_values = cache(int).stage_values
    # local network_inputs = method(int).network_inputs
    local D = ndims(int)
    local S = nbasis(method(int))
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local show_status = method(int).show_status
    local q̄ = sol.q  # 起点 q_n
    local q = cache(int).q̃ # 终点估计 q_{n+1}
    local activation = method(int).basis.activation

    network_inputs = reshape(collect(0:1/40:1),1,41)

    if show_status
        print("\n solution x after solving by Newton \n")
        print(x)
    end
    # @infiltrate
    for k in 1:D
        for i in 1:S
            ps[k][2].W[i] = x[D*(i-1)+k]
            ps[k][1].W[i] = x[D*(S+1)+D*(i-1)+k]
            ps[k][1].b[i] = x[D*(S+1+S)+D*(i-1)+k]
        end

        ps_vec = zeros(3S)
        ps_vec[1:S] = ps[k][2].W[:]
        ps_vec[S+1:2S] = ps[k][1].W[:]
        ps_vec[2S+1:3S] = ps[k][1].b[:]

        for i in eachindex(network_inputs)
            stage_values[i, k] = NN_anstaz(ps_vec, S, activation, network_inputs[i], q̄[k], q[k])
        end

        if show_status
            @show ps[k][2].W[:]
            @show ps[k][1].W[:]
            @show ps[k][1].b[:]
        end
    end

    if show_status
        print("\n stages prediction after solving \n")
        print(stage_values)
        print("\n sol from this step \n")
        print("q:", sol.q, "\n")
        print("p:", sol.p, "\n")

    end

end


function GeometricIntegrators.Integrators.integrate!(sol::GeometricSolution, int::GeometricIntegrator{<:Hardcode_int}, n₁::Int, n₂::Int)
    # check time steps range for consistency
    @assert n₁ ≥ 1
    @assert n₂ ≥ n₁
    @assert n₂ ≤ ntime(sol)

    # copy initial condition from solution to solutionstep and initialize
    solstep = solutionstep(int, sol[n₁-1])
    internal_values = Vector{Matrix}(undef,n₂ - n₁ + 1)
    # loop over time steps
    for n in n₁:n₂
        println("Start integrate at time step n = $(n)")
        # integrate one step and copy solution from cache to solution
        sol[n] = integrate!(solstep, int)

        havenan = false
        for s in current(solstep)
            havenan = havenan || any(isnan, s)
        end

        if havenan
            @warn "Solver encountered NaNs in solution at timestep n=$(n)."
            break
        end

        if hasproperty(cache(int),:stage_values)
            internal_values[n] = deepcopy(cache(int).stage_values)
        end
    end

    return sol, internal_values
end

