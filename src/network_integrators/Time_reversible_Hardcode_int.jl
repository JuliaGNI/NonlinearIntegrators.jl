struct Time_Reversible_Hardcode{T, NNODES, basisType <: Basis{T},
                                ET <: Extrapolation,
                                IPMT <: InitialParametersMethod} <: OneLayerMethod
    common        :: NetworkIntegratorCore{T, NNODES, basisType, ET, IPMT}
    bias_interval :: SVector{2, T}
    dict_amount   :: Int

    function Time_Reversible_Hardcode(basis::Basis{T}, quadrature::QuadratureRule{T};
        extrapolation_substep      :: Int  = 10,
        training_epochs           :: Int  = 50000,
        show_status               :: Bool = true,
        initial_trajectory_method :: ET   = IntegratorExtrapolation(),
        initial_guess_method      :: IPMT = OGA1d(),
        bias_interval = [-pi, pi],
        dict_amount   :: Int = 50000) where {T, ET, IPMT}
        common = NetworkIntegratorCore(basis, quadrature;
            extrapolation_substep=extrapolation_substep,
            training_epochs=training_epochs,
            show_status=show_status,
            initial_trajectory_method=initial_trajectory_method,
            initial_guess_method=initial_guess_method)
        new{T, QuadratureRules.nnodes(quadrature), typeof(basis), ET, IPMT}(
            common, SVector{2,T}(bias_interval), dict_amount)
    end
end

GeometricIntegratorsBase.issymmetric(::Union{Time_Reversible_Hardcode, Type{<:Time_Reversible_Hardcode}}) = true

default_iguess(::Time_Reversible_Hardcode) = IntegratorExtrapolation()
default_iparams(::Time_Reversible_Hardcode) = OGA1d()

struct Time_Reversible_HardcodeCache{ST,S,R,N} <: NetworkIntegratorCache{ST}
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

    function Time_Reversible_HardcodeCache{ST,S,R,N}(ics) where {ST,S,R,N}
        D = length(vec(ics.q))
        x = zeros(ST, D * (1 + 2 * S)) # Last layer Weight S (no bias for now) + q + hidden layer W S/2 + hidden layer bias S/2

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

function GeometricIntegrators.Integrators.Cache{ST}(problem::AbstractProblemIODE, method::Time_Reversible_Hardcode; kwargs...) where {ST}
    Time_Reversible_HardcodeCache{ST, nbasis(method), nnodes(method), extrapolation_substep(method)}(initial_conditions(problem); kwargs...)
end

@inline GeometricIntegrators.Integrators.CacheType(ST, problem::AbstractProblemIODE, method::Time_Reversible_Hardcode) =
    Time_Reversible_HardcodeCache{ST, nbasis(method), nnodes(method), extrapolation_substep(method)}

function initial_trajectory!(sol, history, params, int::GeometricIntegrator{<:Time_Reversible_Hardcode}, initial_trajectory::HermiteExtrapolation)
    local D = length(cache(int).q̃)
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

function initial_trajectory!(sol, history, params, int::GeometricIntegrator{<:Time_Reversible_Hardcode}, initial_trajectory::IntegratorExtrapolation)
    local network_labels = cache(int).network_labels
    local integrator = default_iguess_integrator(method(int))
    local h = int.problem.timestep
    local extrapolation_substep = method(int).extrapolation_substep
    local D = length(cache(int).q̃)
    local problem = int.problem
    local S = nbasis(method(int))
    local x = nlsolution(int)

    tem_ode = similar(problem, [zero(h), h], h / extrapolation_substep, (q=StateVariable(sol.q[:]), p=StateVariable(sol.p[:])))
    tem_sol = integrate(tem_ode, integrator)

    for k in 1:D
        network_labels[:, k] = tem_sol.q[:, k]#[1].s
        cache(int).q̃[k] = tem_sol.q[:, k][end]
        cache(int).p̃[k] = tem_sol.p[:, k][end]
        x[D*S+k] = cache(int).q̃[k]
    end
end
# function apply_NN(t, ps, S, activation)
#     W2 = ps[1:S]
#     W1 = ps[S+1:2S]
#     b1 = ps[2S+1:3S]

#     z1 = W1 .* t .+ b1
#     a1 = activation.(z1)
#     z2 = sum(W2 .* a1)
#     return z2
# end

# function NN_anstaz(ps, S::Int, activation, t, q̄, q)
#     # q_h(t) = (1-t)q_n + t*q_{n+1} + t(1-t)NN(t)
#     return (1.0 - t) * q̄ + t * q + t * (1.0 - t) * apply_NN(t, ps, S, activation)
# end

# VNN_anstaz_zygote(ps, S, activation, t, q̄, q) = Zygote.gradient(tt -> NN_anstaz(ps, S, activation, tt, q̄, q),t)[1]

# VNN_anstaz(ps, S, activation, t, q̄, q) = ForwardDiff.derivative(tt -> NN_anstaz(ps, S, activation, tt, q̄, q), t)
# ∂NN_anstaz_∂params(ps, S, activation, t, q̄, q) = ForwardDiff.gradient(p -> NN_anstaz(p, S, activation, t, q̄, q), ps)
# ∂VNN_anstaz_∂params(ps, S, activation, t, q̄, q) = ForwardDiff.gradient(p -> VNN_anstaz(p, S, activation, t, q̄, q), ps)

# ∂NN_anstaz_∂q̄(ps,S,activation,t,q̄,q) = 1.0 .- t
# ∂NN_anstaz_∂q(ps,S,activation,t,q̄,q) = t

# ∂VNN_anstaz_∂q̄(ps,S,activation,t,q̄,q)= -1.0
# ∂VNN_anstaz_∂q(ps,S,activation,t,q̄,q) = 1.0

function initial_params!(int::GeometricIntegrator{<:Time_Reversible_Hardcode}, InitialParams::OGA1d, sol)
    local S = nbasis(method(int))
    local D = length(cache(int).q̃)
    local quad_nodes = method(int).network_inputs # 0:0.1:1 (1x11)
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local network_labels = cache(int).network_labels' # (D x 11)
    local activation = method(int).basis.activation
    local x = nlsolution(int)
    local show_status = method(int).show_status
    local extrapolation_substep = method(int).extrapolation_substep
    local bias_interval = method(int).bias_interval
    local dict_amount = method(int).dict_amount
    local q̄ = sol.q  # 起点 q_n
    local q̃ = cache(int).q̃ # 终点估计 q_{n+1}


    # 1. Quadrature weights and Ansatz factors (working precision T; the seed is
    #    assembled in T rather than a Float64 island).
    local T = eltype(x)
    local t_vec = T.(quad_nodes[:])
    local quad_weights = simpson_quadrature(extrapolation_substep, T)
    local t_factor = t_vec .* (one(T) .- t_vec) # t(1-t) factor in the Ansatz

    # 2. Symmetric dictionary  g_i(t) = t(1-t)·[σ(w t + b) + σ(w(1-t) + b)]  (w = 1;
    #    symmetry is guaranteed by the formula). Bias grid built without the Float16
    #    range trap.
    B = bias_grid(bias_interval[1], bias_interval[2], dict_amount, T)
    A_dict = hcat(ones(T, length(B)), B)
    gx_quad_sym = zeros(T, length(B), length(t_vec))
    for i in axes(A_dict, 1)
        w, b = A_dict[i, 1], A_dict[i, 2]
        val = activation.(w .* t_vec .+ b) .+ activation.(-w .* t_vec .+ (w + b))
        gx_quad_sym[i, :] = t_factor .* val
    end

    for d in 1:D
        # 3. Fit target: reference trajectory minus the linear part
        #    f_target = q_label(t) - [(1-t)q_n + t·q_{n+1}]
        f_target = network_labels[d, :] .- ((one(T) .- t_vec) .* q̄[d] .+ t_vec .* q̃[d])

        W = zeros(T, S)
        Bias = zeros(T, S)
        selected_indices = Int[]
        xk_low = zeros(T, 0)

        # OGA loop: one symmetric neuron pair per iteration.
        for k = 1:S÷2
            current_residual = isempty(selected_indices) ? f_target :
                f_target .- vec(gx_quad_sym[selected_indices, :]' * xk_low)
            projections = (gx_quad_sym * (current_residual .* quad_weights)) .^ 2

            for idx in selected_indices
                projections[idx] = -one(T) # avoid reselection
            end

            best_idx = argmax(projections)
            push!(selected_indices, best_idx)

            W[2k-1] = A_dict[best_idx, 1]
            Bias[2k-1] = A_dict[best_idx, 2]
            W[2k] = -W[2k-1]
            Bias[2k] = W[2k-1] + Bias[2k-1]

            # 4. Refit the (shared) pair weights by weighted QR least squares
            #    (replaces the (Gk + 1e-14·I) \\ rhs normal equations).
            selected_g = gx_quad_sym[selected_indices, :]
            xk_low = weighted_lstsq(selected_g, quad_weights, f_target)

            # Each pair shares one output weight to preserve time-reversal symmetry.
            ps[d][1].W .= W
            ps[d][1].b .= Bias
            for j in 1:k
                ps[d][2].W[2j-1] = xk_low[j]
                ps[d][2].W[2j] = xk_low[j]
            end

            if show_status
                errs = sum((f_target .- vec(selected_g' * xk_low)) .^ 2)
                println("Dimension $d, Pair $k, Current OGA Residual Error: $errs")
            end
        end
        show_status ? println("Finish OGA for dimension $d") : nothing
    end

    # 5. 将结果映射回非线性求解器的初始向量 x
    # 布局必须与 components! 严格一致
    for k in 1:D
        # 终点 q_{n+1}
        x[D*S+k] = q̃[k]

        # 输出层权重 W2
        for i in 1:S
            x[D*(i-1)+k] = ps[k][2].W[i]
        end

        # 隐藏层 W1, b (只映射独立的部分，即 2i-1)
        for i in 1:Int(S/2)
            idx_W1 = Int(D*(S+1)+D*(i-1)+k)
            idx_b  = Int(D*(S+1+S/2)+D*(i-1)+k)
            x[idx_W1] = ps[k][1].W[2i-1]
            x[idx_b]  = ps[k][1].b[2i-1]
        end
    end

    if show_status
        println("Initial guess for DOF from OGA successfully computed.")
    end
end

function GeometricIntegrators.Integrators.components!(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:Time_Reversible_Hardcode}) where {ST}
    local D = length(cache(int).q̃)
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
        end
        for i in 1:Int(S/2)
            ps[k][1].W[Int(2i-1)] = x[Int(D*(S+1)+D*(i-1)+k)]
            ps[k][1].b[Int(2i-1)] = x[Int(D*(S+1+S/2)+D*(i-1)+k)]
            ps[k][1].W[Int(2i)] = -1 * ps[k][1].W[Int(2i-1)]
            ps[k][1].b[Int(2i)] = ps[k][1].W[Int(2i-1)] + ps[k][1].b[Int(2i-1)]
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

        # Boundary points t=0 and t=1 must share the (plain) element type of the
        # quadrature nodes, NOT the solver type ST: during the Newton solve ST is a
        # ForwardDiff.Dual, and ∂NN_anstaz_∂params itself nests a ForwardDiff.gradient,
        # so passing a Dual `t` triggers a Dual-tag ordering error.
        g0 = ∂NN_anstaz_∂params(ps_vec,S,activation,zero(eltype(quad_nodes)),q̄[d],cache(int).q̃[d])
        dqdW1r₀[:, d] = g0[S+1:2S]
        dqdbr₀[:, d] = g0[2S+1:3S]
        dqdW2r₀[:, d] = g0[1:S]

        g1 = ∂NN_anstaz_∂params(ps_vec,S,activation,one(eltype(quad_nodes)),q̄[d],cache(int).q̃[d])
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


function GeometricIntegrators.Integrators.residual!(b::Vector{ST}, sol, params, int::GeometricIntegrator{<:Time_Reversible_Hardcode}) where {ST}
    local D = length(cache(int).q̃)
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

    for i in 1:1:Int(S/2)
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdW1c[j, Int(2*i-1), k]
                z += method(int).b[j] * P[j][k] * dvdW1c[j, Int(2*i-1), k]
            end
            b[Int(D*(S+1)+D*(i-1)+k)] =  z
        end
    end

    for i in 1:1:Int(S/2)
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdbc[j, Int(2*i-1), k]
                z += method(int).b[j] * P[j][k] * dvdbc[j, Int(2*i-1), k]
            end
            b[Int(D*(S+1+S/2)+D*(i-1)+k)] = z
        end
    end

    show_status ? println(" Residual vector b: \n", b) : nothing
    show_status ? println(" Norm of Residual vector b: ", norm(b)) : nothing
end



function GeometricIntegrators.Integrators.update!(sol, params, int::GeometricIntegrator{<:Time_Reversible_Hardcode}, DT)
    local D = length(cache(int).q̃)
    local quad_nodes = QuadratureRules.nodes(int.method.quadrature)
    local P = cache(int).P
    local F = cache(int).F

    sol.q .= cache(int, DT).q̃

    for k in 1:D
        z = zero(DT)
        for j in eachindex(P, F)
            # dQ/dq_{n+1} = τ, dV/dq_{n+1} = 1/h
            z += timestep(int) * method(int).b[j] * F[j][k] * (quad_nodes[j])
            z += method(int).b[j] * P[j][k]
        end
        sol.p[k] = z
    end
    # sol.p .= cache(int, DT).p̃
end



function record_finer_solution!(sol, int::GeometricIntegrator{<:Time_Reversible_Hardcode})
    local x = nlsolution(int)
    local stage_values = cache(int).stage_values
    # local network_inputs = method(int).network_inputs
    local D = length(cache(int).q̃)
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
        end
        for i in 1:Int(S/2)
            ps[k][1].W[Int(2i-1)] = x[Int(D*(S+1)+D*(i-1)+k)]
            ps[k][1].b[Int(2i-1)] = x[Int(D*(S+1+S/2)+D*(i-1)+k)]
            ps[k][1].W[Int(2i)] = -1 * ps[k][1].W[Int(2i-1)]
            ps[k][1].b[Int(2i)] = ps[k][1].W[Int(2i-1)] + ps[k][1].b[Int(2i-1)]
        end

        ps_vec = zeros(eltype(x), 3S)
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


