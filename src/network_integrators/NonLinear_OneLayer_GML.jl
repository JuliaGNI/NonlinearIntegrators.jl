struct NonLinear_OneLayer_GML{T, NNODES, basisType <: Basis{T},
                               ET <: Extrapolation,
                               IPMT <: InitialParametersMethod} <: OneLayerMethod
    common        :: NetworkIntegratorCore{T, NNODES, basisType, ET, IPMT}
    bias_interval :: SVector{2, T}
    dict_amount   :: Int

    function NonLinear_OneLayer_GML(basis::Basis{T}, quadrature::QuadratureRule{T};
        extrapolation_substep      :: Int  = 10,
        training_epochs           :: Int  = 50000,
        initial_trajectory_method :: ET   = IntegratorExtrapolation(),
        initial_guess_method      :: IPMT = OGA1d(),
        bias_interval = [-pi, pi],
        dict_amount   :: Int = 50000) where {T, ET, IPMT}
        common = NetworkIntegratorCore(basis, quadrature;
            extrapolation_substep=extrapolation_substep,
            training_epochs=training_epochs,
            initial_trajectory_method=initial_trajectory_method,
            initial_guess_method=initial_guess_method)
        new{T, QuadratureRules.nnodes(quadrature), typeof(basis), ET, IPMT}(
            common, SVector{2,T}(bias_interval), dict_amount)
    end
end

struct NonLinear_OneLayer_GMLCache{ST,S,R,N,NEpochs} <: NetworkIntegratorCache{ST}
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

    training_errors::Matrix{ST}
    mse_err::Vector{ST}
    abs_err::Vector{ST}
    training_time::Vector{ST}
    solving_time::Vector{ST}
    integrating_time::Vector{ST}

    function NonLinear_OneLayer_GMLCache{ST,S,R,N,NEpochs}(ics) where {ST,S,R,N,NEpochs}
        D = length(vec(ics.q))
        x = zeros(ST, D * (S + 1 + 2 * S)) # Last layer Weight S (no bias for now) + P + hidden layer W (S*S₁) + hidden layer bias S

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

        stage_values = zeros(ST, 41, D)
        network_labels = zeros(ST, N + 1, D)

        new(x, q̄, p̄, q̃, p̃, ṽ, f̃, s̃, X, Q, P, V, F, ps, r₀, r₁, m, a,
            dqdWc, dqdbc, dvdWc, dvdbc, dqdWr₁, dqdWr₀, dqdbr₁, dqdbr₀,
            stage_values, network_labels)
    end
end

function GeometricIntegrators.Integrators.Cache{ST}(problem::AbstractProblemIODE, method::NonLinear_OneLayer_GML; kwargs...) where {ST}
    NonLinear_OneLayer_GMLCache{ST, nbasis(method), nnodes(method),
        extrapolation_substep(method), method.training_epochs}(initial_conditions(problem); kwargs...)
end

@inline GeometricIntegrators.Integrators.CacheType(ST, problem::AbstractProblemIODE, method::NonLinear_OneLayer_GML) =
    NonLinear_OneLayer_GMLCache{ST, nbasis(method), nnodes(method),
        extrapolation_substep(method), method.training_epochs}


function initial_trajectory!(sol, history, params, int::GeometricIntegrator{<:NonLinear_OneLayer_GML}, initial_trajectory::GeometricIntegratorsBase.HermiteExtrapolation)
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local x = nlsolution(int)
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels

    # TODO: here we should not initialise with the solution q but with the degree of freedom x,
    # obtained e.g. from an L2 projection of q onto the basis

    for i in eachindex(network_inputs)
        soltmp = (
            t=sol.t + (network_inputs[i]-1) * timestep(int),
            q=cache(int).q̃,
            p=cache(int).p̃,
            q̇=cache(int).ṽ,
            ṗ=cache(int).f̃,
        )
        solutionstep!(soltmp, history, problem(int), iguess(int))
        for k in 1:D
            network_labels[i, k] = cache(int).q̃[k]
        end
    end
    soltmp = (
        t=sol.t,
        q=cache(int).q̃,
        p=cache(int).p̃,
        q̇=cache(int).ṽ,
        ṗ=cache(int).f̃,
    )
    solutionstep!(soltmp, history, problem(int), iguess(int))

    for k in 1:D
        x[D*S+k] = cache(int).p̃[k]
    end
end

function initial_trajectory!(sol, history, params, int::GeometricIntegrator{<:NonLinear_OneLayer_GML}, initial_trajectory::IntegratorExtrapolation)
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
        x[D*S+k] = cache(int).p̃[k]
    end
end

# "No initial guess": rather than extrapolating a trajectory, use the previous
# solution as a constant seed. Every stage label is set to the previous qₙ (so the
# subsequent OGA/parameter fit targets a flat trajectory) and the momentum degree of
# freedom is seeded with the previous pₙ. This is the cheapest possible warm start and
# is useful as a baseline against the midpoint/Hermite extrapolations.
function initial_trajectory!(sol, history, params, int::GeometricIntegrator{<:NonLinear_OneLayer_GML}, initial_trajectory::NoExtrapolation)
    local network_labels = cache(int).network_labels
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local x = nlsolution(int)

    for k in 1:D
        network_labels[:, k] .= sol.q[k]
        cache(int).q̃[k] = sol.q[k]
        cache(int).p̃[k] = sol.p[k]
        x[D*S+k] = sol.p[k]
    end
end

function initial_params!(int::GeometricIntegrator{<:NonLinear_OneLayer_GML}, initialParams::TrainingMethod, sol)
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))

    local x = nlsolution(int)
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local extrapolation_substep = method(int).extrapolation_substep
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local nepochs = method(int).training_epochs
    local training_errors = cache(int).training_errors
    local mse_err = cache(int).mse_err
    local abs_err = cache(int).abs_err
    local training_time = cache(int).training_time

    Random.seed!(42)

    for k in 1:D
        @debug "For dimension" k network_labels[:, k]
        labels = reshape(network_labels[:, k], 1, extrapolation_substep + 1)

        PNN = AbstractNeuralNetworks.NeuralNetwork(NN)
        # opt = GeometricMachineLearning.Optimizer(AdamOptimizer(0.001, 0.9, 0.99, 1e-8), ps[k])
        opt = GeometricMachineLearning.Optimizer(GeometricMachineLearning.AdamOptimizerWithDecay(nepochs, 1e-3, 5e-5), PNN.params)
        λ = GeometricMachineLearning.GlobalSection(PNN.params)
        t1 = time()
        for ep in 1:nepochs
            gs = Zygote.gradient(p -> mse_loss(network_inputs, labels, NN, p), PNN.params)[1]
            GeometricMachineLearning.optimization_step!(opt, λ, PNN.params, gs)
            training_errors[k, ep] = mse_loss(network_inputs, labels, NN, PNN.params)
        end
        t2 = time()
        training_time[k] = t2 - t1

        mse_err[k] = training_errors[k, end]
        abs_err[k] = sum(labels - NN(network_inputs, PNN.params)) .^ 2
        @debug "dimension" k "final loss:" mse_err[k] "in" nepochs "epochs"
        @debug "Sum of squared errors for dimension" k ":" abs_err[k]

        for i in 1:S
            x[D*(i-1)+k] = PNN.params[2].W[i]
            x[D*(S+1)+D*(i-1)+k] = PNN.params[1].W[i]
            x[D*(S+1+S)+D*(i-1)+k] = PNN.params[1].b[i]
        end
    end
    @debug "Initial guess from network training" x
end


function initial_params!(int::GeometricIntegrator{<:NonLinear_OneLayer_GML}, initialParams::OGA1d, sol)
    local S = nbasis(method(int))
    local D = length(cache(int).q̃)
    local quad_nodes = method(int).network_inputs
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local network_labels = cache(int).network_labels'
    local activation = method(int).basis.activation
    local x = nlsolution(int)
    local extrapolation_substep = method(int).extrapolation_substep
    local bias_interval = method(int).bias_interval
    local dict_amount = method(int).dict_amount

    # Working precision of the nonlinear solve. The OGA seed is assembled entirely
    # at this precision (no Float64 island), so the whole path is GPU-portable.
    # Robustness at reduced precision comes from (i) a QR least-squares fit on the
    # weighted design matrix (conditioned on κ(Φ), not κ(Φ)² as the normal equations
    # are) and (ii) dictionary normalization plus a coherence guard that keep the
    # greedily selected atoms linearly independent. The resulting parameters feed the
    # working-precision cache; the integrator equations and Newton solve run in T too.
    local T = eltype(x)

    quad_weights = simpson_quadrature(extrapolation_substep, T)

    # Candidate dictionary: neurons with weights ±1 and biases on a uniform grid.
    B = bias_grid(bias_interval[1], bias_interval[2], dict_amount, T)
    A = hcat(vcat(-ones(T, length(B)), ones(T, length(B))), vcat(B, B))
    nodes = vec(T.(quad_nodes))
    quad_nodes_mat = permutedims(hcat(nodes, ones(T, length(nodes))))   # (2 × M)
    gx_quad = activation.(A * quad_nodes_mat)                           # (dict × M)

    # Unit-normalized atoms (quadrature-weighted L² norm) are used only to measure
    # coherence for the dedup guard; atoms below the precision-scaled floor are left
    # unnormalized. Selection itself stays on the raw inner product so that the
    # well-conditioned (Float64/Float32) atom choice is unchanged.
    dict_norms = sqrt.(gx_quad .^ 2 * quad_weights)
    gx_normed = gx_quad ./ ifelse.(dict_norms .< oga_norm_floor(T, maximum(dict_norms)), one(T), dict_norms)
    coherence_cap = one(T) - sqrt(eps(T))

    for d in 1:D
        ps[d][1].W .= zero(T)
        ps[d][1].b .= zero(T)
        ps[d][2].W .= zero(T)

        W = zeros(T, S)         # hidden weights
        Bias = zeros(T, S)      # hidden biases
        selected = Int[]
        blocked = falses(length(dict_norms))
        label = network_labels[d, :]

        for k = 1:S
            # Greedy step: pick the atom most correlated with the current residual,
            # skipping atoms too coherent with those already selected.
            residual = label .- vec(NN(quad_nodes, ps[d]))
            score = ifelse.(blocked, -one(T), abs.(gx_quad * (residual .* quad_weights)))
            best = argmax(score)
            push!(selected, best)
            W[k] = A[best, 1]
            Bias[k] = A[best, 2]

            # Orthogonal projection: refit all selected output weights by weighted
            # QR least squares (no Gram matrix, no Tikhonov ridge).
            xk = weighted_lstsq(gx_quad[selected, :], quad_weights, label)

            ps[d][1].W .= W
            ps[d][1].b .= Bias
            ps[d][2].W[1:k] .= xk

            # Block the chosen atom and its near-duplicates from future selection.
            coh = gx_normed * (gx_normed[best, :] .* quad_weights)
            blocked .|= abs.(coh) .> coherence_cap

            @debug "Sum of squared errors after adding neuron " k sum((label .- vec(NN(quad_nodes, ps[d]))) .^ 2)
        end
        @debug "Finish OGA for dimension" d
    end

    for k in 1:D
        for i in 1:S
            x[D*(i-1)+k] = ps[k][2].W[i]
            x[D*(S+1)+D*(i-1)+k] = ps[k][1].W[i]
            x[D*(S+1+S)+D*(i-1)+k] = ps[k][1].b[i]
        end
    end
    # st = st_tem[1]
    @debug "Initial guess for DOF from OGA " x

end

"""
    initial_params!(int, ::OGA1d_Legacy)

Legacy OGA initial guess for `NonLinear_OneLayer_GML`, kept as a selectable
alternative to the default `OGA1d` for comparison. This is the
pre-refactor algorithm: the dictionary and the greedy least-squares fit are
assembled in `Float64` (a "double-precision island"), the output weights are
obtained from the normal equations `Gk \\ rhs`, and the result is rounded into the
working-precision cache. See the "Orthogonal Greedy Algorithm" section of the
documentation for why this was replaced by a working-precision QR fit. Select it
with `NonLinear_OneLayer_GML(...; initial_guess_method = OGA1d_Legacy())`.
"""
function initial_params!(int::GeometricIntegrator{<:NonLinear_OneLayer_GML}, initialParams::OGA1d_Legacy, sol)
    local S = nbasis(method(int))
    local D = length(cache(int).q̃)
    local quad_nodes = method(int).network_inputs
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local network_labels = cache(int).network_labels'
    local activation = method(int).basis.activation
    local x = nlsolution(int)
    local extrapolation_substep = method(int).extrapolation_substep
    local bias_interval = method(int).bias_interval
    local dict_amount = method(int).dict_amount

    # The OGA initial guess is a seed for the nonlinear solve, so its dictionary
    # and least-squares fit are assembled in double precision for numerical
    # robustness (a reduced-precision Gram matrix is rank-deficient because
    # distinct dictionary neurons collapse onto identical low-precision values).
    # The resulting parameters are stored into the working-precision cache below;
    # the variational integrator equations and the nonlinear solve still run at
    # the working precision.
    quad_weights = simpson_quadrature(extrapolation_substep)# Simpson's rule for 11 quad points 0:0.1:1

    lo = Float64(bias_interval[1])
    hi = Float64(bias_interval[2])
    B = lo:(hi - lo)/dict_amount:hi
    w_list = vcat(-1 * ones(length(B), 1), ones(length(B), 1))
    b_list = vcat(collect(B), collect(B))
    A = hcat(w_list, b_list)
    quad_nodes_mat = hcat(Float64.(quad_nodes'), ones(length(quad_nodes)))'
    gx_quad = activation.(A * quad_nodes_mat)


    for d in 1:D
        W = zeros(S, 1)        # all parameters w
        Bias = zeros(S, 1)      # all parameters b
        C = zeros(S, extrapolation_substep + 1)
        f_weight = network_labels[d, :] .* quad_weights

        for k = 1:S
            #     The subproblem is key to the greedy algorithm, where the
            #     inner products |(u,g) - (f,g)| should be maximized.
            #     Part of the inner products can be computed in advance.

            #select the Optimal basis

            uk_quad = NN(quad_nodes, ps[d])'

            uk_weight = uk_quad .* quad_weights

            loss = -(1 / 2) * (gx_quad * (uk_weight - f_weight)) .^ 2
            argmin_index = argmin(loss)
            W[k] = A[argmin_index[1], :][1]
            Bias[k] = A[argmin_index[1], :][2]

            ak = hcat(W[k], Bias[k])
            C[k, :] = ak * quad_nodes_mat
            selected_g = activation.(C[1:k, :])

            Gk = selected_g * (selected_g .* quad_weights')'
            rhs = selected_g * (network_labels[d, :] .* quad_weights)
            xk = Gk \ rhs

            ps[d][1].W[:] .= W
            ps[d][1].b[:] .= Bias
            ps[d][2].W[1:k] .= xk

            errs = sum(network_labels[d, :] - NN(quad_nodes, ps[d])') .^ 2
            @debug "Sum of squared errors after adding neuron " k ":" errs
        end
        @debug "Finish OGA for dimension" d

    end

    for k in 1:D
        for i in 1:S
            x[D*(i-1)+k] = ps[k][2].W[i]
            x[D*(S+1)+D*(i-1)+k] = ps[k][1].W[i]
            x[D*(S+1+S)+D*(i-1)+k] = ps[k][1].b[i]
        end
    end
    @debug "Initial guess for DOF from OGA (legacy) " x

end

function GeometricIntegrators.Integrators.components!(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:NonLinear_OneLayer_GML}) where {ST}
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local C = cache(int, ST)

    local quad_nodes = QuadratureRules.nodes(int.method.quadrature)

    local q = cache(int, ST).q̃
    local p = cache(int, ST).p̃
    local Q = cache(int, ST).Q
    local V = cache(int, ST).V
    local P = cache(int, ST).P
    local F = cache(int, ST).F
    local X = cache(int, ST).X

    local NN = method(int).basis.NN
    local ps = cache(int, ST).ps

    local r₀ = cache(int, ST).r₀
    local r₁ = cache(int, ST).r₁
    local m = cache(int, ST).m
    local a = cache(int, ST).a
    local dqdWc = cache(int, ST).dqdWc
    local dqdbc = cache(int, ST).dqdbc
    local dvdWc = cache(int, ST).dvdWc
    local dvdbc = cache(int, ST).dvdbc
    local dqdWr₁ = cache(int, ST).dqdWr₁
    local dqdWr₀ = cache(int, ST).dqdWr₀
    local dqdbr₁ = cache(int, ST).dqdbr₁
    local dqdbr₀ = cache(int, ST).dqdbr₀

    local DVDθ = method(int).basis.dvdθ
    local DQDθ = method(int).basis.dqdθ

    # copy x to X
    for i in eachindex(X)
        for k in eachindex(X[i])
            X[i][k] = x[D*(i-1)+k]
        end
    end

    # copy x to p # momenta
    for k in eachindex(p)
        p[k] = x[D*S+k]
    end

    for k in 1:D
        for i in 1:S
            ps[k][2].W[i] = x[D*(i-1)+k]
            ps[k][1].W[i] = x[D*(S+1)+D*(i-1)+k]
            ps[k][1].b[i] = x[D*(S+1+S)+D*(i-1)+k]
        end
    end

    # compute coefficients
    for d in 1:D
        r₀[:, d] = (NN.layers[1])([zero(ST)], ps[d][1])
        r₁[:, d] = (NN.layers[1])([one(ST)], ps[d][1])
        for j in eachindex(quad_nodes)
            m[j, :, d] = (NN.layers[1])([quad_nodes[j]], ps[d][1])
            a[j, :, d] = DVDθ([quad_nodes[j]], NeuralNetworkParameters(ps[d])).L2.W[:]
        end
    end

    # compute the derivatives of the coefficients on the quadrature nodes and at the boundaries
    for d in 1:D
        for j in eachindex(quad_nodes)
            g = DQDθ([quad_nodes[j]], NeuralNetworkParameters(ps[d]))
            dqdWc[j, :, d] = g.L1.W[:]
            dqdbc[j, :, d] = g.L1.b[:]

            gv = DVDθ([quad_nodes[j]], NeuralNetworkParameters(ps[d]))
            dvdWc[j, :, d] = gv.L1.W[:]
            dvdbc[j, :, d] = gv.L1.b[:]
        end

        g0 = DQDθ([zero(ST)], NeuralNetworkParameters(ps[d]))
        dqdWr₀[:, d] = g0.L1.W[:]
        dqdbr₀[:, d] = g0.L1.b[:]

        g1 = DQDθ([one(ST)], NeuralNetworkParameters(ps[d]))
        dqdWr₁[:, d] = g1.L1.W[:]
        dqdbr₁[:, d] = g1.L1.b[:]
    end

    # compute Q : q at quaadurature points
    for i in eachindex(Q)
        for d in eachindex(Q[i])
            y = zero(ST)
            for j in eachindex(X)
                y += m[i, j, d] * X[j][d]
            end
            Q[i][d] = y
        end
    end

    # compute q[t_{n+1}]
    for d in eachindex(q)
        y = zero(ST)
        for i in eachindex(X)
            y += r₁[i, d] * X[i][d]
        end
        q[d] = y
    end

    # compute V volicity at quadrature points
    for i in eachindex(V)
        for k in eachindex(V[i])
            y = zero(ST)
            for j in eachindex(X)
                y += a[i, j, k] * X[j][k]
            end
            V[i][k] = y / timestep(int)
        end
    end

    # compute P=ϑ(Q,V) and F=f(Q,V)
    for i in eachindex(C.Q, C.V, C.P, C.F)
        tᵢ = sol.t + timestep(int) * (method(int).c[i] - 1)
        equations(int).ϑ(C.P[i], tᵢ, C.Q[i], C.V[i], params)
        equations(int).f(C.F[i], tᵢ, C.Q[i], C.V[i], params)
    end
end


function GeometricIntegrators.Integrators.residual!(b::Vector{ST}, sol, params, int::GeometricIntegrator{<:NonLinear_OneLayer_GML}) where {ST}
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local q̄ = sol.q
    local p̄ = sol.p
    local p̃ = cache(int, ST).p̃
    local P = cache(int, ST).P
    local F = cache(int, ST).F
    local X = cache(int, ST).X

    local r₀ = cache(int, ST).r₀
    local r₁ = cache(int, ST).r₁
    local m = cache(int, ST).m
    local a = cache(int, ST).a

    local dqdWc = cache(int, ST).dqdWc
    local dqdbc = cache(int, ST).dqdbc
    local dvdWc = cache(int, ST).dvdWc
    local dvdbc = cache(int, ST).dvdbc
    local dqdWr₁ = cache(int, ST).dqdWr₁
    local dqdWr₀ = cache(int, ST).dqdWr₀
    local dqdbr₁ = cache(int, ST).dqdbr₁
    local dqdbr₀ = cache(int, ST).dqdbr₀

    # compute b = - [(P-AF)], the residual in actual action, vatiation with respect to Q_{n,i}
    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += method(int).b[j] * m[j, i, k] * F[j][k] * timestep(int)
                z += method(int).b[j] * a[j, i, k] * P[j][k]
            end
            b[D*(i-1)+k] = (r₁[i, k] * p̃[k] - r₀[i, k] * p̄[k]) - z
        end
    end

    # the continue constraint from hamilton pontryagin principle
    for k in eachindex(q̄)
        y = zero(ST)
        for j in eachindex(X)
            y += r₀[j, k] * X[j][k]
        end
        b[D*S+k] = q̄[k] - y
    end

    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdWc[j, i, k]
                z += method(int).b[j] * P[j][k] * dvdWc[j, i, k]
            end
            b[D*(S+1)+D*(i-1)+k] = dqdWr₁[i, k] * p̃[k] - z
        end
    end

    for i in 1:S
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdbc[j, i, k]
                z += method(int).b[j] * P[j][k] * dvdbc[j, i, k]
            end
            b[D*(S+1+S)+D*(i-1)+k] = (dqdbr₁[i, k] * p̃[k] - dqdbr₀[i, k] * p̄[k]) - z
        end
    end
    # @debug " Residual vector b: " b
    # @debug " Norm of Residual vector b: " norm(b)

end





function record_finer_solution!(sol, int::GeometricIntegrator{<:NonLinear_OneLayer_GML})
    local x = nlsolution(int)
    local stage_values = cache(int).stage_values
    # local network_inputs = method(int).network_inputs
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local NN = method(int).basis.NN
    local ps = cache(int).ps

    network_inputs = reshape(collect(0:1/40:1),1,41)

    @debug "solution x after solving by Newton" x

    for k in 1:D
        for i in 1:S
            ps[k][2].W[i] = x[D*(i-1)+k]
            ps[k][1].W[i] = x[D*(S+1)+D*(i-1)+k]
            ps[k][1].b[i] = x[D*(S+1+S)+D*(i-1)+k]
        end
        stage_values[:, k] = NN(network_inputs, ps[k])[:]
    end

    @debug "stages prediction after solving" stage_values
    @debug "sol from this step q:", sol.q, "p:", sol.p
end


