"""
    ShallowNetAutodiff <: ShallowNetMethod

Shallow-net variational integrator that computes network derivatives with `ForwardDiff`
instead of the pre-compiled symbolic derivatives used by `ShallowNet`.
The ansatz and optimisation are otherwise identical to `ShallowNet`.

# Constructor

    ShallowNetAutodiff(basis, quadrature; kwargs...)

Keyword arguments are the same as `ShallowNet`:
`initial_trajectory_method`, `initial_guess_method`, `extrapolation_substep`,
`training_epochs`, `show_status`, `bias_interval`, `dict_amount`, `record_grid_points`.
"""
struct ShallowNetAutodiff{T, NNODES, basisType <: Basis{T},
                    ET <: Extrapolation,
                    IPMT <: InitialParametersMethod} <: ShallowNetMethod
    common        :: NetworkIntegratorCore{T, NNODES, basisType, ET, IPMT}
    bias_interval :: SVector{2, T}
    dict_amount   :: Int

    function ShallowNetAutodiff(basis::Basis{T}, quadrature::QuadratureRule{T};
        extrapolation_substep      :: Int  = 10,
        training_epochs           :: Int  = 50000,
        show_status               :: Bool = true,
        initial_trajectory_method :: ET   = IntegratorExtrapolation(),
        initial_guess_method      :: IPMT = OGA1dNormalized(),
        record_grid_points = 41,
        bias_interval = [-pi, pi],
        dict_amount   :: Int = 50000,) where {T, ET, IPMT}
        common = NetworkIntegratorCore(basis, quadrature;
            extrapolation_substep=extrapolation_substep,
            training_epochs=training_epochs,
            show_status = show_status,
            initial_trajectory_method=initial_trajectory_method,
            initial_guess_method=initial_guess_method,
            record_grid_points =  record_grid_points)
        new{T, nnodes(quadrature), typeof(basis), ET, IPMT}(
            common, SVector{2,T}(bias_interval), dict_amount)
    end
end

# Alone among the four integrators, this one's greedy step selects on the *normalized*
# inner product. The rule decides which neurons are picked and hence which Newton basin the
# step lands in, so it is a tuned baseline rather than a free choice.
default_iparams(::ShallowNetAutodiff) = OGA1dNormalized()

struct ShallowNetAutodiffCache{ST,S,R,N} <: NetworkIntegratorCache{ST}
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

    stage_values::Matrix{ST}
    network_labels::Matrix{ST}

    function ShallowNetAutodiffCache{ST,S,R,N}(ics; record_grid_points::Int = 41) where {ST,S,R,N}
        D = length(vec(ics.q))
        x = zeros(ST, D * (1 + 3 * S))

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

        stage_values = zeros(ST, record_grid_points, D)
        network_labels = zeros(ST, N + 1, D)

        new(x, q̄, p̄, q̃, p̃, ṽ, f̃, s̃, X, Q, P, V, F, ps,
            dqdW2c, dvdW2c, dqdW1c, dvdW1c, dqdbc, dvdbc,
            dqdW2r₁, dqdW2r₀, dqdW1r₁, dqdW1r₀, dqdbr₁, dqdbr₀,
            stage_values, network_labels)
    end
end

function GeometricIntegratorsBase.Cache{ST}(problem::AbstractProblemIODE, method::ShallowNetAutodiff; kwargs...) where {ST}
    ShallowNetAutodiffCache{ST, nbasis(method), nnodes(method), extrapolation_substep(method)}(initial_conditions(problem);
        record_grid_points = method.record_grid_points, kwargs...)
end

@inline GeometricIntegratorsBase.CacheType(ST, problem::AbstractProblemIODE, method::ShallowNetAutodiff) =
    ShallowNetAutodiffCache{ST, nbasis(method), nnodes(method), extrapolation_substep(method)}


# The extrapolated trajectory has to land in `network_labels`, because that is what the OGA
# seed reads (`initial_params!` in `src/oga/adapters.jl`). This used to write the extrapolated
# positions into the *output-weight* slots of `x` and leave `network_labels` at zero, so the
# seed fitted the boundary ansatz to an all-zero target and then overwrote the very slots the
# extrapolation had filled. It also put `p̃` into `x[D*S+k]`, which for this ansatz is the
# endpoint *position* unknown rather than the momentum — cf. the `IntegratorExtrapolation`
# method below, which sets `q̃` there.
#
# Note that `solutionstep!` only extrapolates when `iguess(int)` is a framework extrapolation;
# with the default `NoInitialGuess()` it is a no-op. Pass `initialguess = HermiteExtrapolation()`
# to `GeometricIntegrator` for a real Hermite warm start.
function initial_trajectory!(sol, history, params, int::GeometricIntegrator{<:ShallowNetAutodiff}, initial_trajectory_method::HermiteExtrapolation)
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local x = nlsolution(int)
    local network_inputs = method(int).network_inputs
    local network_labels = cache(int).network_labels
    local h = timestep(int)

    for i in eachindex(network_inputs)
        soltmp = (
            t=sol.t + (network_inputs[i] - 1) * h,
            q=cache(int).q̃,
            p=cache(int).p̃,
            q̇=cache(int).ṽ,
            ṗ=cache(int).f̃,
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
        q̇=cache(int).ṽ,
        ṗ=cache(int).f̃,
    )
    solutionstep!(soltmp, history, problem(int), iguess(int))

    for k in 1:D
        x[D*S+k] = cache(int).q̃[k]
    end
end

function initial_trajectory!(sol, history, params, int::GeometricIntegrator{<:ShallowNetAutodiff}, initial_trajectory_method::IntegratorExtrapolation)
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

function apply_NN(t, ps, S, activation)
    W2 = ps[1:S]
    W1 = ps[S+1:2S]
    b1 = ps[2S+1:3S]

    z1 = W1 .* t .+ b1
    a1 = activation.(z1)
    z2 = sum(W2 .* a1)
    return z2
end

function NN_ansatz(ps, S::Int, activation, t, q̄, q)
    # q_h(t) = (1-t)q_n + t*q_{n+1} + t(1-t)NN(t)
    return (one(t) - t) * q̄ + t * q + t * (one(t) - t) * apply_NN(t, ps, S, activation)
end

VNN_ansatz_zygote(ps, S, activation, t, q̄, q) = Zygote.gradient(tt -> NN_ansatz(ps, S, activation, tt, q̄, q),t)[1]

VNN_ansatz(ps, S, activation, t, q̄, q) = ForwardDiff.derivative(tt -> NN_ansatz(ps, S, activation, tt, q̄, q), t)
∂NN_ansatz_∂params(ps, S, activation, t, q̄, q) = ForwardDiff.gradient(p -> NN_ansatz(p, S, activation, t, q̄, q), ps)
∂VNN_ansatz_∂params(ps, S, activation, t, q̄, q) = ForwardDiff.gradient(p -> VNN_ansatz(p, S, activation, t, q̄, q), ps)

∂NN_ansatz_∂q̄(ps,S,activation,t,q̄,q) = one(t) .- t
∂NN_ansatz_∂q(ps,S,activation,t,q̄,q) = t

∂VNN_ansatz_∂q̄(ps,S,activation,t,q̄,q)= -one(t)
∂VNN_ansatz_∂q(ps,S,activation,t,q̄,q) = one(t)

function GeometricIntegratorsBase.components!(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:ShallowNetAutodiff}) where {ST}
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
            g = ∂NN_ansatz_∂params(ps_vec,S,activation,quad_nodes[j],q̄[d],cache(int).q̃[d])
            dqdW2c[j, :, d] = g[1:S]
            dqdW1c[j, :, d] = g[S+1:2S]
            dqdbc[j, :, d] = g[2S+1:3S]

            gv = ∂VNN_ansatz_∂params(ps_vec,S,activation,quad_nodes[j],q̄[d],cache(int).q̃[d])
            dvdW1c[j, :, d] = gv[S+1:2S]
            dvdbc[j, :, d] = gv[2S+1:3S]
            dvdW2c[j, :, d] = gv[1:S]
        end

        # Boundary points t=0 and t=1 must share the (plain) element type of the
        # quadrature nodes, NOT the solver type ST: during the Newton solve ST is a
        # ForwardDiff.Dual, and ∂NN_ansatz_∂params itself nests a ForwardDiff.gradient,
        # so passing a Dual `t` triggers a Dual-tag ordering error.
        g0 = ∂NN_ansatz_∂params(ps_vec,S,activation,zero(eltype(quad_nodes)),q̄[d],cache(int).q̃[d])
        dqdW1r₀[:, d] = g0[S+1:2S]
        dqdbr₀[:, d] = g0[2S+1:3S]
        dqdW2r₀[:, d] = g0[1:S]

        g1 = ∂NN_ansatz_∂params(ps_vec,S,activation,one(eltype(quad_nodes)),q̄[d],cache(int).q̃[d])
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
            Q[i][d] = NN_ansatz(ps_vec, S, activation, quad_nodes[i], q̄[d], q[d])
        end
    end

    # compute V volicity at quadrature points
    for d in 1:D
        ps_vec = zeros(ST, 3S)
        ps_vec[1:S] = ps[d][2].W[:]
        ps_vec[S+1:2S] = ps[d][1].W[:]
        ps_vec[2S+1:3S] = ps[d][1].b[:]
        for i in eachindex(quad_nodes)
            V[i][d] = VNN_ansatz_zygote(ps_vec,S,activation,quad_nodes[i],q̄[d],q[d]) / timestep(int)
        end
    end

    # compute P=ϑ(Q,V) and F=f(Q,V)
    for i in eachindex(C.Q, C.V, C.P, C.F)
        tᵢ = sol.t + timestep(int) * (method(int).c[i] - 1)
        equations(int).ϑ(C.P[i], tᵢ, C.Q[i], C.V[i], params)
        equations(int).f(C.F[i], tᵢ, C.Q[i], C.V[i], params)
    end
end


function GeometricIntegratorsBase.residual!(b::Vector{ST}, sol, params, int::GeometricIntegrator{<:ShallowNetAutodiff}) where {ST}
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
            z += timestep(int) * method(int).b[j] * F[j][k] * (1 - quad_nodes[j])
            z += method(int).b[j] * P[j][k] * (-1)
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

    # show_status ? println(" Residual vector b: \n", b) : nothing
    # show_status ? println(" Norm of Residual vector b: ", norm(b)) : nothing
end



function GeometricIntegratorsBase.update!(sol, params, int::GeometricIntegrator{<:ShallowNetAutodiff}, DT)
    local D = length(cache(int).q̃)
    local quad_nodes = QuadratureRules.nodes(int.method.quadrature)
    local P = cache(int).P
    local F = cache(int).F

    sol.q .= cache(int, DT).q̃

    for k in 1:D
        z = zero(eltype(sol.p))
        for j in eachindex(P, F)
            # dQ/dq_{n+1} = τ, dV/dq_{n+1} = 1/h
            z += timestep(int) * method(int).b[j] * F[j][k] * (quad_nodes[j])
            z += method(int).b[j] * P[j][k]
        end
        sol.p[k] = z
    end
    # sol.p .= cache(int, DT).p̃
end



function record_finer_solution!(sol, int::GeometricIntegrator{<:ShallowNetAutodiff})
    local x = nlsolution(int)
    local stage_values = cache(int).stage_values
    # local network_inputs = method(int).network_inputs
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local show_status = method(int).show_status
    local q̄ = sol.q  # start point q_n
    local q = cache(int).q̃ # endpoint estimate q_{n+1}
    local activation = method(int).basis.activation

    local N_plot = method(int).record_grid_points
    local T = eltype(x)
    network_inputs = reshape(collect(range(zero(T), one(T), N_plot)), 1, N_plot)

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

        ps_vec = zeros(eltype(x), 3S)
        ps_vec[1:S] = ps[k][2].W[:]
        ps_vec[S+1:2S] = ps[k][1].W[:]
        ps_vec[2S+1:3S] = ps[k][1].b[:]

        for i in eachindex(network_inputs)
            stage_values[i, k] = NN_ansatz(ps_vec, S, activation, network_inputs[i], q̄[k], q[k])
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


