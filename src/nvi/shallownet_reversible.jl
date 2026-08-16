"""
    ShallowNetReversible <: ShallowNetMethod

Time-symmetric variant of `ShallowNet`. The variational integrator is
constructed so that reversing the time direction recovers the original trajectory,
giving `issymmetric(method) == true`. Uses a symmetric quadrature / ansatz
structure while keeping the same `ShallowNetBasis` basis.

# Constructor

    ShallowNetReversible(basis, quadrature; kwargs...)

Keyword arguments are the same as `ShallowNet`. Only the [`OGA`](@ref)
seeds are meaningful as `initial_guess_method`; `TrainingMethod` is not specialised
for this integrator. The basis must have an even number of neurons — they come in
mirrored pairs.
"""
struct ShallowNetReversible{T, NNODES, basisType <: Basis{T},
                                ET <: Extrapolation,
                                IPMT <: InitialParametersMethod} <: ShallowNetMethod
    common        :: NetworkIntegratorCore{T, NNODES, basisType, ET, IPMT}
    bias_interval :: SVector{2, T}
    dict_amount   :: Int

    function ShallowNetReversible(basis::Basis{T}, quadrature::QuadratureRule{T};
        extrapolation_substep      :: Int  = 10,
        training_epochs           :: Int  = 50000,
        show_status               :: Bool = false,
        initial_trajectory_method :: ET   = IntegratorExtrapolation(),
        initial_guess_method      :: IPMT = OGA1d(),
        record_grid_points        :: Int  = 41,
        bias_interval = [-pi, pi],
        dict_amount   :: Int = 50000) where {T, ET, IPMT}
        # The ansatz pairs every neuron with its reflection about t = 1/2 and stores only
        # the independent half, so `S` must be even. Caught here rather than several call
        # levels down in `components!`, which now indexes with `S ÷ 2` and would silently
        # drop the last neuron for odd `S` rather than complain.
        iseven(basis.S) || throw(ArgumentError(
            "ShallowNetReversible requires a basis with an even number of neurons, " *
            "got S = $(basis.S). Neurons come in mirrored pairs and only the S/2 independent " *
            "hidden parameters are stored in the nonlinear solution vector."))
        require_symbolic_derivatives(basis, "ShallowNetReversible")

        common = NetworkIntegratorCore(basis, quadrature;
            extrapolation_substep=extrapolation_substep,
            training_epochs=training_epochs,
            show_status=show_status,
            initial_trajectory_method=initial_trajectory_method,
            initial_guess_method=initial_guess_method,
            record_grid_points = record_grid_points)
        new{T, nnodes(quadrature), typeof(basis), ET, IPMT}(
            common, SVector{2,T}(bias_interval), dict_amount)
    end
end

GeometricIntegratorsBase.issymmetric(::Union{ShallowNetReversible, Type{<:ShallowNetReversible}}) = true

# The cache lives in `network_integrator_core.jl` and is shared with this integrator's
# sibling; the only thing that differs is the number of unknowns per dimension, passed
# below. `CacheType` returns `SymbolicShallowNetCache{ST}` — a *concrete* type, which it was
# not while the basis size and sub-step count were (runtime-computed) type parameters.

function GeometricIntegratorsBase.Cache{ST}(problem::AbstractProblemIODE, method::ShallowNetReversible; kwargs...) where {ST}
    local S = nbasis(method)
    SymbolicShallowNetCache{ST}(initial_conditions(problem), 2 * S + 1, S, nnodes(method),
        extrapolation_substep(method);
        record_grid_points = method.record_grid_points, kwargs...)
end

@inline GeometricIntegratorsBase.CacheType(ST, problem::AbstractProblemIODE, method::ShallowNetReversible) =
    SymbolicShallowNetCache{ST}

function GeometricIntegratorsBase.components!(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:ShallowNetReversible}) where {ST}
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

    local tbuf = cache(int, ST).tbuf
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
        end
        for i in 1:(S ÷ 2)
            ps[k][1].W[2i-1] = x[D*(S+1)+D*(i-1)+k]
            ps[k][1].b[2i-1] = x[D*(S+1+S÷2)+D*(i-1)+k]
            ps[k][1].W[2i] = -ps[k][1].W[2i-1]
            ps[k][1].b[2i] = ps[k][1].W[2i-1] + ps[k][1].b[2i-1]
        end
    end

    # Coefficients and their parameter derivatives, at the quadrature nodes and the two
    # interval boundaries. See the equivalent block in shallownet.jl: merging the two passes
    # over `d` lets `DVDθ` be evaluated once per node instead of twice, builds
    # `NeuralNetworkParameters(ps[d])` once per dimension instead of four times per node, and
    # replaces the `[:]`-copy-then-slice-assign with `copyto!` into a view.
    for d in 1:D
        psd = NeuralNetworkParameters(ps[d])

        tbuf[1] = zero(ST)
        copyto!(view(r₀, :, d), (NN.layers[1])(tbuf, ps[d][1]))
        g0 = DQDθ(tbuf, psd)
        copyto!(view(dqdWr₀, :, d), g0.L1.W)
        copyto!(view(dqdbr₀, :, d), g0.L1.b)

        tbuf[1] = one(ST)
        copyto!(view(r₁, :, d), (NN.layers[1])(tbuf, ps[d][1]))
        g1 = DQDθ(tbuf, psd)
        copyto!(view(dqdWr₁, :, d), g1.L1.W)
        copyto!(view(dqdbr₁, :, d), g1.L1.b)

        for j in eachindex(quad_nodes)
            tbuf[1] = quad_nodes[j]
            copyto!(view(m, j, :, d), (NN.layers[1])(tbuf, ps[d][1]))

            g = DQDθ(tbuf, psd)
            copyto!(view(dqdWc, j, :, d), g.L1.W)
            copyto!(view(dqdbc, j, :, d), g.L1.b)

            gv = DVDθ(tbuf, psd)
            copyto!(view(a, j, :, d), gv.L2.W)
            copyto!(view(dvdWc, j, :, d), gv.L1.W)
            copyto!(view(dvdbc, j, :, d), gv.L1.b)
        end
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


function GeometricIntegratorsBase.residual!(b::Vector{ST}, sol, params, int::GeometricIntegrator{<:ShallowNetReversible}) where {ST}
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

    for i in 1:(S ÷ 2)
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdWc[j, 2i-1, k]
                z += method(int).b[j] * P[j][k] * dvdWc[j, 2i-1, k]
            end
            b[D*(S+1)+D*(i-1)+k] = dqdWr₁[2i-1, k] * p̃[k] - z
        end
    end

    for i in 1:(S ÷ 2)
        for k in 1:D
            z = zero(ST)
            for j in eachindex(P, F)
                z += timestep(int) * method(int).b[j] * F[j][k] * dqdbc[j, 2i-1, k]
                z += method(int).b[j] * P[j][k] * dvdbc[j, 2i-1, k]
            end
            b[D*(S+1+S÷2)+D*(i-1)+k] = (dqdbr₁[2i-1, k] * p̃[k] - dqdbr₀[2i-1, k] * p̄[k]) - z
        end
    end
    # `@debug`, not `show_status ? println(...)`: `residual!` runs once per Newton iteration
    # *and* once per ForwardDiff Jacobian column, with `b` a vector of `Dual`s, so printing it
    # under a flag that defaulted to `true` dumped O(iterations × unknowns) residual vectors
    # per time step. The non-reversible siblings had already dropped these two lines.
    @debug "residual" b norm_b = norm(b)
end



# No `update!` override here: this integrator uses the shared DT-form `update!` in
# `network_integrator_core.jl`, of which the override that used to sit here was a verbatim
# copy. (The two *autodiff* integrators do override it — they recompute `p` from the
# quadrature rather than reading it out of the cache.)



function record_finer_solution!(sol, int::GeometricIntegrator{<:ShallowNetReversible})
    local x = nlsolution(int)
    local stage_values = cache(int).stage_values
    # local network_inputs = method(int).network_inputs
    local D = length(cache(int).q̃)
    local S = nbasis(method(int))
    local NN = method(int).basis.NN
    local ps = cache(int).ps

    local N_plot = method(int).record_grid_points
    local T = eltype(x)
    network_inputs = reshape(collect(range(zero(T), one(T), N_plot)), 1, N_plot)

    @debug "solution x after solving by Newton" x

    for k in 1:D
        for i in 1:S
            ps[k][2].W[i] = x[D*(i-1)+k]
        end
        for i in 1:(S ÷ 2)
            ps[k][1].W[2i-1] = x[D*(S+1)+D*(i-1)+k]
            ps[k][1].b[2i-1] = x[D*(S+1+S÷2)+D*(i-1)+k]
            ps[k][1].W[2i] = -ps[k][1].W[2i-1]
            ps[k][1].b[2i] = ps[k][1].W[2i-1] + ps[k][1].b[2i-1]
        end
        stage_values[:, k] = NN(network_inputs, ps[k])[:]
        @debug "parameters after solving" dim = k W2 = ps[k][2].W W1 = ps[k][1].W b1 = ps[k][1].b
    end

    @debug "stages prediction after solving" stage_values q = sol.q p = sol.p

end


