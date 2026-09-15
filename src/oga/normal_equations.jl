# ---- The original-paper reference implementation -----------------------------
#
# Kept verbatim in behaviour (only relocated and renamed) so that every claim about the
# modern variants can be measured against the algorithm as published, rather than against
# a paraphrase of it. See `OGA1dNormalEquations` for what it does differently and why that
# matters.
#
# This is deliberately *not* built on the shared core: the point of a reference is that it
# does not inherit later fixes. `NormalEquationsFit` provides the same arithmetic inside
# the composable `OGA` for anyone who wants the Gram solve with the modern guard rails.

"""
    initial_params!(int, ::OGA1dNormalEquations, sol)

Reference OGA initial guess for `ShallowNet`: the implementation from the
original paper, kept as a selectable baseline.

The dictionary and the greedy least-squares fit are assembled in `Float64` (a
"double-precision island"), the output weights come from the normal equations
`Gk \\ rhs`, and the result is rounded into the working-precision cache. There is no norm
floor, no coherence guard, no ridge and no rank detection. See the "Orthogonal Greedy
Algorithm" section of the documentation for why the working-precision QR fit of
[`OGA1d`](@ref) replaced it as the default, and [`OGA1dNormalEquations`](@ref) for the
failure mode this variant exhibits at 16 bits.

Defined only for `ShallowNet` — it is a comparison baseline, not a production
seed.
"""
function initial_params!(int::GeometricIntegrator{<:ShallowNet},
        initialParams::OGA1dNormalEquations, sol)
    local S = nbasis(method(int))
    local D = length(cache(int).q̃)
    local quad_nodes = method(int).network_inputs
    local NN = method(int).basis.NN
    local ps = cache(int).ps
    local network_labels = cache(int).network_labels'
    local activation = method(int).basis.activation
    local x = nlsolution(int)
    # Kept as a local named `nstages`: renaming it to `extrapolation_substep` would shadow
    # the accessor function of that name for the rest of the body.
    local nstages = extrapolation_substep(method(int))
    local bias_interval = method(int).bias_interval
    local dict_amount = method(int).dict_amount

    # The OGA initial guess is a seed for the nonlinear solve, so its dictionary
    # and least-squares fit are assembled in double precision for numerical
    # robustness (a reduced-precision Gram matrix is rank-deficient because
    # distinct dictionary neurons collapse onto identical low-precision values).
    # The resulting parameters are stored into the working-precision cache below;
    # the variational integrator equations and the nonlinear solve still run at
    # the working precision.
    quad_weights = simpson_quadrature(nstages)# Simpson's rule for 11 quad points 0:0.1:1

    lo = Float64(bias_interval[1])
    hi = Float64(bias_interval[2])
    B = lo:((hi - lo) / dict_amount):hi
    w_list = vcat(-1 * ones(length(B), 1), ones(length(B), 1))
    b_list = vcat(collect(B), collect(B))
    A = hcat(w_list, b_list)
    quad_nodes_mat = hcat(Float64.(quad_nodes'), ones(length(quad_nodes)))'
    gx_quad = activation.(A * quad_nodes_mat)

    for d in 1:D
        W = zeros(S, 1)        # all parameters w
        Bias = zeros(S, 1)      # all parameters b
        C = zeros(S, nstages + 1)
        f_weight = network_labels[d, :] .* quad_weights

        for k in 1:S
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
            x[D * (i - 1) + k] = ps[k][2].W[i]
            x[D * (S + 1) + D * (i - 1) + k] = ps[k][1].W[i]
            x[D * (S + 1 + S) + D * (i - 1) + k] = ps[k][1].b[i]
        end
    end
    @debug "Initial guess for DOF from OGA (reference, normal equations) " x
end
