# Shared abstract type and traits for the two spline variational integrators.

"""
    VISplineMethod <: LODEMethod

Abstract supertype for the fixed-knot (`VISplineFixed`) and free-knot (`VISplineFree`)
B-spline variational integrators.
"""
abstract type VISplineMethod <: LODEMethod end

abstract type VISplineCache{ST} <: IODEIntegratorCache{ST} end

GeometricIntegratorsBase.nlsolution(c::VISplineCache) = c.x

function GeometricIntegratorsBase.reset!(cache::VISplineCache, _, q, p)
    copyto!(cache.q̄, q)
    copyto!(cache.p̄, p)
end

# --- Shared traits ---

GeometricIntegratorsBase.isexplicit( ::Union{VISplineMethod, Type{<:VISplineMethod}}) = false
GeometricIntegratorsBase.isimplicit( ::Union{VISplineMethod, Type{<:VISplineMethod}}) = true
GeometricIntegratorsBase.issymmetric(::Union{VISplineMethod, Type{<:VISplineMethod}}) = missing
GeometricIntegratorsBase.issymplectic(::Union{VISplineMethod, Type{<:VISplineMethod}}) = true

default_solver(::VISplineMethod)           = Newton()
default_iguess(::VISplineMethod)           = GeometricIntegratorsBase.NoInitialGuess()
default_iguess_integrator(::VISplineMethod) = GeometricIntegratorsBase.ImplicitMidpoint()

# --- Shared integrator step (Newton solve + update) ---

function GeometricIntegratorsBase.residual!(
        b::AbstractVector{ST}, x::AbstractVector{ST}, sol, params,
        int::GeometricIntegrator{<:VISplineMethod}) where {ST}
    @assert axes(x) == axes(b)
    GeometricIntegratorsBase.components!(x, sol, params, int)
    GeometricIntegratorsBase.residual!(b, sol, params, int)
end

function GeometricIntegratorsBase.update!(
        sol, params, x::AbstractVector{DT},
        int::GeometricIntegrator{<:VISplineMethod}) where {DT}
    GeometricIntegratorsBase.components!(x, sol, params, int)
    sol.q .= cache(int, DT).q̃
    sol.p .= cache(int, DT).p̃
end

function GeometricIntegratorsBase.integrate_step!(
        sol, history, params,
        int::GeometricIntegrator{<:VISplineMethod, <:AbstractProblemIODE})
    solve_with_status!(nlsolution(int), solver(int), solverstate(int), (sol, params, int))
    GeometricIntegratorsBase.update!(sol, params, nlsolution(int), int)
end

# --- Shared initial trajectory (IntegratorExtrapolation) ---
#
# Integrates a sub-problem over [0, h] with `extrapolation_substep` sub-steps using
# ImplicitMidpoint, storing the resulting q trajectory in `cache.network_labels`.
# Also seeds `p̃` (and x[D*S+k]) from the sub-integrator's final momentum.

function _vi_spline_initial_trajectory!(sol, history, params, int)
    N  = method(int).extrapolation_substep
    S  = nbasis(method(int))
    D  = length(cache(int).q̃)
    h  = timestep(int)
    x  = nlsolution(int)

    tem_ode = similar(int.problem, [zero(h), h], h / N,
        (q = StateVariable(sol.q[:]), p = StateVariable(sol.p[:])))
    tem_sol = integrate(tem_ode, default_iguess_integrator(method(int)))

    for k in 1:D
        cache(int).network_labels[:, k] .= tem_sol.q[:, k]
        cache(int).q̃[k] = tem_sol.q[end, k]
        cache(int).p̃[k] = tem_sol.p[end, k]
        x[D * S + k]    = cache(int).p̃[k]
    end
end

# --- Shared least-squares B-spline coefficient fit ---
#
# Fits B-spline coefficients to the extrapolated trajectory stored in `network_labels`.
# `eval_times` are normalized times in [0,1] for the N+1 label points.

function _fit_bspline_to_labels!(x, cache, order, knots, D, S)
    N = size(cache.network_labels, 1) - 1
    T = eltype(knots)
    eval_times = [T(i) / T(N) for i in 0:N]
    B_mat = bspline_eval_matrix(order, knots, eval_times; T)   # (N+1) × S
    Q_fac = qr(B_mat)
    for k in 1:D
        c = Q_fac \ cache.network_labels[:, k]
        for i in 1:S
            x[D * (i - 1) + k] = c[i]
        end
    end
end
