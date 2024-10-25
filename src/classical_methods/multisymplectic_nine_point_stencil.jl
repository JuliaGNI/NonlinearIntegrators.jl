struct NinePointStencil <: TimeDependentPDEMethod end

struct NinePointStencilCache{ST,N} <: TimeDependentPDEMethodCache{{ST,N}}
    x::Vector{ST}

    function NinePointStencilCache{ST,N}(; kwargs...) where {ST,N}
        x = zeros(ST, N)
        q̄ = zeros(ST, N)
        p̄ = zeros(ST, N)

        q̃ = zeros(ST, N)
        p̃ = zeros(ST, N)

        return new(x,)
    end
end

function Cache{ST}(problem::AbstractProblemIODE, method::NinePointStencil; kwargs...) where {ST}
    NinePointStencilCache{ST, ngrids(problem)}(; kwargs...)
end

nlsolution(cache::NinePointStencilCache) = cache.x

@inline CacheType(ST, problem::AbstractProblemIODE, method::NinePointStencil) = NinePointStencilCache{ST, ngrids(problem)}

function initial_guess(int,history,params,int::GeometricIntegrator{<:NinePointStencil, <:AbstractProblemIODE})
    local x = nlsolution(int)
    local x̄₁ = cache(int).q̄
    local v̄₁ = cache(int).p̄
    local Δt = int.problem.tstep

    x = x̄₁ .+ Δt * v̄₁
end

function components!(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:CGVI}) where {ST}
    
end

function residual(b::AbstractVector{ST}, x::AbstractVector{ST}, int::GeometricIntegrator{<:IPRK, <:AbstractProblemPODE}) where {ST}
    local x̄₀ = int.solstep.history.q[end]

    local x̄₁ = cache(int).q̄
    local v̄₁ = cache(int).p̄
    local N = length(x̄₁)

    b[1] = (1 / (4 * Δt^2)) * (-x[end] + 2x̄₁[end] - x̄₀[end]) + (1 / (2 * Δt^2)) * (-x[1] + 2x̄₁[1] - x̄₀[1]) + (1 / (4 * Δt^2)) * (-x[2] + 2x̄₁[2] - x̄₀[2])
    b[1] += (1 / (4 * Δx^2)) * (x[end] - 2x[1] + x[2]) + (1 / (2 * Δx^2)) * (x̄₁[end] - 2x̄₁[1] + x̄₁[2]) + (1 / (4 * Δx^2)) * (x̄₀[end] - 2x̄₀[1] + x̄₀[2]) 
    for i in 2:N-1
        b[i] = (1 / (4 * Δt^2)) * (-x[i-1] + 2x̄₁[i-1] - x̄₀[i-1]) + (1 / (2 * Δt^2)) * (-x[i] + 2x̄₁[i] - x̄₀[i]) + (1 / (4 * Δt^2)) * (-x[i+1] + 2x̄₁[i+1] - x̄₀[i+1])
        b[i] += (1 / (4 * Δx^2)) * (x[i-1] - 2x[i] + x[i+1]) + (1 / (2 * Δx^2)) * (x̄₁[i-1] - 2x̄₁[i] + x̄₁[i+1]) + (1 / (4 * Δx^2)) * (x̄₀[i-1] - 2x̄₀[i] + x̄₀[i+1]) 
    end
    b[end] = (1 / (4 * Δt^2)) * (-x[end-1] + 2x̄₁[end-1] - x̄₀[end-1]) + (1 / (2 * Δt^2)) * (-x[end] + 2x̄₁[end] - x̄₀[end]) + (1 / (4 * Δt^2)) * (-x[1] + 2x̄₁[1] - x̄₀[1])
    b[end] += (1 / (4 * Δx^2)) * (x[end-1] - 2x[end] + x[1]) + (1 / (2 * Δx^2)) * (x̄₁[end-1] - 2x̄₁[end] + x̄₁[1]) + (1 / (4 * Δx^2)) * (x̄₀[end-1] - 2x̄₀[end] + x̄₀[1]) 
end

function update!(sol, params, int::GeometricIntegrator{<:CGVI}, DT)
    sol.q .= cache(int, DT).q̃
    sol.p .= cache(int, DT).p̃
end

function update!(sol, params, x::AbstractVector{DT}, int::GeometricIntegrator{<:CGVI}) where {DT}
    # compute vector field at internal stages
    components!(x, sol, params, int)

    # compute final update
    update!(sol, params, int, DT)
end

function integrate_step!(sol, history, params, int::GeometricIntegrator{<:NinePointStencil, <:AbstractProblemIODE})
    local x = nlsolution(int)

    prob = NonlinearProblem(residual, x, int)
    x = solve(prob,reltol = 1e-12, abstol = 1e-12)

    update!(sol, params, nlsolution(int), int)
end
