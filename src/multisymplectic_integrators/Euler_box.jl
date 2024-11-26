using GeometricIntegrators:ODEMethod

struct EulerBox <: LODEMethod end

struct EulerBoxCache{DT,NX,D}
    u::Vector{DT,NX,D}
    v::Vector{DT,NX,D}
    w::Vector{DT,NX,D}
    p::Vector{DT,NX,D}
    function EulerBoxCache{DT,NX,D}() where {DT,NX,D} 
        u = zeros(DT,NX,D)
        v = zeros(DT,NX,D)
        w = zeros(DT,NX,D)
        p = zeros(DT,NX,D)
        new{DT,NX,D}(u,v,w,p)
    end
end

function Cache{ST}(problem, method::EulerBox; kwargs...) where {ST}
    EulerBoxCache{ST, ndims(problem)}(; kwargs...)
end

@inline CacheType(ST, problem::AbstractProblem, method::EulerBox) = EulerBoxCache{ST, ndims(problem)}

function update!(sol, params, _, int::GeometricIntegrator{<:EulerBox})
    # compute final update
    sol.v .+= timestep(int) 
end
