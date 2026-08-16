# Inference and allocation regression gates for the Newton hot path.
#
# `residual!` runs once per Newton iteration *and* once per ForwardDiff Jacobian column, so it
# is where a type instability or a stray temporary actually costs something. Before the audit
# the suite had exactly one `@inferred` (on `oga_fit`, in oga_kernels.jl) and no allocation
# assertion anywhere, which is how the two integrators that allocated ~212 kB per residual
# evaluation stayed that way.
#
# The budgets below are *ceilings*, set roughly 1.5× above the measured figure so that ordinary
# variation between Julia versions and platforms does not turn this into a flaky test. They are
# here to catch a regression of the kind the audit found — a factor of two or more — not to pin
# an exact byte count. Update them deliberately, with the measurement, when the hot path
# legitimately changes.

using GeometricIntegratorsBase: solutionstep, nlsolution, residual!, initial_guess!, current
using GeometricSolutions: timesteps

# Drive one integrator to the point where `residual!` can be called repeatedly with a fixed
# argument, which is the only way to measure it in isolation: measuring a whole `integrate`
# call conflates per-call cost with how many Newton iterations that configuration happened to
# take, and those differ by more than an order of magnitude between integrators.
function residual_probe(make, ::Type{T}) where {T}
    prob = ho_problem(T)
    int = GeometricIntegrator(prob, make(T);
        regularization_factor = T(1e-5), max_iterations = MAX_NEWTON_ITERATIONS)
    sol = GeometricIntegratorsBase.GeometricSolution(prob)
    solstep = solutionstep(int, sol[0])
    GeometricIntegratorsBase.reset!(solstep, timesteps(sol)[1])
    s = current(solstep)
    params = GeometricIntegratorsBase.parameters(prob)
    initial_guess!(s, nothing, params, int)
    x = nlsolution(int)
    b = similar(x)
    return (; int, s, params, x, b)
end

# Measured on Float64, S = 4, R = 8, D = 1 (bytes per `residual!` call), before → after the
# audit:
#
#   ShallowNet                     21 344 → 11 424   (1.9×)
#   ShallowNetReversible           26 176 → 11 424   (2.3×)
#   ShallowNetAutodiff            211 968 → 51 584   (4.1×)
#   ShallowNetAutodiffReversible  216 800 → 51 584   (4.2×)
const RESIDUAL_ALLOC_BUDGET = Dict(
    "ShallowNet"                   => 17_000,
    "ShallowNetReversible"         => 17_000,
    "ShallowNetAutodiff"           => 78_000,
    "ShallowNetAutodiffReversible" => 78_000,
)

# The measurement has to happen inside a function taking `p` as an argument, not inline in a
# `@testset` body. `@testset` wraps its body in a closure, so an inline `@allocated` also
# measures the boxed access to every captured variable — which inflated ShallowNet from 11 424
# to 31 411 bytes here and would have made the budget meaningless.
function residual_bytes(p, n = 10)
    residual!(p.b, p.x, p.s, p.params, p.int)                 # compile / first touch
    return (@allocated for _ in 1:n
        residual!(p.b, p.x, p.s, p.params, p.int)
    end) ÷ n
end

@testset "residual! hot path" begin
    for row in NETWORK_INTEGRATORS
        haskey(RESIDUAL_ALLOC_BUDGET, row.name) || continue   # DenseNet has no pinned budget

        @testset "$(row.name)" begin
            p = residual_probe(row.make, Float64)

            @testset "allocations" begin
                bytes = residual_bytes(p)
                @debug "$(row.name) residual! bytes" bytes
                @test bytes <= RESIDUAL_ALLOC_BUDGET[row.name]
            end

            @testset "inference" begin
                # `residual!` returns nothing useful; what matters is that inference gets
                # through the call without falling back to `Any`. `@inferred` still fails on a
                # `Nothing`-returning call if the call itself cannot be inferred.
                @test (@inferred residual!(p.b, p.x, p.s, p.params, p.int)) isa Any
            end
        end
    end
end

@testset "oga_fit is allocation-stable across dimensions" begin
    # The OGA seed builds a dictionary-sized `Ψ` per call. This does not assert a small number
    # — the matrix is genuinely large — only that a second call with identical arguments costs
    # the same as the first, i.e. that nothing has started growing per invocation.
    nodes, weights, y = oga_testcase(Float64)
    σ = relu_k(3)
    args = (OGA1d(), σ, nodes, weights, y, 4)
    kw = (; bias_interval = [-Float64(pi), Float64(pi)], dict_amount = 200)
    oga_fit(args...; kw...)
    a1 = @allocated oga_fit(args...; kw...)
    a2 = @allocated oga_fit(args...; kw...)
    @debug "oga_fit bytes" a1 a2
    @test a1 == a2
end
