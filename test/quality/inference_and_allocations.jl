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

# Measured on Float64, S = 4, R = 8, D = 1 (bytes per `residual!` call), before the audit → after →
# **re-measured on Julia 1.11.9 against the current stack**, which is what the ceilings below are set
# from:
#
#   ShallowNet                     21 344 → 11 424 → 11 424
#   ShallowNetReversible           26 176 → 11 424 → 11 424
#   ShallowNetAutodiff            211 968 → 51 584 → 54 656
#   ShallowNetAutodiffReversible  216 800 → 51 584 → 54 656
#
# The two autodiff rows are 3072 bytes dearer than recorded and the two symbolic rows have not moved
# at all. Both are worth stating rather than quietly re-recording, because the *prediction* was the
# other way round: `NeuralNetworkParameters` 0.2.2 took `SymbolicNeuralNetworks`' single-sample split
# from 768 bytes to 560, and the symbolic `residual!` here calls `DQDθ` on a length-one `Vector`, so
# that is the path it takes — and it moves this figure by nothing. Measured, not assumed.
#
# One ceiling per row, on every Julia version. That was not true between `SymbolicNeuralNetworks`
# 0.6.0 and 0.7.0: the two symbolic rows cost 28 096 bytes on Julia 1.10 there — 1.85× the 15 168 they
# cost under SNN 0.5, while 1.11 and later stayed at 11 424 either way — and this file carried a
# 1.10-only ceiling of 42 000 for them. The cause was a `map` over a closure that 1.10 does not always
# elide, on the walk that splits a generated function's flat result back into the nesting of the
# parameters. It is fixed in two independent halves and `Project.toml` requires both:
# `SymbolicNeuralNetworks` 0.7.0 for the batched walk and `NeuralNetworkParameters` 0.2.1 for the
# un-batched one.
#
# **That whole comparison is now history rather than a live concern**, since 1.10 is no longer a
# supported version here. It is kept because the shape of it recurs: a defect that shows on one Julia
# version and not another, in a walk two packages away, found from a budget in this file.
const RESIDUAL_ALLOC_BUDGET = Dict(
    "ShallowNet" => 17_000,
    "ShallowNetReversible" => 17_000,
    "ShallowNetAutodiff" => 78_000,
    "ShallowNetAutodiffReversible" => 78_000
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
                # A weak gate, deliberately kept as a tripwire rather than relied on:
                # `@inferred` compares the *return* type against `typeof(result)`, and
                # `residual!` returns `nothing`, so it passes as long as inference reaches
                # `Nothing` — which it can do with runtime dispatch throughout the body. What
                # it does catch is the call becoming uninferable altogether, e.g. a cache
                # lookup that stops folding. The allocation budget above is the assertion that
                # actually bites; `jet_residual.jl` is the one that checks dispatch directly.
                @test (@inferred residual!(p.b, p.x, p.s, p.params, p.int)) isa Any
            end
        end
    end
end

# Repeated `@allocated` samples of one `oga_fit` call, measured inside a function for the same
# reason `residual_bytes` is: a `@testset` body is a closure, and an inline `@allocated` would
# fold the boxed access to every captured variable into the number.
function oga_fit_bytes(args, kw, n = 3)
    oga_fit(args...; kw...)                                   # compile / first touch
    return [(@allocated oga_fit(args...; kw...)) for _ in 1:n]
end

@testset "oga_fit is allocation-stable across dimensions" begin
    # The OGA seed builds a dictionary-sized `Ψ` per call. This does not assert a small number
    # — the matrix is genuinely large — only that repeated calls with identical arguments cost
    # the same, i.e. that nothing has started growing per invocation.
    #
    # Three samples and a tolerance, not `a1 == a2`. Exact byte equality between two
    # `@allocated` calls is not a property of this code: the same assertion read 222 374 vs
    # 222 438 on Windows / 1.13.0-rc3 and 155 894 vs 155 846 on Windows / nightly — tens of
    # bytes, and in *opposite* directions, which is measurement noise rather than growth.
    # (The two totals differ by 30% between those builds, so the absolute figure is not a
    # stable quantity either, which is why nothing is pinned here.)
    #
    # Growth is what the test is for, and growth compounds: over `n` samples it widens the
    # spread by `n` times the per-call leak, where the noise floor stays put. So the assertion
    # is on the spread, at ~4 kB against a ~200 kB measurement — two orders of magnitude above
    # the noise seen, and far below the per-call `Ψ` that a regression here would add.
    nodes, weights, y = oga_testcase(Float64)
    σ = relu_k(3)
    args = (OGA1d(), σ, nodes, weights, y, 4)
    kw = (; bias_interval = [-Float64(pi), Float64(pi)], dict_amount = 200)

    bytes = oga_fit_bytes(args, kw)
    spread = maximum(bytes) - minimum(bytes)
    @debug "oga_fit bytes" bytes spread
    @test spread ≤ max(4096, minimum(bytes) ÷ 100)
end
