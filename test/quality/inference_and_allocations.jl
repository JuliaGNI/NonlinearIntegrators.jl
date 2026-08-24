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

# Julia 1.10 pays more than 1.12+ for the same call, and `SymbolicNeuralNetworks` 0.6 widened the
# gap. The symbolic bases go through generated kernels (`DQDθ`, `DVDθ`, `V_func`), and 0.6 laid
# their equation sets out over `NeuralNetworkParameters.ParameterLayout` instead of a local
# `FlatSlice`; something in that path stops folding on 1.10 and does not on 1.12 or later.
# Measured per `residual!` call, Float64, S = 4, R = 8, D = 1:
#
#                                 1.10 / ANN 0.6.4   1.10 / ANN 0.7   1.11 - 1.13 / ANN 0.7
#   ShallowNet                              15 168           28 096                  11 424
#   ShallowNetReversible                    15 168           28 096                  11 424
#   ShallowNetAutodiff                      51 584           51 584        52 096 - 54 656
#   ShallowNetAutodiffReversible            51 584           51 584        52 096 - 54 656
#
# Only the two symbolic rows move, and only on 1.10: the autodiff rows go through `ForwardDiff`
# rather than a generated kernel and are unchanged. 1.11 was measured (11 424, same as 1.13) so
# that the cutoff below is a measurement and not a guess — CI's matrix skips 1.11. The container
# constructor itself allocates 0 bytes under both stacks on every version tried, so this is not
# the `NetworkParameters` rename. 28 096 is byte-identical on macOS aarch64, macOS x86_64 and
# Linux x86_64, so it is deterministic rather than noise and a fixed ceiling is safe.
#
# The 1.10 ceiling keeps the same ~1.5x margin over what 1.10 actually costs, so a *further*
# regression there still trips. The tight ceiling stays in force on 1.11 and later, which is
# where the number this package can control is visible. Recorded under *Open Issues* →
# *Upstream* in the CHANGELOG and reported as SymbolicNeuralNetworks#55; remove this override when
# that closes.
if VERSION < v"1.11"
    RESIDUAL_ALLOC_BUDGET["ShallowNet"]           = 42_000
    RESIDUAL_ALLOC_BUDGET["ShallowNetReversible"] = 42_000
end

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
