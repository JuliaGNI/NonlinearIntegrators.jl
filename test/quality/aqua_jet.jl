# Aqua.jl and JET.jl static checks.
#
# Gated on the Julia version. JET's analysis output moves between Julia releases and the CI
# matrix includes `nightly` and a `^1.13.0-0` prerelease, so running it everywhere would turn
# an upstream change into a red build here. Restricting it to stable releases from 1.12 on
# means exactly one CI job runs it, which still catches a regression introduced in this
# package. Set `NI_STATIC_ANALYSIS=true` to force it on locally.
const RUN_STATIC_ANALYSIS = get(ENV, "NI_STATIC_ANALYSIS", "") == "true" ||
                            (isempty(VERSION.prerelease) && VERSION >= v"1.12")

if RUN_STATIC_ANALYSIS
    using Aqua
    using JET
    # `GeometricBase.update!` is the generic the ambiguity exclusion below names.
    using GeometricBase
end

@testset "Aqua" begin
    if !RUN_STATIC_ANALYSIS
        @test_skip "Aqua static analysis runs on stable Julia ≥ 1.12"
    else
        # `piracies = false`: this package extends ~15 `GeometricIntegratorsBase` generics
        # (`components!`, `residual!`, `update!`, `Cache`, `CacheType`, `initial_guess!`,
        # `integrate!`, …) on its *own* method and cache types, which is the framework's
        # intended extension mechanism. Aqua's heuristic cannot distinguish that from piracy
        # when both the function and one argument type are foreign.
        Aqua.test_all(NonlinearIntegrators; piracies = false, ambiguities = false)

        @testset "ambiguities" begin
            # `exclude = [GeometricBase.update!]`. Three ambiguities remain, all of the form
            #
            #   GeometricOptimizers: update!(::BFGSState, ::Gradient, ::XT, ::Any)
            #   here:                update!(sol, params, ::AbstractVector{DT}, ::GeometricIntegrator{...})
            #
            # Our signature is the one `GeometricIntegratorsBase` itself uses for
            # `ImplicitMidpoint`/`CrankNicolson`/`ImplicitEuler`, i.e. the documented extension
            # point, and resolving it would mean typing `sol` and `params`, which the framework
            # deliberately leaves free. The ambiguous call — a BFGS optimizer state and a
            # gradient passed alongside a `GeometricIntegrator` — is not reachable.
            #
            # This is an exclusion for *one function*, not for the check: an ambiguity
            # introduced anywhere else still fails here. Five further ambiguities that Aqua
            # reported before the audit were this package's own doing and were fixed by giving
            # the DT-form update a name of its own (`update_solution!`); see
            # `network_integrator_core.jl`.
            Aqua.test_ambiguities(NonlinearIntegrators;
                exclude = [GeometricBase.update!])
        end
    end
end

@testset "JET" begin
    # Deliberately not asserted in the suite. `test/quality/jet_residual.jl` analyses
    # `residual!` for runtime dispatch and reports **clean** for all four integrators in every
    # environment that can be constructed by hand — a plain session, a `Pkg.test`-equivalent
    # environment, with and without `--check-bounds=yes`, `--pkgimages=no`, `-O0`, `-g1`,
    # `--depwarn`, `--code-coverage=none`. Inside the `Pkg.test` process itself, and in a
    # subprocess spawned from it, the same analysis reports twelve dispatches whose shape
    # (`view(r, :, d)` inferring as `SubArray{Float64,N,Matrix{Float64},I,true} where {N,I}`)
    # is inference widening under accumulated specialisations, not a property of the code.
    #
    # A gate that fails only inside the harness, for a reason that cannot be reproduced or
    # explained, is worse than no gate: it trains people to ignore it. What is asserted instead
    # is the *behaviour* that the type stability buys — the `@allocated` budgets and `@inferred`
    # in `inference_and_allocations.jl`, both of which do run in the suite.
    #
    # Run the analysis by hand; the exact command is in that file's header (it needs a temp
    # environment, because `test/Project.toml` does not carry the package itself).
    @test_skip "JET: run test/quality/jet_residual.jl by hand; see the note above"
end
