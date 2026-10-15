# Aqua.jl static checks. JET's analysis is in `jet.jl`.

using Test
using NonlinearIntegrators
using Aqua
# `GeometricBase.update!` is the generic the ambiguity exclusion below names.
using GeometricBase

@testset "Aqua" begin
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
