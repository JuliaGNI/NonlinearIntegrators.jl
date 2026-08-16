# Unit tests for VISE / VISEBasis (the symbolic-regression integrator).
#
# Restricted to Float64: VISE compiles its basis functions with Symbolics.jl and the compiled
# functions evaluate in Float64 regardless of the type parameter. VISE also keeps its own
# `integrate!` override — it is not a `NetworkIntegratorMethod` and so does not pick up the
# shared one — and returns a 3-tuple `(sol, internal_values, x_list)` rather than the shared
# 2-tuple.
#
# The ansatz here is `q(t) = W₁·cos(W₂·t + W₃)`, which is *exact* for the harmonic oscillator:
# with the default parameters (m = 1, k = 0.5) the solution is `0.5·cos(√0.5·t)`, and `init_w`
# starts Newton at exactly that point. So this file can assert real accuracy rather than mere
# finiteness — which is what it used to do, on a single time step, making it the weakest guard
# in the suite and no basis at all for the type-stability work on `VISECache`/`VISEBasis`.

vise_method() = VISE(build_vise_basis(Float64), gauss(Float64, 4),
                     [Float64[0.5, sqrt(0.5), 0.0]])

@testset "VISE (Float64)" begin
    params = HarmonicOscillator.default_parameters(Float64)

    @testset "accuracy over $nsteps step(s)" for nsteps in (1, 5)
        tend = 0.1 * nsteps
        prob = HarmonicOscillator.lodeproblem([0.5], [0.0];
            timespan = (0.0, tend), timestep = 0.1, parameters = params)

        sol, internal_values, x_list = integrate(prob, vise_method())

        @test eltype(sol.q[end]) == Float64
        qend = collect(sol.q[:, 1])[end]
        qref = HarmonicOscillator.exact_solution_q(tend, 0.5, 0.0, 0.0, params)
        @debug "VISE" nsteps q_end=qend q_ref=qref abs_err=abs(qend - qref)
        # The ansatz spans the exact solution, so the only error is the Newton residual.
        @test abs(qend - qref) < 1e-12

        # `internal_values` and `x_list` are sized `n₂-n₁+1` and must be *fully* populated.
        @test length(internal_values) == nsteps
        @test length(x_list) == nsteps
        @test all(i -> isassigned(internal_values, i), eachindex(internal_values))
        @test all(i -> isassigned(x_list, i), eachindex(x_list))
        # Concrete element types, not `Vector{Matrix}` / `Vector{Vector}`.
        @test isconcretetype(eltype(internal_values))
        @test isconcretetype(eltype(x_list))
    end

    # Regression guard for the restart indexing bug: `integrate!` sized its two output vectors
    # `n₂-n₁+1` but indexed them by `n`, so any `n₁ > 1` left the first `n₁-1` slots `#undef`
    # and ran off the end. The network integrators had this fixed; the VISE copy had not.
    @testset "restart from n₁ > 1" begin
        prob = HarmonicOscillator.lodeproblem([0.5], [0.0];
            timespan = (0.0, 0.3), timestep = 0.1, parameters = params)
        int = GeometricIntegrator(prob, vise_method())
        sol = GeometricIntegratorsBase.GeometricSolution(prob)

        # Step 1 first, so that `sol[1]` holds a real state for the restart to continue from —
        # starting at n₁ = 2 against an unwritten `sol[1]` gives a degenerate Newton system.
        GeometricIntegratorsBase.integrate!(sol, int, 1, 1)
        _, internal_values, x_list = GeometricIntegratorsBase.integrate!(sol, int, 2, 3)

        # Sized n₂-n₁+1 = 2. Indexing these by `n` rather than `n-n₁+1`, as the code used to,
        # writes past the end on the second iteration.
        @test length(internal_values) == 2
        @test all(i -> isassigned(internal_values, i), eachindex(internal_values))
        @test all(i -> isassigned(x_list, i), eachindex(x_list))
    end

    # These four used to be defined *unqualified*, which created a shadowing
    # `NonlinearIntegrators.isexplicit` instead of extending the framework generic — so the
    # framework answered `missing` for all of them, including where the intent was a definite
    # `false`/`true`. Asserted through `GeometricIntegratorsBase` precisely so that a bare
    # definition cannot pass.
    @testset "traits reach the framework" begin
        m = vise_method()
        @test GeometricIntegratorsBase.isexplicit(m) === false
        @test GeometricIntegratorsBase.isimplicit(m) === true
        @test GeometricIntegratorsBase.issymmetric(m) === missing
        @test GeometricIntegratorsBase.issymplectic(m) === missing
    end
end
