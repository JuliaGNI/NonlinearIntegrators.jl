# Unit test for PR_Integrator / PR_Basis (the symbolic regression integrator).
#
# Unlike the neural-network integrators, PR_Integrator uses Symbolics.jl to compile
# basis functions; the compiled functions evaluate in Float64 regardless of the type
# parameter T, so this test is restricted to Float64. PR_Integrator keeps its own
# `integrate!` override — it is not a `NetworkIntegratorMethod` and so does not pick up the
# shared one — and returns a 3-tuple (sol, internal_values, x_list) rather than the
# shared (sol, internal_values).
#
# The initial parameters `init_w` are set close to the exact HO solution
# q(t) = A·cos(ω·t) with A=0.5, ω≈0.707 (=√0.5, the default HO spring constant),
# φ=0, giving Newton a convergent starting point.

@testset "PR_Integrator (Float64)" begin
    prb = build_pr_basis(Float64)
    init_w = [Float64[0.5, sqrt(0.5), 0.0]]
    method = PR_Integrator(prb, gauss(Float64, 4), init_w)

    prob = HarmonicOscillator.lodeproblem([Float64(0.5)], [Float64(0.0)];
        timespan = (Float64(0.0), Float64(0.1)), timestep = Float64(0.1))

    sol, internal_values, x_list = integrate(prob, method)

    @test eltype(sol.q[end]) == Float64
    @test all(isfinite, collect(sol.q[:, 1])[end])
    @test internal_values isa AbstractArray
    @test x_list isa AbstractArray
end
