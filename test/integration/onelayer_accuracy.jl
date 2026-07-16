# Float64 accuracy regression guard for the OGA-seeded one-layer variational
# integrator on the canonical default Harmonic Oscillator problem, asserting the
# network solution matches the analytic reference to ~machine precision. The final
# accuracy is set by the Newton solve of the variational equations, not by the OGA
# seed, so a moderate dictionary already reaches < 1e-12 (a dict_amount of 400 000
# only slows the seed build without improving accuracy). Solver options are passed
# through integrate(...).
@testset "NonLinear_OneLayer_GML OGA accuracy (Float64)" begin
    @debug "NonLinear_OneLayer_GML accuracy: Float64, S=4, R=8, dict_amount=4000"
    HO_lode = lodeproblem()
    HO_ref  = exact_solution(podeproblem())

    net = build_onelayer_basis(Float64; S = 4)
    method = NonLinear_OneLayer_GML(net, gauss(Float64, 8); bias_interval = [-pi, pi], dict_amount = 4000)

    sol, _ = integrate(HO_lode, method; regularization_factor = 1e-5, max_iterations = 10000)

    rel_err = relative_maximum_error(sol.q, HO_ref.q)
    @debug "NonLinear_OneLayer_GML accuracy (Float64)" relative_max_error=rel_err
    @test rel_err < 1e-12
end
