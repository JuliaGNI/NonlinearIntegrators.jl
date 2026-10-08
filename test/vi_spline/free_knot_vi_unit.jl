# VISplineFree and the network-to-spline conversion (src/vi_spline/). Each check is one of the
# statements of benchmark/theory/relu_cgvi_equivalence_en.md:
#   - without knots the integrator is CGVI(P_k) on its rule (Theorem 1's linear side);
#   - with the knots frozen it is CGVI on S_k(Z) with the rule split at Z (Theorem 2);
#   - with free knots the knot equation converges and the step map is symplectic (Corollary 5.3);
#   - a ReLUᵏ network is a spline of degree k with its kinks as knots (Proposition 5.2).

using Test
using LinearAlgebra
using GeometricIntegrators: GeometricIntegrator, integrate, CGVI
using GeometricIntegrators.Integrators.CompactBasisFunctions: Lagrange
using GeometricProblems.HarmonicOscillator: lodeproblem as ho_lodeproblem
import GeometricProblems.HenonHeilesPotential as hh
using NonlinearIntegrators
using NonlinearIntegrators: cache, knot_space, relu_network_to_spline, nn_spline_max_error
using QuadratureRules: GaussLegendreQuadrature, QuadratureRule, nodes
using SimpleSplines: BSplineBasis, GeneralMesh

qs(sol, N) = [sol.q[n][d] for n in 0:N for d in eachindex(sol.q[0])]

@testset "VISplineFree" begin
    k, R, h, N = 3, 4, 1.0, 20
    quad = GaussLegendreQuadrature(R)
    prob = ho_lodeproblem([0.5], [0.0]; timespan = (0.0, N * h), timestep = h)

    # no knots: CGVI(P_k) on the same rule
    cgvi = integrate(prob, CGVI(Lagrange(nodes(GaussLegendreQuadrature(k + 1))), quad))
    @test maximum(abs, qs(integrate(prob, VISplineFree(k, 0, quad)), N) .- qs(cgvi, N)) < 1e-13

    # frozen knots (the uniform seed, never moved): CGVI on S_k(Z) with the rule split at Z
    m = 2
    method = VISplineFree(k, m, quad; maxouter = 0)
    _, c, b = knot_space(method, collect((1:m) ./ (m + 1)))
    spline = CGVI(BSplineBasis(GeneralMesh([0; (1:m) ./ (m + 1); 1]), k), QuadratureRule(2R, c, b))
    @test maximum(abs, qs(integrate(prob, method), N) .- qs(integrate(prob, spline), N)) < 1e-13

    # free knots, two dimensions: the knot equation converges and the step map is symplectic
    function step(z)
        pr = hh.lodeproblem(z[1:2], z[3:4]; timespan = (0.0, h), timestep = h,
            parameters = hh.default_parameters())
        int = GeometricIntegrator(pr, VISplineFree(k, 1, quad))
        sol = integrate(int)
        vcat(sol.q[1], sol.p[1]), cache(int).fallbacks[1]
    end
    z₀, ε = [0.1, 0.1, 0.1, 0.1], 1e-6
    @test last(step(z₀)) == 0
    J = reduce(hcat, (first(step(z₀ .+ ε .* (1:4 .== j))) .- first(step(z₀ .- ε .* (1:4 .== j)))) ./ 2ε
                     for j in 1:4)
    Ω = [zeros(2, 2) I(2); -I(2) zeros(2, 2)]
    @test maximum(abs, J' * Ω * J .- Ω) < 1e-8
end

@testset "ReLUᵏ network = spline" begin
    # kinks inside, left of, right of [0, 1], and negative input weights
    a = [0.7, -1.3, 0.4, 2.0, -0.5, 0.9]
    w = [1.5, -2.0, 0.8, 1.0, -0.6, 3.0]
    b = [-0.3, 1.2, 0.5, -1.4, 0.21, -2.4]
    for k in 1:5
        sp = relu_network_to_spline(a, w, b; degree = k)
        @test nn_spline_max_error(a, w, b, sp, range(0, 1; length = 201)) < 1e-12
    end
end
