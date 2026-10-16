# Unit tests for `FreeKnotSpline`, the variational integrator on a cubic spline with free knots.
#
# Every reference here is independent of the method: the closed-form solution of a linear
# oscillator, or a fine RK4 run for the pendulum. The thresholds sit between the cubic step and
# the free-knot step. The cubic (S = 0) step has an endpoint error of 5.8e-10 to 5.9e-10 after one
# step of h = 0.5 on these problems, and the free-knot step 1e-13. So a step that collapses onto
# the cubic — knots merged, at 0 or 1, or dead — fails every accuracy test in this file.

using Test
using NonlinearIntegrators
using GeometricIntegratorsBase
using GeometricEquations: LODEProblem
using GeometricIntegratorsBase: nlsolution
using GeometricProblems.HarmonicOscillator

# The knots and the truncated-cubic coefficients of the last step, from `x = (a, c, τ, p̃)`.
function knots_and_weights(x, S, D)
    c = reshape(x[(4D + 1):(4D + S * D)], S, D)
    τ = x[(4D + S * D + 1):(4D + S * D + S)]
    τ, c
end

# A live step: every knot inside (0, 1) and away from its neighbours, every truncated cubic used.
function live(τ, c)
    gaps = diff(vcat(0.0, sort(τ), 1.0))
    minimum(gaps) > 1e-3 && minimum(abs, c) > 1e-3 * maximum(abs, c)
end

@testset "FreeKnotSpline" begin
    @testset "S = 0 is rejected" begin
        @test_throws ArgumentError FreeKnotSpline(0)
    end

    @testset "harmonic oscillator, S = $S" for S in (4, 6)
        params = HarmonicOscillator.default_parameters()
        prob = HarmonicOscillator.lodeproblem([0.5], [0.0];
            timespan = (0.0, 1.0), timestep = 0.5, parameters = params)
        int = GeometricIntegrator(prob, FreeKnotSpline(S))
        sol = integrate(int)
        q = collect(sol.q[:, 1])
        for n in 1:2
            qref = HarmonicOscillator.exact_solution_q(0.5n, 0.5, 0.0, 0.0, params)
            @test abs(q[n + 1] - qref) < 1e-12
        end
        τ, c = knots_and_weights(nlsolution(int), S, 1)
        @test live(τ, c)
    end

    # Near a zero of q, the truncated cubics are about 1e-5 and change sign, and the gain of the
    # knots over the cubic falls below the round-off of the action. A descent that stops on the
    # action alone then leaves Newton next to the dead-knot point, and the step collapses onto
    # the cubic: the error jumps by 1e-11 at the fifth step, t = 2.5. Sixteen steps cross two
    # zeros of q.
    @testset "harmonic oscillator across zeros of q, S = 4" begin
        params = HarmonicOscillator.default_parameters()
        prob = HarmonicOscillator.lodeproblem([0.5], [0.0];
            timespan = (0.0, 8.0), timestep = 0.5, parameters = params)
        sol = integrate(prob, FreeKnotSpline(4))
        q = collect(sol.q[:, 1])
        qref = [HarmonicOscillator.exact_solution_q(0.5n, 0.5, 0.0, 0.0, params)
                for n in 0:16]
        @test maximum(abs, q .- qref) < 5e-12
    end

    @testset "pendulum, S = 4" begin
        k = 0.5
        ϑ(p, t, q, v, params) = (p .= v)
        f(f, t, q, v, params) = (f .= -params.k .* sin.(q))
        ω(ω, t, q, v, params) = (ω .= [0 -1; 1 0])
        l(t, q, v, params) = v[1]^2 / 2 + params.k * cos(q[1])
        v̄(v, t, q, p, params) = (v .= p)
        prob = LODEProblem(ϑ, f, ω, l, (0.0, 1.0), 0.5, [1.0], [0.0];
            v̄ = v̄, parameters = (k = k,))
        sol = integrate(prob, FreeKnotSpline(4))
        q = collect(sol.q[:, 1])

        # RK4 at h/40000: its error, of order (1.25e-5)⁴, is far below the thresholds.
        function rk4(q, p, T; n = 40000)
            dt = T / n
            for _ in 1:n
                k1q, k1p = p, -k * sin(q)
                k2q, k2p = p + dt / 2 * k1p, -k * sin(q + dt / 2 * k1q)
                k3q, k3p = p + dt / 2 * k2p, -k * sin(q + dt / 2 * k2q)
                k4q, k4p = p + dt * k3p, -k * sin(q + dt * k3q)
                q += dt / 6 * (k1q + 2k2q + 2k3q + k4q)
                p += dt / 6 * (k1p + 2k2p + 2k3p + k4p)
            end
            q
        end
        @test abs(q[2] - rk4(1.0, 0.0, 0.5)) < 1e-12
        @test abs(q[3] - rk4(1.0, 0.0, 1.0)) < 1e-11
    end

    @testset "two degrees of freedom share the knots" begin
        # q̈ = -ω²q with ω = (1, 0.6), from q = (0.5, 0.2), p = (0, 0.1)
        ωs = [1.0, 0.6]
        ϑ(p, t, q, v, params) = (p .= v)
        f(f, t, q, v, params) = (f .= -params.ω .^ 2 .* q)
        ω(ω, t, q, v, params) = (ω .= 0)
        l(t, q, v, params) = sum(v .^ 2) / 2 - sum(params.ω .^ 2 .* q .^ 2) / 2
        v̄(v, t, q, p, params) = (v .= p)
        q₀, p₀ = [0.5, 0.2], [0.0, 0.1]
        prob = LODEProblem(ϑ, f, ω, l, (0.0, 1.0), 0.5, q₀, p₀;
            v̄ = v̄, parameters = (ω = ωs,))
        int = GeometricIntegrator(prob, FreeKnotSpline(4))
        sol = integrate(int)
        exact(t) = q₀ .* cos.(ωs .* t) .+ p₀ ./ ωs .* sin.(ωs .* t)
        @test maximum(abs, collect(sol.q[2]) .- exact(1.0)) < 1e-11
        τ, c = knots_and_weights(nlsolution(int), 4, 2)
        @test live(τ, c)
    end
end
