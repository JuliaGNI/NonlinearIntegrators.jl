# Unit tests for the native B-spline implementation in src/vi_spline/bspline_utils.jl.
#
# Three verification strategies:
#   1. Analytic checks  — known exact values for simple cases (linear B-splines, partition
#      of unity, boundary values, symmetry).
#   2. Numerical checks — bspline_deriv and bspline_knot_deriv compared against
#      ForwardDiff automatic differentiation through bspline_value.
#   3. Precision checks — Float32 and Float64 results agree to single precision.
#
# A separate script (scripts/compare_bspline_bsplinekit.jl) compares against BSplineKit.jl
# using the scripts sub-environment; it is not run as part of the main test suite because
# BSplineKit is not a package dependency.

using Test
using ForwardDiff
using NonlinearIntegrators: bspline_value, bspline_deriv, bspline_knot_deriv,
                             bspline_mixed_deriv, build_knot_vector,
                             uniform_internal_knots, bspline_eval_matrix

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Finite-difference time derivative of B_{i,k}(t)
function fd_bspline_deriv(i, k, t, knots; T = eltype(knots), ε = T(1e-7))
    (bspline_value(i, k, t + ε, knots; T) - bspline_value(i, k, t - ε, knots; T)) / (T(2) * ε)
end

# ForwardDiff time derivative of B_{i,k}(t)
function ad_bspline_deriv(i, k, t::T, knots; kwargs...) where {T}
    ForwardDiff.derivative(s -> bspline_value(i, k, s, knots; T), t)
end

# ForwardDiff knot-position derivative of B_{i,k}(t) w.r.t. knots[j]
# Use Vector{typeof(kj)} so the Dual element type flows through bspline_value.
function ad_bspline_knot_deriv(i, k, t, knots, j; kwargs...)
    ForwardDiff.derivative(knots[j]) do kj
        kv = Vector{typeof(kj)}(knots)
        kv[j] = kj
        bspline_value(i, k, t, kv)
    end
end

# ForwardDiff mixed derivative ∂(B'_{i,k})/∂knots[j]
function ad_bspline_mixed_deriv(i, k, t, knots, j; kwargs...)
    ForwardDiff.derivative(knots[j]) do kj
        kv = Vector{typeof(kj)}(knots)
        kv[j] = kj
        bspline_deriv(i, k, t, kv)
    end
end

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@testset "B-spline utils" begin

    # ------------------------------------------------------------------
    # Partition of unity: Σᵢ B_{i,k}(t) = 1 for t in domain
    @testset "Partition of unity (order $k)" for k in [2, 3, 4]
        for T in (Float32, Float64)
            τ = uniform_internal_knots(3; T)
            knots = build_knot_vector(k, τ; T)
            S = length(knots) - k
            for t in T[0.0, 0.1, 0.3, 0.5, 0.72, 0.9, 1.0]
                s = sum(bspline_value(i, k, t, knots; T) for i in 1:S)
                @test s ≈ T(1) atol = T(1e-5)
            end
        end
    end

    # ------------------------------------------------------------------
    # Non-negativity of basis functions
    @testset "Non-negativity (order $k)" for k in [2, 3, 4]
        T = Float64
        τ = uniform_internal_knots(4; T)
        knots = build_knot_vector(k, τ; T)
        S = length(knots) - k
        for t in range(T(0), T(1), length = 41)
            for i in 1:S
                @test bspline_value(i, k, t, knots; T) ≥ -1e-12
            end
        end
    end

    # ------------------------------------------------------------------
    # Exact linear B-spline (order 2) values at a known knot sequence
    @testset "Linear B-spline exact values" begin
        T = Float64
        # Uniform knots: [0,0, 0.25, 0.5, 0.75, 1,1]  →  S=5 basis functions
        knots = T[0, 0, 0.25, 0.5, 0.75, 1, 1]
        k = 2

        # B₁ should equal 1 at t=0 and 0 at t≥0.25
        @test bspline_value(1, k, T(0), knots; T) ≈ T(1)
        @test bspline_value(1, k, T(0.25), knots; T) ≈ T(0)

        # B₅ should equal 1 at t=1 and 0 at t≤0.75
        @test bspline_value(5, k, T(1), knots; T) ≈ T(1)
        @test bspline_value(5, k, T(0.75), knots; T) ≈ T(0)

        # B₃ peaks at t=0.5: B₃(0.5) = 1/(0.5-0.25) * (0.5-0.25) + (0.75-0.5)/(0.75-0.5) ...
        # For uniform linear B-splines B_i(τᵢ) = 1 at its peak
        @test bspline_value(3, k, T(0.5), knots; T) ≈ T(1)
    end

    # ------------------------------------------------------------------
    # bspline_eval_matrix rows sum to 1
    @testset "eval_matrix partition of unity" begin
        T = Float64
        τ = uniform_internal_knots(5; T)
        knots = build_knot_vector(4, τ; T)
        pts = collect(range(T(0), T(1), length = 21))
        B = bspline_eval_matrix(4, knots, pts; T)
        @test all(≈(T(1); atol = 1e-12), sum(B; dims = 2))
    end

    # ------------------------------------------------------------------
    # bspline_deriv vs ForwardDiff (time derivative)
    @testset "bspline_deriv vs ForwardDiff (order $k)" for k in [2, 3, 4]
        T = Float64
        τ = uniform_internal_knots(3; T)
        knots = build_knot_vector(k, τ; T)
        S = length(knots) - k
        # Evaluate away from knot positions to avoid distributional singularities
        test_pts = T[0.05, 0.18, 0.37, 0.53, 0.68, 0.82, 0.95]
        for t in test_pts, i in 1:S
            analytic = bspline_deriv(i, k, t, knots; T)
            ad       = ad_bspline_deriv(i, k, t, knots)
            @test analytic ≈ ad atol = 1e-8
        end
    end

    # ------------------------------------------------------------------
    # bspline_knot_deriv vs ForwardDiff (knot-position derivative)
    @testset "bspline_knot_deriv vs ForwardDiff (order $k)" for k in [3, 4]
        T = Float64
        τ = uniform_internal_knots(3; T)
        knots = build_knot_vector(k, τ; T)
        S = length(knots) - k
        m = length(knots)
        # Test at interior t values, and for each knot position j
        test_pts = T[0.08, 0.22, 0.44, 0.61, 0.77, 0.91]
        for t in test_pts, i in 1:S, j in (k + 1):(m - k)
            analytic = bspline_knot_deriv(i, k, t, knots, j; T)
            ad       = ad_bspline_knot_deriv(i, k, t, knots, j)
            @test analytic ≈ ad atol = 1e-7
        end
    end

    # ------------------------------------------------------------------
    # bspline_mixed_deriv vs ForwardDiff (mixed derivative)
    @testset "bspline_mixed_deriv vs ForwardDiff (order $k)" for k in [3, 4]
        T = Float64
        τ = uniform_internal_knots(3; T)
        knots = build_knot_vector(k, τ; T)
        S = length(knots) - k
        m = length(knots)
        test_pts = T[0.09, 0.31, 0.52, 0.74, 0.88]
        for t in test_pts, i in 1:S, j in (k + 1):(m - k)
            analytic = bspline_mixed_deriv(i, k, t, knots, j; T)
            ad       = ad_bspline_mixed_deriv(i, k, t, knots, j)
            @test analytic ≈ ad atol = 1e-7
        end
    end

    # ------------------------------------------------------------------
    # Float32 vs Float64 consistency
    @testset "Float32 / Float64 consistency (order $k)" for k in [2, 3, 4]
        τ64 = uniform_internal_knots(3; T = Float64)
        τ32 = uniform_internal_knots(3; T = Float32)
        knots64 = build_knot_vector(k, τ64; T = Float64)
        knots32 = build_knot_vector(k, τ32; T = Float32)
        S = length(knots64) - k
        for t64 in Float64[0.1, 0.35, 0.6, 0.85]
            t32 = Float32(t64)
            for i in 1:S
                v64 = bspline_value(i, k, t64, knots64; T = Float64)
                v32 = bspline_value(i, k, t32, knots32; T = Float32)
                @test Float32(v64) ≈ v32 atol = 1f-5
            end
        end
    end

    # ------------------------------------------------------------------
    # build_knot_vector and uniform_internal_knots basic checks
    @testset "build_knot_vector" begin
        T = Float64
        τ = uniform_internal_knots(3; T)
        @test τ ≈ [0.25, 0.5, 0.75]
        knots = build_knot_vector(4, τ; T)
        @test length(knots) == 2 * 4 + 3            # 11
        @test all(knots[1:4] .== T(0))
        @test all(knots[8:11] .== T(1))
        @test knots[5:7] ≈ τ
    end
end
