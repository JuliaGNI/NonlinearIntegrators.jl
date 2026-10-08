# Compares the native B-spline implementation (src/vi_spline/bspline_utils.jl)
# against BSplineKit.jl as an independent external reference.
#
# BSplineKit is not a main-package dependency (it lives in the scripts sub-environment),
# so this file is NOT included by runtests.jl.  Run it manually from the repo root:
#
#   julia --project=scripts test/vi_spline/compare_bspline_bsplinekit.jl
#
# It exits with code 0 on success and 1 on any failure.

using Pkg
# Activate the scripts environment which carries BSplineKit
Pkg.activate(joinpath(@__DIR__, "..", "..", "scripts"))

using Test
using BSplineKit
using Printf

# Load the native implementation directly (without activating the main package)
include(joinpath(@__DIR__, "..", "..", "src", "vi_spline", "bspline_utils.jl"))

@testset "Native B-spline vs BSplineKit" begin
    for T in (Float64, Float32), order in [2, 3, 4]
        @testset "T=$T order=$order" begin
            n_internal = 4
            τ = uniform_internal_knots(n_internal; T)
            knots_native = build_knot_vector(order, τ; T)
            S = length(knots_native) - order

            # BSplineKit expects breakpoints (distinct boundary + interior values)
            breakpoints = Float64[0.0; Float64.(τ); 1.0]
            B_kit = BSplineBasis(BSplineOrder(order), breakpoints)
            @assert length(B_kit) == S

            tol_val = T === Float32 ? 2f-5 : 1e-12
            tol_der = T === Float32 ? 2f-4 : 1e-10

            test_pts = T[0.05, 0.15, 0.27, 0.41, 0.55, 0.68, 0.82, 0.93]

            for t in test_pts, i in 1:S
                # Value comparison
                val_native = bspline_value(i, order, t, knots_native; T)
                val_kit    = T(B_kit[i](Float64(t)))
                @test val_native ≈ val_kit atol = tol_val

                # Time-derivative comparison
                coefs         = zeros(Float64, S)
                coefs[i]      = 1.0
                sp_kit        = BSplineKit.Spline(B_kit, coefs)
                der_kit       = T((BSplineKit.Derivative(1) * sp_kit)(Float64(t)))
                der_native    = bspline_deriv(i, order, t, knots_native; T)
                @test der_native ≈ der_kit atol = tol_der
            end
        end
    end
end
