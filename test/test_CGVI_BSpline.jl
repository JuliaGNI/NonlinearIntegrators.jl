using Pkg

# cd("IntegratorNN/GeometricIntegrators.jl")
# cd("..")
cd("IntegratorNN")

Pkg.activate(".")

# using NonlinearIntegrators

# using GeometricIntegrators
# using NonlinearIntegrators
# using QuadratureRules
# using CompactBasisFunctions
# using GeometricProblems: HarmonicOscillator
# using GeometricProblems
# using StaticArrays
# using BSplineKit

# int_step = 4.
# int_timespan = 40.

# HO_iode = GeometricProblems.HarmonicOscillator.lodeproblem(tspan = (0,int_timespan),tstep = int_step)
# HO_pref = HarmonicOscillator.exact_solution(HarmonicOscillator.podeproblem(tspan = (0,int_timespan),tstep = int_step))

# QGau = QuadratureRules.GaussLegendreQuadrature(8)
# # knots_seq = 0:0.2:1
# # SplineB = BSplineDirichlet(4, knots_seq)

# # Spline_results= integrate(HO_iode, CGVI_BSpline(SplineB, QGau))
# # Spline_results[1].q

# # relative_maximum_error(Spline_results[1].q, HO_pref.q)



# ### Nonlinear Case
# NL_SplineB = Nonlinear_BSpline_Basis(4, QGau.nodes)
# Spline_results= integrate(HO_iode, Nonlinear_BSpline_Integrator(NL_SplineB, QGau))

# Nonlinear_BSpline_Integrator(NL_SplineB, QGau)

using BSplineKit
knots = [0,0.1,0.2,0.5,0.7,1.0]
B = BSplineBasis(BSplineOrder(2), knots)
S = BSplineKit.Spline(B,rand(length(B)))
S(1.0)