using Pkg

# cd("IntegratorNN/GeometricIntegrators.jl")
# cd("..")
cd("IntegratorNN")

Pkg.activate(".")

using NonlinearIntegrators

using GeometricIntegrators
using NonlinearIntegrators
using QuadratureRules
using CompactBasisFunctions
using GeometricProblems: HarmonicOscillator
using GeometricProblems
using StaticArrays
using BSplineKit


int_step = 5.
int_timespan = 20

HO_iode = GeometricProblems.HarmonicOscillator.lodeproblem(tspan = (0,int_timespan),tstep = int_step)
HO_pref = HarmonicOscillator.exact_solution(HarmonicOscillator.podeproblem(tspan = (0,int_timespan),tstep = int_step))

QGau4 = QuadratureRules.GaussLegendreQuadrature(8)
knots_seq = 0:0.2:1
SplineB = BSplineDirichlet(3, knots_seq)
CGVI_BSpline(SplineB, QGau4)
Spline_results= integrate(HO_iode, CGVI_BSpline(SplineB, QGau4))
Spline_results[1].q