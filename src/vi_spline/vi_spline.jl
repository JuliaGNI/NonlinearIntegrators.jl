# B-spline variational integrators, on SimpleSplines.
#
# Fixed knots need no integrator of their own: GeometricIntegrators' `CGVI` with a SimpleSplines
# `BSplineBasis` and a `CompositeQuadrature` is the Galerkin VI on S_k (Theorem 2 of
# benchmark/theory/relu_cgvi_equivalence_en.md), e.g.
#
#     CGVI(BSplineBasis(UniformMesh(M, 0 .. 1), k), CompositeQuadrature(GaussLegendreQuadrature(k + 1), M))
#
# The native Cox–de Boor implementation and the fixed-knot integrator it carried are under
# `obsolete/vi_spline/`.

include("nn_to_spline.jl")
include("free_knot_vi.jl")
