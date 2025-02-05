using CompactBases
using QuadratureRules

QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
QGau4_nodes = QuadratureRules.nodes(QGau4)

pushfirst!(QGau4_nodes, 0.0)
push!(QGau4_nodes, 1.0)

SplineB = BSplineDirichlet(3, QGau4_nodes)