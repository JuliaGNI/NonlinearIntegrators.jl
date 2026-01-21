## Linear and Nonlinear Network Basis

The basis in this package are all subtypes of [`NetworkBasis`](@ref), which is a subtype of [`Basis`](@ref) from [`CompactBasisFunctions.jl`](@ref):

- [`NetworkBasis`](@ref),
- [`DenseNetBasis`](@ref),
- [`OneLayerNetBasis`](@ref).

```@example Basis Types
julia> using CompactBasisFunctions:Basis
julia> NetworkBasis <: Basis #true
julia> DenseNetBasis <: NetworkBasis #true
julia> OneLayerNetBasis <: NetworkBasis #true
```

Basis (as well as integrators) are then implemented with [`GeometricMachineLearning.jl`(GML)](@ref) and [`Lux.jl`](@ref), but only GML versions are frequently used and updated.

 