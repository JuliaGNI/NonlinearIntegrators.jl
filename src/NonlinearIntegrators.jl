module NonlinearIntegrators

    using GeometricIntegrators
    import GeometricIntegrators.Integrators: IODEIntegratorCache
    import GeometricIntegrators.Integrators: CacheDict, Cache, default_solver, default_iguess,CacheType
    import GeometricIntegrators.Integrators: create_internal_stage_vector,parameters
    import GeometricIntegrators.Integrators: cache,nlsolution,solver,method,iguess,problem,current,update!

    using NonlinearIntegrators
    using QuadratureRules
    using CompactBasisFunctions
    using Zygote
    using Random
    using Optimisers
    using Lux
    using Statistics
    using Base
    using StaticArrays
    using SimpleSolvers
    import SimpleSolvers: solve!
    import GeometricMachineLearning
    using SymbolicNeuralNetworks
    using AbstractNeuralNetworks
    using LinearAlgebra
    using BSplineKit

    include("methods.jl")
    export OneLayerMethod, DenseNetMethod, NetworkIntegratorMethod
    export IntegratorExtrapolation
    export InitialParametersMethod, TrainingMethod, OGA1d

    include("network_integrators/utilities.jl")

    include("network_basis/NetworkBasis.jl")
    export NetworkBasis, DenseNetBasis, OneLayerNetBasis

    include("network_basis/DenseNet_GML.jl")
    include("network_basis/DenseNet_Lux.jl")
    export DenseNet_GML,DenseNet_Lux

    include("network_basis/OneLayerNetwork_GML.jl")
    include("network_basis/OneLayerNetwork_Lux.jl")
    include("network_basis/OneLayerNetwork.jl")
    export OneLayerNetwork_GML,OneLayerNetwork_Lux,OneLayerNetwork

    include("network_basis/OneLayerVectorValueNet_Lux.jl")
    include("network_basis/OneLayerVectorValueNet_GML.jl")
    export OneLayerVectorValueNet_Lux,OneLayerVectorValueNet_GML

    include("network_integrators/NonLinear_OneLayer_GML.jl")
    include("network_integrators/NonLinear_OneLayer_Lux.jl")

    include("network_integrators/NonLinear_DenseNet_GML.jl")
    include("network_integrators/NonLinear_DenseNet_Lux.jl")
    include("network_integrators/Linear_DenseNet_GML.jl")
    export NonLinear_OneLayer_GML,NonLinear_OneLayer_Lux,
            NonLinear_DenseNet_GML,NonLinear_DenseNet_Lux,
            Linear_DenseNet_GML

    include("network_integrators/NonLinear_OneLayer_VectorValue_Lux.jl")
    include("network_integrators/NonLinear_OneLayer_VectorValue_GML.jl")
    export NonLinear_OneLayer_VectorValue_Lux,NonLinear_OneLayer_VectorValue_GML

    # BSpline
    include("BSpline/BSplineBases.jl")
    include("BSpline/CGVI_SplineBases.jl")
    export BSplineDirichlet, CGVI_BSpline

end