module NonlinearIntegrators

    using GeometricIntegrators
    using NonlinearIntegrators
    using QuadratureRules
    using CompactBasisFunctions
    using Zygote
    using Random
    using Optimisers
    using Lux
    using Statistics
    using Base
    using Reexport

    include("network_integrators/methods.jl")
    export OneLayerMethod, DenseNetMethod, NetworkIntegratorMethod

    include("network_basis/DenseNet_GML.jl")
    include("network_basis/DenseNet_Lux.jl")
    export DenseNet_GML,DenseNet_Lux

    include("network_basis/OneLayerNetwork_GML.jl")
    include("network_basis/OneLayerNetwork_Lux.jl")
    include("network_basis/OneLayerNetwork.jl")
    export OneLayerNetwork_GML,OneLayerNetwork_Lux,OneLayerNetwork

    include("network_integrators/NonLinear_OneLayer_GML.jl")
    include("network_integrators/NonLinear_OneLayer_Lux.jl")
    include("network_integrators/NonLinear_DenseNet_GML.jl")
    export NonLinear_OneLayer_GML,NonLinear_OneLayer_Lux,NonLinear_DenseNet_GML




end