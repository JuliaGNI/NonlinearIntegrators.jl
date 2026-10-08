using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    @safetestset "JET" include("quality/jet.jl")
    @safetestset "Basis smoke" include("bases_smoke.jl")
    @safetestset "Method smoke" include("methods_smoke.jl")
    @safetestset "Parameter flattening" include("nvi/parameter_flattening_unit.jl")
    @safetestset "OGA kernels" include("oga/oga_kernels.jl")
    @safetestset "VISE" include("vise/vise_unit.jl")
    @safetestset "VISplineFree" include("vi_spline/free_knot_vi_unit.jl")
    @safetestset "Scripts archives" include("scripts_archives_tests.jl")
    @safetestset "Inference and allocations" include("quality/inference_and_allocations.jl")
    @safetestset "Plots" include("plots_tests.jl")
    @safetestset "ShallowNet accuracy" include("integration/shallownet_accuracy.jl")
end
if "slow" in GROUPS
    @safetestset "Network integrators" include("nvi/network_integrators_unit.jl")
    @safetestset "Dispatch variants" include("nvi/dispatch_variants_unit.jl")
end
