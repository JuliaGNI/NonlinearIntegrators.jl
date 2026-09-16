using JLD2
using CairoMakie
using Statistics
using Printf

include(joinpath(@__DIR__, "result_summary_config.jl"))

resultsdir = joinpath(@__DIR__, "results", "shallownet_autodiff")
figdir     = joinpath(@__DIR__, "..", "docs", "src", "nvi", "figures")
mdfile     = joinpath(@__DIR__, "..", "docs", "src", "nvi", "shallownet_autodiff.md")
mkpath(figdir)

# ── Harmonic Oscillator ───────────────────────────────────────────────────────
HO_relu_data, HO_relu_best_err, HO_relu_best_by_key, HO_relu_fewest_by_key, HO_relu_fewest_data =
    load_relu_tensor_ex(resultsdir, "NVI_AD", "HO", "HO")
HO_tanh_data, HO_tanh_best_err, HO_tanh_best_by_key, HO_tanh_fewest_by_key, HO_tanh_fewest_data =
    load_tanh_tensor_ex(resultsdir, "NVI_AD", "HO", "HO")

save_relu_error_trend(figdir, "shallownet_autodiff_HO_relu_error_trend",
    HO_relu_data, "ShallowNetAutodiff — Harmonic Oscillator (ReLU)")
save_tanh_error_trend(figdir, "shallownet_autodiff_HO_tanh_error_trend",
    HO_tanh_data, "ShallowNetAutodiff — Harmonic Oscillator (tanh)")

save_relu_error_trend(figdir, "shallownet_autodiff_HO_relu_fewest_error_trend",
    HO_relu_fewest_data, "ShallowNetAutodiff — Harmonic Oscillator (ReLU, fewest unconverged)")
save_tanh_error_trend(figdir, "shallownet_autodiff_HO_tanh_fewest_error_trend",
    HO_tanh_fewest_data, "ShallowNetAutodiff — Harmonic Oscillator (tanh, fewest unconverged)")

HO_relu_fignames        = save_relu_best_figures_ex(figdir, "shallownet_autodiff_HO", HO_relu_best_by_key)
HO_tanh_fignames        = save_tanh_best_figures_ex(figdir, "shallownet_autodiff_HO", HO_tanh_best_by_key)
HO_relu_fewest_fignames = save_relu_fewest_figures(figdir, "shallownet_autodiff_HO", HO_relu_fewest_by_key)
HO_tanh_fewest_fignames = save_tanh_fewest_figures(figdir, "shallownet_autodiff_HO", HO_tanh_fewest_by_key)

let io = IOBuffer()
    print_relu_table_ex(HO_relu_data, "ShallowNetAutodiff HO", io;
        figdir_rel="figures", fignames=HO_relu_fignames,
        fewest_fignames=HO_relu_fewest_fignames,
        best_by_key=HO_relu_best_by_key, fewest_by_key=HO_relu_fewest_by_key)
    inject_md_table(mdfile, "HO_RELU_TABLE", String(take!(io)))
end
let io = IOBuffer()
    print_tanh_table_ex(HO_tanh_data, "ShallowNetAutodiff HO", io;
        figdir_rel="figures", fignames=HO_tanh_fignames,
        fewest_fignames=HO_tanh_fewest_fignames,
        best_by_key=HO_tanh_best_by_key, fewest_by_key=HO_tanh_fewest_by_key)
    inject_md_table(mdfile, "HO_TANH_TABLE", String(take!(io)))
end

# ── Double Pendulum ───────────────────────────────────────────────────────────
DP_relu_data, DP_relu_best_err, DP_relu_best_by_key, DP_relu_fewest_by_key, DP_relu_fewest_data =
    load_relu_tensor_ex(resultsdir, "NVI_AD", "DP", "DP")
DP_tanh_data, DP_tanh_best_err, DP_tanh_best_by_key, DP_tanh_fewest_by_key, DP_tanh_fewest_data =
    load_tanh_tensor_ex(resultsdir, "NVI_AD", "DP", "DP")

save_relu_error_trend(figdir, "shallownet_autodiff_DP_relu_error_trend",
    DP_relu_data, "ShallowNetAutodiff — Double Pendulum (ReLU)")
save_tanh_error_trend(figdir, "shallownet_autodiff_DP_tanh_error_trend",
    DP_tanh_data, "ShallowNetAutodiff — Double Pendulum (tanh)")

save_relu_error_trend(figdir, "shallownet_autodiff_DP_relu_fewest_error_trend",
    DP_relu_fewest_data, "ShallowNetAutodiff — Double Pendulum (ReLU, fewest unconverged)")
save_tanh_error_trend(figdir, "shallownet_autodiff_DP_tanh_fewest_error_trend",
    DP_tanh_fewest_data, "ShallowNetAutodiff — Double Pendulum (tanh, fewest unconverged)")

DP_relu_fignames        = save_relu_best_figures_ex(figdir, "shallownet_autodiff_DP", DP_relu_best_by_key)
DP_tanh_fignames        = save_tanh_best_figures_ex(figdir, "shallownet_autodiff_DP", DP_tanh_best_by_key)
DP_relu_fewest_fignames = save_relu_fewest_figures(figdir, "shallownet_autodiff_DP", DP_relu_fewest_by_key)
DP_tanh_fewest_fignames = save_tanh_fewest_figures(figdir, "shallownet_autodiff_DP", DP_tanh_fewest_by_key)

let io = IOBuffer()
    print_relu_table_ex(DP_relu_data, "ShallowNetAutodiff DP", io;
        figdir_rel="figures", fignames=DP_relu_fignames,
        fewest_fignames=DP_relu_fewest_fignames,
        best_by_key=DP_relu_best_by_key, fewest_by_key=DP_relu_fewest_by_key)
    inject_md_table(mdfile, "DP_RELU_TABLE", String(take!(io)))
end
let io = IOBuffer()
    print_tanh_table_ex(DP_tanh_data, "ShallowNetAutodiff DP", io;
        figdir_rel="figures", fignames=DP_tanh_fignames,
        fewest_fignames=DP_tanh_fewest_fignames,
        best_by_key=DP_tanh_best_by_key, fewest_by_key=DP_tanh_fewest_by_key)
    inject_md_table(mdfile, "DP_TANH_TABLE", String(take!(io)))
end
