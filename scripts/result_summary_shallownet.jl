using JLD2
using CairoMakie
using Statistics
using Printf

# Loads parameter lists, figure style constants, and shared helper functions:
# load_relu_tensor, load_tanh_tensor, save_relu_error_trend, save_tanh_error_trend,
# save_hams_ts, save_relu_best_figures, save_tanh_best_figures,
# print_relu_table, print_tanh_table, inject_md_table,
# and extended variants: load_relu_tensor_ex, load_tanh_tensor_ex,
# save_relu_best_figures_ex, save_tanh_best_figures_ex,
# save_relu_fewest_figures, save_tanh_fewest_figures,
# print_relu_table_ex, print_tanh_table_ex.
include(joinpath(@__DIR__, "result_summary_config.jl"))

resultsdir = joinpath(@__DIR__, "results", "shallownet")
figdir     = joinpath(@__DIR__, "..", "docs", "src", "nvi", "figures")
mdfile     = joinpath(@__DIR__, "..", "docs", "src", "nvi", "shallownet.md")
mkpath(figdir)

# ── Harmonic Oscillator ───────────────────────────────────────────────────────
HO_relu_data, HO_relu_best_err, HO_relu_best_by_key, HO_relu_fewest_by_key, HO_relu_fewest_data =
    load_relu_tensor_ex(resultsdir, "NVI", "HO", "HO")
HO_tanh_data, HO_tanh_best_err, HO_tanh_best_by_key, HO_tanh_fewest_by_key, HO_tanh_fewest_data =
    load_tanh_tensor_ex(resultsdir, "NVI", "HO", "HO")

save_relu_error_trend(figdir, "shallownet_HO_relu_error_trend",
    HO_relu_data, "ShallowNet — Harmonic Oscillator (ReLU)")
save_tanh_error_trend(figdir, "shallownet_HO_tanh_error_trend",
    HO_tanh_data, "ShallowNet — Harmonic Oscillator (tanh)")

save_relu_error_trend(figdir, "shallownet_HO_relu_fewest_error_trend",
    HO_relu_fewest_data, "ShallowNet — Harmonic Oscillator (ReLU, fewest unconverged)")
save_tanh_error_trend(figdir, "shallownet_HO_tanh_fewest_error_trend",
    HO_tanh_fewest_data, "ShallowNet — Harmonic Oscillator (tanh, fewest unconverged)")

HO_relu_fignames        = save_relu_best_figures_ex(figdir, "shallownet_HO", HO_relu_best_by_key)
HO_tanh_fignames        = save_tanh_best_figures_ex(figdir, "shallownet_HO", HO_tanh_best_by_key)
HO_relu_fewest_fignames = save_relu_fewest_figures(figdir, "shallownet_HO", HO_relu_fewest_by_key)
HO_tanh_fewest_fignames = save_tanh_fewest_figures(figdir, "shallownet_HO", HO_tanh_fewest_by_key)

let io = IOBuffer()
    print_relu_table_ex(HO_relu_data, "ShallowNet HO", io;
        figdir_rel="figures", fignames=HO_relu_fignames,
        fewest_fignames=HO_relu_fewest_fignames,
        best_by_key=HO_relu_best_by_key, fewest_by_key=HO_relu_fewest_by_key)
    inject_md_table(mdfile, "HO_RELU_TABLE", String(take!(io)))
end
let io = IOBuffer()
    print_tanh_table_ex(HO_tanh_data, "ShallowNet HO", io;
        figdir_rel="figures", fignames=HO_tanh_fignames,
        fewest_fignames=HO_tanh_fewest_fignames,
        best_by_key=HO_tanh_best_by_key, fewest_by_key=HO_tanh_fewest_by_key)
    inject_md_table(mdfile, "HO_TANH_TABLE", String(take!(io)))
end

# ── Double Pendulum ───────────────────────────────────────────────────────────
if run_dp
    DP_relu_data, DP_relu_best_err, DP_relu_best_by_key, DP_relu_fewest_by_key, DP_relu_fewest_data =
        load_relu_tensor_ex(resultsdir, "NVI", "DP", "DP")
    DP_tanh_data, DP_tanh_best_err, DP_tanh_best_by_key, DP_tanh_fewest_by_key, DP_tanh_fewest_data =
        load_tanh_tensor_ex(resultsdir, "NVI", "DP", "DP")

    save_relu_error_trend(figdir, "shallownet_DP_relu_error_trend",
        DP_relu_data, "ShallowNet — Double Pendulum (ReLU)")
    save_tanh_error_trend(figdir, "shallownet_DP_tanh_error_trend",
        DP_tanh_data, "ShallowNet — Double Pendulum (tanh)")

    save_relu_error_trend(figdir, "shallownet_DP_relu_fewest_error_trend",
        DP_relu_fewest_data, "ShallowNet — Double Pendulum (ReLU, fewest unconverged)")
    save_tanh_error_trend(figdir, "shallownet_DP_tanh_fewest_error_trend",
        DP_tanh_fewest_data, "ShallowNet — Double Pendulum (tanh, fewest unconverged)")

    DP_relu_fignames        = save_relu_best_figures_ex(figdir, "shallownet_DP", DP_relu_best_by_key)
    DP_tanh_fignames        = save_tanh_best_figures_ex(figdir, "shallownet_DP", DP_tanh_best_by_key)
    DP_relu_fewest_fignames = save_relu_fewest_figures(figdir, "shallownet_DP", DP_relu_fewest_by_key)
    DP_tanh_fewest_fignames = save_tanh_fewest_figures(figdir, "shallownet_DP", DP_tanh_fewest_by_key)

    let io = IOBuffer()
        print_relu_table_ex(DP_relu_data, "ShallowNet DP", io;
            figdir_rel="figures", fignames=DP_relu_fignames,
            fewest_fignames=DP_relu_fewest_fignames,
            best_by_key=DP_relu_best_by_key, fewest_by_key=DP_relu_fewest_by_key)
        inject_md_table(mdfile, "DP_RELU_TABLE", String(take!(io)))
    end
    let io = IOBuffer()
        print_tanh_table_ex(DP_tanh_data, "ShallowNet DP", io;
            figdir_rel="figures", fignames=DP_tanh_fignames,
            fewest_fignames=DP_tanh_fewest_fignames,
            best_by_key=DP_tanh_best_by_key, fewest_by_key=DP_tanh_fewest_by_key)
        inject_md_table(mdfile, "DP_TANH_TABLE", String(take!(io)))
    end
end
