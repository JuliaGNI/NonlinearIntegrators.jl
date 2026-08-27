using JLD2
using CairoMakie
using Statistics
using Printf

# Loads parameter lists, figure style constants, and shared helper functions:
# load_relu_tensor, load_tanh_tensor, save_relu_error_trend, save_tanh_error_trend,
# save_hams_ts, save_relu_best_figures, save_tanh_best_figures,
# print_relu_table, print_tanh_table, inject_md_table.
include(joinpath(@__DIR__, "result_summary_config.jl"))

resultsdir = joinpath(@__DIR__, "results", "densenet")
figdir     = joinpath(@__DIR__, "..", "docs", "src", "nvi", "figures")
mdfile     = joinpath(@__DIR__, "..", "docs", "src", "nvi", "densenet.md")
mkpath(figdir)

# ── Harmonic Oscillator ───────────────────────────────────────────────────────
HO_relu_tensor, HO_relu_best_err, HO_relu_best_by_key =
    load_relu_tensor(resultsdir, "NVI_Dense", "HO", "HO")
HO_tanh_tensor, HO_tanh_best_err, HO_tanh_best_by_key =
    load_tanh_tensor(resultsdir, "NVI_Dense", "HO", "HO")

save_relu_error_trend(figdir, "densenet_HO_relu_error_trend",
    HO_relu_tensor, "DenseNet — Harmonic Oscillator (ReLU)")
save_tanh_error_trend(figdir, "densenet_HO_tanh_error_trend",
    HO_tanh_tensor, "DenseNet — Harmonic Oscillator (tanh)")

HO_relu_fignames = save_relu_best_figures(figdir, "densenet_HO", HO_relu_best_by_key)
HO_tanh_fignames = save_tanh_best_figures(figdir, "densenet_HO", HO_tanh_best_by_key)

let io = IOBuffer()
    print_relu_table(HO_relu_tensor, "DenseNet HO", io;
        figdir_rel="figures", fignames=HO_relu_fignames)
    inject_md_table(mdfile, "HO_RELU_TABLE", String(take!(io)))
end
let io = IOBuffer()
    print_tanh_table(HO_tanh_tensor, "DenseNet HO", io;
        figdir_rel="figures", fignames=HO_tanh_fignames)
    inject_md_table(mdfile, "HO_TANH_TABLE", String(take!(io)))
end

# ── Double Pendulum ───────────────────────────────────────────────────────────
DP_relu_tensor, DP_relu_best_err, DP_relu_best_by_key =
    load_relu_tensor(resultsdir, "NVI_Dense", "DP", "DP")
DP_tanh_tensor, DP_tanh_best_err, DP_tanh_best_by_key =
    load_tanh_tensor(resultsdir, "NVI_Dense", "DP", "DP")

save_relu_error_trend(figdir, "densenet_DP_relu_error_trend",
    DP_relu_tensor, "DenseNet — Double Pendulum (ReLU)")
save_tanh_error_trend(figdir, "densenet_DP_tanh_error_trend",
    DP_tanh_tensor, "DenseNet — Double Pendulum (tanh)")

DP_relu_fignames = save_relu_best_figures(figdir, "densenet_DP", DP_relu_best_by_key)
DP_tanh_fignames = save_tanh_best_figures(figdir, "densenet_DP", DP_tanh_best_by_key)

let io = IOBuffer()
    print_relu_table(DP_relu_tensor, "DenseNet DP", io;
        figdir_rel="figures", fignames=DP_relu_fignames)
    inject_md_table(mdfile, "DP_RELU_TABLE", String(take!(io)))
end
let io = IOBuffer()
    print_tanh_table(DP_tanh_tensor, "DenseNet DP", io;
        figdir_rel="figures", fignames=DP_tanh_fignames)
    inject_md_table(mdfile, "DP_TANH_TABLE", String(take!(io)))
end
