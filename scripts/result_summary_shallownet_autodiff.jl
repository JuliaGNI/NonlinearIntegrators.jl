using JLD2
using CairoMakie
using Statistics
using Printf

# Loads parameter lists, figure style constants, and shared helper functions:
# load_relu_tensor, load_tanh_tensor, save_relu_error_trend, save_tanh_error_trend,
# save_hams_ts, print_relu_table, print_tanh_table.
include(joinpath(@__DIR__, "result_summary_config.jl"))

resultsdir = joinpath(@__DIR__, "results", "shallownet_autodiff")
figdir     = joinpath(@__DIR__, "..", "docs", "src", "nvi", "figures")
mkpath(figdir)

# ── Harmonic Oscillator ───────────────────────────────────────────────────────
HO_relu_tensor, HO_relu_best_err = load_relu_tensor(resultsdir, "NVI_AD", "HO", "HO")
HO_tanh_tensor, HO_tanh_best_err = load_tanh_tensor(resultsdir, "NVI_AD", "HO", "HO")

save_relu_error_trend(figdir, "shallownet_autodiff_HO_relu_error_trend",
    HO_relu_tensor, "ShallowNetAutodiff — Harmonic Oscillator (ReLU)")
save_tanh_error_trend(figdir, "shallownet_autodiff_HO_tanh_error_trend",
    HO_tanh_tensor, "ShallowNetAutodiff — Harmonic Oscillator (tanh)")

print_relu_table(HO_relu_tensor, "ShallowNetAutodiff HO")
print_tanh_table(HO_tanh_tensor, "ShallowNetAutodiff HO")

# ── Double Pendulum ───────────────────────────────────────────────────────────
DP_relu_tensor, DP_relu_best_err = load_relu_tensor(resultsdir, "NVI_AD", "DP", "DP")
DP_tanh_tensor, DP_tanh_best_err = load_tanh_tensor(resultsdir, "NVI_AD", "DP", "DP")

save_relu_error_trend(figdir, "shallownet_autodiff_DP_relu_error_trend",
    DP_relu_tensor, "ShallowNetAutodiff — Double Pendulum (ReLU)")
save_tanh_error_trend(figdir, "shallownet_autodiff_DP_tanh_error_trend",
    DP_tanh_tensor, "ShallowNetAutodiff — Double Pendulum (tanh)")

print_relu_table(DP_relu_tensor, "ShallowNetAutodiff DP")
print_tanh_table(DP_tanh_tensor, "ShallowNetAutodiff DP")
