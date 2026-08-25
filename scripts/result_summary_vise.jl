using JLD2
using CairoMakie
using Statistics
using Printf

# Loads parameter lists, figure style constants, and save_hams_ts helper.
include(joinpath(@__DIR__, "result_summary_config.jl"))

resultsdir = joinpath(@__DIR__, "results", "vise")
figdir     = joinpath(@__DIR__, "..", "docs", "src", "nvi", "figures")
mkpath(figdir)

# Parameter grids for VISE (h from the shared list; R from R_list_sum)
# Results are indexed over h × R only (no activation, reg, fabs, xsuc axes).

# ── Shared: load error tensor and best run ────────────────────────────────────

function load_vise_tensor(resultsdir, problem_prefix, jld2_key_prefix)
    nh = length(h_list); nR = length(R_list_sum); nd = length(dtype_list)
    tensor = fill(NaN, nh, nR, nd)
    best_val = Inf
    best_err = Float64[]

    for (hi, h) in enumerate(h_list), (Ri, R) in enumerate(R_list_sum),
        (di, dtype) in enumerate(dtype_list)

        fname = joinpath(resultsdir, "VISE_$(problem_prefix)_h$(h)R$(R)_$(dtype).jld2")
        isfile(fname) || continue
        try
            d = load(fname)
            val = d["$(jld2_key_prefix)_max_hams_err"]
            tensor[hi, Ri, di] = val
            if val < best_val
                best_val = val
                best_err = d["$(jld2_key_prefix)_hams_err"]
            end
        catch e
            println("Failed to load $fname: $e")
        end
    end
    tensor, best_err
end

function save_vise_error_trend(figdir, figname, tensor, di, dtype, title)
    fig = Figure(size=sum_size_trend)
    Label(fig[0, 1], "$(title) ($(dtype))", fontsize=sum_title_size, tellwidth=false)
    ax = Axis(fig[1, 1],
        xlabel="Time Step h", ylabel="Maximum Hamiltonian Error",
        xscale=log10, yscale=log10,
        xlabelsize=sum_label_size, ylabelsize=sum_label_size,
        xticklabelsize=sum_tick_size, yticklabelsize=sum_tick_size)

    palette = cgrad(:tab10, length(R_list_sum), categorical=true)
    for (Ri, R) in enumerate(R_list_sum)
        vals = tensor[:, Ri, di]
        valid_mask = isfinite.(vals)
        any(valid_mask) || continue
        scatterlines!(ax, h_list[valid_mask], vals[valid_mask],
            label="R=$(R)", color=palette[Ri], markersize=6, linewidth=2)
    end
    axislegend(ax, position=:rb, labelsize=18)

    for ext in ("pdf", "png")
        save(joinpath(figdir, "$(figname)_$(dtype).$(ext)"), fig)
    end
end

function print_vise_table(tensor, di, dtype, header)
    println("\n=== $(header) — $(dtype) ===")
    println("| h | R | Max Hamiltonian Error |")
    println("|---|---|-----------------------|")
    for (hi, h) in enumerate(h_list), (Ri, R) in enumerate(R_list_sum)
        val = tensor[hi, Ri, di]
        str = isfinite(val) ? @sprintf("%.3e", val) : "—"
        println("| $(h) | $(R) | $(str) |")
    end
end

# ── Harmonic Oscillator ───────────────────────────────────────────────────────
HO_tensor, HO_best_err = load_vise_tensor(resultsdir, "HO", "HO")
save_hams_ts(figdir, "vise_HO_hamiltonian_error_ts", HO_best_err, "VISE — HO Best Run Hamiltonian Error")
for (di, dtype) in enumerate(dtype_list)
    save_vise_error_trend(figdir, "vise_HO_error_trend", HO_tensor, di, dtype, "VISE — Harmonic Oscillator")
    print_vise_table(HO_tensor, di, dtype, "VISE Harmonic Oscillator")
end

# ── Perturbed Pendulum ────────────────────────────────────────────────────────
PP_tensor, PP_best_err = load_vise_tensor(resultsdir, "PP", "PP")
save_hams_ts(figdir, "vise_PP_hamiltonian_error_ts", PP_best_err, "VISE — PP Best Run Hamiltonian Error")
for (di, dtype) in enumerate(dtype_list)
    save_vise_error_trend(figdir, "vise_PP_error_trend", PP_tensor, di, dtype, "VISE — Perturbed Pendulum")
    print_vise_table(PP_tensor, di, dtype, "VISE Perturbed Pendulum")
end

# ── Hénon-Heiles Potential ────────────────────────────────────────────────────
HH_tensor, HH_best_err = load_vise_tensor(resultsdir, "HH", "HH")
save_hams_ts(figdir, "vise_HH_hamiltonian_error_ts", HH_best_err, "VISE — HH Best Run Hamiltonian Error")
for (di, dtype) in enumerate(dtype_list)
    save_vise_error_trend(figdir, "vise_HH_error_trend", HH_tensor, di, dtype, "VISE — Hénon-Heiles Potential")
    print_vise_table(HH_tensor, di, dtype, "VISE Hénon-Heiles Potential")
end
