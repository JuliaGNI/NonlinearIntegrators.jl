# Shared configuration and helpers for all result_summary_*.jl scripts.
# Include this file after loading packages:
#   include(joinpath(@__DIR__, "result_summary_config.jl"))
#
# Requires JLD2, CairoMakie, Statistics, Printf to be loaded.

# ── Key parameter axes (used for plot/table indices) ─────────────────────────
const h_list     = [0.05, 0.1, 0.2, 0.5, 1.0]
const S_list_sum = [4, 6, 8]
const k_list_sum = [2, 3, 4]

run_dp = false

# Figure style (kept in sync with run_config.jl)
const sum_label_size = 22
const sum_tick_size  = 18
const sum_title_size = 22
const sum_size_1d    = (800, 900)
const sum_size_trend = (1100, 1200)

# ── Data loaders ─────────────────────────────────────────────────────────────

"""
Scan `resultsdir` for all ReLU `.jld2` files matching `method_prefix` and
`problem_prefix`. All secondary parameters (R, λ, fabs, xsuc, solver, dtype)
are aggregated over. Returns `(data, best_err)` where `data` is a
`Dict{NTuple{3,Int}, Vector{Float64}}` keyed by `(hi, Si, ki)` indices into
`(h_list, S_list_sum, k_list_sum)`.
"""
function load_relu_tensor(resultsdir, method_prefix, problem_prefix, jld2_key_prefix)
    data     = Dict{NTuple{3,Int}, Vector{Float64}}()
    best_val = Inf
    best_err = Float64[]

    pat = Regex("^$(method_prefix)_$(problem_prefix)_h([0-9.e+\\-]+)S(\\d+)R\\d+reluk=(\\d+).*\\.jld2\$")
    for fname in readdir(resultsdir, join=true)
        endswith(fname, ".jld2") || continue
        m = match(pat, basename(fname))
        m === nothing && continue

        h  = parse(Float64, m[1])
        S  = parse(Int,     m[2])
        k  = parse(Int,     m[3])
        hi = findfirst(x -> x ≈ h, h_list)
        Si = findfirst(==(S), S_list_sum)
        ki = findfirst(==(k), k_list_sum)
        (hi === nothing || Si === nothing || ki === nothing) && continue

        try
            d   = load(fname)
            val = d["$(jld2_key_prefix)_max_hams_err"]
            isfinite(val) || continue
            push!(get!(data, (hi, Si, ki), Float64[]), val)
            if val < best_val
                best_val = val
                best_err = d["$(jld2_key_prefix)_hams_err"]
            end
        catch e
            println("Failed to load $fname: $e")
        end
    end
    data, best_err
end

"""
Scan `resultsdir` for all tanh `.jld2` files matching `method_prefix` and
`problem_prefix`. Returns `(data, best_err)` where `data` is a
`Dict{NTuple{2,Int}, Vector{Float64}}` keyed by `(hi, Si)` indices into
`(h_list, S_list_sum)`.
"""
function load_tanh_tensor(resultsdir, method_prefix, problem_prefix, jld2_key_prefix)
    data     = Dict{NTuple{2,Int}, Vector{Float64}}()
    best_val = Inf
    best_err = Float64[]

    pat = Regex("^$(method_prefix)_$(problem_prefix)_h([0-9.e+\\-]+)S(\\d+)R\\d+tanh.*\\.jld2\$")
    for fname in readdir(resultsdir, join=true)
        endswith(fname, ".jld2") || continue
        m = match(pat, basename(fname))
        m === nothing && continue

        h  = parse(Float64, m[1])
        S  = parse(Int,     m[2])
        hi = findfirst(x -> x ≈ h, h_list)
        Si = findfirst(==(S), S_list_sum)
        (hi === nothing || Si === nothing) && continue

        try
            d   = load(fname)
            val = d["$(jld2_key_prefix)_max_hams_err"]
            isfinite(val) || continue
            push!(get!(data, (hi, Si), Float64[]), val)
            if val < best_val
                best_val = val
                best_err = d["$(jld2_key_prefix)_hams_err"]
            end
        catch e
            println("Failed to load $fname: $e")
        end
    end
    data, best_err
end

# ── Aggregation helpers ───────────────────────────────────────────────────────

function _valid_stats(vals::Vector{Float64})
    isempty(vals) ? (NaN, NaN, NaN) : (mean(vals), maximum(vals), minimum(vals))
end

relu_stats(data, hi, Si, ki) = _valid_stats(get(data, (hi, Si, ki), Float64[]))
tanh_stats(data, hi, Si)     = _valid_stats(get(data, (hi, Si),     Float64[]))

# ── Figure: ReLU error trend ──────────────────────────────────────────────────

"""
Generate and save an error-trend figure for the ReLU sweep (mean ± [min, max] bands vs h).
One line per (S, k) pair present in `relu_data`.
"""
function save_relu_error_trend(figdir, figname, relu_data, title)
    fig = Figure(size=sum_size_trend)
    Label(fig[0, 1], title, fontsize=sum_title_size, tellwidth=false)
    ax_mean = Axis(fig[1, 1],
        xlabel="Time Step h", ylabel="Mean Maximum Hamiltonian Error",
        xscale=log10, yscale=log10,
        xlabelsize=sum_label_size, ylabelsize=sum_label_size,
        xticklabelsize=sum_tick_size, yticklabelsize=sum_tick_size)
    ax_min = Axis(fig[2, 1],
        xlabel="Time Step h", ylabel="Minimum Maximum Hamiltonian Error",
        xscale=log10, yscale=log10,
        xlabelsize=sum_label_size, ylabelsize=sum_label_size,
        xticklabelsize=sum_tick_size, yticklabelsize=sum_tick_size)

    palette  = cgrad(:tab10, length(S_list_sum) * length(k_list_sum), categorical=true)
    idx      = 1
    any_mean = false
    any_min  = false
    for (Si, S) in enumerate(S_list_sum), (ki, k) in enumerate(k_list_sum)
        stats = [relu_stats(relu_data, hi, Si, ki) for hi in eachindex(h_list)]
        means = [s[1] for s in stats]
        maxs  = [s[2] for s in stats]
        mins  = [s[3] for s in stats]
        valid = isfinite.(means)
        any(valid) || (idx += 1; continue)
        c = palette[idx]
        scatterlines!(ax_mean, h_list[valid], means[valid], label="S$(S) k$(k)", color=c, markersize=6, linewidth=2)
        errorbars!(ax_mean, h_list[valid], means[valid], means[valid] .- mins[valid], maxs[valid] .- means[valid], color=c, linewidth=2, whiskerwidth=10)
        scatterlines!(ax_min,  h_list[valid], mins[valid], label="S$(S) k$(k)", color=c, markersize=6, linewidth=2)
        any_mean = true
        any_min  = true
        idx += 1
    end
    any_mean && axislegend(ax_mean, position=:rb, labelsize=18)
    any_min  && axislegend(ax_min,  position=:rb, labelsize=18)

    for ext in ("pdf", "png")
        save(joinpath(figdir, "$(figname).$(ext)"), fig)
    end
end

# ── Figure: tanh error trend ──────────────────────────────────────────────────

"""
Generate and save an error-trend figure for the tanh sweep (mean ± [min, max] bands vs h).
One line per S value present in `tanh_data`.
"""
function save_tanh_error_trend(figdir, figname, tanh_data, title)
    fig = Figure(size=sum_size_trend)
    Label(fig[0, 1], title, fontsize=sum_title_size, tellwidth=false)
    ax_mean = Axis(fig[1, 1],
        xlabel="Time Step h", ylabel="Mean Maximum Hamiltonian Error",
        xscale=log10, yscale=log10,
        xlabelsize=sum_label_size, ylabelsize=sum_label_size,
        xticklabelsize=sum_tick_size, yticklabelsize=sum_tick_size)
    ax_min = Axis(fig[2, 1],
        xlabel="Time Step h", ylabel="Minimum Maximum Hamiltonian Error",
        xscale=log10, yscale=log10,
        xlabelsize=sum_label_size, ylabelsize=sum_label_size,
        xticklabelsize=sum_tick_size, yticklabelsize=sum_tick_size)

    palette  = cgrad(:tab10, length(S_list_sum), categorical=true)
    any_mean = false
    any_min  = false
    for (Si, S) in enumerate(S_list_sum)
        stats = [tanh_stats(tanh_data, hi, Si) for hi in eachindex(h_list)]
        means = [s[1] for s in stats]
        maxs  = [s[2] for s in stats]
        mins  = [s[3] for s in stats]
        valid = isfinite.(means)
        any(valid) || continue
        c = palette[Si]
        scatterlines!(ax_mean, h_list[valid], means[valid], label="S$(S)", color=c, markersize=6, linewidth=2)
        errorbars!(ax_mean, h_list[valid], means[valid], means[valid] .- mins[valid], maxs[valid] .- means[valid], color=c, linewidth=2, whiskerwidth=10)
        scatterlines!(ax_min,  h_list[valid], mins[valid], label="S$(S)", color=c, markersize=6, linewidth=2)
        any_mean = true
        any_min  = true
    end
    any_mean && axislegend(ax_mean, position=:rb, labelsize=18)
    any_min  && axislegend(ax_min,  position=:rb, labelsize=18)

    for ext in ("pdf", "png")
        save(joinpath(figdir, "$(figname).$(ext)"), fig)
    end
end

# ── Figure: Hamiltonian error time series ─────────────────────────────────────

"""
Save a time-series plot of the Hamiltonian error for the best (lowest max error) run.
"""
function save_hams_ts(figdir, figname, hams_err, title)
    isempty(hams_err) && return
    fig = Figure(size=sum_size_1d)
    Label(fig[0, 1], title, fontsize=sum_title_size, tellwidth=false)
    ax = Axis(fig[1, 1],
        xlabel="Step index", ylabel="Relative Hamiltonian Error",
        yscale=log10,
        xlabelsize=sum_label_size, ylabelsize=sum_label_size,
        xticklabelsize=sum_tick_size, yticklabelsize=sum_tick_size)
    lines!(ax, ifelse.(hams_err .> 0, hams_err, NaN))
    for ext in ("pdf", "png")
        save(joinpath(figdir, "$(figname).$(ext)"), fig)
    end
end

# ── Markdown injection ───────────────────────────────────────────────────────

"""
Replace content between `<!-- MARKER_START -->` and `<!-- MARKER_END -->` in
`mdfile` with `content`. Creates the markers if absent (appends to end).
"""
function inject_md_table(mdfile, marker, content)
    raw     = read(mdfile, String)
    s_tag   = "<!-- $(marker)_START -->"
    e_tag   = "<!-- $(marker)_END -->"
    block   = "$(s_tag)\n$(content)\n$(e_tag)"
    if occursin(s_tag, raw) && occursin(e_tag, raw)
        new = replace(raw, Regex("$(s_tag).*?$(e_tag)", "s") => block)
    else
        new = raw * "\n" * block * "\n"
    end
    write(mdfile, new)
end

# ── Table: ReLU activation ────────────────────────────────────────────────────

"""
Print a Markdown table of min max-error per (S, k) for each h value.
Aggregates over all secondary parameters (R, λ, fabs, xsuc, solver, dtype).
Pass `io` to write to a file instead of stdout.
"""
function print_relu_table(relu_data, header, io=stdout)
    println(io, "\n=== $(header) — ReLU ===")
    for (hi, h) in enumerate(h_list)
        println(io, "\n### h=$(h)")
        println(io, "| S | k | min max Hamiltonian error |")
        println(io, "|---|---|--------------------------|")
        for (Si, S) in enumerate(S_list_sum), (ki, k) in enumerate(k_list_sum)
            vals = get(relu_data, (hi, Si, ki), Float64[])
            val  = isempty(vals) ? "—" : @sprintf("%.3e", minimum(vals))
            println(io, "| $(S) | $(k) | $(val) |")
        end
    end
end

# ── Table: tanh activation ────────────────────────────────────────────────────

"""
Print a Markdown table of min max-error per S for each h value.
Aggregates over all secondary parameters (R, λ, fabs, xsuc, solver, dtype).
Pass `io` to write to a file instead of stdout.
"""
function print_tanh_table(tanh_data, header, io=stdout)
    println(io, "\n=== $(header) — tanh ===")
    for (hi, h) in enumerate(h_list)
        println(io, "\n### h=$(h)")
        println(io, "| S | min max Hamiltonian error |")
        println(io, "|---|--------------------------|")
        for (Si, S) in enumerate(S_list_sum)
            vals = get(tanh_data, (hi, Si), Float64[])
            val  = isempty(vals) ? "—" : @sprintf("%.3e", minimum(vals))
            println(io, "| $(S) | $(val) |")
        end
    end
end
