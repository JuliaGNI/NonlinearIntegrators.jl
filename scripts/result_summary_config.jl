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
are aggregated over. Returns `(data, best_err, best_by_key)` where:
- `data` is a `Dict{NTuple{3,Int}, Vector{Float64}}` keyed by `(hi, Si, ki)`.
- `best_err` is the global best hams_err series (lowest max error across all keys).
- `best_by_key` is a `Dict{NTuple{3,Int}, Vector{Float64}}` with the best
  hams_err series for each individual `(hi, Si, ki)` key.
"""
function load_relu_tensor(resultsdir, method_prefix, problem_prefix, jld2_key_prefix)
    data        = Dict{NTuple{3,Int}, Vector{Float64}}()
    best_by_key = Dict{NTuple{3,Int}, Vector{Float64}}()
    best_val_by_key = Dict{NTuple{3,Int}, Float64}()
    best_val    = Inf
    best_err    = Float64[]

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
            key = (hi, Si, ki)
            push!(get!(data, key, Float64[]), val)
            if val < get(best_val_by_key, key, Inf)
                best_val_by_key[key] = val
                best_by_key[key]     = d["$(jld2_key_prefix)_hams_err"]
            end
            if val < best_val
                best_val = val
                best_err = d["$(jld2_key_prefix)_hams_err"]
            end
        catch e
            println("Failed to load $fname: $e")
        end
    end
    data, best_err, best_by_key
end

"""
Scan `resultsdir` for all tanh `.jld2` files matching `method_prefix` and
`problem_prefix`. Returns `(data, best_err, best_by_key)` where:
- `data` is a `Dict{NTuple{2,Int}, Vector{Float64}}` keyed by `(hi, Si)`.
- `best_err` is the global best hams_err series.
- `best_by_key` is a `Dict{NTuple{2,Int}, Vector{Float64}}` with the best
  hams_err series for each individual `(hi, Si)` key.
"""
function load_tanh_tensor(resultsdir, method_prefix, problem_prefix, jld2_key_prefix)
    data        = Dict{NTuple{2,Int}, Vector{Float64}}()
    best_by_key = Dict{NTuple{2,Int}, Vector{Float64}}()
    best_val_by_key = Dict{NTuple{2,Int}, Float64}()
    best_val    = Inf
    best_err    = Float64[]

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
            key = (hi, Si)
            push!(get!(data, key, Float64[]), val)
            if val < get(best_val_by_key, key, Inf)
                best_val_by_key[key] = val
                best_by_key[key]     = d["$(jld2_key_prefix)_hams_err"]
            end
            if val < best_val
                best_val = val
                best_err = d["$(jld2_key_prefix)_hams_err"]
            end
        catch e
            println("Failed to load $fname: $e")
        end
    end
    data, best_err, best_by_key
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

# ── Per-entry best-run figures ────────────────────────────────────────────────

"""
Save one Hamiltonian time-series PNG per `(hi, Si, ki)` key in `best_by_key`.
Files are named `{figbase}_relu_h{h}_S{S}_k{k}_best.png` and saved to `figdir`.
Returns a `Dict{NTuple{3,Int}, String}` mapping each key to its figure filename
(relative to `figdir`, suitable for embedding in Markdown).
"""
function save_relu_best_figures(figdir, figbase, best_by_key)
    fignames = Dict{NTuple{3,Int}, String}()
    for ((hi, Si, ki), hams_err) in best_by_key
        isempty(hams_err) && continue
        h = h_list[hi]; S = S_list_sum[Si]; k = k_list_sum[ki]
        fname = "$(figbase)_relu_h$(h)_S$(S)_k$(k)_best"
        save_hams_ts(figdir, fname, hams_err,
            "Hamiltonian Error — h=$(h), S=$(S), k=$(k) (best run)")
        fignames[(hi, Si, ki)] = "$(fname).png"
    end
    fignames
end

"""
Save one Hamiltonian time-series PNG per `(hi, Si)` key in `best_by_key`.
Files are named `{figbase}_tanh_h{h}_S{S}_best.png` and saved to `figdir`.
Returns a `Dict{NTuple{2,Int}, String}` mapping each key to its figure filename.
"""
function save_tanh_best_figures(figdir, figbase, best_by_key)
    fignames = Dict{NTuple{2,Int}, String}()
    for ((hi, Si), hams_err) in best_by_key
        isempty(hams_err) && continue
        h = h_list[hi]; S = S_list_sum[Si]
        fname = "$(figbase)_tanh_h$(h)_S$(S)_best"
        save_hams_ts(figdir, fname, hams_err,
            "Hamiltonian Error — h=$(h), S=$(S) (best run)")
        fignames[(hi, Si)] = "$(fname).png"
    end
    fignames
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
Emit one HTML grid table per h value.  Each cell shows:
  config label (S=x, k=y), min max-error, and the best-run figure (if available).
Columns = S values, rows = k values.

- `figdir_rel`: path to figures dir relative to the Markdown file (e.g. `"figures"`).
- `fignames`: `Dict{NTuple{3,Int}, String}` from `save_relu_best_figures`.
"""
function print_relu_table(relu_data, header, io=stdout;
                          figdir_rel=nothing, fignames=nothing)
    println(io, "\n## $(header) — ReLU\n")
    for (hi, h) in enumerate(h_list)
        println(io, "### h = $(h)\n")
        println(io, "<table>")
        # Header row: blank corner + one column per S
        print(io, "<thead><tr><th></th>")
        for S in S_list_sum
            print(io, "<th>S = $(S)</th>")
        end
        println(io, "</tr></thead>")
        println(io, "<tbody>")
        for (ki, k) in enumerate(k_list_sum)
            print(io, "<tr><th>k = $(k)</th>")
            for (Si, S) in enumerate(S_list_sum)
                vals = get(relu_data, (hi, Si, ki), Float64[])
                print(io, "<td>")
                if isempty(vals)
                    print(io, "—")
                else
                    val_str = @sprintf("%.3e", minimum(vals))
                    if figdir_rel !== nothing && fignames !== nothing
                        fname = get(fignames, (hi, Si, ki), nothing)
                        if fname !== nothing
                            print(io, "<strong>S=$(S), k=$(k), Max Error = $(val_str)<br/><img src=\"$(figdir_rel)/$(fname)\" style=\"width:100%;min-width:180px\"/>")
                        else
                            print(io, "<strong>S=$(S), k=$(k), Max Error = $(val_str)</strong>")
                        end
                    else
                        print(io, "<strong>S=$(S), k=$(k), Max Error = $(val_str)</strong>")
                    end
                end
                print(io, "</td>")
            end
            println(io, "</tr>")
        end
        println(io, "</tbody></table>\n")
    end
end

# ── Table: tanh activation ────────────────────────────────────────────────────

"""
Emit one HTML grid table per h value.  Each cell shows:
  config label (S=x), min max-error, and the best-run figure (if available).
Columns = S values, single body row.

- `figdir_rel`: path to figures dir relative to the Markdown file.
- `fignames`: `Dict{NTuple{2,Int}, String}` from `save_tanh_best_figures`.
"""
function print_tanh_table(tanh_data, header, io=stdout;
                          figdir_rel=nothing, fignames=nothing)
    println(io, "\n## $(header) — tanh\n")
    for (hi, h) in enumerate(h_list)
        println(io, "### h = $(h)\n")
        println(io, "<table>")
        print(io, "<thead><tr>")
        for S in S_list_sum
            print(io, "<th>S = $(S)</th>")
        end
        println(io, "</tr></thead>")
        println(io, "<tbody><tr>")
        for (Si, S) in enumerate(S_list_sum)
            vals = get(tanh_data, (hi, Si), Float64[])
            print(io, "<td>")
            if isempty(vals)
                print(io, "—")
            else
                val_str = @sprintf("%.3e", minimum(vals))
                if figdir_rel !== nothing && fignames !== nothing
                    fname = get(fignames, (hi, Si), nothing)
                    if fname !== nothing
                        print(io, "<strong>S=$(S), Max Error = $(val_str)<br/><img src=\"$(figdir_rel)/$(fname)\" style=\"width:100%;min-width:180px\"/>")
                    else
                        print(io, "<strong>S=$(S), Max Error = $(val_str)</strong>")
                    end
                else
                    print(io, "<strong>S=$(S), Max Error = $(val_str)</strong>")
                end
            end
            print(io, "</td>")
        end
        println(io, "</tr></tbody></table>\n")
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Extended functions (solver-status aware)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Config string helper ──────────────────────────────────────────────────────

"""
Parse a JLD2 filename base into a short config display string.
Extracts S, R, k (relu only), regularization factor, solver name, and dtype.
"""
function parse_config_str(fname_base; is_relu=true)
    m_S      = match(r"S(\d+)",                                        fname_base)
    m_R      = match(r"R(\d+)",                                        fname_base)
    m_k      = is_relu ? match(r"reluk=(\d+)",                         fname_base) : nothing
    m_reg    = match(r"reg=([0-9.e+\-]+)",                             fname_base)
    m_solver = match(r"_(backtracking|static|strongwolfe|dogleg)_",    fname_base)
    m_dtype  = match(r"_(Float\d+)(?:\.jld2)?$",                       fname_base)

    S_str      = m_S      !== nothing ? "S=$(m_S[1])"   : ""
    R_str      = m_R      !== nothing ? "R=$(m_R[1])"   : ""
    k_str      = m_k      !== nothing ? "k=$(m_k[1])"   : ""
    reg_str    = m_reg    !== nothing ? "λ=$(m_reg[1])" : ""
    solver_str = m_solver !== nothing ? m_solver[1]     : ""
    dtype_str  = m_dtype  !== nothing ? m_dtype[1]      : ""

    parts = filter(!isempty, [S_str, R_str, k_str, reg_str, solver_str, dtype_str])
    join(parts, " · ")
end

# ── Extended data loaders ─────────────────────────────────────────────────────

"""
Extended ReLU loader. Returns `(data, best_err, best_by_key, fewest_by_key, fewest_data)`:
- `data` / `best_err`: same as `load_relu_tensor`.
- `best_by_key[key]`: NamedTuple `(hams_err, solver_status, config_str, max_err)`.
- `fewest_by_key[key]`: NamedTuple `(hams_err, solver_status, config_str, pct)` for
  the run with fewest unconverged steps (tie-break: lowest max error).
- `fewest_data[key]`: one-element `Vector{Float64}` with max hams_err of that run,
  suitable for passing to `save_relu_error_trend`.
"""
function load_relu_tensor_ex(resultsdir, method_prefix, problem_prefix, jld2_key_prefix)
    data                 = Dict{NTuple{3,Int}, Vector{Float64}}()
    best_by_key          = Dict{NTuple{3,Int}, Any}()
    fewest_by_key        = Dict{NTuple{3,Int}, Any}()
    fewest_data          = Dict{NTuple{3,Int}, Vector{Float64}}()
    best_val_by_key      = Dict{NTuple{3,Int}, Float64}()
    fewest_pct_by_key    = Dict{NTuple{3,Int}, Float64}()
    fewest_maxerr_by_key = Dict{NTuple{3,Int}, Float64}()
    best_val             = Inf
    best_err             = Float64[]

    pat = Regex("^$(method_prefix)_$(problem_prefix)_h([0-9.e+\\-]+)S(\\d+)R\\d+reluk=(\\d+).*\\.jld2\$")
    for fname in readdir(resultsdir, join=true)
        endswith(fname, ".jld2") || continue
        m = match(pat, basename(fname))
        m === nothing && continue

        h  = parse(Float64, m[1]); S = parse(Int, m[2]); k = parse(Int, m[3])
        hi = findfirst(x -> x ≈ h, h_list)
        Si = findfirst(==(S), S_list_sum)
        ki = findfirst(==(k), k_list_sum)
        (hi === nothing || Si === nothing || ki === nothing) && continue

        try
            d             = load(fname)
            val           = d["$(jld2_key_prefix)_max_hams_err"]
            isfinite(val) || continue
            key           = (hi, Si, ki)
            hams_err      = d["$(jld2_key_prefix)_hams_err"]
            solver_status = get(d, "$(jld2_key_prefix)_solver_status", Bool[])
            pct           = get(d, "$(jld2_key_prefix)_pct_unconverged", NaN)
            config_str    = parse_config_str(basename(fname); is_relu=true)

            push!(get!(data, key, Float64[]), val)

            if val < get(best_val_by_key, key, Inf)
                best_val_by_key[key] = val
                best_by_key[key] = (hams_err=hams_err, solver_status=solver_status,
                                    config_str=config_str, max_err=val)
            end
            if val < best_val; best_val = val; best_err = hams_err; end

            cur_pct = isnan(pct) ? Inf : pct
            old_pct = get(fewest_pct_by_key, key, Inf)
            if cur_pct < old_pct ||
               (cur_pct == old_pct && val < get(fewest_maxerr_by_key, key, Inf))
                fewest_pct_by_key[key]    = cur_pct
                fewest_maxerr_by_key[key] = val
                fewest_by_key[key] = (hams_err=hams_err, solver_status=solver_status,
                                      config_str=config_str, pct=cur_pct)
                fewest_data[key]   = [val]
            end
        catch e
            println("Failed to load $fname: $e")
        end
    end
    data, best_err, best_by_key, fewest_by_key, fewest_data
end

"""
Extended tanh loader. Returns `(data, best_err, best_by_key, fewest_by_key, fewest_data)`.
Same structure as `load_relu_tensor_ex` but keyed by `(hi, Si)`.
"""
function load_tanh_tensor_ex(resultsdir, method_prefix, problem_prefix, jld2_key_prefix)
    data                 = Dict{NTuple{2,Int}, Vector{Float64}}()
    best_by_key          = Dict{NTuple{2,Int}, Any}()
    fewest_by_key        = Dict{NTuple{2,Int}, Any}()
    fewest_data          = Dict{NTuple{2,Int}, Vector{Float64}}()
    best_val_by_key      = Dict{NTuple{2,Int}, Float64}()
    fewest_pct_by_key    = Dict{NTuple{2,Int}, Float64}()
    fewest_maxerr_by_key = Dict{NTuple{2,Int}, Float64}()
    best_val             = Inf
    best_err             = Float64[]

    pat = Regex("^$(method_prefix)_$(problem_prefix)_h([0-9.e+\\-]+)S(\\d+)R\\d+tanh.*\\.jld2\$")
    for fname in readdir(resultsdir, join=true)
        endswith(fname, ".jld2") || continue
        m = match(pat, basename(fname))
        m === nothing && continue

        h  = parse(Float64, m[1]); S = parse(Int, m[2])
        hi = findfirst(x -> x ≈ h, h_list)
        Si = findfirst(==(S), S_list_sum)
        (hi === nothing || Si === nothing) && continue

        try
            d             = load(fname)
            val           = d["$(jld2_key_prefix)_max_hams_err"]
            isfinite(val) || continue
            key           = (hi, Si)
            hams_err      = d["$(jld2_key_prefix)_hams_err"]
            solver_status = get(d, "$(jld2_key_prefix)_solver_status", Bool[])
            pct           = get(d, "$(jld2_key_prefix)_pct_unconverged", NaN)
            config_str    = parse_config_str(basename(fname); is_relu=false)

            push!(get!(data, key, Float64[]), val)

            if val < get(best_val_by_key, key, Inf)
                best_val_by_key[key] = val
                best_by_key[key] = (hams_err=hams_err, solver_status=solver_status,
                                    config_str=config_str, max_err=val)
            end
            if val < best_val; best_val = val; best_err = hams_err; end

            cur_pct = isnan(pct) ? Inf : pct
            old_pct = get(fewest_pct_by_key, key, Inf)
            if cur_pct < old_pct ||
               (cur_pct == old_pct && val < get(fewest_maxerr_by_key, key, Inf))
                fewest_pct_by_key[key]    = cur_pct
                fewest_maxerr_by_key[key] = val
                fewest_by_key[key] = (hams_err=hams_err, solver_status=solver_status,
                                      config_str=config_str, pct=cur_pct)
                fewest_data[key]   = [val]
            end
        catch e
            println("Failed to load $fname: $e")
        end
    end
    data, best_err, best_by_key, fewest_by_key, fewest_data
end

# ── Extended Hamiltonian time-series figure ───────────────────────────────────

"""
Save a Hamiltonian error time-series plot with optional red × markers at steps
where the solver did not converge (`solver_status[i] == false`).
"""
function save_hams_ts_ex(figdir, figname, hams_err, title; solver_status=nothing)
    isempty(hams_err) && return
    fig = Figure(size=sum_size_1d)
    Label(fig[0, 1], title, fontsize=sum_title_size, tellwidth=false)
    ax = Axis(fig[1, 1],
        xlabel="Step index", ylabel="Relative Hamiltonian Error",
        yscale=log10,
        xlabelsize=sum_label_size, ylabelsize=sum_label_size,
        xticklabelsize=sum_tick_size, yticklabelsize=sum_tick_size)
    lines!(ax, ifelse.(hams_err .> 0, hams_err, NaN))
    if solver_status !== nothing && !isempty(solver_status)
        fail_idx   = findall(!, solver_status)
        valid_fail = filter(i -> i <= length(hams_err) && hams_err[i] > 0, fail_idx)
        if !isempty(valid_fail)
            scatter!(ax, valid_fail, hams_err[valid_fail];
                color=:red, marker=:xcross, markersize=12, label="not converged")
            axislegend(ax, position=:rt, labelsize=16)
        end
    end
    for ext in ("pdf", "png")
        save(joinpath(figdir, "$(figname).$(ext)"), fig)
    end
end

# ── Extended per-entry best-run figures ──────────────────────────────────────

"""
Save best-error Hamiltonian time-series figures using the extended `best_by_key`
structure from `load_relu_tensor_ex`. Passes `solver_status` to show red × markers.
Returns `Dict{NTuple{3,Int}, String}` of PNG filenames.
"""
function save_relu_best_figures_ex(figdir, figbase, best_by_key)
    fignames = Dict{NTuple{3,Int}, String}()
    for (key, entry) in best_by_key
        (hi, Si, ki) = key
        isempty(entry.hams_err) && continue
        h = h_list[hi]; S = S_list_sum[Si]; k = k_list_sum[ki]
        fname = "$(figbase)_relu_h$(h)_S$(S)_k$(k)_best"
        save_hams_ts_ex(figdir, fname, entry.hams_err,
            "Hamiltonian Error — h=$(h), S=$(S), k=$(k) (best run)";
            solver_status=entry.solver_status)
        fignames[key] = "$(fname).png"
    end
    fignames
end

"""
Save fewest-unconverged Hamiltonian time-series figures using the extended
`fewest_by_key` structure from `load_relu_tensor_ex`.
Returns `Dict{NTuple{3,Int}, String}` of PNG filenames.
"""
function save_relu_fewest_figures(figdir, figbase, fewest_by_key)
    fignames = Dict{NTuple{3,Int}, String}()
    for (key, entry) in fewest_by_key
        (hi, Si, ki) = key
        isempty(entry.hams_err) && continue
        h = h_list[hi]; S = S_list_sum[Si]; k = k_list_sum[ki]
        fname = "$(figbase)_relu_h$(h)_S$(S)_k$(k)_fewest"
        save_hams_ts_ex(figdir, fname, entry.hams_err,
            "Hamiltonian Error — h=$(h), S=$(S), k=$(k) (fewest unconverged)";
            solver_status=entry.solver_status)
        fignames[key] = "$(fname).png"
    end
    fignames
end

"""
Save best-error Hamiltonian time-series figures using the extended `best_by_key`
structure from `load_tanh_tensor_ex`.
Returns `Dict{NTuple{2,Int}, String}` of PNG filenames.
"""
function save_tanh_best_figures_ex(figdir, figbase, best_by_key)
    fignames = Dict{NTuple{2,Int}, String}()
    for (key, entry) in best_by_key
        (hi, Si) = key
        isempty(entry.hams_err) && continue
        h = h_list[hi]; S = S_list_sum[Si]
        fname = "$(figbase)_tanh_h$(h)_S$(S)_best"
        save_hams_ts_ex(figdir, fname, entry.hams_err,
            "Hamiltonian Error — h=$(h), S=$(S) (best run)";
            solver_status=entry.solver_status)
        fignames[key] = "$(fname).png"
    end
    fignames
end

"""
Save fewest-unconverged Hamiltonian time-series figures using the extended
`fewest_by_key` structure from `load_tanh_tensor_ex`.
Returns `Dict{NTuple{2,Int}, String}` of PNG filenames.
"""
function save_tanh_fewest_figures(figdir, figbase, fewest_by_key)
    fignames = Dict{NTuple{2,Int}, String}()
    for (key, entry) in fewest_by_key
        (hi, Si) = key
        isempty(entry.hams_err) && continue
        h = h_list[hi]; S = S_list_sum[Si]
        fname = "$(figbase)_tanh_h$(h)_S$(S)_fewest"
        save_hams_ts_ex(figdir, fname, entry.hams_err,
            "Hamiltonian Error — h=$(h), S=$(S) (fewest unconverged)";
            solver_status=entry.solver_status)
        fignames[key] = "$(fname).png"
    end
    fignames
end

# ── Extended tables ───────────────────────────────────────────────────────────

"""
Extended ReLU table with two-column cell layout when `fewest_fignames` is provided.
Each cell shows best-error figure (left) and fewest-unconverged figure (right),
with config info below each. Falls back to the original single-figure layout when
`fewest_fignames` is `nothing`.

Extra kwargs:
- `fewest_fignames`: `Dict{NTuple{3,Int}, String}` from `save_relu_fewest_figures`.
- `best_by_key` / `fewest_by_key`: extended loader outputs for config strings.
"""
function print_relu_table_ex(relu_data, header, io=stdout;
                             figdir_rel=nothing, fignames=nothing,
                             fewest_fignames=nothing,
                             best_by_key=nothing, fewest_by_key=nothing)
    println(io, "\n## $(header) — ReLU\n")
    for (hi, h) in enumerate(h_list)
        println(io, "### h = $(h)\n")
        println(io, "<table>")
        print(io, "<thead><tr><th></th>")
        for S in S_list_sum; print(io, "<th>S = $(S)</th>"); end
        println(io, "</tr></thead>")
        println(io, "<tbody>")
        for (ki, k) in enumerate(k_list_sum)
            print(io, "<tr><th>k = $(k)</th>")
            for (Si, S) in enumerate(S_list_sum)
                key  = (hi, Si, ki)
                vals = get(relu_data, key, Float64[])
                print(io, "<td>")
                if isempty(vals)
                    print(io, "—")
                elseif figdir_rel !== nothing && fignames !== nothing
                    val_str    = @sprintf("%.3e", minimum(vals))
                    best_fig   = get(fignames,         key, nothing)
                    fewest_fig = fewest_fignames !== nothing ? get(fewest_fignames, key, nothing) : nothing
                    best_cfg   = best_by_key    !== nothing ? get(best_by_key,    key, nothing) : nothing
                    fewest_cfg = fewest_by_key  !== nothing ? get(fewest_by_key,  key, nothing) : nothing

                    if fewest_fig !== nothing
                        pct_str = (fewest_cfg !== nothing && isfinite(fewest_cfg.pct)) ?
                            @sprintf("%.1f%%", fewest_cfg.pct * 100) : "n/a"
                        print(io, "<table style=\"width:100%\"><tr>")
                        print(io, "<td style=\"text-align:center;vertical-align:top;width:50%\">")
                        print(io, "<strong>Best error: $(val_str)</strong><br/>")
                        if best_fig !== nothing
                            print(io, "<img src=\"$(figdir_rel)/$(best_fig)\" style=\"width:100%;min-width:130px\"/><br/>")
                        end
                        if best_cfg !== nothing; print(io, "<small>$(best_cfg.config_str)</small>"); end
                        print(io, "</td>")
                        print(io, "<td style=\"text-align:center;vertical-align:top;width:50%\">")
                        print(io, "<strong>Fewest unconverged: $(pct_str)</strong><br/>")
                        print(io, "<img src=\"$(figdir_rel)/$(fewest_fig)\" style=\"width:100%;min-width:130px\"/><br/>")
                        if fewest_cfg !== nothing; print(io, "<small>$(fewest_cfg.config_str)</small>"); end
                        print(io, "</td>")
                        print(io, "</tr></table>")
                    elseif best_fig !== nothing
                        print(io, "<strong>S=$(S), k=$(k), Max Error = $(val_str)</strong><br/>")
                        print(io, "<img src=\"$(figdir_rel)/$(best_fig)\" style=\"width:100%;min-width:180px\"/>")
                        if best_cfg !== nothing; print(io, "<br/><small>$(best_cfg.config_str)</small>"); end
                    else
                        print(io, "<strong>S=$(S), k=$(k), Max Error = $(val_str)</strong>")
                    end
                else
                    print(io, "<strong>S=$(S), k=$(k), Max Error = $(@sprintf("%.3e", minimum(vals)))</strong>")
                end
                print(io, "</td>")
            end
            println(io, "</tr>")
        end
        println(io, "</tbody></table>\n")
    end
end

"""
Extended tanh table with two-column cell layout when `fewest_fignames` is provided.
See `print_relu_table_ex` for kwargs description.
"""
function print_tanh_table_ex(tanh_data, header, io=stdout;
                             figdir_rel=nothing, fignames=nothing,
                             fewest_fignames=nothing,
                             best_by_key=nothing, fewest_by_key=nothing)
    println(io, "\n## $(header) — tanh\n")
    for (hi, h) in enumerate(h_list)
        println(io, "### h = $(h)\n")
        println(io, "<table>")
        print(io, "<thead><tr>")
        for S in S_list_sum; print(io, "<th>S = $(S)</th>"); end
        println(io, "</tr></thead>")
        println(io, "<tbody><tr>")
        for (Si, S) in enumerate(S_list_sum)
            key  = (hi, Si)
            vals = get(tanh_data, key, Float64[])
            print(io, "<td>")
            if isempty(vals)
                print(io, "—")
            elseif figdir_rel !== nothing && fignames !== nothing
                val_str    = @sprintf("%.3e", minimum(vals))
                best_fig   = get(fignames,         key, nothing)
                fewest_fig = fewest_fignames !== nothing ? get(fewest_fignames, key, nothing) : nothing
                best_cfg   = best_by_key    !== nothing ? get(best_by_key,    key, nothing) : nothing
                fewest_cfg = fewest_by_key  !== nothing ? get(fewest_by_key,  key, nothing) : nothing

                if fewest_fig !== nothing
                    pct_str = (fewest_cfg !== nothing && isfinite(fewest_cfg.pct)) ?
                        @sprintf("%.1f%%", fewest_cfg.pct * 100) : "n/a"
                    print(io, "<table style=\"width:100%\"><tr>")
                    print(io, "<td style=\"text-align:center;vertical-align:top;width:50%\">")
                    print(io, "<strong>Best error: $(val_str)</strong><br/>")
                    if best_fig !== nothing
                        print(io, "<img src=\"$(figdir_rel)/$(best_fig)\" style=\"width:100%;min-width:130px\"/><br/>")
                    end
                    if best_cfg !== nothing; print(io, "<small>$(best_cfg.config_str)</small>"); end
                    print(io, "</td>")
                    print(io, "<td style=\"text-align:center;vertical-align:top;width:50%\">")
                    print(io, "<strong>Fewest unconverged: $(pct_str)</strong><br/>")
                    print(io, "<img src=\"$(figdir_rel)/$(fewest_fig)\" style=\"width:100%;min-width:130px\"/><br/>")
                    if fewest_cfg !== nothing; print(io, "<small>$(fewest_cfg.config_str)</small>"); end
                    print(io, "</td>")
                    print(io, "</tr></table>")
                elseif best_fig !== nothing
                    print(io, "<strong>S=$(S), Max Error = $(val_str)</strong><br/>")
                    print(io, "<img src=\"$(figdir_rel)/$(best_fig)\" style=\"width:100%;min-width:180px\"/>")
                    if best_cfg !== nothing; print(io, "<br/><small>$(best_cfg.config_str)</small>"); end
                else
                    print(io, "<strong>S=$(S), Max Error = $(val_str)</strong>")
                end
            else
                print(io, "<strong>S=$(S), Max Error = $(@sprintf("%.3e", minimum(vals)))</strong>")
            end
            print(io, "</td>")
        end
        println(io, "</tr></tbody></table>\n")
    end
end
