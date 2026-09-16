# Shared configuration and helper functions for all run_*.jl scripts.
# Include this file after loading packages in each run script:
#   include(joinpath(@__DIR__, "run_config.jl"))
#
# Requires CairoMakie and JLD2 to be loaded by the calling script.

# ── Solver / integrator options ───────────────────────────────────────────────
max_iterations = 10000
dict_amount    = 40000   # OGA dictionary size

# ── Figure style ─────────────────────────────────────────────────────────────
fig_label_size = 22   # axis label font size
fig_tick_size  = 18   # tick label font size
fig_title_size = 22   # figure super-title font size

# 1-D problem figure size (HO: q, p, Hamiltonian error stacked vertically)
fig_size_1d = (800, 900)

# 2-D problem figure size (DP / HenonHeiles: 2-column layout)
fig_size_2d = (1200, 900)

# ── Helpers: figure ──────────────────────────────────────────────────────────

"""
Save a 3-panel PDF for a 1-D problem (q, p, Hamiltonian error over time).
Pass `solver_status` (Vector{Bool}, one entry per step) to mark unconverged
steps with red × on the Hamiltonian error panel.
"""
function plot_1d!(outdir, fname, ts, sol_q, sol_p, hams_err, title;
                  solver_status=nothing)
    fig = Figure(size=fig_size_1d)
    Label(fig[0, 1], title, fontsize=fig_title_size, tellwidth=false)
    ax1 = Axis(fig[1, 1], xlabel="Time", ylabel="q(t)",
               xlabelsize=fig_label_size, ylabelsize=fig_label_size,
               xticklabelsize=fig_tick_size, yticklabelsize=fig_tick_size)
    lines!(ax1, ts, sol_q)
    ax2 = Axis(fig[2, 1], xlabel="Time", ylabel="p(t)",
               xlabelsize=fig_label_size, ylabelsize=fig_label_size,
               xticklabelsize=fig_tick_size, yticklabelsize=fig_tick_size)
    lines!(ax2, ts, sol_p)
    ax3 = Axis(fig[3, 1], xlabel="Time", ylabel="Relative Hamiltonian Error",
               xlabelsize=fig_label_size, ylabelsize=fig_label_size,
               xticklabelsize=fig_tick_size, yticklabelsize=fig_tick_size)
    lines!(ax3, ts, hams_err)
    if solver_status !== nothing && !isempty(solver_status)
        fail_idx   = findall(!, solver_status)
        valid_fail = filter(i -> i + 1 <= length(ts) && hams_err[i + 1] > 0, fail_idx)
        if !isempty(valid_fail)
            scatter!(ax3, ts[valid_fail .+ 1], hams_err[valid_fail .+ 1];
                color=:red, marker=:xcross, markersize=12, label="not converged")
            axislegend(ax3, position=:lt, labelsize=fig_tick_size)
        end
    end
    save(joinpath(outdir, "$(fname).pdf"), fig)
end

"""
Save a 5-panel PDF for a 2-D problem (q₁, q₂, p₁, p₂, Hamiltonian error).
Pass `solver_status` (Vector{Bool}, one entry per step) to mark unconverged
steps with red × on the Hamiltonian error panel.
"""
function plot_2d!(outdir, fname, ts, q1, q2, p1, p2, hams_err, title;
                  solver_status=nothing)
    fig = Figure(size=fig_size_2d)
    Label(fig[0, 1:2], title, fontsize=fig_title_size, tellwidth=false)
    ax1 = Axis(fig[1, 1], xlabel="Time", ylabel="q₁",
               xlabelsize=fig_label_size, ylabelsize=fig_label_size,
               xticklabelsize=fig_tick_size, yticklabelsize=fig_tick_size)
    lines!(ax1, ts, q1)
    ax2 = Axis(fig[1, 2], xlabel="Time", ylabel="q₂",
               xlabelsize=fig_label_size, ylabelsize=fig_label_size,
               xticklabelsize=fig_tick_size, yticklabelsize=fig_tick_size)
    lines!(ax2, ts, q2)
    ax3 = Axis(fig[2, 1], xlabel="Time", ylabel="p₁",
               xlabelsize=fig_label_size, ylabelsize=fig_label_size,
               xticklabelsize=fig_tick_size, yticklabelsize=fig_tick_size)
    lines!(ax3, ts, p1)
    ax4 = Axis(fig[2, 2], xlabel="Time", ylabel="p₂",
               xlabelsize=fig_label_size, ylabelsize=fig_label_size,
               xticklabelsize=fig_tick_size, yticklabelsize=fig_tick_size)
    lines!(ax4, ts, p2)
    ax5 = Axis(fig[3, 1:2], xlabel="Time", ylabel="Relative Hamiltonian Error",
               xlabelsize=fig_label_size, ylabelsize=fig_label_size,
               xticklabelsize=fig_tick_size, yticklabelsize=fig_tick_size)
    lines!(ax5, ts, hams_err)
    if solver_status !== nothing && !isempty(solver_status)
        fail_idx   = findall(!, solver_status)
        valid_fail = filter(i -> i + 1 <= length(ts) && hams_err[i + 1] > 0, fail_idx)
        if !isempty(valid_fail)
            scatter!(ax5, ts[valid_fail .+ 1], hams_err[valid_fail .+ 1];
                color=:red, marker=:xcross, markersize=12, label="not converged")
            axislegend(ax5, position=:lt, labelsize=fig_tick_size)
        end
    end
    save(joinpath(outdir, "$(fname).pdf"), fig)
end

# ── Helpers: JLD2 saving ──────────────────────────────────────────────────────

"""
Save JLD2 for a 1-D problem result. `prefix` controls field names (e.g. "HO").
"""
function save_1d_jld2(outdir, fname, sol_q, sol_p, internal_sol, qerror, hams_err,
                       solver_status; prefix="HO")
    pct = isempty(solver_status) ? NaN : count(!, solver_status) / length(solver_status)
    record = Dict(
        "$(prefix)_sol_q"           => sol_q,
        "$(prefix)_sol_p"           => sol_p,
        "$(prefix)_internal_sol"    => internal_sol,
        "$(prefix)_qerror"          => qerror,
        "$(prefix)_hams_err"        => hams_err,
        "$(prefix)_max_hams_err"    => maximum(hams_err),
        "$(prefix)_solver_status"   => solver_status,
        "$(prefix)_pct_unconverged" => pct,
    )
    save(joinpath(outdir, "$(fname).jld2"), record)
end

"""
Save JLD2 for a 2-D problem result. `prefix` controls field names (e.g. "DP").
"""
function save_2d_jld2(outdir, fname, sol_q1, sol_q2, sol_p1, sol_p2, internal_sol, qerror, hams_err,
                       solver_status; prefix="DP")
    pct = isempty(solver_status) ? NaN : count(!, solver_status) / length(solver_status)
    record = Dict(
        "$(prefix)_sol_q1"          => sol_q1,
        "$(prefix)_sol_q2"          => sol_q2,
        "$(prefix)_sol_p1"          => sol_p1,
        "$(prefix)_sol_p2"          => sol_p2,
        "$(prefix)_internal_sol"    => internal_sol,
        "$(prefix)_qerror"          => qerror,
        "$(prefix)_hams_err"        => hams_err,
        "$(prefix)_max_hams_err"    => maximum(hams_err),
        "$(prefix)_solver_status"   => solver_status,
        "$(prefix)_pct_unconverged" => pct,
    )
    save(joinpath(outdir, "$(fname).jld2"), record)
end
