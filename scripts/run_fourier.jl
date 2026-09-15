# The bases the ansatz search in `../figures` found, applied to a whole trajectory.
#
#   julia --project=scripts scripts/run_fourier.jl [perturbed-pendulum|henon-heiles]... \
#       [--runs-dir dir]
#
#   perturbed pendulum, t ∈ [0, 1000]   odd harmonics of one free ω
#   Hénon–Heiles,       t ∈ [0,  300]   combinations of two basic frequencies
#
# These are not integrators: nothing is stepped, and there is no `h`. One global fit over the
# whole window represents the trajectory, and the question the archive answers is how many
# numbers that takes compared with a fixed basis on the same interval.
#
# The intervals differ on purpose. The pendulum's orbit is exactly periodic, so its ansatz
# holds over any window and 1000 shows that. Hénon–Heiles is quasi-periodic on a two-torus:
# on [0,60] the two frequencies are closer than one Rayleigh width and the ansatz cannot be
# identified at all, and on [0,1000] the construction breaks down for reasons not understood
# — every initialisation stalls near 1e-2 although the orbit is regular. On [0,300] it works,
# so that is the window recorded here.
#
# The reference is the same `Gauss(8)` solve on the `h/40` grid that every other figure in
# this directory is measured against, and the fit is made to it. Figures are rendered from
# the archive by `figures.jl`, with the extension's `plot_solution` — the same routine the
# VISE and NVI figures use.

include(joinpath(@__DIR__, "experiments.jl"))
include(joinpath(@__DIR__, "basis_fits.jl"))

# The perturbation coefficient of the pendulum, needed to turn the fitted velocity into the
# momentum: q̇ = ∂H/∂p = p - qA.
const PPD_PERTURBATION = let params = PPD.default_parameters()
    0.3 * params.ϵ * sin(2 * params.ϕ) + 0.7 * params.ϵ * sin(3 * params.ϕ)
end

# `max|coefficient| / scale` above this is cancellation, not a representation.
const MAX_COEFFICIENT = 10.0

struct FourierRun
    name::String
    label::String                       # names the curve in the legend
    problem_label::String
    problem::Function                   # (timespan, timestep) -> problem
    hamiltonian::Function
    dimension::Int
    final_time::Float64
    sample_step::Float64                # spacing of the marked samples
    momentum::Function                  # (q, q̇) -> p
    fit::Function                       # (times, Y) -> vector of (dof, freqs, ω)
end

# ---- the two runs ------------------------------------------------------------

pendulum_fits(times, Y) = map(1:7) do m
    freqs, ω = odd_harmonic_fit(times, Y, m)
    (dof = 2 + 2m, freqs = freqs, ω = (ω,))
end

function lattice_fits(times, Y)
    ω₀ = basic_frequencies(times, Y)
    @printf("  periodogram: (%.6f, %.6f)\n", ω₀...)
    map([(3, 4), (4, 5), (5, 6), (6, 8)]) do (N, M)
        freqs, ω = lattice_fit(times, Y, N, M, ω₀)
        (dof = 2 + 2 * (1 + 2 * length(freqs)), freqs = freqs, ω = ω)
    end
end

const FOURIER_RUNS = (
    FourierRun("perturbed-pendulum", "Odd harmonics of one free ω", "Perturbed pendulum",
        (timespan, timestep) -> PPD.hodeproblem(; timespan = timespan, timestep = timestep),
        PPD.hamiltonian, 1, 1000.0, 10.0,
        (q, q̇) -> q̇ .+ PPD_PERTURBATION .* q,
        pendulum_fits),
    FourierRun("henon-heiles", "Two-frequency lattice", "Hénon–Heiles",
        (timespan, timestep) -> HH.hodeproblem([0.1, 0.1], [0.1, 0.1];
            timespan = timespan, timestep = timestep),
        HH.hamiltonian, 2, 300.0, 5.0,
        (q, q̇) -> q̇,
        lattice_fits)
)

function fourier_run(name)
    index = findfirst(r -> r.name == name, FOURIER_RUNS)
    index === nothing && throw(ArgumentError(
        "no Fourier run named `$(name)`; this study has " *
        join((r.name for r in FOURIER_RUNS), ", ")))
    FOURIER_RUNS[index]
end

# ---- one run -----------------------------------------------------------------

function run_fourier(run::FourierRun)
    banner("$(run.problem_label) — $(run.label), T = $(number_label(run.final_time))")

    prob = run.problem((0.0, run.final_time), run.sample_step)
    params = prob.parameters

    # The reference every other figure here is measured against, and the data the ansatz is
    # fitted to.
    ref_sol = reference_solution(run.problem, run.final_time, run.sample_step)
    ref_t, ref_q, ref_p = solution_data(ref_sol, run.dimension)
    report("reference", "Gauss(8) on h/$(REFERENCE_SUBSTEPS), $(length(ref_t)) points")

    times = collect(ref_t)
    Y = reduce(hcat, ref_q)
    scale = maximum(abs, Y)

    best = nothing
    for candidate in run.fit(times, Y)
        C = fit_coefficients(trig_columns(times, candidate.freqs), Y)
        app, _ = evaluate_fit(candidate.freqs, C, times)
        err = maximum(abs, Y .- app) / scale
        coeff = maximum(abs, C) / scale
        @printf("  DOF = %4d   error = %.3e   max|c|/scale = %6.2f   ω = %s\n",
            candidate.dof, err, coeff,
            join((@sprintf("%.7f", x) for x in candidate.ω), ", "))
        flush(stdout)
        coeff > MAX_COEFFICIENT && continue
        if best === nothing || err < best.err
            best = (dof = candidate.dof, freqs = candidate.freqs, ω = candidate.ω,
                C = C, err = err, coeff = coeff)
        end
    end

    best === nothing && error("every fit for $(run.name) was rejected as cancellation")
    report("degrees of freedom", best.dof)
    report_error("relative maximum error", best.err)

    # ---- the series the figure needs -----------------------------------------
    #
    # `t` are the marked samples and `continuous_t` the curve between them, exactly as for an
    # integrator: there the samples are the steps, here they are where the representation is
    # sampled. Both come from the same expansion.
    t = collect(0.0:run.sample_step:run.final_time)
    q_t, q̇_t = evaluate_fit(best.freqs, best.C, t)
    p_t = run.momentum(q_t, q̇_t)

    q_fine, _ = evaluate_fit(best.freqs, best.C, times)

    H = [run.hamiltonian(t[n], q_t[n, :], p_t[n, :], params) for n in eachindex(t)]
    herr = relative_invariant_error(H)
    report_error("max |ΔH/H₀|", maximum(herr))

    data = Dict{String, Any}(
        # A global fit, so there is no `"timestep"` key at all — and nothing is stepped, so a `Δt`
        # in the title would name a quantity the method does not have. The renderer omits it when
        # the key is absent, which is what lets these share one figure function with the stepped
        # runs instead of needing a near-copy of it.
        "kind" => "solution",
        "problem" => run.name,
        "label" => "$(run.label), $(best.dof) DOF",
        "problem_label" => run.problem_label,
        "final_time" => run.final_time,
        "dimension" => run.dimension,
        "sample_step" => run.sample_step,
        "degrees_of_freedom" => best.dof,
        "frequencies" => best.freqs,
        "basic_frequencies" => collect(best.ω),
        "coefficients" => best.C,
        "reference_error" => best.err,
        "coefficient_ratio" => best.coeff,
        "t" => t,
        "q" => [q_t[:, d] for d in 1:run.dimension],
        "p" => [p_t[:, d] for d in 1:run.dimension],
        "continuous_t" => times,
        "continuous_q" => [q_fine[:, d] for d in 1:run.dimension],
        "hamiltonian_error" => herr,
        "max_hamiltonian_error" => maximum(herr),
        "reference_t" => ref_t,
        "reference_q" => ref_q,
        "reference_p" => ref_p,
        "reference_substeps" => REFERENCE_SUBSTEPS,
        "comparisons" => Dict{String, Any}()
    )

    # Eighty oscillations in one panel is a block of ink, so the pendulum also gets a figure over
    # the first fifth of the interval. Recorded here rather than in `figures.jl` because which
    # window reads well is a property of this run — and under `"windows"`, the same key the network
    # runs use, rather than a `"figure_window"` of its own: one run drawn over several intervals is
    # one mechanism, and it had two names for no reason.
    data["windows"] = run.final_time > 400 ? [run.final_time / 5] : Float64[]

    stem = study_stem(run.name, "fourier", "T$(number_label(run.final_time))")
    report_path("archive", store_run!(stem, data))
    return data
end

function main(args)
    names, _ = parse_arguments(args)
    runs = isempty(names) ? FOURIER_RUNS : Tuple(fourier_run(n) for n in names)
    for run in runs
        run_fourier(run)
    end
    banner("done")
end

main(ARGS)
