# Does the choice of orthogonal-greedy seed change what the neural variational integrator
# converges to?
#
#   julia --project=scripts scripts/run_oga_seeds.jl [--final-time T] [--steps h,h,…] \
#       [--runs-dir dir]
#
# The seed is `initial_guess_method`, and it is not a detail. The discrete Euler–Lagrange equations
# of a network ansatz are non-convex, and Newton is started from *this* point at every single time
# step — so a seed decides which solution each step lands in, and whether it lands at all.
# `ShallowNet` defaults to `OGA1d()`, and every historical script in this directory took that
# default without ever comparing it against the five alternatives the package ships.
#
# For each seed, at each time step: the maximum relative Hamiltonian error over the run, the error
# against a `Gauss(8)` reference, and the wall clock. Writes one archive; `figures.jl` draws the
# error against the time step, all seeds on one pair of axes.
#
# The comparison is run on the deck's own ReLU³ configuration, which is what makes the answer apply
# to the figures rather than to a configuration chosen for the study.

include(joinpath(@__DIR__, "experiments.jl"))

using Logging

const CONFIG = OGA_STUDY_CONFIGURATION

function build_method(seed_constructor, R)
    ShallowNet(
        ShallowNetBasis{Float64}(CONFIG.σ, CONFIG.S),
        QuadratureRules.GaussLegendreQuadrature(R);
        show_status = false,
        bias_interval = [-pi, pi],
        dict_amount = NVI_DICT_AMOUNT,
        initial_guess_method = seed_constructor())
end

"""
    run_seed(name, seed_constructor, steps, final_time) -> (errors, reference_errors, seconds)

One seed over the whole step ladder. A step that fails contributes `NaN` rather than aborting the
study: whether a seed *fails* is part of what is being measured, and losing the other five to the
first failure would hide exactly that.
"""
function run_seed(name, seed_constructor, steps, final_time)
    println("  ", name, "  (", oga_label(seed_constructor()), ")")

    problem_fn = (timespan, timestep) -> HO.lodeproblem(;
        timespan = timespan, timestep = timestep)

    errors = Float64[]
    reference_errors = Float64[]
    seconds = Float64[]

    for h in steps
        prob = problem_fn((0.0, final_time), h)
        t_start = time()
        try
            sol, _ = Logging.with_logger(Logging.NullLogger()) do
                integrate(prob, build_method(seed_constructor, CONFIG.R); NVI_SOLVER_OPTIONS...)
            end
            elapsed = time() - t_start
            ref = reference_solution(problem_fn, final_time, h)
            err = maximum(relative_invariant_error(sol, HO.hamiltonian, prob.parameters))
            ref_err = coarse_grid_error(sol, ref, REFERENCE_SUBSTEPS)

            push!(errors, err)
            push!(reference_errors, ref_err)
            push!(seconds, elapsed)
            @printf("    h = %-6g  max |ΔH/H₀| = %-11.3e  vs reference = %-11.3e  (%.1f s, %d steps)\n",
                h, err, ref_err, elapsed, round(Int, final_time / h))
        catch exception
            exception isa InterruptException && rethrow()
            push!(errors, NaN)
            push!(reference_errors, NaN)
            push!(seconds, time() - t_start)
            @printf("    h = %-6g  failed: %s\n", h, typeof(exception))
        end
    end

    return errors, reference_errors, seconds
end

function main(args)
    names, options = parse_arguments(args, ("--steps", "--final-time"))
    isempty(names) || throw(ArgumentError(
        "this study takes no positional arguments, got $(join(names, ", "))"))
    steps = option_steps(options, OGA_STUDY_STEPS)
    final_time = option_final_time(options, OGA_STUDY_FINAL_TIME)

    label = network_label(CONFIG.S, CONFIG.R, CONFIG.name)
    banner("OGA seed study — $(label), T = $(final_time), h ∈ $(collect(steps))")

    labels = String[]
    errors = Vector{Float64}[]
    reference_errors = Vector{Float64}[]
    timings = Vector{Float64}[]

    for (name, constructor) in OGA_VARIANTS
        e, r, s = run_seed(name, constructor, steps, final_time)
        push!(labels, name)
        push!(errors, e)
        push!(reference_errors, r)
        push!(timings, s)
    end

    # The summary that answers the question, rather than leaving it in the log.
    banner("Summary — max |ΔH/H₀| by seed and time step")
    @printf("  %-22s %s\n", "seed", join([@sprintf("%-11s", "h = $(h)") for h in steps]))
    for (l, e) in zip(labels, errors)
        @printf("  %-22s %s\n", l,
            join([isfinite(x) ? @sprintf("%-11.3e", x) : @sprintf("%-11s", "failed")
                  for x in e]))
    end

    finite = [filter(isfinite, e) for e in errors]
    spread = if all(isempty, finite)
        NaN
    else
        allvals = reduce(vcat, filter(!isempty, finite))
        maximum(allvals) / minimum(allvals)
    end
    report("failures", "$(count(!isfinite, reduce(vcat, errors))) of $(length(reduce(vcat, errors)))")
    report_error("spread between best and worst seed (ratio)", spread)

    data = Dict{String, Any}(
        "kind" => "convergence",
        # No `h^p` guides. They are anchored to the largest error plotted, and two of these seeds
        # diverge to `1e23`, which puts the guides twenty decades above the data and makes them
        # decoration. The question this figure answers is *which seeds work at all*, not at what
        # order — so the study says so, rather than the renderer having to know.
        "reference_orders" => Int[],
        "configuration" => label,
        "title" =>
            "OGA seed variants — $(label), harmonic oscillator, " *
            "T = $(number_label(final_time))",
        "problem" => CONFIG.problem,
        "S" => CONFIG.S,
        "R" => CONFIG.R,
        "activation" => CONFIG.name,
        "final_time" => final_time,
        "timesteps" => collect(steps),
        "labels" => labels,
        "errors" => errors,
        "reference_errors" => reference_errors,
        "seconds" => timings
    )

    stem = study_stem(CONFIG.problem, "oga-seeds", CONFIG.name)
    report_path("archive", store_run!(stem, data))

    banner("done")
end

main(ARGS)
