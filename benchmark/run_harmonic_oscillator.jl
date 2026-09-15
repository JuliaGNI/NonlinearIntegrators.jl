# Benchmark the shallow-net integrator on the harmonic oscillator.
#
#   julia --project=benchmark benchmark/run_harmonic_oscillator.jl [quick|full]
#
# Mode defaults to "quick" (also settable via SHALLOWNET_BENCH_PRESET). Writes
# results/harmonic_oscillator_<mode>.csv plus a markdown report and plots.

include(joinpath(@__DIR__, "shallownet_benchmark_common.jl"))
using GeometricProblems.HarmonicOscillator

const NAME = "harmonic_oscillator"

function build_prob(::Type{T}, timespan, timestep) where {T}
    HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)];
        timespan = timespan, timestep = timestep,
        parameters = HarmonicOscillator.default_parameters(T))
end

ham(t, q, p, params) = HarmonicOscillator.hamiltonian(t, q, p, params)

let mode = pick_mode()
    # Network width is chosen per problem, because the accuracy the ansatz can reach — and
    # therefore whether the solve has a target it can meet at all — depends on it. Measured
    # at Float64/tanh/DogLeg over 10 steps of dt = 0.1, `ref_err` against the sweep's own
    # Gauss(8) reference: S = 4 reaches 2.8e-06 in 1000 iterations, S = 10 reaches 3.2e-14 in
    # 112, and S = 12 reaches 4.4e-13 in *nine*. A network too narrow to represent the
    # trajectory floors its residual above the convergence target and then iterates to the
    # cap without getting there.
    over = mode == "quick" ? (; Ss = [10]) : (;)
    csv = run_sweep(; problem_name = NAME, build_prob = build_prob,
        hamiltonian = ham, mode = mode, over...)
    write_report(read_results(csv);
        title = "Shallow-net benchmark — Harmonic Oscillator ($(mode))",
        mode = mode, outdir = RESULTS_DIR, prefix = "$(NAME)_$(mode)")
end
