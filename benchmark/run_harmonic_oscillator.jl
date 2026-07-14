# Benchmark the one-layer GML integrator on the harmonic oscillator.
#
#   julia --project=benchmark benchmark/run_harmonic_oscillator.jl [quick|full]
#
# Mode defaults to "quick" (also settable via GML_BENCH_PRESET). Writes
# results/harmonic_oscillator_<mode>.csv plus a markdown report and plots.

include(joinpath(@__DIR__, "gml_benchmark_common.jl"))
using GeometricProblems.HarmonicOscillator

const NAME = "harmonic_oscillator"

build_prob(::Type{T}, timespan, timestep) where {T} =
    HarmonicOscillator.lodeproblem([T(0.5)], [T(0.0)];
        timespan = timespan, timestep = timestep,
        parameters = HarmonicOscillator.default_parameters(T))

ham(t, q, p, params) = HarmonicOscillator.hamiltonian(t, q, p, params)

let mode = pick_mode()
    csv = run_sweep(; problem_name = NAME, build_prob = build_prob,
                    hamiltonian = ham, mode = mode)
    write_report(read_results(csv);
        title = "One-layer GML benchmark — Harmonic Oscillator ($(mode))",
        mode = mode, outdir = RESULTS_DIR, prefix = "$(NAME)_$(mode)")
end
