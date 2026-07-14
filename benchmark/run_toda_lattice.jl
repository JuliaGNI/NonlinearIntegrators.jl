# Benchmark the one-layer GML integrator on the Toda lattice with N = 16 (D = 16).
#
#   julia --project=benchmark benchmark/run_toda_lattice.jl [quick|full]
#
# NOTE: with N = 16 (> 10) the Lagrangian system is built without symbolic
# simplification, so each problem construction is comparatively heavy; problems are
# cached per (T, dt) by the sweep engine.

include(joinpath(@__DIR__, "gml_benchmark_common.jl"))
using GeometricProblems.TodaLattice

const NAME = "toda_lattice"
const N_TODA = 16

function build_prob(::Type{T}, timespan, timestep) where {T}
    d  = TodaLattice.lodeproblem(N_TODA)            # Float64 defaults, to read the ics
    q0 = T.(d.ics.q); p0 = T.(d.ics.p)
    TodaLattice.lodeproblem(q0, p0;                 # infers N = length(q0) = 16
        timespan = timespan, timestep = timestep,
        parameters = TodaLattice.default_parameters(T))
end

ham(t, q, p, params) = TodaLattice.hamiltonian(t, q, p, params, N_TODA)

let mode = pick_mode()
    # quick mode uses a larger network for this harder problem
    over = mode == "quick" ? (; Rs = [16], Ss = [8]) : (;)
    csv = run_sweep(; problem_name = NAME, build_prob = build_prob,
                    hamiltonian = ham, mode = mode, over...)
    write_report(read_results(csv);
        title = "One-layer GML benchmark — Toda Lattice (N=16, $(mode))",
        mode = mode, outdir = RESULTS_DIR, prefix = "$(NAME)_$(mode)")
end
