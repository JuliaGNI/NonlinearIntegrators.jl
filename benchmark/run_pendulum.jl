# Benchmark the one-layer GML integrator on the (mathematical) pendulum.
#
#   julia --project=benchmark benchmark/run_pendulum.jl [quick|full]
#
# NOTE: GeometricProblems' Pendulum has no `lodeproblem`; it exposes a *degenerate*
# 2-component IODE (`iodeproblem`, ϑ: p₁=ml²q₂, p₂=0). The GML method accepts any
# `AbstractProblemIODE`, so we use it here — this case deliberately stresses the solver.

include(joinpath(@__DIR__, "gml_benchmark_common.jl"))
using GeometricProblems.Pendulum

const NAME = "pendulum"

function build_prob(::Type{T}, timespan, timestep) where {T}
    d  = Pendulum.iodeproblem()                     # Float64 defaults, to read the ics
    q0 = T.(d.ics.q); p0 = T.(d.ics.p)
    Pendulum.iodeproblem(q0, p0;
        timespan = timespan, timestep = timestep,
        parameters = Pendulum.default_parameters(T))
end

ham(t, q, p, params) = Pendulum.hamiltonian(t, q, p, params)

let mode = pick_mode()
    csv = run_sweep(; problem_name = NAME, build_prob = build_prob,
                    hamiltonian = ham, mode = mode)
    write_report(read_results(csv);
        title = "One-layer GML benchmark — Pendulum ($(mode))",
        mode = mode, outdir = RESULTS_DIR, prefix = "$(NAME)_$(mode)")
end
