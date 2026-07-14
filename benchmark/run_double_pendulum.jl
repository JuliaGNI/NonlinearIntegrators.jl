# Benchmark the one-layer GML integrator on the double pendulum (D = 2).
#
#   julia --project=benchmark benchmark/run_double_pendulum.jl [quick|full]

include(joinpath(@__DIR__, "gml_benchmark_common.jl"))
using GeometricProblems.DoublePendulum

const NAME = "double_pendulum"

function build_prob(::Type{T}, timespan, timestep) where {T}
    d  = DoublePendulum.lodeproblem()               # Float64 defaults, to read the ics
    q0 = T.(d.ics.q); p0 = T.(d.ics.p)
    DoublePendulum.lodeproblem(q0, p0;
        timespan = timespan, timestep = timestep,
        parameters = DoublePendulum.default_parameters(T))
end

ham(t, q, p, params) = DoublePendulum.hamiltonian(t, q, p, params)

let mode = pick_mode()
    # quick mode uses a larger network for this harder problem
    over = mode == "quick" ? (; Rs = [16], Ss = [8]) : (;)
    csv = run_sweep(; problem_name = NAME, build_prob = build_prob,
                    hamiltonian = ham, mode = mode, over...)
    write_report(read_results(csv);
        title = "One-layer GML benchmark — Double Pendulum ($(mode))",
        mode = mode, outdir = RESULTS_DIR, prefix = "$(NAME)_$(mode)")
end
