# Benchmark the shallow-net integrator on the double pendulum (D = 2).
#
#   julia --project=benchmark benchmark/run_double_pendulum.jl [quick|full]

include(joinpath(@__DIR__, "shallownet_benchmark_common.jl"))
using GeometricProblems.DoublePendulum

const NAME = "double_pendulum"

function build_prob(::Type{T}, timespan, timestep) where {T}
    d = DoublePendulum.lodeproblem()               # Float64 defaults, to read the ics
    q0 = T.(d.ics.q)
    p0 = T.(d.ics.p)
    DoublePendulum.lodeproblem(q0, p0;
        timespan = timespan, timestep = timestep,
        parameters = DoublePendulum.default_parameters(T))
end

ham(t, q, p, params) = DoublePendulum.hamiltonian(t, q, p, params)

let mode = pick_mode()
    # A larger quadrature order and network for this harder problem. Measured at
    # Float64/tanh/DogLeg over 10 steps of dt = 0.1, `ref_err` falls monotonically with width
    # and then flattens: 5.9e-08 at S = 8, 8.4e-10 at S = 10, 9.3e-10 at S = 12. S = 10 is
    # where the gain stops.
    over = mode == "quick" ? (; Rs = [16], Ss = [10]) : (;)
    csv = run_sweep(; problem_name = NAME, build_prob = build_prob,
        hamiltonian = ham, mode = mode, over...)
    write_report(read_results(csv);
        title = "Shallow-net benchmark — Double Pendulum ($(mode))",
        mode = mode, outdir = RESULTS_DIR, prefix = "$(NAME)_$(mode)")
end
