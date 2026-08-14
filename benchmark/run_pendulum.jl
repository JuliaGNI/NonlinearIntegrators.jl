# Benchmark the shallow-net integrator on the (mathematical) pendulum.
#
#   julia --project=benchmark benchmark/run_pendulum.jl [quick|full]
#
# NOTE: GeometricProblems' Pendulum has no `lodeproblem`; it exposes a *degenerate*
# 2-component IODE (`iodeproblem`, ϑ: p₁=ml²q₂, p₂=0). The shallow-net method accepts any
# `AbstractProblemIODE`, so we use it here — this case deliberately stresses the solver.

include(joinpath(@__DIR__, "shallownet_benchmark_common.jl"))
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
    # S = 8 is a measured optimum, not a default: unlike the other problems this one gets
    # *worse* with a wider network. At Float64/tanh/DogLeg over 10 steps of dt = 0.1,
    # `ref_err` is 8.0e-05 at S = 4, 2.9e-07 at S = 8, 5.8e-05 at S = 10 and 1.4e+03 at
    # S = 12 — i.e. the solve diverges. The degenerate ϑ (p₂ = 0) leaves the parameter
    # Jacobian singular, and widening the network enlarges its null space.
    over = mode == "quick" ? (; Ss = [8]) : (;)
    csv = run_sweep(; problem_name = NAME, build_prob = build_prob,
                    hamiltonian = ham, mode = mode, over...)
    write_report(read_results(csv);
        title = "Shallow-net benchmark — Pendulum ($(mode))",
        mode = mode, outdir = RESULTS_DIR, prefix = "$(NAME)_$(mode)")
end
