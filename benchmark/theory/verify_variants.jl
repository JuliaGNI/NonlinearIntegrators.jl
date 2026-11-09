# Structural properties of the shallow-network integrators stated in the NVI paper
# (sections "Shallow neural networks with fixed boundary values" and "Time-reversible shallow
# neural networks"), on Hénon–Heiles (two degrees of freedom), one step of size h:
#   symp = max |JᵀΩJ − Ω|, J the Jacobian of the one-step map Φ by central differences;
#   back = max |ρΦ(ρΦ(z₀)) − z₀|, ρ(q, p) = (q, −p): zero for a time-reversible integrator;
#   max res = largest residual of the nonlinear solves involved;
#   jump = max |Δx| / 2ε over the perturbed solves (x = network parameters of the step).
#
#   julia --project=benchmark benchmark/theory/verify_variants.jl
#
# Writes benchmark/results/verify_variants.csv.

include(joinpath(@__DIR__, "..", "shallownet_benchmark_common.jl"))   # relu_k
using NonlinearIntegrators
# in a module of its own, as in relu_k_sweep.jl: integration.jl defines a `RESULTS_DIR` of its own
module HardProblems
include(joinpath(@__DIR__, "..", "hard_problems", "integration.jl"))
end
using .HardProblems: integrate_counting, WarnCounter
using Logging: with_logger
using QuadratureRules: GaussLegendreQuadrature
import GeometricProblems.HenonHeilesPotential as hh
using LinearAlgebra
using Printf

const Ω = [zeros(2, 2) I(2); -I(2) zeros(2, 2)]
const ρ = Diagonal([1.0, 1.0, -1.0, -1.0])
const SOLVER_OPTS = (regularization_factor = 1e-10, max_iterations = 100)

"One step from z: the new state, the residual of the nonlinear solve and its solution x."
function step(method, z, h)
    prob = hh.lodeproblem(z[1:2], z[3:4]; timespan = (0.0, h), timestep = h,
        parameters = hh.default_parameters())
    r = with_logger(WarnCounter()) do
        integrate_counting(prob, method; record = true, SOLVER_OPTS...)
    end
    r.status == "ok" || error("step failed: $(r.status)")
    (vcat(collect(r.sol.q[end]), collect(r.sol.p[end])), r.max_res, only(r.x))
end

const VARIANTS = (ShallowNet = ShallowNet, ShallowNetReversible = ShallowNetReversible,
    ShallowNetAutodiff = ShallowNetAutodiff)

z₀, ε, S, R = [0.1, 0.1, 0.1, 0.1], 1e-4, 4, 4
rows = String["variant,activation,h,symp,back,max_res,jump,seconds"]
for (act, σ) in (("relu3", relu_k(3)), ("tanh", tanh)), h in (0.2, 1.0), (name, M) in pairs(VARIANTS)
    method = M(ShallowNetBasis{Float64}(σ, S), GaussLegendreQuadrature(Float64, R);
        show_status = false, bias_interval = [-π, π], dict_amount = 4000)
    t = @elapsed begin
        plus = [step(method, z₀ .+ ε .* (1:4 .== j), h) for j in 1:4]
        minus = [step(method, z₀ .- ε .* (1:4 .== j), h) for j in 1:4]
        J = reduce(hcat, (plus[j][1] .- minus[j][1]) ./ 2ε for j in 1:4)
        symp = maximum(abs, J' * Ω * J .- Ω)
        # parameter jump: |Δx| / 2ε, large when the perturbed solves end on different branches
        jump = maximum(maximum(abs, plus[j][3] .- minus[j][3]) / 2ε for j in 1:4)
        fwd = step(method, z₀, h)
        bwd = step(method, ρ * fwd[1], h)
        back = maximum(abs, ρ * bwd[1] .- z₀)
        res = maximum(r[2] for r in vcat(plus, minus, [fwd, bwd]))
    end
    @printf("%-22s %-5s h = %.1f  symp = %.2e  back = %.2e  max res = %.2e  jump = %.2e  (%.1f s)\n",
        name, act, h, symp, back, res, jump, t)
    push!(rows, @sprintf("%s,%s,%g,%.3e,%.3e,%.3e,%.3e,%.1f", name, act, h, symp, back, res, jump, t))
end
out = joinpath(@__DIR__, "..", "results", "verify_variants.csv")
write(out, join(rows, "\n") * "\n")
println("wrote ", out)
