# Are the nonlinear equations of the shallow-network integrators the stationarity conditions of
# their discrete action? At a random point x of the unknowns, one step of Hénon–Heiles, the
# residual of the integrator is compared with the gradient (ForwardDiff) of the discrete action
# written out independently of the integrator code:
#   ShallowNet, ShallowNetReversible (Hamilton–Pontryagin form, multipliers p̄ = pₙ, p̃ = pₙ₊₁):
#     G(x) = h Σⱼ αⱼ L(q(cⱼ), q̇(cⱼ)) + p̄ ⋅ q(0) − p̃ ⋅ q(1),  rows (a, W, b) = −∂G/∂(a, W, b);
#     for ShallowNetReversible the neuron 2i is the mirror of neuron 2i−1:
#     W₂ᵢ = −W₂ᵢ₋₁, b₂ᵢ = W₂ᵢ₋₁ + b₂ᵢ₋₁, and only (W₂ᵢ₋₁, b₂ᵢ₋₁) are unknowns;
#   ShallowNetAutodiff (fixed boundary values, q(t) = (1−t)q̄ + t qₙ₊₁ + t(1−t)N(t)):
#     K(x) = h Σⱼ αⱼ L(q(cⱼ), q̇(cⱼ)),  rows (a, W, b) = ∂K/∂(a, W, b), row p = p̄ + ∂K/∂q̄.
#
#   julia --project=benchmark benchmark/theory/verify_variants_residual.jl
#
# Writes benchmark/results/verify_variants_residual.csv (column `err` = max |residual − gradient|).

include(joinpath(@__DIR__, "..", "shallownet_benchmark_common.jl"))   # relu_k
using NonlinearIntegrators
using GeometricIntegratorsBase
using QuadratureRules: GaussLegendreQuadrature
import GeometricProblems.HenonHeilesPotential as hh
import NonlinearIntegrators: ForwardDiff
using Random
using Printf

const D, S, R, h = 2, 4, 4, 0.5
const q̄, p̄ = [0.1, 0.1], [0.1, -0.2]
const PARAMS = hh.default_parameters()

L(q, v) = hh.lagrangian(0.0, q, v, PARAMS)
σ′(σ, z) = ForwardDiff.derivative(σ, z)

# unknowns of all three integrators: x[D(i-1)+k] = aᵢ (dimension k), x[DS+k] = p̃ₖ or qₙ₊₁,ₖ,
# then the hidden weights and biases (S of each, or S/2 for the reversible network)
function hidden(x, k, mirrored)
    if mirrored
        W1 = [x[D * (S + 1) + D * (i - 1) + k] for i in 1:(S ÷ 2)]
        b1 = [x[D * (S + 1 + S ÷ 2) + D * (i - 1) + k] for i in 1:(S ÷ 2)]
        (collect(Iterators.flatten((w, -w) for w in W1)),
            collect(Iterators.flatten((b, w + b) for (w, b) in zip(W1, b1))))
    else
        ([x[D * (S + 1) + D * (i - 1) + k] for i in 1:S],
            [x[D * (S + 1 + S) + D * (i - 1) + k] for i in 1:S])
    end
end

N(σ, a, W, b, t) = sum(a[i] * σ(W[i] * t + b[i]) for i in 1:S)
N′(σ, a, W, b, t) = sum(a[i] * W[i] * σ′(σ, W[i] * t + b[i]) for i in 1:S)

# discrete action of ShallowNet / ShallowNetReversible
function G(x, σ, c, α, mirrored)
    nets = [(a = [x[D * (i - 1) + k] for i in 1:S], Wb = hidden(x, k, mirrored)) for k in 1:D]
    q(t) = [N(σ, n.a, n.Wb..., t) for n in nets]
    v(t) = [N′(σ, n.a, n.Wb..., t) / h for n in nets]
    p̃ = x[(D * S + 1):(D * S + D)]
    h * sum(α[j] * L(q(c[j]), v(c[j])) for j in eachindex(c)) + p̄ ⋅ q(0.0) - p̃ ⋅ q(1.0)
end

# discrete action of ShallowNetAutodiff, as a function of the unknowns and of q̄
function K(x, q₀, σ, c, α)
    nets = [(a = [x[D * (i - 1) + k] for i in 1:S], Wb = hidden(x, k, false)) for k in 1:D]
    q₁ = x[(D * S + 1):(D * S + D)]
    q(t) = [(1 - t) * q₀[k] + t * q₁[k] + t * (1 - t) * N(σ, nets[k].a, nets[k].Wb..., t)
            for k in 1:D]
    v(t) = [(q₁[k] - q₀[k] + (1 - 2t) * N(σ, nets[k].a, nets[k].Wb..., t) +
             t * (1 - t) * N′(σ, nets[k].a, nets[k].Wb..., t)) / h for k in 1:D]
    h * sum(α[j] * L(q(c[j]), v(c[j])) for j in eachindex(c))
end

function code_residual(int, x)
    b = zero(x)
    sol = (t = h, q = copy(q̄), p = copy(p̄))
    copyto!(cache(int).q̃, x[(D * S + 1):(D * S + D)])   # endpoint estimate read by ShallowNetAutodiff
    GeometricIntegratorsBase.residual!(b, copy(x), sol, NamedTuple(PARAMS), int)
    b
end

prob = hh.lodeproblem(q̄, p̄; timespan = (0.0, h), timestep = h, parameters = PARAMS)
rows = String["variant,activation,rows,err,max_abs_residual"]
for (act, σ) in (("relu3", relu_k(3)), ("tanh", tanh)),
    M in (ShallowNet, ShallowNetReversible, ShallowNetAutodiff)

    method = M(ShallowNetBasis{Float64}(σ, S), GaussLegendreQuadrature(Float64, R))
    integ = GeometricIntegrator(prob, method)
    c, α = method.c, method.b
    x = randn(MersenneTwister(1), length(nlsolution(integ)))
    r = code_residual(integ, x)
    a_rows = 1:(D * S)
    Wb_rows = (D * (S + 1) + 1):length(x)
    if M === ShallowNetAutodiff
        g = ForwardDiff.gradient(y -> K(y, q̄, σ, c, α), x)
        gq = ForwardDiff.gradient(y -> K(x, y, σ, c, α), q̄)
        checks = (("a", r[a_rows] .- g[a_rows]), ("Wb", r[Wb_rows] .- g[Wb_rows]),
            ("p", r[(D * S + 1):(D * S + D)] .- (p̄ .+ gq)))
    else
        g = -ForwardDiff.gradient(y -> G(y, σ, c, α, M === ShallowNetReversible), x)
        checks = (("a", r[a_rows] .- g[a_rows]), ("Wb", r[Wb_rows] .- g[Wb_rows]))
    end
    for (name, d) in checks
        @printf("%-22s %-5s %-4s max |residual − gradient| = %.2e   (max |residual| = %.2e)\n",
            M, act, name, maximum(abs, d), maximum(abs, r))
        push!(rows, @sprintf("%s,%s,%s,%.3e,%.3e", M, act, name, maximum(abs, d), maximum(abs, r)))
    end
end
out = joinpath(@__DIR__, "..", "results", "verify_variants_residual.csv")
write(out, join(rows, "\n") * "\n")
println("wrote ", out)
