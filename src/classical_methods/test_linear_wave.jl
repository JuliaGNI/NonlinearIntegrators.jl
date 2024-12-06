cd("MultiSymplectic.jl")
using Pkg
Pkg.activate(".")
using GeometricIntegrators
using GeometricProblems.LinearWave
using Plots
problem = LinearWave.hodeproblem(tspan = (0, 20), tstep = 0.005)
sol = integrate(problem, RK4())


@gif for i in 1:1000
    plot(sol.q[i, :],ylims=(-3,3))
end

xs = -0.5:1/(257):0.5
ts = 0:0.005:0.02
surface(xs,ts,hcat(sol.q[1:5, :]...)',xlabel = "space", ylabel = "time ")


D = 2
S = 6

W = zeros(S,D)        # all parameters w
Bias = zeros(S,1)      # all parameters b
C = zeros(S,nstages+1)

unn = Lux.Chain(Dense(2, S, tanh), Dense(S, 1, use_bias=false))
ps, st = Lux.setup(Random.default_rng(), unn)

quad = simpson_quadrature(250,a = -0.5, b = 0.5)
quad.quad_weights
quad.quad_nodes
function simpson_quadrature(N::Int;a::Float64=0.0,b::Float64=1.0)
    if N % 2 != 0
        error("N must be even for Simpson's rule.")
    end
    
    # Step size
    h = (b-a) / N
    x = collect(a:h:b)    
    # Generate weights
    w = zeros(Float64, N + 1)
    for i in 1:(N + 1)
        if i == 1 || i == N + 1
            w[i] = h / 3 # First and last weights
        elseif i % 2 == 0
            w[i] = 4 * h / 3 # Even-indexed weights
        else
            w[i] = 2 * h / 3 # Odd-indexed weights
        end
    end
    
    return (quad_weights = w,quad_nodes =x)
end