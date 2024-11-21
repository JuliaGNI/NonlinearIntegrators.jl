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

