# The plotting API: `continuous_solution` and `Trajectory` in `src/`, `plot_solution`,
# `plot_convergence` and `plot_theme` in the `NonlinearIntegratorsPlots` extension.
#
# Loading CairoMakie activates the `Makie` weakdep and with it the extension — the same idiom
# `GeometricProblems/test/plots_tests.jl` and `ElectromagneticFields` use. The `plot_*` tests are
# smoke tests: each must build and return a `Figure` without error, which is what a plot function
# can usefully be asserted to do.
#
# `continuous_solution` is *not* smoke-tested. It is arithmetic with two off-by-one traps in it —
# the duplicated step boundary and the assumed grid size — so it gets real assertions, including
# against a non-default `record_grid_points`, which is the case a re-hardcoded 41 would fail.

using CairoMakie
using GeometricSolutions: GeometricSolution
import GeometricSolutions
import GeometricEquations

# `import`, not `using`: `testsetup.jl` does `using GeometricProblems.HarmonicOscillator`, which
# already brings a `plot_solution` into this scope. That clash is the reason these live in a
# submodule at all, so the test reaches them the way a caller in the same position has to.
import NonlinearIntegrators.Diagnostics as NIP
# The diagnostics this extension does not duplicate.
import GeometricProblems.Diagnostics as GPD

@testset "$(rpad("Plotting", 80))" begin
    params = HarmonicOscillator.default_parameters(Float64)
    h = 0.1
    nsteps = 5
    prob = HarmonicOscillator.lodeproblem([0.5], [0.0];
        timespan = (0.0, h * nsteps), timestep = h, parameters = params)
    # The ansatz spans the exact solution and `init_w` starts Newton at it, so this run is
    # accurate to the residual floor — the same setup `vise_unit.jl` relies on.
    method = VISE(build_vise_basis(Float64), gauss(Float64, 4),
        [Float64[0.5, sqrt(0.5), 0.0]])
    sol, internal_values, _ = integrate(prob, method)

    @testset "continuous_solution" begin
        G = size(internal_values[1], 1)
        @test G == 41

        t, q = continuous_solution(internal_values, h)

        # One point per interior grid point per step: the left endpoint of each step is dropped
        # because it repeats the previous step's right endpoint.
        @test length(t) == nsteps * (G - 1)
        @test length(q) == length(t)
        @test issorted(t)
        @test all(isfinite, q)

        # The grid starts one sub-step in, not at t₀, and ends exactly at the final time.
        @test t[1] ≈ h / (G - 1)
        @test t[end] ≈ h * nsteps
        @test all(≈(h / (G - 1)), diff(t))

        # Every step boundary is hit exactly once, and the value there agrees with the discrete
        # solution — this is what fails if row 1 is kept, or if the grid spacing is wrong.
        for n in 1:nsteps
            k = n * (G - 1)
            @test t[k] ≈ n * h
            @test q[k] ≈ collect(sol.q[:, 1])[n + 1] atol=1e-10
        end

        # The tuple `integrate` returns is accepted directly, in both of its shapes.
        t2, q2 = continuous_solution((sol, internal_values), h)
        @test t2 == t && q2 == q
        t3, q3 = continuous_solution((sol, internal_values, nothing), h)
        @test t3 == t && q3 == q

        # `t₀` shifts the whole grid.
        t4, _ = continuous_solution(internal_values, h; t₀ = 7.0)
        @test t4 ≈ t .+ 7.0
    end

    @testset "continuous_solution honours record_grid_points" begin
        # With the grid size read off the array this is 5 × 20 points; with a hard-coded 41 the
        # returned `t` and `q` disagree in length, or the indexing runs off the end.
        method21 = VISE(build_vise_basis(Float64), gauss(Float64, 4),
            [Float64[0.5, sqrt(0.5), 0.0]]; record_grid_points = 21)
        _, internal21, _ = integrate(prob, method21)
        @test size(internal21[1], 1) == 21

        t, q = continuous_solution(internal21, h)
        @test length(t) == nsteps * 20
        @test length(q) == length(t)
        @test t[1] ≈ h / 20
        @test t[end] ≈ h * nsteps
    end

    @testset "continuous_solution rejects what it cannot do" begin
        @test_throws ArgumentError continuous_solution(Matrix{Float64}[], h)
        @test_throws ArgumentError continuous_solution([zeros(1, 1)], h)
        @test_throws ArgumentError continuous_solution(internal_values, h; dof = 2)
        # A ragged record is a bug upstream, not something to average over.
        @test_throws ArgumentError continuous_solution(
            [internal_values[1], zeros(7, 1)], h)
        @test_throws ArgumentError continuous_solution((sol,), h)
    end

    @testset "plot functions return a Figure" begin
        ham = HarmonicOscillator.hamiltonian
        imp = integrate(prob, ImplicitMidpoint())
        exact = HarmonicOscillator.exact_solution(
            HarmonicOscillator.podeproblem([0.5], [0.0];
            timespan = (0.0, h * nsteps), timestep = h / 10, parameters = params))

        @test NIP.plot_solution(sol, internal_values) isa Figure
        @test NIP.plot_solution((sol, internal_values, nothing)) isa Figure
        @test NIP.plot_solution(sol, nothing) isa Figure
        @test NIP.plot_solution(sol, internal_values;
            hamiltonian = ham, parameters = prob.parameters,
            reference = exact,
            comparisons = ["Implicit midpoint" => imp],
            training_region = 0.2,
            label = "VISE") isa Figure
        @test NIP.plot_solution(sol, internal_values; latex = false) isa Figure

        @test NIP.plot_solution(sol, internal_values; title = "a title") isa Figure
        @test NIP.plot_solution(sol, internal_values; figsize = (700, 300)) isa Figure

        # The single-solution diagnostics this extension deliberately does *not* implement, checked
        # here so that a `GeometricProblems` change which broke them for a `lodeproblem` shows up in
        # this suite rather than in a caller's figure.
        @test HarmonicOscillator.plot_phase_portrait(sol) isa Figure

        # ...and this one is broken upstream *today*, which is what the check found.
        #
        # `GeometricProblems.Diagnostics.plot_energy_error` cannot compute the energy of a
        # partitioned or implicit solution. Its `_invariant_error` branches on
        # `sol isa Union{SolutionPODE, SolutionPDAE}` to decide whether to pass `p` to the
        # invariant — and that test is `false` for a `GeometricSolution` of a `LODEProblem`, even
        # though `SolutionPODE`'s own definition names `LODEProblem`: the alias is written
        #
        #     const SolutionPODE{dType, …, probType, perType} =
        #         GeometricSolution{dType, …, probType, perType} where {probType <: Union{…}}
        #
        # with `probType` bound both as an alias parameter and by the `where`, so the constraint
        # does not apply the way it reads. The q-only branch is therefore always taken, `p` is never
        # passed, and `HarmonicOscillator.hamiltonian(t, q, params)` — the three-argument method,
        # which expects `q = [q, v]` — indexes `q[2]` on a one-element vector.
        #
        # Two assertions rather than one `@test_broken`: the first records the symptom, the second
        # pins the cause, so a fix upstream flips the `@test_broken` to a pass and the second one
        # says why. Measured on GeometricProblems 0.8.3 / GeometricSolutions 0.6.5.
        @test_broken GPD.plot_energy_error(sol; energy = ham) isa Figure
        @test !(sol isa GeometricSolutions.SolutionPODE)   # the cause; should become `false`
        @test typeof(sol.problem) <: GeometricEquations.LODEProblem
    end

    @testset "plot_solution with more than one degree of freedom" begin
        prob2 = CoupledHarmonicOscillator.lodeproblem(;
            timespan = (0.0, h * nsteps), timestep = h)
        sol2 = integrate(prob2, ImplicitMidpoint())
        @test length(sol2.q[0]) == 2
        @test NIP.plot_solution(sol2, nothing) isa Figure
        @test NIP.plot_solution(sol2, nothing;
            hamiltonian = CoupledHarmonicOscillator.hamiltonian,
            parameters = prob2.parameters) isa Figure
    end

    @testset "the error panel bridges a sample the log axis cannot take" begin
        # A representation that conserves its invariant to round-off hits an exact `0` repeatedly,
        # not only at `t₀` — the global Fourier fit of the perturbed pendulum does so at 22 of its
        # 101 samples. Masking those with `NaN` and keeping the full time vector, which is what
        # this did first, breaks the polyline at every one of them, and the panel comes out as
        # fragments and isolated dots instead of a curve. Dropping the sample from the *time*
        # vector as well is what bridges it, and asserted here rather than left to the eye,
        # because the difference between a fragmented panel and a whole one is invisible while
        # only `t₀` is zero.
        t = collect(0.0:1.0:6.0)
        err = [0.0, 1e-16, 0.0, 2e-16, 3e-16, 0.0, 1e-16]
        traj = Trajectory("round-off", t, [sin.(t)], [cos.(t)]; invariant_error = err)
        fig = NIP.plot_solution(traj)

        # `q`, `p`, error — the error panel is the last, and the only logarithmic one.
        ax_err = filter(x -> x isa Axis, fig.content)[end]
        @test ax_err.yscale[] === log10

        # One plot, carrying every plottable sample and nothing else: no `NaN` to break it, and
        # each surviving point still at its own time.
        @test length(ax_err.scene.plots) == 1
        points = ax_err.scene.plots[1][1][]
        @test length(points) == count(>(0), err)
        @test all(p -> isfinite(p[1]) && isfinite(p[2]), points)
        @test [p[1] for p in points] == t[findall(>(0), err)]
    end

    @testset "plot_theme" begin
        # The shared theme is something the caller applies, not something the extension installs.
        # Asserted because it is copied from `GeometricExamples/src/common.jl` by hand, and a figure
        # made with a theme that has quietly drifted is exactly the kind of difference nobody
        # notices until two of them sit side by side.
        @test NIP.plot_theme() isa Theme
        @test NIP.plot_theme().fontsize[] == 18
        @test NIP.plot_theme().Axis.xlabelsize[] == 22
        @test NIP.plot_theme().Axis.ylabelsize[] == 22
        @test NIP.plot_theme().Axis.xticklabelsize[] == 16
        @test NIP.plot_theme().Axis.titlesize[] == 20
        @test NIP.plot_theme().Lines.linewidth[] == 2
        @test NIP.plot_theme().Scatter.markersize[] == 10

        # And the figures build under it, which is how they are actually made.
        @test with_theme(NIP.plot_theme()) do
            NIP.plot_solution(sol, internal_values)
        end isa Figure
    end

    @testset "plot_convergence" begin
        hs = [0.1, 0.2, 0.4, 0.8]
        errs = [hs .^ 2, hs .^ 3]

        @test NIP.plot_convergence(hs, errs; labels = ["second", "third"]) isa Figure
        @test NIP.plot_convergence(hs, errs;
            labels = ["second", "third"],
            linestyles = [:solid, :dash],
            reference_orders = (2, 3),
            title = "test") isa Figure
        # Per-series step vectors, for a study where not every configuration ran everywhere.
        @test NIP.plot_convergence([hs, hs[1:3]], [hs .^ 2, hs[1:3] .^ 3];
            labels = ["a", "b"]) isa Figure
        # A series that failed at some steps must not take the whole figure down with it: a
        # logarithmic axis cannot plot a zero or a NaN.
        @test NIP.plot_convergence(hs, [[NaN, 0.0, 0.04, 0.16]];
            labels = ["partial"]) isa Figure
        @test NIP.plot_convergence(hs, [hs .^ 2]; labels = ["only"],
            reference_orders = ()) isa Figure

        @test_throws ArgumentError NIP.plot_convergence(hs, errs; labels = ["one label"])
    end
end
