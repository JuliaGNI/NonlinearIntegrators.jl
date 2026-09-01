# Variational Integrator with Symbolic Expression (VISE)

## Basic Usage
At first, we have to load several packages:
```
using Symbolics
using CompactBasisFunctions
using NonlinearIntegrators
using QuadratureRules
using GeometricProblems
using GeometricIntegratorsBase
```
Additional package would be required if we want to use Nested Sindy expression.
```
using Symbolics
```
Like what we would do in *GeometricIntegrators.jl*, we first define the problem we want to solve. The type of problem here is restricted to _Lagrangian ordinary equations_, i.e. models have a regular Lagrangian and its Lagrangian equation of motions.

```
T = 100.0
h_step = 2.0
HHlode = GeometricProblems.HenonHeilesPotential.lodeproblem([0.1,0.1],[0.1,0.1],timespan = (0,T),timestep = h_step)
```

For each dimension of generalized coordinates __q__ = (q<sub>1</sub>, q<sub>2</sub>), we prescribe a expression that approximates the coordinates.
```
@variables W1[1:4] W2[1:4] t
q₁_expr = W1[1] *cos(W1[2]* t + W1[3]) + W1[4]
q₂_expr = W2[1] *cos(W2[2]* t + W2[3]) + W2[4]
```
then define a basis object for variational Galerkin integrator, in this case, a Polynomial-Radial basis for the two dimension problem.
```
vise_basis = VISEBasis{Float64}([q₁_expr,q₂_expr], [W1,W2], t,2)
```
along with a chosen quadrature rule, and initial guess for the parameters in the expression, we define the integrator, and start the game :)

```
vise_method = VISE(vise_basis, QGau8,[[0.14831,1.0,-0.64812,- 0.018712],[0.14298,- 0.97215,0.7615,-0.0013983]]) 
vise_sol = integrate(HHlode, vise_method)
```

One good thing about continuous Galerkin variational integrators is that you could record the coordinates values between two discrete steps for free!

The second element of the returned tuple holds one such record per time step, sampled on a
uniform grid over the step. `record_grid_points` sets its size — 41 points by default, which is
the `h_step/40` spacing the plot below assumes, and the same keyword the network integrators
take:

```
vise_method = VISE(vise_basis, QGau8, init_w; record_grid_points = 81)
```

(`init_w` being the initial-parameter vector spelled out in full above.)

Turning that record into something plottable is what [`continuous_solution`](@ref) is for:

```julia
t, q = continuous_solution(vise_sol, h_step)     # the whole returned tuple is accepted
t, q = continuous_solution(vise_sol[2], h_step; dof = 2)
```

Two details make it worth a function rather than a one-liner, and both used to be got wrong by
hand. Row 1 of every step is that step's *left* endpoint, so concatenating the records whole
duplicates each interior step boundary — which is why the grid comes back starting at
`h_step/(G-1)` and not at `0`, the initial condition being in the solution rather than the record.
And the grid size is `record_grid_points`, read off the array, not the 41 it happens to default to.

## Plotting

Most of what a run of these integrators wants plotted, `GeometricProblems` already plots, and this
package does not duplicate any of it:

| want | use |
|:--|:--|
| a phase portrait, a trajectory, a set of traces | the per-problem recipes, e.g. `GeometricProblems.HarmonicOscillator.plot_phase_portrait` |
| one method's error against the time step, with its expected order | `GeometricProblems.Diagnostics.plot_convergence` |
| the relative error of an invariant over time | see the caveat below |

!!! warning "`plot_energy_error` does not work on a LODE solution today"

    `GeometricProblems.Diagnostics.plot_energy_error` and `plot_invariant_error` cannot compute the
    energy of a *partitioned or implicit* solution, which is every solution this package produces.
    Their `_invariant_error` branches on `sol isa Union{SolutionPODE, SolutionPDAE}` to decide
    whether to pass `p` to the invariant, and that test is `false` for a `GeometricSolution` of a
    `LODEProblem` even though `SolutionPODE`'s definition names `LODEProblem` — the alias binds
    `probType` both as a parameter and in its `where` clause, so the constraint does not apply as it
    reads. The `q`-only branch is always taken, `p` is never passed, and a Hamiltonian expecting
    `(t, q, p, params)` is called with three arguments.

    Measured on GeometricProblems 0.8.3 / GeometricSolutions 0.6.5, and guarded by a `@test_broken`
    in `test/plots_tests.jl` so that a fix upstream is noticed here.

    Until then, take the relative Hamiltonian error from the third panel of
    [`plot_solution`](@ref), which computes it from `q` *and* `p`, or compute it directly:

    ```julia
    using NonlinearIntegrators: relative_invariant_error
    H = [hamiltonian(sol.t[n], sol.q[n], sol.p[n], prob.parameters) for n in 0:ntime(sol)]
    ΔH = relative_invariant_error(H)
    ```

The `NonlinearIntegratorsPlots` extension adds the two figures those cannot make. It loads with any
Makie backend, and is reached through the `NonlinearIntegrators.Diagnostics` submodule rather than
the package's top level, because `plot_solution` is a name every `GeometricProblems` problem
submodule already exports and `plot_convergence` one that its own `Diagnostics` does.

### `plot_solution` — several integrators, and the continuous solution

```julia
using CairoMakie
import NonlinearIntegrators.Diagnostics as NIP

# The shared theme of this ecosystem: larger fonts and thicker lines than the Makie defaults,
# identical to the copy in `GeometricExamples/src/common.jl`. The extension sets no font size of
# its own, so this is where they come from.
set_theme!(NIP.plot_theme())

HHlode_ref = GeometricProblems.HenonHeilesPotential.lodeproblem([0.1, 0.1], [0.1, 0.1];
    timespan = (0, TT), timestep = h_step / 40)
ref_sol = integrate(HHlode_ref, Gauss(8))

fig = NIP.plot_solution(vise_sol;
    hamiltonian = GeometricProblems.HenonHeilesPotential.hamiltonian,
    parameters = HHlode.parameters,
    reference = "Reference (Gauss(8))" => ref_sol,
    comparisons = ["Implicit midpoint" => integrate(HHlode, ImplicitMidpoint())],
    label = "VISE",
    title = "Hénon–Heiles, Δt = $(h_step)")
save("henon-heiles.pdf", fig)
```

It takes the whole tuple `integrate` returned, so the continuous solution comes along without being
asked for, and it returns a `Figure` and never saves one. For `D` degrees of freedom the layout is a
`D`×2 grid of `qᵈ(t)` and `pᵈ(t)` with the relative Hamiltonian error spanning the width beneath;
the error panel appears only when `hamiltonian` and `parameters` are both given, rather than being
drawn empty.

No `GeometricProblems` recipe takes more than one solution, and none knows about `internal_values`
— which is why this one is here and why it is the only solution plot that is.

### `plot_convergence` — several series, several reference slopes

`GeometricProblems.Diagnostics.plot_convergence` is the right thing for one method against its
expected order. This one takes a series per configuration and a reference slope per order, for the
case where the claim of the figure is which *family* of method follows which slope.

### Archived results

Where a result has been archived rather than plotted straight away, build a [`Trajectory`](@ref)
from the stored vectors and pass that instead of a solution. It is the type these functions
actually consume, and it holds plain arrays, so an archive of one does not pin the version of
`GeometricSolutions` that wrote it.
