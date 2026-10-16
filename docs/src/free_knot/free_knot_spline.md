# Free-Knot Spline Integrator

[`FreeKnotSpline`](@ref) is a variational integrator whose ansatz on each step is a cubic spline
with `S` free interior knots:

```math
q_d(t) = \sum_{j=0}^{3} a_{jd}\, t^j + \sum_{i=1}^{S} c_{id}\, (t - \tau_i)_+^3 ,
\qquad t \in [0, 1] .
```

It is the shallow ReLU³ network of [`ShallowNet`](@ref) with every source of rank deficiency
removed:

- **The scaling gauge.** A ReLU³ neuron ``c\,\sigma(w t + b)`` is unchanged under
  ``(w, b, c) \mapsto (\lambda w, \lambda b, \lambda^{-3} c)``. Here every neuron has ``w = 1``.
  With the cubic explicit, the sign of ``w`` carries no information either, because
  ``(\tau - t)_+^3 = (t - \tau)_+^3 - (t - \tau)^3``.
- **The polynomial collapse.** A neuron whose kink lies outside the step is a cubic on the step.
  Here the cubic is explicit, and the knots are free parameters of the step.
- **Quadrature on the whole step.** A Gauss rule on ``[0, 1]`` is exact for polynomials, not for
  a piecewise cubic. Free knots then fit the quadrature error. Here the rule is applied on each
  piece between consecutive knots, so the discrete action is smooth in the knots, and exact for a
  quadratic Lagrangian with Gauss 4.

## Basic usage

```julia
using NonlinearIntegrators
using GeometricIntegratorsBase
using GeometricProblems.HarmonicOscillator

prob = HarmonicOscillator.lodeproblem([0.5], [0.0]; timespan = (0.0, 10.0), timestep = 0.5)
sol = integrate(prob, FreeKnotSpline(4))
```

The problem must be a Lagrangian problem. Of its functions, the step evaluates the Lagrangian
`l(t, q, v, params)`, through ForwardDiff, and the initial guess `v̄` only, so `l` must accept dual
numbers. Nothing checks `ϑ`, `f` or `ω` against `l`. All degrees of freedom share the knots.

## The solve

The discrete Euler–Lagrange system has an accurate, non-degenerate stationary point, with the
knots spread near-uniformly and all ``c_i \ne 0``. But it also has stationary points where the
knots merge, sit at 0 or 1, or leave the step, and where the trajectory is the cubic step. Plain
Newton on the full system finds the accurate point only from uniform knots, and not reliably from
those. Each step therefore solves in four stages:

1. **The cubic step** (``S = 0``), by Newton. Its endpoint ``Q`` is fixed for stages 2 and 3.
2. **The reduced action.** For given knots, the action with fixed endpoints ``q(0) = \bar q`` and
   ``q(1) = Q`` is minimised over ``(a_1, a_2, c)``. This gives ``f(\tau)``.
3. **Descent on the knots.** ``f`` is minimised from uniform knots, by Newton with the
   absolute-eigenvalue Hessian of ``f`` and a step cap. The knots are parametrised by their gaps,
   so they stay ordered and inside ``(0, 1)``. A step is accepted if it passes Armijo on ``f``,
   or, where the Hessian of ``f`` is positive definite, if it halves the gradient of ``f``.
4. **Newton on the full system**, from that point. This also frees ``q(1)``.

For a short step, the action with fixed endpoints is a minimum in the trajectory. For a quadratic
Lagrangian, ``f(\tau) - A(q_{\text{exact}})`` is half the squared energy-norm error of the best
spline with knots ``\tau``. So stage 3 descends towards the best free-knot spline. A collapsed
point is no better than the cubic, so the descent does not settle there. If stage 4 still ends
with a knot closer than ``10^{-3}`` to a neighbour or to an end of the step, the step raises an
error.

**Why the second test is needed.** The gain of the knots over the cubic is of the order of the
squared error, about ``10^{-20}`` against ``|f| \approx 3 \cdot 10^{-2}`` on the oscillator below.
That is below the round-off of ``f``, so near the minimum Armijo on ``f`` compares round-off and
stops the descent at an arbitrary point. Away from a zero of ``q`` that is harmless. Near one, the
``c_i`` are about ``10^{-5}`` and change sign, the dead-knot point ``c = 0`` is close, and stage 4
can converge to it: the step is then the cubic step. With Armijo alone, the fifth step of
``h = 0.5`` from ``(q, p) = (0.5, 0)``, at ``q = 0.078``, has the cubic's local error,
``1.0 \cdot 10^{-11}``, against ``10^{-13}`` for its neighbours. The gradient of ``f`` comes from
automatic differentiation, and it still resolves the minimum there.

## Measured

Harmonic oscillator (``m = 1``, ``k = 0.5``, ``\bar q = 0.5``) and pendulum
(``L = \dot q^2 / 2 + 0.5 \cos q``, ``\bar q = 1``), ``h = 0.5``. Endpoint error of
[`FreeKnotSpline`](@ref) after one and after two steps (`scripts/free_knot_spline_study.jl`,
section 5; the pendulum against RK4 in 256-bit arithmetic):

| ``S`` | oscillator | pendulum |
|:--|:--|:--|
| 4 | ``9.6 \cdot 10^{-14}``, ``3.6 \cdot 10^{-13}`` | ``9.2 \cdot 10^{-14}``, ``4.6 \cdot 10^{-13}`` |
| 5 | ``3.4 \cdot 10^{-14}``, ``1.3 \cdot 10^{-13}`` | ``3.2 \cdot 10^{-14}``, ``1.6 \cdot 10^{-13}`` |
| 6 | ``2.2 \cdot 10^{-14}``, ``6.0 \cdot 10^{-14}`` | ``1.3 \cdot 10^{-14}``, ``5.5 \cdot 10^{-14}`` |

In the prototype, on the same two problems and steps, the descent was started from four knot
seeds: uniform knots, and the raw, normalised and orthogonal greedy selections of the OGA seeds.
Every seed reached the same point, in 4 to 19 iterations, for ``S = 4, 5, 6`` on both problems:
48 of 48 cases. Plain Newton from the same seeds reached it in 8 of 48 cases, all from uniform
knots. The cubic step there has an endpoint error of ``5.9 \cdot 10^{-10}`` (oscillator) and
``5.8 \cdot 10^{-10}`` (pendulum) after one step.

At the prototype's points the Jacobian of the full system has full rank, with row and column
equilibration, at ``\sqrt{\varepsilon}``. Its smallest scaled singular value is ``10^{-7}`` to
``10^{-6}``: the ``\tau`` columns scale with ``c \approx 2 \cdot 10^{-4}``, so the system is
ill-conditioned but not singular. Row equilibration alone misreads this as a rank loss of
``S - 1``.

The prototype is `scripts/free_knot_descent.jl`. Its descent accepts on Armijo alone, which is
enough for these two steps from ``p = 0``. `scripts/free_knot_rank.jl` holds the two
alternatives that fall short: Gauss 8 on the whole step, and plain Newton with per-piece quadrature.

Over many steps, with the integrator itself (`scripts/free_knot_spline_study.jl`, oscillator,
``S = 4``):

| | |
|:--|:--|
| error at ``t = 4``, ``h = 1``, ``0.5``, ``0.25`` | ``5.6 \cdot 10^{-11}``, ``6.9 \cdot 10^{-13}``, ``1.8 \cdot 10^{-15}`` |
| ``h = 0.5``, 200 steps: largest energy error | ``3.2 \cdot 10^{-13}`` |
| ``h = 0.5``, 200 steps: largest error in ``q`` | ``5.8 \cdot 10^{-11}`` |
| time per step, ``h = 0.5`` | 12 ms |

The rate between ``h = 1`` and ``0.5`` is 6.3. The error at ``h = 0.25`` is at round-off.

## Limits

- **Cost.** Each step runs nested ForwardDiff and allocates. It is a research method, much slower
  than the network integrators.
- **No framework solver.** The step runs its own Newton solves, so a `solver` keyword passed to
  `GeometricIntegrator` has no effect.
- **Not shown:** symplecticity, behaviour for large ``h`` or stiff problems, and ``S > 6``.
  Stages 2 and 3 hold ``q(1)`` at the cubic's endpoint; a step far from the cubic may need an
  outer loop on ``q(1)``.
