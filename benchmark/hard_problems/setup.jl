# Problems, reference solutions and invariants of the hard-problems benchmark (PLAN.md §1, §3).
#
# Only builds on GeometricProblems; no problem is redefined here. P2 (charged particle) is
# skipped for now, see the Obsidian note "GeometricProblems/Phase 0 follow-ups".
#
# A case is a NamedTuple
#   name        file stem of the results, e.g. "P1_eps0.01"
#   problem     "P1", "P3", "P4"
#   quick       part of the quick preset
#   Ω           time-scale normalisation: h = c / Ω for c ∈ STEP_FACTORS (ω ≈ 1 for P1 and P4,
#               the mean motion for P3, as in the plan)
#   T, Terr     final time, and the final time of the q error (Terr ≤ T)
#   build       (timespan, timestep) -> LODEProblem
#   reference   :gauss (Gauss(8) with dt = h_min / 50 and dt / 2; the latter is the reference,
#               their difference is reported as `selfcheck`) or
#               (t -> q(t)) for an exact solution
#   invariants  names and functions (t, q, p, params) -> value, each reported as
#               max_n |I_n - I_0| / |I_0|, except the angles in `angles`, reported as
#               max_n |I_n - I_0| (wrapped to (-π, π])

using GeometricIntegrators
using GeometricProblems
using GeometricSolutions

import GeometricProblems.FrequencyModulatedOscillator as fmo
import GeometricProblems.HenonHeilesPotential as hh
import GeometricProblems.KeplerProblem as kepler

const STEP_FACTORS = [0.1, 0.3, 1.0, 3.0, 10.0]   # h Ω
const REF_FACTOR = 50                             # dt_ref = h_min / REF_FACTOR

# ---- P1: frequency-modulated oscillator --------------------------------------------------------

function p1_case(ε; quick)
    params = merge(fmo.default_parameters(), (ε = ε,))
    (name = "P1_eps$(ε)", problem = "P1", quick = quick, Ω = 1.0, T = 2π / ε, Terr = 2π / ε,
        build = (ts, h) -> fmo.lodeproblem([1.0], [0.0]; timespan = ts, timestep = h,
            parameters = params),
        reference = :gauss,
        invariants = (J = fmo.adiabatic_invariant,), angles = ())
end

# ---- P3: Kepler problem ------------------------------------------------------------------------

kepler_perihelion(t, q, p, params) = (R = kepler.runge_lenz_vector(t, q, p, params); atan(R[2], R[1]))

function p3_case(e; quick)
    q₀, p₀ = kepler.initial_condition(e)
    params = kepler.default_parameters()
    (name = "P3_e$(e)", problem = "P3", quick = quick, Ω = 1.0, T = 20π, Terr = 20π,
        build = (ts, h) -> kepler.lodeproblem(q₀, p₀; timespan = ts, timestep = h,
            parameters = params),
        reference = t -> first(kepler.exact_solution(t, 0.0, q₀, p₀, params)),
        invariants = (H = kepler.hamiltonian, L = kepler.angular_momentum,
            ϖ = kepler_perihelion),
        angles = (:ϖ,))
end

# ---- P4: Hénon–Heiles --------------------------------------------------------------------------

function p4_case(label, q₀, p₀; quick)
    (name = "P4_$(label)", problem = "P4", quick = quick, Ω = 1.0, T = 1000.0, Terr = 100.0,
        build = (ts, h) -> hh.lodeproblem(q₀, p₀; timespan = ts, timestep = h,
            parameters = hh.default_parameters()),
        reference = :gauss,
        invariants = (H = hh.hamiltonian,), angles = ())
end

const CASES = [
    p1_case(1E-2; quick = true),
    p1_case(1E-3; quick = false),
    p3_case(0.9; quick = true),
    p3_case(0.99; quick = false),
    p4_case("E0.02", [0.1, 0.1], [0.1, 0.1]; quick = true),
    p4_case("E0.135", [0.2, 0.2], [0.3, 0.3]; quick = false)
]

step_sizes(case) = STEP_FACTORS ./ case.Ω

# ---- solution helpers --------------------------------------------------------------------------

qmatrix(sol) = reduce(hcat, collect.(collect(sol.q[:])))   # D × (N + 1)
pmatrix(sol) = reduce(hcat, collect.(collect(sol.p[:])))
times(sol) = collect(sol.t[:])

relerr(Q, Qref) = maximum(abs, Q .- Qref) / maximum(abs, Qref)

wrap(x) = rem2pi(x, RoundNearest)

function invariant_errors(case, t, Q, P, params)
    map(keys(case.invariants)) do k
        I = [case.invariants[k](t[n], Q[:, n], P[:, n], params) for n in eachindex(t)]
        k in case.angles ? maximum(abs.(wrap.(I .- I[1]))) : maximum(abs.(I .- I[1])) / abs(I[1])
    end
end

# ---- reference solutions -----------------------------------------------------------------------

"""
    Reference(case)

Positions of the reference solution of `case` on the interval (0, Terr): the Gauss(8) solution
with dt_ref / 2, sampled on the grid n dt_ref, and the relative difference `selfcheck` to the
Gauss(8) solution with dt_ref (zero for an exact solution).
"""
struct Reference{RT}
    dt::Float64
    Q::Matrix{Float64}          # on the grid n dt, empty for an exact solution
    exact::RT
    selfcheck::Float64
end

function Reference(case)
    case.reference === :gauss || return Reference(0.0, zeros(0, 0), case.reference, 0.0)
    dt = minimum(step_sizes(case)) / REF_FACTOR
    Q₁ = qmatrix(integrate(case.build((0.0, case.Terr), dt), Gauss(8)))
    Q₂ = qmatrix(integrate(case.build((0.0, case.Terr), dt / 2), Gauss(8)))[:, 1:2:end]
    n = min(size(Q₁, 2), size(Q₂, 2))
    Reference(dt, Q₂[:, 1:n], nothing, relerr(Q₁[:, 1:n], Q₂[:, 1:n]))
end

"Reference positions at the times `t` (all multiples of `ref.dt` for a Gauss reference)."
function (ref::Reference)(t)
    ref.exact === nothing || return reduce(hcat, ref.exact.(t))
    idx = round.(Int, t ./ ref.dt)
    all(abs.(idx .* ref.dt .- t) .< 1E-9 * max(1, maximum(t))) ||
        error("times are not on the reference grid dt = $(ref.dt)")
    ref.Q[:, idx .+ 1]
end
