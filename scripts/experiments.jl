# The experiment registry: which problems are run, with which ansätze, at which time steps, and
# with which solver options.
#
# `include`d by every `run_*.jl` driver. Nothing runs at top level, so loading this file is free
# and two drivers can share it without ordering constraints.
#
# `archives.jl` beside it holds the layer that does not need any of this — where output goes, how a
# run is archived, argument parsing, reporting. `figures.jl` includes only that one, and the split
# is by dependency rather than by size: building an ansatz here pulls in Symbolics,
# GeometricIntegrators, GeometricProblems and SimpleSolvers, none of which a script that draws PDFs
# out of plain vectors has any use for.
#
# ---- why this file exists ----------------------------------------------------
#
# The scripts this replaces were three copies of one file (harmonic oscillator, perturbed
# pendulum, Hénon–Heiles), each containing three copies of one 30-line block (h = 1, 2, 5), and
# none of them ran a simulation: they loaded `.jld2` archives produced elsewhere by code that no
# longer exists. Everything that actually varies between the nine VISE runs is a row of
# `VISE_PROBLEMS` and `VISE_STEPS` below, and everything that varies between the NVI runs is a
# row of `NVI_RUNS`.
#
# ---- provenance of the parameters -------------------------------------------
#
# The ansätze, initial weights and quadrature orders were recovered from three places, since no
# single surviving file had all of them. **This file is now the only record of them**: the two it
# was reconstructed from have been deleted, and the third was never in a repository.
#
#   * `scripts/test_vise.jl` (deleted) — the original driver, 96 % commented out. The ansätze, the
#     `init_w` vectors, and the `Backtracking`/`iter1000`/`fabs`/`fsuc` solver settings that the
#     archive filenames encode.
#   * `tem_file.jl` in the talk directory these scripts came from — which comparison integrators
#     each panel carried, and the `Gauss(8)` reference at `h/40`.
#   * the archive filenames themselves, `Backtracking2_R<n>_h<h>_iter1000_fabs<a>_fsuc<s>_TT200`
#     — the per-step quadrature order `R`, which varies with `h` and is not recorded anywhere
#     else. This is why `VISEProblem.quadrature` is a `Dict` from time step to `R` and errors on a
#     step it has no entry for, rather than defaulting.
#
# `Papers/Neural Variational Integrators/document.tex` (§ "Note that similar notation …") is the
# authority for the `S`/`R`/`Q` naming, which `NonlinearIntegrators.network_label` implements.

include(joinpath(@__DIR__, "archives.jl"))

# `relu_k` — type-generic `max(zero(x), x)^k`, never `max(0.0, x)`, which is how three historical
# inline copies of it silently upcast a `Float32` run. Included rather than repeated: the copy here
# used to be justified by not wanting to reach "across a repository boundary", and now that these
# scripts live in the package beside `oga_activations.jl` there is no boundary to reach across.
include(joinpath(@__DIR__, "oga_activations.jl"))

using GeometricIntegrators
using GeometricIntegratorsBase
using GeometricProblems
using GeometricSolutions
using CompactBasisFunctions
using QuadratureRules
using SimpleSolvers
using Symbolics

# ---- naming ------------------------------------------------------------------
#
# `figure_stem`, `window_stem`, `study_stem`, `galerkin_label` and `network_label` are the package's
# own, in `src/plots.jl`, so that the plotting extension can name a figure and a driver can find
# that figure's archive from one definition rather than two that drift.

# ---- solver settings ---------------------------------------------------------
#
# **No `f_abstol` or `f_suctol` here, deliberately.** The archive filenames record
# `fabs`/`fsuc` values of `8eps()` and `2eps()`, and an earlier version of this file carried them
# forward. It should not have: `GeometricIntegratorsBase.default_options` computes
#
#     f_abstol = max(8, solversize(method, problem)) * eps(datatype(problem))
#
# which *scales with the size of the solver system*, because the norm of a residual sitting on its
# round-off floor grows with the number of components it sums. A hand-written `8eps()` is that
# formula's floor, so it is right only for the smallest systems and too tight for everything else —
# and VISE's system grows with the number of weights, Hénon–Heiles carrying eight against the
# harmonic oscillator's three. Pinning it asks a larger system for a residual below its own floor,
# which is how a solve ends up iterating to the cap and reporting stagnation instead of converging.
#
# `default_options` *merges* what it is handed since GeometricIntegratorsBase 0.5, so naming
# `max_iterations` and `linesearch` here keeps the sized `f_abstol` and the `f_stall_window` —
# which is exactly what the original driver's `default_options` override silently discarded.
#
# `Backtracking2` in the filenames is the line search; `max_iterations = 1000` is `iter1000`.
const VISE_SOLVER_OPTIONS = (
    max_iterations = 1000,
    linesearch = SimpleSolvers.Backtracking()
)

# Sub-steps of the reference solution per macro step. Also the spacing the continuous solution
# is recorded on (`record_grid_points = 41` → 40 intervals), so the two grids line up.
const REFERENCE_SUBSTEPS = 40

# ---- the VISE experiments ----------------------------------------------------

"""
    VISEProblem

One symbolic-ansatz experiment: the problem, the ansatz, where Newton starts, and the quadrature
order to use at each time step.

`quadrature` maps a time step to the number of Gauss–Legendre nodes `R`. It is a mapping and not
a constant because the originals varied it per step and the archive filenames are the only record
of how.
"""
struct VISEProblem
    name::String
    label::String
    dimension::Int
    problem::Function          # (timespan, timestep) -> LODEProblem
    hamiltonian::Function      # (t, q, p, params) -> H
    weight_count::Vector{Int}  # weights per degree of freedom
    ansatz::Function           # (W, t) -> Vector{Num}, one expression per dof
    init_w::Vector{Vector{Float64}}
    final_time::Float64
    quadrature::Dict{Float64, Int}
end

const HO = GeometricProblems.HarmonicOscillator
const PPD = GeometricProblems.PerturbedPendulum
const HH = GeometricProblems.HenonHeilesPotential

# Harmonic oscillator. The ansatz `W₁ sin(W₂t + W₃)` *spans* the exact solution
# `0.5 cos(√0.5 t)`, which is what makes this run a check and not just a picture: the only error
# left is the nonlinear solver's residual. `init_w` is the expression a symbolic regression found
# on the first 30 time units, not the analytic optimum — the figure is about extrapolation beyond
# the fitting window, so starting at the exact answer would defeat it.
const VISE_HARMONIC_OSCILLATOR = VISEProblem(
    "harmonic-oscillator",
    "Harmonic oscillator",
    1,
    (timespan, timestep) -> HO.lodeproblem(; timespan = timespan, timestep = timestep),
    HO.hamiltonian,
    [3],
    (W, t) -> [W[1][1] * sin(W[1][2] * t + W[1][3])],
    [[-0.5000433352162222, 0.705350078478666, -1.5678140333370576]],
    200.0,
    Dict(1.0 => 8, 2.0 => 16, 5.0 => 8, 10.0 => 16)
)

# Perturbed pendulum: a non-separable Hamiltonian, so the same three-parameter ansatz is now an
# approximation rather than exact. `H(q,p) = p²/2 - ω²cos q - q p A(ϵ,ϕ)` with the package
# defaults ω = 0.5, ϵ = 0.5, ϕ = π/3.
const VISE_PERTURBED_PENDULUM = VISEProblem(
    "perturbed-pendulum",
    "Perturbed pendulum",
    1,
    (timespan, timestep) -> PPD.lodeproblem(; timespan = timespan, timestep = timestep),
    PPD.hamiltonian,
    [3],
    (W, t) -> [W[1][1] * cos(W[1][2] * t + W[1][3])],
    [[-0.51941, -0.47405, 2.8713]],
    200.0,
    Dict(1.0 => 8, 2.0 => 16, 5.0 => 16, 10.0 => 16)
)

# Hénon–Heiles: two degrees of freedom, four weights each — the constant term matters because
# neither coordinate oscillates about zero. Initial conditions `[0.1, 0.1]`, `[0.1, 0.1]` rather
# than the package defaults, as in the original driver. `T = 60` and not 200: beyond that the
# trajectory leaves the region the ansatz was fitted in and the panel is unreadable.
const VISE_HENON_HEILES = VISEProblem(
    "henon-heiles",
    "Hénon–Heiles",
    2,
    (timespan, timestep) -> HH.lodeproblem([0.1, 0.1], [0.1, 0.1];
        timespan = timespan, timestep = timestep),
    HH.hamiltonian,
    [4, 4],
    (W, t) -> [W[d][1] * cos(W[d][2] * t + W[d][3]) + W[d][4] for d in 1:2],
    [[0.14831, 1.0, -0.64812, -0.018712],
        [0.14298, -0.97215, 0.7615, -0.0013983]],
    60.0,
    Dict(1.0 => 16, 2.0 => 16, 5.0 => 16)
)

# The basis an ansatz search found for this problem: the odd harmonics of one free fundamental.
# (That search is a talk's figure script and did not move here with the harness; `basis_fits.jl`
# beside this file is the fitting machinery it used.) The orbit of a one-degree-of-freedom autonomous system librating in a well is
# exactly periodic, and the potential is symmetric, so only the odd harmonics carry anything —
# fitted globally over `t ∈ [0,1000]` the first three of them reach `1e-8` with eight numbers.
# `VISE_PERTURBED_PENDULUM`'s ansatz is the first term of exactly this series, which is what makes
# the two comparable: one extra weight, same family, and the convergence cell shows what it buys.
#
# `init_w` extends the existing seed by the third-harmonic amplitude the global fit found,
# `A₃/A₁ ≈ 1.5e-3` — the residual of the one-term fit, which is what that ratio is.
const VISE_PERTURBED_PENDULUM_ODD = VISEProblem(
    "perturbed-pendulum-odd",
    "Perturbed pendulum",
    1,
    (timespan, timestep) -> PPD.lodeproblem(; timespan = timespan, timestep = timestep),
    PPD.hamiltonian,
    [4],
    (W, t) -> [W[1][1] * cos(W[1][2] * t + W[1][3]) +
               W[1][4] * cos(3 * (W[1][2] * t + W[1][3]))],
    [[-0.51941, -0.47405, 2.8713, -7.8e-4]],
    200.0,
    Dict(1.0 => 8, 2.0 => 16, 5.0 => 16, 10.0 => 16)
)

# Ansätze the convergence study carries *in addition* to the problem's own, so that the earlier
# curves stay in the figure as the reference they are.
const CONVERGENCE_VISE_EXTRA = Dict{String, Vector{Pair{String, VISEProblem}}}(
    "perturbed-pendulum" => ["odd-harmonic" => VISE_PERTURBED_PENDULUM_ODD]
)

const VISE_PROBLEMS = (VISE_HARMONIC_OSCILLATOR, VISE_PERTURBED_PENDULUM, VISE_HENON_HEILES)

# The shared ladder every method runs on.
const VISE_STEPS = (1.0, 2.0, 5.0)

# `h = 10` in addition, for the two problems where it was measured to work. VISE at `h = 10` with
# `R = 16` reaches `1.33e-13` on the harmonic oscillator — *better* than `h = 5` with `R = 8`
# (`1.03e-8`), because the ansatz there is exact and the quadrature order, not the step, is what
# limits it. On the perturbed pendulum it reaches `4.23e-5`.
#
# Hénon–Heiles is excluded: `T = 60` at `h = 10` is six steps, which is not a trajectory.
const VISE_EXTRA_STEPS = Dict(
    "harmonic-oscillator" => (10.0,),
    "perturbed-pendulum" => (10.0,)
)

function vise_steps(experiment::VISEProblem)
    Tuple(sort(unique(vcat(collect(VISE_STEPS),
        collect(get(VISE_EXTRA_STEPS, experiment.name, ()))))))
end

# `findfirst` returns `nothing` for a name that is not in the table, and indexing with that throws
# a bare `MethodError` naming neither the argument nor the alternatives. Four call sites in this
# harness had that shape; a mistyped problem name is the most likely thing a caller does wrong, so
# it gets the list.
function vise_problem(name)
    index = findfirst(p -> p.name == name, VISE_PROBLEMS)
    index === nothing && throw(ArgumentError(
        "no VISE problem named `$(name)`; this study has " *
        join((p.name for p in VISE_PROBLEMS), ", ")))
    VISE_PROBLEMS[index]
end

"""
    build_vise_method(experiment, timestep) -> VISE

The integrator for one `(experiment, timestep)` pair.

The symbolic variables are created here rather than stored on the `VISEProblem` because
`Symbolics.@variables` and `VISEBasis`'s `build_function` are the expensive part of construction
and there is no reason to keep the intermediate expressions alive between runs.
"""
function build_vise_method(experiment::VISEProblem, timestep::Float64)
    haskey(experiment.quadrature, timestep) ||
        error("no quadrature order recorded for $(experiment.name) at h = $(timestep); " *
              "the surviving record of R is the archive filenames, so a new step size needs a " *
              "deliberate choice rather than a default.")

    t = only(@variables tvar)
    W = [only(@variables($(Symbol(:W, d))[1:n]))
         for (d, n) in enumerate(experiment.weight_count)]
    exprs = experiment.ansatz(W, t)

    basis = VISEBasis{Float64}(exprs, W, t, experiment.dimension)
    quadrature = QuadratureRules.GaussLegendreQuadrature(experiment.quadrature[timestep])

    VISE(basis, quadrature, experiment.init_w)
end

# ---- the NVI (neural variational integrator) experiments --------------------

"""
    NVIRun

One neural-variational-integrator run.

`S` is the number of hidden neurons, `R` the number of Gauss–Legendre quadrature nodes, and
`Q = 2R` the quadrature *order* — a label only, never a constructor argument. This is the
`SsRrQuσ` convention of the paper, and getting it wrong is how the figure
`S6R10Q16tanh_h2.0.pdf` came to be mislabelled: `R = 10` means `Q = 20`.
"""
struct NVIRun
    problem::String           # "harmonic-oscillator" | "double-pendulum"
    architecture::Symbol      # :shallow | :dense
    S::Int
    S₁::Int                   # first hidden layer; `:dense` only, 0 for `:shallow`
    R::Int
    activation_name::String
    activation::Function
    timestep::Float64
    final_time::Float64
    # Windows for the "same run, several time intervals" figures. Empty means one figure over the
    # whole run.
    windows::Vector{Float64}
end

nvi_order(run::NVIRun) = 2 * run.R

# `SsRrQuσ` for a shallow net, `DenseS₁xSRrQuσ` for a dense one — `NonlinearIntegrators`' own
# `network_label`, which is also what the convergence study and the OGA seed study use. There were
# four inline copies of this format string before, and the one thing they have to agree on is that
# `Q = 2R`: a legend that says `Q16` at `R = 10` is the mislabelling one published figure carried.
function nvi_label(run::NVIRun)
    network_label(run.S, run.R, run.activation_name;
        S₁ = run.architecture === :dense ? run.S₁ : nothing)
end

nvi_stem(run::NVIRun) = figure_stem(run.problem, nvi_label(run), run.timestep)

const NVI_DICT_AMOUNT = 4000

# `T = 1000` for the harmonic oscillator, read off the original figures (their x axis runs to
# 1000); `T = 40` for the double pendulum, from the script that made `nn_Double_Pendulum.png`.
# How each problem is named in a figure title.
const NVI_PROBLEM_LABELS = Dict(
    "harmonic-oscillator" => "Harmonic oscillator",
    "perturbed-pendulum" => "Perturbed pendulum",
    "henon-heiles" => "Hénon–Heiles",
    "double-pendulum" => "Double pendulum"
)

const NVI_FINAL_TIME = Dict(
    "harmonic-oscillator" => 1000.0,
    "perturbed-pendulum" => 200.0,   # as the VISE runs of the same problem
    "henon-heiles" => 60.0,          # as the VISE runs; beyond that the trajectory is unreadable
    "double-pendulum" => 40.0
)

# The two network configurations the deck shows, each now run at **every** step in `VISE_STEPS`,
# so the neural and the symbolic integrators are compared at the same time steps. The originals
# were at mixed steps — ReLU³ at h ∈ {1, 2} and tanh at h ∈ {2, 4} — which made the two families
# not directly comparable and is why they are uniform here.
const NVI_CONFIGURATIONS = (
    (problem = "harmonic-oscillator", architecture = :shallow, S = 4, S₁ = 0, R = 8,
        name = "relu3", σ = relu_k(3)),
    (problem = "harmonic-oscillator", architecture = :shallow, S = 6, S₁ = 0, R = 10,
        name = "tanh", σ = tanh),
    # The perturbed pendulum, measured to work at every step including `h = 10`: ReLU³ reaches
    # `1.26e-08` at `h = 1` — three orders better than the symbolic ansatz's `1.16e-05` on the same
    # problem and step.
    (problem = "perturbed-pendulum", architecture = :shallow, S = 4, S₁ = 0, R = 8,
        name = "relu3", σ = relu_k(3)),
    (problem = "perturbed-pendulum", architecture = :shallow, S = 6, S₁ = 0, R = 10,
        name = "tanh", σ = tanh),
    # Hénon–Heiles, where the network ansatz succeeds and the three-term symbolic one does not:
    # `8.37e-04` (ReLU³) and `3.97e-04` (tanh) at `h = 1`, against VISE's `3.91e-01`. Two degrees of
    # freedom, so `ShallowNet` carries a network per coordinate.
    (problem = "henon-heiles", architecture = :shallow, S = 4, S₁ = 0, R = 8,
        name = "relu3", σ = relu_k(3)),
    (problem = "henon-heiles", architecture = :shallow, S = 6, S₁ = 0, R = 10,
        name = "tanh", σ = tanh),
    # The dense network — two hidden layers, `S₁` then `S`. This is a different integrator
    # (`DenseNet`, seeded by `LSGD`/`TrainingMethod` rather than the orthogonal greedy algorithm),
    # and it is here because two figures in `results/` were made with it:
    #
    #   NVI_Densefabs1.78e-15_fsuc1.78e-15_iter1000_h1.0_R4tanh_harmonic_oscillator.pdf
    #   NVI_Densefabs4.44e-16_fsuc4.44e-16_iter10000_h5.0_R24tanh_harmonic_oscillator.pdf
    #
    # `S₁ = S = 5` is from the deleted `scripts/test_densenet.jl`; the filenames record
    # only the tolerances, the iteration cap, `h` and `R`, and the legends only `R`. `R` is
    # therefore per step here — 4 at `h = 1`, 24 at `h = 5` — see `DENSE_QUADRATURE`.
    (problem = "harmonic-oscillator", architecture = :dense, S = 5, S₁ = 5, R = 0,
        name = "tanh", σ = tanh),
    # The double pendulum. `S = 8`, `R = 8`, tanh were read out of the commented block of a
    # `results/test_NonLinear_OneLayer_GML.jl` that produced the shipped `nn_Double_Pendulum.png`.
    # That file is in `results/`, which is git-ignored, so it exists on one machine and in no
    # repository: **these three numbers are the record, not a citation of one.**
    #
    # **It cannot run at h ∈ {1, 2, 5} and this is a property of the problem.** At these initial
    # conditions the double-pendulum LODE is singular for the large-step solves: `ImplicitMidpoint`
    # and `Gauss(8)` both fail with a `SingularException` at every `h ≥ 0.5`, and the network —
    # whose warm start goes through implicit midpoint — completes two steps at `h = 2.0` and then
    # fails the same way for any `T ≥ 10`. Measured at `T = 40`: `h = 1.0` fails with a
    # `DomainError`, `h = 0.5` completes all 80 steps at `max |ΔH/H₀| = 4.3e-2`, `h = 0.25` all 160
    # at `5.0e-3`.
    #
    # So it keeps its own step, named in `NVI_STEP_OVERRIDES`, and `run_nvi.jl` says so when it
    # runs. The original settings are not recoverable in any case: that block was already commented
    # out, and the file's own re-render does not match the shipped PNG.
    (problem = "double-pendulum", architecture = :shallow, S = 8, S₁ = 0, R = 8,
        name = "tanh", σ = tanh),
    # `S6R6Q12tanh` at `h = 4`: the run behind `harmonic-oscillator-nvi-{100,500,2000}.pdf`, which
    # were one run plotted over three growing intervals. `T = 2000` and `S = R = 6` are from the
    # archive `nvi/nvi_h4.0_tanh_S6R6_HO.jld2` those figures were drawn from.
    (problem = "harmonic-oscillator", architecture = :shallow, S = 6, S₁ = 0, R = 6,
        name = "tanh", σ = tanh),
    # `S4R4Q8relu3`: the run behind the old `nn_harmonic_oscillator.png`, whose frame pairs one
    # harmonic-oscillator run with one double-pendulum run. Its `max |ΔH/H₀|` of 3.136e-2 at `h = 4`
    # is the number visible in that figure (0.032), which is how the settings were confirmed.
    (problem = "harmonic-oscillator", architecture = :shallow, S = 4, S₁ = 0, R = 4,
        name = "relu3", σ = relu_k(3))
)

# `R` for the dense-net runs, per time step. 4 at `h = 1` and 24 at `h = 5` are the two the original
# figures record; **8 at `h = 2` is a choice**, filling the gap between them, since nothing recorded
# says what a dense run at `h = 2` used.
const DENSE_QUADRATURE = Dict(1.0 => 4, 2.0 => 8, 5.0 => 24)

# Extra steps a configuration runs at *beyond* `VISE_STEPS`, because a published figure used them.
# `h = 4` for `S6R10Q20tanh` is `S6R10Q16tanh_h4.0.pdf` — under the mislabelled name; `Q = 2R = 20`.
const NVI_EXTRA_STEPS = Dict(
    ("harmonic-oscillator", 6, 10, "tanh") => (4.0,),
    # `h = 4` is the step the old `nn_harmonic_oscillator.png` was run at.
    ("harmonic-oscillator", 4, 4, "relu3") => (4.0,),
    # `h = 10` for the perturbed pendulum, where it works: `2.43e-02` (ReLU³) and `1.64e-03` (tanh).
    #
    # **Not** for the harmonic oscillator, where it does not. Measured at `T = 200`: ReLU³ diverges to
    # `9.69e+25` and tanh reaches `8.76e-01`, i.e. no trajectory at all — while VISE at the same step
    # reaches `1.33e-13` and `CGVI(8)` `6.80e-04`. So `h = 10` is a step the *symbolic and polynomial*
    # integrators handle on this problem and the networks do not, which is a result rather than a
    # reason to include a diverged curve in a figure.
    ("perturbed-pendulum", 4, 8, "relu3") => (10.0,),
    ("perturbed-pendulum", 6, 10, "tanh") => (10.0,)
)

# Per-configuration final times, where a run needs one other than `NVI_FINAL_TIME`. The
# `S6R6Q12tanh` run goes to `T = 2000` because its figures do.
const NVI_FINAL_TIME_OVERRIDES = Dict(
    ("harmonic-oscillator", 6, 6, "tanh") => 2000.0,
    # The dense-net figures' x axis runs to 100, not 1000. Keyed with `R = 0` because the dense
    # configuration's `R` is per step (see `DENSE_QUADRATURE`) and the entry names the configuration.
    ("harmonic-oscillator", 5, 0, "tanh") => 100.0,
    # The old `nn_harmonic_oscillator.png` runs to 100.
    ("harmonic-oscillator", 4, 4, "relu3") => 100.0
)

# Windows for the several-figures-of-one-run case: one run, plotted over three growing intervals, the
# point being that a trajectory which looks perfect over `t ∈ [0,100]` has visibly drifted by
# `t = 2000`. The originals (`harmonic-oscillator-nvi-{100,500,2000}.pdf`) were an `S6R6Q12tanh` run
# at `h = 4`; this is the same configuration at `h = 5`, which is on the shared `VISE_STEPS` ladder
# and so comparable with everything else, rather than a step of its own.
const NVI_WINDOWS = Dict(
    ("harmonic-oscillator", 6, 6, "tanh", 5.0) => [100.0, 500.0, 2000.0]
)

# A configuration that cannot run at `VISE_STEPS` names the steps it can. Keyed by
# `(problem, S, R, activation)`.
const NVI_STEP_OVERRIDES = Dict(
    ("double-pendulum", 8, 8, "tanh") => (0.5,)
)

function nvi_steps(c)
    key = (c.problem, c.S, c.R, c.name)
    haskey(NVI_STEP_OVERRIDES, key) && return NVI_STEP_OVERRIDES[key]
    Tuple(sort(unique(vcat(collect(VISE_STEPS),
        collect(get(NVI_EXTRA_STEPS, key, ()))))))
end

function nvi_final_time(c)
    get(NVI_FINAL_TIME_OVERRIDES, (c.problem, c.S, c.R, c.name),
        NVI_FINAL_TIME[c.problem])
end

nvi_windows(c, h) = get(NVI_WINDOWS, (c.problem, c.S, c.R, c.name, h), Float64[])

# `R` is per-step for the dense nets and constant for the shallow ones.
nvi_nodes(c, h) = c.architecture === :dense ? DENSE_QUADRATURE[h] : c.R

const NVI_RUNS = Tuple(NVIRun(c.problem, c.architecture, c.S, c.S₁, nvi_nodes(c, h),
                           c.name, c.σ, h, nvi_final_time(c), nvi_windows(c, h))
for c in NVI_CONFIGURATIONS
for h in nvi_steps(c))

const DP = GeometricProblems.DoublePendulum

# The double-pendulum initial conditions, read out of the commented block of a
# `results/test_NonLinear_OneLayer_GML.jl` that made the shipped `nn_Double_Pendulum.png`
# (verified at the time: its harmonic-oscillator sibling reproduced a byte-identical PNG). As
# above, `results/` is git-ignored, so these two vectors are the only surviving record of them —
# do not "restore them from the original" if they are ever doubted, because there is no original
# in any repository to restore from.
const DOUBLE_PENDULUM_Q₀ = [0.7853981633974483, 1.5707963267948966]
const DOUBLE_PENDULUM_P₀ = [0.2776801836348979, 0.39269908169872414]

function nvi_problem(run::NVIRun, timestep = run.timestep)
    timespan = (0.0, run.final_time)
    if run.problem == "harmonic-oscillator"
        return HO.lodeproblem(; timespan = timespan, timestep = timestep)
    elseif run.problem == "perturbed-pendulum"
        return PPD.lodeproblem(; timespan = timespan, timestep = timestep)
    elseif run.problem == "henon-heiles"
        return HH.lodeproblem([0.1, 0.1], [0.1, 0.1];
            timespan = timespan, timestep = timestep)
    elseif run.problem == "double-pendulum"
        return DP.lodeproblem(DOUBLE_PENDULUM_Q₀, DOUBLE_PENDULUM_P₀;
            timespan = timespan, timestep = timestep)
    end
    error("unknown problem $(run.problem)")
end

function nvi_hamiltonian(run::NVIRun)
    run.problem == "harmonic-oscillator" ? HO.hamiltonian :
    run.problem == "perturbed-pendulum" ? PPD.hamiltonian :
    run.problem == "henon-heiles" ? HH.hamiltonian :
    run.problem == "double-pendulum" ? DP.hamiltonian :
    error("unknown problem $(run.problem)")
end

nvi_dimension(run::NVIRun) = run.problem in ("double-pendulum", "henon-heiles") ? 2 : 1

"""
    build_nvi_method(run) -> ShallowNet

The shallow-network integrator for one run. `dict_amount` is [`NVI_DICT_AMOUNT`](@ref) and not
the 400 000 the historical scripts passed: the package's own accuracy test records that "a
`dict_amount` of 400 000 only slows the seed build without improving accuracy", and the
difference here is minutes per run.
"""
function build_nvi_method(run::NVIRun)
    quadrature = QuadratureRules.GaussLegendreQuadrature(run.R)

    if run.architecture === :dense
        # `DenseNet` takes neither `bias_interval` nor `dict_amount`: it has no orthogonal-greedy
        # seed, and is initialised by training instead. `TrainingMethod()` is what the deleted
        # `scripts/test_densenet.jl` used, against the `LSGD()` default.
        #
        # The package's own test suite says of this integrator that "its Training/LSGD
        # initial-guess methods are not stable enough for an accuracy guard", so its rows assert
        # dispatch and finiteness only. Expect it to be the least reliable run here.
        net = DenseNetBasis{Float64}(run.activation, run.S₁, run.S)
        return DenseNet(net, quadrature;
            show_status = false,
            initial_guess_method = TrainingMethod())
    end

    net = ShallowNetBasis{Float64}(run.activation, run.S)
    ShallowNet(net, quadrature;
        show_status = false,
        bias_interval = [-pi, pi],
        dict_amount = NVI_DICT_AMOUNT)
end

const NVI_SOLVER_OPTIONS = (
    regularization_factor = 1e-5,
    max_iterations = 1000
)

# ---- the OGA seed study ------------------------------------------------------
#
# Does the choice of orthogonal-greedy seed change what the network integrator converges to?
#
# The seed is `initial_guess_method`, and it is not a detail: the discrete Euler–Lagrange equations
# of a network ansatz are solved by Newton from *this* starting point, so a seed decides which
# solution of a non-convex problem the step lands in — and whether it lands at all. `ShallowNet`
# defaults to `OGA1d()`, and every historical script here took that default without ever comparing
# it. `run_oga_seeds.jl` compares them.
#
# The six variants differ along three axes, which `oga_label` prints as
# `dictionary/selection/fit`:
#
#   * the **dictionary** searched — a bias grid at fixed weight (`BiasGrid1d`), a weight-and-bias
#     grid (`WeightBiasGrid2d`), or angles on a sphere (`AngularGrid`);
#   * the **selection** rule — the raw inner product against the residual, or the normalised one;
#   * the **fit** — a weighted QR, or the normal equations.
const OGA_VARIANTS = (
    ("OGA1d", OGA1d),
    ("OGA1dNormalized", OGA1dNormalized),
    ("OGA1dStable", OGA1dStable),
    ("OGA2d", OGA2d),
    ("OGASphere", OGASphere),
    ("OGA1dNormalEquations", OGA1dNormalEquations)
)

# The configuration the seed study varies the seed *of*. One configuration and not all of them: the
# question is whether the seed matters, and answering it on the deck's own ReLU³ run is what makes
# the answer apply to the figures.
const OGA_STUDY_CONFIGURATION = (problem = "harmonic-oscillator", S = 4, R = 8,
    name = "relu3", σ = relu_k(3))

# Long enough for a drift to show, short enough to run six seeds over four steps. The convergence
# ladder is the same one the `nvi-hamiltonian-*` study uses, truncated at the top: `h = 5` is in
# `VISE_STEPS` and belongs here for comparability.
const OGA_STUDY_FINAL_TIME = 100.0
const OGA_STUDY_STEPS = (0.25, 0.5, 1.0, 2.0, 5.0)

# ---- the convergence study ---------------------------------------------------
#
# `nvi-hamiltonian-tanh.pdf` and `nvi-hamiltonian-relu3.pdf`: the maximum relative Hamiltonian
# error against the time step, for several neural configurations and the polynomial Galerkin
# integrators they are compared to, with `h²` and `h³` reference slopes.
#
# The configurations are read off the legends of the original figures. What is *not* recorded
# anywhere is the final time those runs used, and it matters: the plotted quantity is a maximum
# over the run, so a longer run can only push it up. `CONVERGENCE_FINAL_TIME` is therefore a
# deliberate choice and not a reconstruction. `100` is what the package's own benchmarks use;
# `--final-time` prints what it used, and anything but the default has to be said in the caption,
# because two curves computed over different windows are not comparable.
const CONVERGENCE_FINAL_TIME = 100.0

const CONVERGENCE_STEPS = (0.03125, 0.0625, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)

# The polynomial Galerkin family, indexed by the number of quadrature nodes `R` alone — which is
# the only free choice: the basis is Lagrange on those same `R` Gauss nodes, so the polynomial
# degree is `R - 1` and the quadrature order `2R`. Labelled `CGVI(R)` by `galerkin_label` above.
const GALERKIN_ORDERS = (2, 3, 4)

convergence_label(S, R, activation_name) = network_label(S, R, activation_name)

# ---- the convergence matrix ---------------------------------------------------
#
# Two problems × three integrators, one figure per cell:
#
#            │ CGVI              NVI                        VISE
#   ─────────┼─────────────────────────────────────────────────────────────────
#   harmonic │ R ∈ {2,3,4}       tanh and ReLU³ families    the symbolic ansatz
#   pendulum │ R ∈ {2,3,4}       tanh and ReLU³ families    the symbolic ansatz
#
# The `cgvi` cell is the reference family on its own axes; the `nvi` and `vise` cells each carry it
# again, dashed, because a convergence claim about a nonlinear method is only readable against the
# linear one at the same quadrature order.
#
# What the two problems buy: on the harmonic oscillator the VISE ansatz *spans* the exact solution,
# so its curve is a flat line at the solver's residual floor and says nothing about order. The
# perturbed pendulum is the case where the same ansatz is an approximation, so it is the one where
# VISE has an order to measure at all.

struct ConvergenceProblem
    name::String
    label::String
    problem::Function              # (timespan, timestep) -> LODEProblem
    hamiltonian::Function
    vise::VISEProblem              # the symbolic ansatz for this problem
end

const CONVERGENCE_PROBLEMS = (
    ConvergenceProblem("harmonic-oscillator", "harmonic oscillator",
        (timespan, timestep) -> HO.lodeproblem(; timespan = timespan, timestep = timestep),
        HO.hamiltonian, VISE_HARMONIC_OSCILLATOR),
    ConvergenceProblem("perturbed-pendulum", "perturbed pendulum",
        (timespan, timestep) -> PPD.lodeproblem(; timespan = timespan, timestep = timestep),
        PPD.hamiltonian, VISE_PERTURBED_PENDULUM)
)

# The neural configurations, by activation. Read off the legends of the original figures.
const CONVERGENCE_NVI = (
    (name = "tanh", σ = tanh, configurations = [(6, 10), (6, 8), (6, 6)]),
    (name = "relu3", σ = relu_k(3),
        configurations = [(4, 10), (4, 8), (4, 6), (4, 4), (5, 8), (5, 4), (6, 4)])
)

# The quadrature orders VISE is run at in the convergence study. Constant across the ladder, unlike
# `VISEProblem.quadrature`, which records what the *original* per-step runs used: a convergence study
# has to hold everything but `h` fixed, or it is measuring two things at once.
const CONVERGENCE_VISE_ORDERS = (8, 16)

const CONVERGENCE_INTEGRATORS = ("cgvi", "nvi", "vise")

# ---- diagnostics -------------------------------------------------------------
#
# Two of these are the package's own, in `src/plots.jl`, because they are not specific to any
# experiment and each existed here in a second copy:
#
#   `relative_invariant_error(sol, hamiltonian, parameters)`  — was `relative_hamiltonian_error`
#   `coarse_grid_error(sol, ref_sol, substeps)`               — normalises by the maximum over the
#       *whole* reference rather than per step, which is what keeps an oscillator's zero crossings
#       from reporting a bounded absolute error as an arbitrarily large relative one.
#
# What stays here needs `GeometricIntegrators` — `Gauss`, `CGVI`, `Lagrange` — which is deliberately
# not a dependency of the package.

"""
    reference_solution(problem_fn, final_time, timestep; substeps = REFERENCE_SUBSTEPS)

A `Gauss(8)` solution on a grid `substeps` times finer than the macro step — the reference every
panel is measured against. Eighth order at `h/40` is far below the accuracy any of the compared
integrators reach, which is what lets it stand in for the exact solution where none is known.
"""
function reference_solution(problem_fn, final_time, timestep; substeps = REFERENCE_SUBSTEPS)
    integrate(problem_fn((0.0, final_time), timestep / substeps), Gauss(8))
end

"""
    galerkin_method(R) -> CGVI

The polynomial Galerkin variational integrator on a Lagrange basis at `R` Gauss–Legendre nodes,
with the matching `R`-node quadrature. This is `PpRrQu` of the paper with `p = R - 1`, `r = R`,
`u = 2R`.
"""
function galerkin_method(R::Int)
    quadrature = QuadratureRules.GaussLegendreQuadrature(R)
    CGVI(Lagrange(QuadratureRules.nodes(quadrature)), quadrature)
end

# The archive layer — `archive_path`, `store_run!`, `load_run`, `load_runs`, `solution_data` — and
# the reporting helpers are in `archives.jl`, included at the top of this file. They are what
# `figures.jl` needs without any of the machinery above it.
