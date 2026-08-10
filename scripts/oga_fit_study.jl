# Tier A: OGA *seed quality*, measured without an integrator.
#
# This calls `oga_fit` directly — no Newton solve, no time stepping, no variational
# equations. That separation is the point. End-to-end convergence conflates the quality of
# the seed with the behaviour of the solve, and the resulting confound is exactly what made
# the reduced-precision failures hard to attribute: a run that fails looks the same whether
# the greedy fit went rank-deficient or the Jacobian did.
#
# What is measured, per (dictionary × selection × fit × activation × precision × target):
#
#   * `fit_err`   — the quadrature-weighted L² fit error, recomputed in `Float64` from the
#                   returned parameters, so precisions are comparable on one scale.
#   * `cond`      — condition number of the seed's design matrix, in `Float64`. This is the
#                   proxy for whether the Newton system it feeds is solvable.
#   * `sigma_min` — its smallest singular value: how close the seed is to rank-deficient.
#   * `neurons`   — how many of the requested `S` the greedy loop could actually place.
#   * `rejected`  — candidate atoms refused for adding no new direction.
#
# `cond`/`sigma_min`/`fit_err` are computed in `Float64` *after* the fit returns, purely for
# reporting; nothing derived from them re-enters the fit. Every case is an `S ≤ 8`,
# `nnodes = 11` problem, so the whole grid runs in seconds.
#
# Run with:
#   julia --project=scripts scripts/oga_fit_study.jl

using NonlinearIntegrators
using LinearAlgebra
using Printf

const NI = NonlinearIntegrators

include(joinpath(@__DIR__, "oga_activations.jl"))
include(joinpath(@__DIR__, "oga_report.jl"))

const S_NEURONS = 4
const NNODES = 10                # ⇒ 11 nodes, matching the integrators' default
const DICT_AMOUNT = 400
const BIAS = (-pi, pi)

# ---- the axes ----------------------------------------------------------------

# `bias_amount` is reduced on the 2-D and angular grids so the total dictionary size stays
# comparable to the 1-D one: the greedy step is linear in the number of atoms, so a fair
# comparison holds that count roughly fixed rather than handing the richer dictionaries a
# larger budget.
const DICTIONARIES = [
    ("grid1d",   BiasGrid1d()),
    ("grid2d",   WeightBiasGrid2d(octaves = (-3, 3), weight_amount = 6, bias_amount = 56)),
    ("angular",  AngularGrid(radii = (0.25, 1.0, 4.0), amount = 266)),
    ("refined",  Refined(BiasGrid1d(); iterations = 4)),
]

const SELECTIONS = [
    ("raw",        RawProjection()),
    ("normalized", NormalizedProjection()),
    ("orthogonal", OrthogonalProjection()),
]

const FITS = [
    ("qr",          WeightedQR()),
    ("incqr",       IncrementalQR()),
    ("pivqr",       PivotedQR()),
    ("tsvd",        TruncatedSVD()),
    ("normaleq",    NormalEquationsFit(ridge = true)),
    # The reference implementation's arithmetic: Gram solve, no ridge, in a Float64 island.
    ("normaleq+f64", NormalEquationsFit(ridge = false, island = true)),
]

# ---- targets ----------------------------------------------------------------
#
# The label sets the integrators actually fit are one time step of a trajectory sampled at
# the network's input nodes. These stand in for that: a slowly varying step (small `dt`),
# an oscillatory one (a step long enough to contain most of a period — where the seed is
# known to struggle), a monotone exponential, and a two-frequency signal standing in for a
# chaotic segment.
const TARGETS = [
    ("smooth",      t -> cos(3t)),
    ("oscillatory", t -> cos(12t)),
    ("exponential", t -> exp(2t) / 4),
    ("twofreq",     t -> 0.4cos(7t) + 0.3sin(11t + 1)),
]

# ---- metrics ----------------------------------------------------------------

# Rebuild the seed's design matrix in `Float64` from the returned neuron parameters, and
# report the fit error and the conditioning. All `S` neurons are included, including any
# zero-weight placeholders: they are part of what the Newton solve sees, so they belong in
# the conditioning estimate.
function seed_metrics(σ, W, b, c, nodes64::Vector{Float64}, w64::Vector{Float64}, y64::Vector{Float64})
    sw = sqrt.(w64)
    Φ = Matrix{Float64}(undef, length(nodes64), length(W))
    for i in eachindex(W), j in eachindex(nodes64)
        Φ[j, i] = σ(Float64(W[i]) * nodes64[j] + Float64(b[i])) * sw[j]
    end
    any(!isfinite, Φ) && return (NaN, NaN, NaN)

    resid = sqrt(sum(abs2, Φ * Float64.(c) .- sw .* y64))
    σs = svdvals(Φ)
    σmax, σmin = maximum(σs), minimum(σs)
    return (resid, σmin > 0 ? σmax / σmin : Inf, σmin)
end

const CSV_HEADER = "study,target,T,activation,dictionary,selection,fit,S,dict_amount," *
                   "status,fit_err,cond,sigma_min,neurons,rejected,secs"

function main()
    mkpath(RESULTS_DIR)
    csvpath = joinpath(RESULTS_DIR, "oga_fit_study.csv")

    total = length(TARGETS) * 3 * length(OGA_ACTIVATIONS_ALL) *
            length(DICTIONARIES) * length(SELECTIONS) * length(FITS)
    println("="^100)
    println("Tier A — OGA seed quality (no integrator, no Newton): $total cases")
    println("="^100)
    @printf("%-12s %-8s %-8s %-9s %-11s %-13s | %-10s %-10s %-10s %-4s %-4s\n",
            "target", "T", "act", "dict", "selection", "fit",
            "fit_err", "cond", "σ_min", "neu", "rej")
    println("-"^100)

    idx = 0
    open(csvpath, "w") do io
        println(io, CSV_HEADER)
        flush(io)
        for (tname, f) in TARGETS, T in (Float16, Float32, Float64)
            nodes = T.((0:NNODES) ./ NNODES)
            weights = NI.simpson_quadrature(NNODES, T)
            nodes64 = Float64.(nodes)
            w64 = Float64.(weights)
            y64 = f.(nodes64)
            y = T.(y64)

            for (aname, σ) in OGA_ACTIVATIONS_ALL, (dname, dict) in DICTIONARIES,
                (sname, sel) in SELECTIONS, (fname, fit) in FITS
                idx += 1
                oga = OGA(dict, sel, fit)
                status = "ok"
                err = cnd = smin = NaN
                neurons = rejected = -1
                secs = NaN
                try
                    secs = @elapsed r = oga_fit(oga, σ, nodes, weights, y, S_NEURONS;
                                                bias_interval = [T(BIAS[1]), T(BIAS[2])],
                                                dict_amount = DICT_AMOUNT)
                    neurons, rejected = r.neurons, r.rejected
                    if !(all(isfinite, r.c) && all(isfinite, r.W) && all(isfinite, r.b))
                        status = "nonfinite"
                    else
                        err, cnd, smin = seed_metrics(σ, r.W, r.b, r.c, nodes64, w64, y64)
                        isfinite(err) || (status = "nonfinite")
                    end
                catch e
                    status = first(string(nameof(typeof(e))), 14)
                end

                @printf("%-12s %-8s %-8s %-9s %-11s %-13s | %-10s %-10s %-10s %-4s %-4s\n",
                        tname, string(T), aname, dname, sname, fname,
                        isfinite(err) ? @sprintf("%.3e", err) : status,
                        isfinite(cnd) ? @sprintf("%.2e", cnd) : "—",
                        isfinite(smin) ? @sprintf("%.2e", smin) : "—",
                        neurons < 0 ? "—" : string(neurons),
                        rejected < 0 ? "—" : string(rejected))

                println(io, join(("fit_study", tname, string(T), aname, dname, sname, fname,
                                  string(S_NEURONS), string(DICT_AMOUNT), status,
                                  csvnum(err), csvnum(cnd), csvnum(smin),
                                  string(neurons), string(rejected), csvnum(secs)), ","))
                flush(io)
            end
        end
    end
    println("-"^100)
    println("Wrote $(csvpath)")
    write_fit_study_report(csvpath)
    return csvpath
end

main()
