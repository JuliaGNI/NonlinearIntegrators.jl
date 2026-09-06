# Fitting the ansätze the basis search in `../figures` found.
#
#   odd harmonics of one free ω     a₀ + Σ_{k odd} [a_k cos(kωt) + b_k sin(kωt)]
#   two-frequency lattice           combinations k₁ω₁ + k₂ω₂ of two basic frequencies
#
# Both are linear in their coefficients once the frequencies are fixed, so they are fitted by
# separable (variable-projection) least squares: an inner linear solve, and an outer search
# over the frequencies only. What makes them worth the trouble is that the frequencies are
# *parameters*: the cost does not grow with the length of the interval, while a fixed basis
# has to resolve every one of its ~ωT/π oscillations.
#
# The callers report `max|coefficient| / scale` for every fit and reject anything above a
# threshold. Left to itself a frequency search will drive two basic frequencies together
# until the columns are nearly parallel and then fit by cancellation between coefficients
# hundreds of times the amplitude of the signal — an excellent residual and a worthless
# representation, and it is not visible in the error alone.

using LinearAlgebra
using Printf

function trig_columns(times, freqs)
    A = ones(length(times), 1 + 2 * length(freqs))
    for (k, ω) in pairs(freqs)
        @views A[:, 2k] .= cos.(ω .* times)
        @views A[:, 2k + 1] .= sin.(ω .* times)
    end
    A
end

# derivative of the same expansion, for the velocities
function trig_derivative(times, freqs)
    A = zeros(length(times), 1 + 2 * length(freqs))
    for (k, ω) in pairs(freqs)
        @views A[:, 2k] .= -ω .* sin.(ω .* times)
        @views A[:, 2k + 1] .= ω .* cos.(ω .* times)
    end
    A
end

function fit_coefficients(A, Y)
    (F = qr(A, ColumnNorm());
        reduce(hcat, [F \ Y[:, c] for c in axes(Y, 2)]))
end

fit_residual(A, Y) = (C = fit_coefficients(A, Y); sum(abs2, Y .- A * C))

function evaluate_fit(freqs, C, times)
    (trig_columns(times, freqs) * C, trig_derivative(times, freqs) * C)
end

function golden(f, a, b; iterations = 45)
    φ = (sqrt(5) - 1) / 2
    c, d = b - φ * (b - a), a + φ * (b - a)
    fc, fd = f(c), f(d)
    for _ in 1:iterations
        if fc < fd
            b, d, fd = d, c, fc
            c = b - φ * (b - a)
            fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + φ * (b - a)
            fd = f(d)
        end
    end
    (a + b) / 2
end

# Written out rather than broadcast: the obvious one-liner allocates two vectors and copies a
# column of `Y` for every trial frequency, which for a grid this size is gigabytes of churn
# and turns a five-second scan into a ten-minute one.
function periodogram(times, Y, ωs)
    power = zeros(length(ωs))
    npoints = length(times)
    for (i, ω) in pairs(ωs)
        total = 0.0
        for c in axes(Y, 2)
            re = 0.0
            im = 0.0
            @inbounds @simd for j in 1:npoints
                sn, cs = sincos(ω * times[j])
                re += cs * Y[j, c]
                im += sn * Y[j, c]
            end
            total += re^2 + im^2
        end
        power[i] = total
    end
    power
end

"""
    odd_harmonic_fit(times, Y, m; bracket) -> frequencies, ω

`m` odd harmonics of one fundamental, which is what the exactly periodic orbit of a
one-degree-of-freedom system in a symmetric potential calls for.
"""
function odd_harmonic_fit(times, Y, m; bracket = (0.40, 0.55))
    ks = collect(1:2:(2m - 1))
    obj(ω) = fit_residual(trig_columns(times, [k * ω for k in ks]), Y)
    ω = golden(obj, bracket...; iterations = 80)
    for w in (1e-2, 2e-3, 5e-4, 1e-4, 2e-5, 4e-6, 1e-6)
        ω = golden(obj, ω * (1 - w), ω * (1 + w); iterations = 60)
    end
    [k * ω for k in ks], ω
end

"""
    lattice_fit(times, Y, N, M, ω₀; sweeps) -> frequencies, (ω₁, ω₂)

Combinations `n ω₁ + m ω₂`, `0 ≤ n ≤ N`, `|m| ≤ M`, for quasi-periodic motion on a two-torus:
two nonlinear parameters, however many lines are kept.

The starting bracket is scaled to the Rayleigh width `2π/T`. The residual as a function of a
frequency oscillates on that scale, so a bracket wider than one lobe leaves golden section to
return whichever lobe it happened to land in — which is what made the first long-interval
fits stall.
"""
function lattice_fit(times, Y, N, M, ω₀; sweeps = 14)
    mult = [(n, m) for n in 0:N for m in (-M):M if !(n == 0 && m ≤ 0)]

    function frequencies(a, b)
        f = Float64[]
        for (n, m) in mult
            v = n * a + m * b
            v ≤ 1e-4 && continue
            any(abs(v - g) < 1e-9 for g in f) && continue
            push!(f, v)
        end
        f
    end

    obj(a, b) = fit_residual(trig_columns(times, frequencies(a, b)), Y)

    rayleigh = 2π / (times[end] - times[begin])
    a, b = ω₀
    w = 0.4 * rayleigh
    for _ in 1:sweeps
        a = golden(x -> obj(x, b), a - w, a + w)
        b = golden(x -> obj(a, x), max(b - w, 1e-4), b + w)
        w *= 0.55
    end

    frequencies(a, b), (a, b)
end

"""
    basic_frequencies(times, Y; slow) -> (ωfast, ωslow)

The dominant line, and the strongest slow line of the residual once it has been removed.
"""
function basic_frequencies(times, Y; ωmax = 3.0, slow = 0.4)
    ωgrid = collect(range(0.004, ωmax, length = 12_000))
    ωfast = ωgrid[argmax(periodogram(times, Y, ωgrid))]
    A = trig_columns(times, [ωfast])
    R = Y .- A * fit_coefficients(A, Y)
    mask = ωgrid .< slow
    ωfast, ωgrid[mask][argmax(periodogram(times, R, ωgrid)[mask])]
end
