# Unit tests for the OGA subsystem's numerical kernels (`src/oga/`).
#
# These run the greedy fit and its factorisations *directly*, without an integrator, a
# Newton solve, or a time step. That matters for two reasons: the kernels are where the
# reduced-precision behaviour is decided, and an end-to-end test cannot distinguish a bad
# seed from a bad solve.
#
# Parametrised over `Float16` as well as `TEST_TYPES`. The suite excludes `Float16`
# elsewhere because the *end-to-end* path is ill-conditioned at half precision — but these
# are pure linear algebra, and half precision is exactly the case they exist to get right.

using NonlinearIntegrators
using LinearAlgebra
using Test

const NI = NonlinearIntegrators
const OGA_TYPES = (Float16, TEST_TYPES...)

# All fits, and all selection rules, as they are actually configured.
oga_fits() = (WeightedQR(), IncrementalQR(), PivotedQR(), TruncatedSVD(),
              NormalEquationsFit(), NormalEquationsFit(ridge = false))
oga_selections() = (RawProjection(), NormalizedProjection(), OrthogonalProjection())

# A smooth target on the unit interval and the Simpson weights the integrators use.
function oga_testcase(::Type{T}; n = 10) where {T}
    nodes = T.((0:n) ./ n)
    weights = NI.simpson_quadrature(n, T)
    y = T.(cos.(3 .* Float64.(nodes)))
    return (nodes, weights, y)
end

@testset "grids" begin
    @testset "bias_grid ($T)" for T in OGA_TYPES
        B = NI.bias_grid(-pi, pi, 8, T)
        @test eltype(B) === T
        @test length(B) == 9
        @test B[1] == T(-pi) && B[end] == T(pi)
        @test issorted(B)

        # The Float16 trap: `T(70000)` overflows to `Inf`, so a naive
        # `lo:(hi-lo)/n:hi` had a zero step and threw `ArgumentError`.
        big = NI.bias_grid(-pi, pi, 70000, T)
        @test length(big) == 70001
        @test all(isfinite, big)
    end

    @testset "weight_grid ($T)" for T in OGA_TYPES
        W = NI.weight_grid(-3, 3, 6, T)
        @test eltype(W) === T
        @test length(W) == 7
        @test W[1] == T(0.125) && W[4] == one(T) && W[end] == T(8)
        @test all(>(zero(T)), W)
        # Octaves outside the range of `T` saturate rather than corrupting the grid.
        @test all(isfinite, NI.weight_grid(-3, 3, 70000, T))
    end
end

@testset "generic dense kernels" begin
    @testset "jacobi_svd matches LAPACK (Float64)" begin
        for trial in 1:3
            A = Float64[i^(j-1) + (i == j) for i in 1:11, j in 1:4]   # Vandermonde-ish
            trial == 2 && (A[:, 3] .= A[:, 1] .+ 1e-13)               # near-dependent
            trial == 3 && (A[:, 3] .= A[:, 1])                        # exactly dependent
            σ, U, V = NI.jacobi_svd(A)
            @test sort(σ, rev = true) ≈ svd(A).S rtol = 1e-10
            @test norm(A - U * Diagonal(σ) * V') < 1e-10 * norm(A)
            @test norm(V'V - I) < 1e-12
        end
    end

    # Regression: the convergence test used to be `abs(β) ≤ eps(T)*sqrt(α*γ)`. At Float16
    # the product of two squared column norms overflows once the norms exceed ~16, making
    # the threshold `Inf`, so every column pair tested as "already orthogonal" and the
    # routine returned the *unrotated* matrix — wrong singular values, no error raised.
    @testset "jacobi_svd does not overflow its convergence test (Float16)" begin
        # Column norms ≈ 33 and 66, so `α ≈ 1100` and `γ ≈ 4400` and `α*γ` is well past the
        # 65504 ceiling — the configuration that used to make the test vacuous.
        A = Float16[20 40; 20.5 41.5; 19 38; 1 3]
        σ, U, V = NI.jacobi_svd(A)
        @test norm(Float64.(U)' * Float64.(U) - I) < 0.05      # columns really orthogonal
        @test sort(Float64.(σ), rev = true) ≈ svd(Float64.(A)).S rtol = 0.05
    end

    @testset "rank detection ($T)" for T in OGA_TYPES
        # Third column duplicates the first: mathematical rank 2 out of 3.
        A = T[1 0 1; 0 1 0; 1 1 1; 2 0 2]
        y = T[1, 2, 3, 4]
        rtol = eps(T) * 4
        best = norm(Float64.(A) * (pinv(Float64.(A)) * Float64.(y)) - Float64.(y))

        xp = NI.pivoted_qr_lstsq(A, y, rtol)
        xt = NI.truncated_svd_lstsq(A, y, rtol)
        for x in (xp, xt)
            @test eltype(x) === T
            @test all(isfinite, x)
            # Both attain the achievable residual — dropping a dependent direction costs
            # nothing, since it added no reachable component in the first place.
            @test norm(Float64.(A) * Float64.(x) - Float64.(y)) ≈ best rtol = 64 * sqrt(eps(T))
        end
        # Pivoted QR truncates by *column*, so a dependent column gets a zero coefficient.
        @test count(!iszero, xp) ≤ 2
        # The truncated SVD instead returns the minimum-norm solution, which spreads weight
        # across the duplicated pair rather than zeroing either — the same residual with a
        # smaller ‖x‖, so all three coefficients are legitimately nonzero here.
        @test norm(Float64.(xt)) ≤ norm(Float64.(xp)) * (1 + 64 * sqrt(eps(T)))
    end
end

# `IncrementalQR` reads its answer from the maintained factorisation, so the state has to be
# populated the way the greedy loop populates it: one column at a time.
function oga_state_for(Â::AbstractMatrix{T}) where {T}
    qr = NI.IncrementalQRState{T}(size(Â, 1), size(Â, 2))
    for j in axes(Â, 2)
        NI.oga_qr_append!(qr, Â[:, j])
    end
    return qr
end

@testset "fits" begin
    @testset "exact recovery on a well-conditioned system ($T)" for T in OGA_TYPES
        # `Â x = ŷ` with an exactly representable solution, so any correct solver hits it.
        Â = T[1 0; 0 1; 1 1; 1 -1]
        x = T[0.5, 0.25]
        ŷ = Â * x
        for fit in oga_fits()
            x̂ = NI.oga_solve(fit, Â, ŷ, oga_state_for(Â))
            @test eltype(x̂) === T
            @test Float64.(x̂) ≈ Float64.(x) rtol = 32 * sqrt(eps(T))
        end
    end

    @testset "every fit stays finite on a rank-deficient design ($T)" for T in OGA_TYPES
        # Duplicate columns: the normal equations are exactly singular, and at `Float16` even
        # `\` *throws* rather than returning garbage. This is the property the whole
        # subsystem exists for — no fit may return NaN/Inf, and none may let an exception
        # escape onto the seed path.
        Â = T[1 1 0; 1 1 1; 1 1 2; 0 0 1]
        ŷ = T[1, 2, 3, 1]
        for fit in oga_fits()
            x̂ = NI.oga_solve(fit, Â, ŷ, oga_state_for(Â))
            @test eltype(x̂) === T
            @test length(x̂) == size(Â, 2)
            @test all(isfinite, x̂)
        end
    end

    @testset "IncrementalQR reproduces WeightedQR ($T)" for T in OGA_TYPES
        nodes, weights, y = oga_testcase(T)
        a = oga_fit(OGA(BiasGrid1d(), RawProjection(), WeightedQR()), x -> max(zero(x), x)^3,
                    nodes, weights, y, 4; bias_interval = [-T(pi), T(pi)], dict_amount = 200)
        b = oga_fit(OGA(BiasGrid1d(), RawProjection(), IncrementalQR()), x -> max(zero(x), x)^3,
                    nodes, weights, y, 4; bias_interval = [-T(pi), T(pi)], dict_amount = 200)
        if T === Float16
            # At half precision the two factorisations round differently enough to pick
            # different atoms from a dictionary far finer than `T` can resolve. Only the
            # quality of the fit is comparable, not the path taken to it.
            @test Float64(b.residual) ≤ Float64(a.residual) * 4 + sqrt(eps(T))
        else
            @test a.atoms == b.atoms
            @test Float64.(a.c) ≈ Float64.(b.c) rtol = 64 * sqrt(eps(T))
        end
    end

    @testset "incremental QR is a valid factorisation ($T)" for T in OGA_TYPES
        qr = NI.IncrementalQRState{T}(5, 3)
        cols = (T[1, 0, 0, 0, 0], T[1, 1, 0, 0, 0], T[0, 0, 1, 1, 0])
        for col in cols
            @test NI.oga_qr_append!(qr, col) > zero(T)
        end
        @test qr.k == 3
        Q = qr.Q[:, 1:3]
        @test Float64.(Q' * Q) ≈ I rtol = 32 * sqrt(eps(T))
        @test Float64.(Q * qr.R[1:3, 1:3]) ≈ Float64.(hcat(cols...)) rtol = 32 * sqrt(eps(T))

        # A column already in the span adds no rank and is refused.
        k = qr.k
        @test NI.oga_qr_append!(qr, T[2, 2, 0, 0, 0]; min_gain = sqrt(eps(T))) ≤ sqrt(eps(T)) * T(3)
        @test qr.k == k
    end
end

@testset "selection rules" begin
    # The orthogonal/normalised criteria are *provably* optimal for the first atom: they
    # maximise the residual reduction of a single-atom refit. Raw projection is not, since
    # it ranks by amplitude as well as alignment. Checking against brute force is what
    # distinguishes a correct score from a plausible one.
    @testset "normalised and orthogonal selection find the optimal first atom" begin
        T = Float64
        nodes, weights, y = oga_testcase(T)
        σ = x -> max(zero(x), x)^3
        sw = sqrt.(weights)
        ŷ = sw .* y

        A = NI.oga_atoms(BiasGrid1d(), [-pi, pi], 400, T)
        best, besti = Inf, 0
        for i in axes(A, 1)
            g = σ.(A[i, 1] .* nodes .+ A[i, 2]) .* sw
            n² = sum(abs2, g)
            n² == 0 && continue
            nr = norm(ŷ .- (dot(g, ŷ) / n²) .* g)
            nr < best && ((best, besti) = (nr, i))
        end

        for sel in (NormalizedProjection(), OrthogonalProjection())
            r = oga_fit(OGA(BiasGrid1d(), sel, IncrementalQR()), σ, nodes, weights, y, 1;
                        bias_interval = [-pi, pi], dict_amount = 400)
            @test r.atoms == [besti]
            @test r.residual ≈ best rtol = 1e-10
        end
        raw = oga_fit(OGA(BiasGrid1d(), RawProjection(), WeightedQR()), σ, nodes, weights, y, 1;
                      bias_interval = [-pi, pi], dict_amount = 400)
        @test raw.residual > best        # not optimal, by design — it is the pinned legacy rule
    end
end

@testset "guard rails" begin
    @testset "coherence guard blocks near-duplicate atoms ($T)" for T in OGA_TYPES
        nodes, weights, y = oga_testcase(T)
        σ = x -> max(zero(x), x)^3
        # A dictionary far finer than `T` can resolve, so neighbouring atoms round together.
        r = oga_fit(OGA(BiasGrid1d(), RawProjection(), WeightedQR()), σ, nodes, weights, y, 4;
                    bias_interval = [-T(pi), T(pi)], dict_amount = 20000)
        @test length(unique(r.atoms)) == length(r.atoms)     # never the same atom twice
    end

    @testset "gain floor refuses dependent atoms instead of selecting them ($T)" for T in OGA_TYPES
        nodes, weights, y = oga_testcase(T)
        σ = x -> max(zero(x), x)^3
        r = oga_fit(OGA(BiasGrid1d(), OrthogonalProjection(), IncrementalQR()), σ,
                    nodes, weights, y, 4; bias_interval = [-T(pi), T(pi)], dict_amount = 20000)
        # Every accepted atom contributed a genuinely new direction...
        @test all(>(zero(T)), r.gains)
        @test length(r.gains) == length(r.atoms)
        # ...and where atoms had to be refused, the placeholders are distinct rather than
        # all-zero, so the Newton Jacobian does not inherit the rank deficiency.
        if r.neurons < 4
            @test r.rejected > 0
            @test length(unique(collect(zip(Float64.(r.W), Float64.(r.b))))) == 4
        end
    end

    @testset "atoms with a non-finite norm are excluded (Float16, ReLU⁴)" begin
        T = Float16
        nodes, weights, y = oga_testcase(T)
        # `σ(b)⁴` for `b ≈ π` overflows Float16, giving atoms of infinite norm. They must be
        # skipped, not normalised by `Inf` into `NaN`.
        r = oga_fit(OGA1d(), x -> max(zero(x), x)^4, nodes, weights, y, 4;
                    bias_interval = [-T(pi), T(pi)], dict_amount = 400)
        @test all(isfinite, r.c)
        @test all(isfinite, r.W) && all(isfinite, r.b)
        @test isfinite(r.residual)
    end
end

@testset "dictionaries" begin
    @testset "2-D grid restricted to |w| = 1 reproduces the 1-D grid ($T)" for T in OGA_TYPES
        # The backward-compatibility claim: the weight axis is a strict generalisation, so
        # collapsing it must recover the original dictionary exactly.
        A1 = NI.oga_atoms(BiasGrid1d(), [-T(pi), T(pi)], 20, T)
        A2 = NI.oga_atoms(WeightBiasGrid2d(octaves = (0, 0), weight_amount = 0, signed = true),
                          [-T(pi), T(pi)], 20, T)
        @test eltype(A2) === T
        @test size(A1) == size(A2)
        @test sort(collect(zip(A1[:, 1], A1[:, 2]))) == sort(collect(zip(A2[:, 1], A2[:, 2])))
    end

    @testset "angular grid lies on its radii ($T)" for T in OGA_TYPES
        A = NI.oga_atoms(AngularGrid(radii = (1.0,), amount = 16), [-T(pi), T(pi)], 400, T)
        @test eltype(A) === T
        @test size(A, 1) == 17
        @test all(abs(sqrt(A[i, 1]^2 + A[i, 2]^2) - one(T)) < 8 * sqrt(eps(T)) for i in axes(A, 1))
    end

    @testset "off-grid refinement improves the greedy step it optimises ($T)" for T in OGA_TYPES
        nodes, weights, y = oga_testcase(T)
        σ = x -> max(zero(x), x)^3
        # A coarse dictionary, where polishing has something to gain.
        kw = (bias_interval = [-T(pi), T(pi)], dict_amount = 12)

        # The refinement maximises the *selection score*, which is exactly the residual
        # objective for a single atom — so at `S = 1` the improvement is guaranteed. It is
        # not guaranteed after four atoms: a better first atom can lead to a worse quartet,
        # the ordinary myopia of any greedy method. Testing `S = 1` tests the mechanism;
        # testing `S = 4` would be testing a property refinement does not have.
        for S in (1, 4)
            plain = oga_fit(OGA(BiasGrid1d(), NormalizedProjection(), IncrementalQR()),
                            σ, nodes, weights, y, S; kw...)
            fine = oga_fit(OGA(Refined(BiasGrid1d()), NormalizedProjection(), IncrementalQR()),
                           σ, nodes, weights, y, S; kw...)
            @test all(isfinite, fine.c)
            @test all(isfinite, fine.W) && all(isfinite, fine.b)
            S == 1 && @test Float64(fine.residual) ≤ Float64(plain.residual) + 8 * sqrt(eps(T))
        end
    end
end

@testset "neuron symmetry" begin
    # The two time-reversible integrators depend on the seed placing neurons in pairs
    # related by `t ↦ 1 - t`, i.e. `(w, b) ↦ (-w, w + b)`, and — for the shared variant —
    # on both members of a pair carrying the *same* output weight. That sharing is what
    # actually enforces time-reversal symmetry of the ansatz; with independent weights the
    # pair can drift apart. Nothing else in the suite checks the structure directly.
    @testset "$(nameof(typeof(sym))) ($T)" for sym in (MirrorPairs(), SharedMirrorPairs()),
                                               T in OGA_TYPES
        nodes, weights, y = oga_testcase(T)
        modulation = nodes .* (one(T) .- nodes)          # the t(1-t) ansatz factor
        r = oga_fit(OGA(BiasGrid1d(), RawProjection(), WeightedQR()), x -> max(zero(x), x)^3,
                    nodes, weights, y, 4; bias_interval = [-T(pi), T(pi)], dict_amount = 200,
                    modulation = modulation, symmetry = sym)

        for k in 1:2
            @test r.W[2k] == -r.W[2k-1]
            @test r.b[2k] == r.W[2k-1] + r.b[2k-1]
        end
        if sym isa SharedMirrorPairs
            @test r.c[2] == r.c[1]
            @test r.c[4] == r.c[3]
        end
        # Two neuron slots per greedy step, so at most half as many atoms as neurons.
        @test length(r.atoms) ≤ 2
        @test r.neurons % 2 == 0
    end

    # An odd count cannot be honoured by a symmetry that places neurons two at a time: the
    # loop would run `nneurons ÷ 2` steps and leave the last neuron at `(0, 0)`, and
    # `_fill_unused!` fills pairs too, so it cannot repair it. Rejected up front rather
    # than half-honoured — otherwise the duplicate neuron reappears as a rank-deficient
    # Newton Jacobian, several call levels from the cause.
    @testset "an odd neuron count is rejected ($(nameof(typeof(sym))))" for sym in
                                                (MirrorPairs(), SharedMirrorPairs())
        nodes, weights, y = oga_testcase(Float64)
        @test_throws ArgumentError oga_fit(OGA1d(), x -> max(zero(x), x)^3, nodes, weights,
                                           y, 5; bias_interval = [-pi, pi],
                                           dict_amount = 200, symmetry = sym)
    end

    @testset "an odd neuron count is fine without a symmetry" begin
        nodes, weights, y = oga_testcase(Float64)
        r = oga_fit(OGA1d(), x -> max(zero(x), x)^3, nodes, weights, y, 5;
                    bias_interval = [-pi, pi], dict_amount = 200, symmetry = NoSymmetry())
        @test length(r.W) == 5
    end
end

@testset "precision discipline" begin
    # The central invariant: a run at `T` must stay at `T`. A `Float64` creeping in through
    # a bare literal or a promoting fallback would otherwise show up only as suspiciously
    # good half-precision accuracy.
    @testset "oga_fit returns eltype T for every configuration ($T)" for T in OGA_TYPES
        nodes, weights, y = oga_testcase(T)
        σ = x -> max(zero(x), x)^3
        dicts = (BiasGrid1d(), WeightBiasGrid2d(bias_amount = 40),
                 AngularGrid(amount = 60), Refined(BiasGrid1d()))
        for dict in dicts, sel in oga_selections(), fit in oga_fits()
            r = oga_fit(OGA(dict, sel, fit), σ, nodes, weights, y, 4;
                        bias_interval = [-T(pi), T(pi)], dict_amount = 60)
            @test eltype(r.W) === T
            @test eltype(r.b) === T
            @test eltype(r.c) === T
            @test typeof(r.residual) === T
            @test eltype(r.gains) === T
            @test all(isfinite, r.c)
        end
    end

    @testset "oga_fit is type stable ($T)" for T in OGA_TYPES
        nodes, weights, y = oga_testcase(T)
        σ = x -> max(zero(x), x)^3
        @test (@inferred oga_fit(OGA1d(), σ, nodes, weights, y, 4;
                                 bias_interval = [-T(pi), T(pi)],
                                 dict_amount = 60)) isa NI.OGAResult{T}
    end

    @testset "an upcasting activation is rejected, not silently honoured ($T)" for T in (Float16, Float32)
        nodes, weights, y = oga_testcase(T)
        # The documented trap: `max(0.0, x)` promotes the whole evaluation to Float64.
        bad = x -> max(0.0, x)^3
        @test_throws ArgumentError NI.oga_check_precision(bad, T)
        @test_throws ArgumentError oga_fit(OGA1d(), bad, nodes, weights, y, 4;
                                          bias_interval = [-T(pi), T(pi)], dict_amount = 60)
        @test NI.oga_check_precision(x -> max(zero(x), x)^3, T) === nothing
    end
end

@testset "OGA1d regression pin" begin
    # `OGA1d` must keep selecting the atoms the pre-refactor implementation selected: the
    # docs record that normalising before selection steers the Newton solve into a
    # different, empirically worse basin, so this sequence is load-bearing rather than
    # incidental. Values captured from the implementation that passes the Float64
    # end-to-end accuracy guard (`test/integration/onelayer_accuracy.jl`, < 1e-12).
    @testset "$T" for T in (Float64, Float32)
        nodes, weights, y = oga_testcase(T)
        r = oga_fit(OGA1d(), x -> max(zero(x), x)^3, nodes, weights, y, 4;
                    bias_interval = [-T(pi), T(pi)], dict_amount = 400)
        @test r.atoms == [802, 401, 711, 286]
        @test Float64.(r.W) == [1.0, -1.0, 1.0, -1.0]
        @test Float64.(r.b) ≈ [pi, pi, 1.7121679962064373, 1.335176877775662] rtol = 8 * sqrt(eps(T))
        @test Float64.(r.c) ≈ [-0.49460912014278396, 0.5665083163888716,
                               1.4395563049941698, -3.5558966254259077] rtol = 64 * sqrt(eps(T))
        @test Float64(r.residual) ≈ 0.0029557412009774313 rtol = 64 * sqrt(eps(T))
    end
end
