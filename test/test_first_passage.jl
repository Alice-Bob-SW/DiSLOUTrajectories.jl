# Paper: s(t) (Eq. 18).
function _test_survival_probability(c, λ, G, time)
    return DiSLOUTrajectories._survival_probability!(
        similar(c), similar(c), c, λ, G, time, DiSLOUTrajectories._matrix_inf_norm(G)
    )
end

# Paper: s(t) (Eq. 18).
_test_survival(cache, c, time) =
    _test_survival_probability(c, cache.Λ, cache.G, time)

# Paper: -ṡ(t) (Eq. B.1).
function _test_jump_rate(state, matrices)
    norms = DiSLOUTrajectories._matrix_inf_norm.(matrices)
    return DiSLOUTrajectories._jump_rate!(similar(state), state, matrices, norms)
end

# Paper: s(t), -ṡ(t) (Eq. B.2).
_test_survival_and_rate!(
    scratch, state, λ, G,
    Gnorm = DiSLOUTrajectories._matrix_inf_norm(G)
) =
    DiSLOUTrajectories._survival_and_rate!(scratch, state, λ, G, Gnorm)

@testset "paper first-passage names select the retained kernels" begin
    @test DiSLOUTrajectories._validated_first_passage_method(:automatic) === :log_survival
    @test DiSLOUTrajectories._validated_first_passage_method(:survival) === :survival
    @test DiSLOUTrajectories._validated_first_passage_method(:log_survival_predictor) ===
        :log_survival_predictor
    @test_throws ArgumentError DiSLOUTrajectories._validated_first_passage_method(:linear)
end

@testset "trajectory invalid predictor bracket falls back without another draw" begin
    # Mutation target: accepting the reduced no-event prediction, or drawing a fresh
    # threshold for the full fallback, changes the event stream or the next RNG value.
    H = QuantumObject(zeros(ComplexF64, 2, 2))
    c_ops = [QuantumObject(Matrix(Diagonal(sqrt.([0.1, 10.0]))))]
    cache = DiSLOUTrajectories._DiagonalCache(H, c_ops)
    ψ = ComplexF64[sqrt(0.51), sqrt(0.49)]
    times = [0.0, 0.2]

    threshold = rand(Xoshiro(4))
    c = cache.Vfac \ ψ
    c ./= sqrt(real(dot(c, cache.G, c)))
    guess = DiSLOUTrajectories._predict_first_passage!(
        DiSLOUTrajectories._FirstPassagePredictor(cache; m = 1),
        c, cache.Λ, cache.G, threshold, last(times)
    )
    @test !guess.jumped
    @test guess.t == last(times)
    @test _test_survival(cache, c, last(times)) < threshold

    function run(method, rng)
        data = DiSLOUTrajectories._validated_gauge_data(zeros(ComplexF64, 1, 1), 1)
        prepared = DiSLOUTrajectories._prepare_layer1(H.data, [c_ops[1].data], nothing, data, 0.5)
        acc = DiSLOUTrajectories._TrajectoryAccumulator(cache.Ne, length(times), cache.Nc)
        diagnostics = DiSLOUTrajectories._GaugeDiagnostics(1)
        record = DiSLOUTrajectories._TrajectoryRecord(cache.N, cache.Ne, length(times), 0, false)
        predictor = method === :log_survival_predictor ?
            DiSLOUTrajectories._FirstPassagePredictor(cache; m = 1) : nothing
        DiSLOUTrajectories._run_exact_trajectory!(
            acc, diagnostics, prepared, ψ, times,
            last(times), rng, DiSLOUTrajectories._WorkBuffers(cache);
            first_passage_method = method,
            trajectory_predictor = predictor, record
        )
        return acc, record
    end

    full_rng = Xoshiro(4); predictor_rng = Xoshiro(4)
    full, full_record = run(:log_survival, full_rng)
    predicted, predictor_record = run(:log_survival_predictor, predictor_rng)
    @test !isempty(full_record.col_times)
    @test length(predictor_record.col_times) == length(full_record.col_times) &&
        all(
        isapprox.(
            predictor_record.col_times, full_record.col_times;
            atol = 1.0e-12, rtol = 0
        )
    )
    @test predictor_record.col_which == full_record.col_which
    @test (predicted.njumps_total, predicted.jumps_by_channel) ==
        (full.njumps_total, full.jumps_by_channel)
    @test rand(full_rng) == rand(predictor_rng)
end

@testset "all first-passage methods solve the complete survival" begin
    H, c_ops = random_system(N = 6, seed = 22)
    cache = DiSLOUTrajectories._DiagonalCache(H, c_ops)
    @test DiSLOUTrajectories._FirstPassageBuffers(cache).predictor === nothing
    buffers = DiSLOUTrajectories._FirstPassageBuffers(cache, DiSLOUTrajectories._FirstPassagePredictor(cache))
    ψ = normalize!(randn(Xoshiro(3), ComplexF64, 6))
    c = cache.Vfac \ ψ
    threshold = 0.63
    roots = [
        DiSLOUTrajectories._find_first_passage!(
            buffers, cache, c, threshold, 1.5;
            method = m, survival_rtol = 1.0e-10, time_rtol = 1.0e-12,
            time_atol = 0.0
        ).time
            for m in (:survival, :log_survival, :log_survival_predictor)
    ]
    @test maximum(roots) - minimum(roots) <= 1.0e-10
    @test all(abs(_test_survival(cache, c, t) - threshold) <= 1.0e-9 for t in roots)
end

@testset "first-passage predictor validates controls" begin
    for tolerance in (0.0, -1.0, Inf, NaN)
        @test_throws ArgumentError DiSLOUTrajectories._FirstPassagePredictor(4; tol = tolerance)
    end
    for iterations in (0, -1)
        @test_throws ArgumentError DiSLOUTrajectories._FirstPassagePredictor(4; maxiter = iterations)
    end
end

# Mutations caught: reversing the comparison, dropping deterministic index ties,
# or mishandling NaN, infinities, and signed zero changes the selected modes.
@testset "predictor top-m selection matches Base ordering without allocations" begin
    rng = Xoshiro(801)
    cases = Vector{Float64}[
        [NaN, -Inf, -1.0, -0.0, 0.0, 1.0, Inf],
        [3.0, 3.0, 3.0, 2.0, 2.0, 3.0],
        [NaN, NaN, Inf, Inf, -Inf, -Inf],
        randn(rng, 32),
    ]
    for values in cases
        N = length(values)
        for m in unique((1, min(2, N), N))
            expected = collect(1:N)
            partialsortperm!(
                expected, values, 1:m;
                rev = true, initialized = true
            )
            actual = zeros(Int, N)
            DiSLOUTrajectories._topm_indices!(actual, values, m)
            @test actual[1:m] == expected[1:m]
        end
    end

    values = randn(rng, 2048)
    indices = zeros(Int, length(values))
    DiSLOUTrajectories._topm_indices!(indices, values, 8)
    values .= randn(rng, length(values))
    expected = collect(eachindex(values))
    partialsortperm!(
        expected, values, 1:8;
        rev = true, initialized = true
    )
    DiSLOUTrajectories._topm_indices!(indices, values, 8)
    @test indices[1:8] == expected[1:8]
    @test (@allocated DiSLOUTrajectories._topm_indices!(indices, values, 8)) == 0
end

@testset "first-passage survival is exact in a reduced invariant space" begin
    H, c_ops = random_system(N = 6, seed = 4)
    cache = DiSLOUTrajectories._DiagonalCache(H, c_ops)
    H_eff = DiSLOUTrajectories._effective_hamiltonian(cache.H, cache.C)
    reduced = DiSLOUTrajectories._get_reduced!(cache, [1, 2])
    ψ = cache.V[:, 1] * (0.7 + 0.2im) + cache.V[:, 2] * (-0.4 + 0.5im)
    ψ ./= norm(ψ)
    coordinates = reduced.G_I \ (reduced.V_I' * ψ)
    coordinates ./= sqrt(real(dot(coordinates, reduced.G_I, coordinates)))

    @test _test_survival_probability(
        coordinates, reduced.λ_I, reduced.G_I, 0.0
    ) ≈ 1.0 atol = 1.0e-10
    for time in (0.0, 0.2, 0.8, 2.0, 5.0)
        @test _test_survival_probability(
            coordinates, reduced.λ_I, reduced.G_I, time
        ) ≈
            direct_survival(H_eff, ψ, time) atol = 1.0e-9
    end
end

@testset "first-passage survival and rate match dense evolution" begin
    H, c_ops = random_system(N = 6, seed = 5)
    cache = DiSLOUTrajectories._DiagonalCache(H, c_ops)
    H_eff = DiSLOUTrajectories._effective_hamiltonian(cache.H, cache.C)
    ψ = normalize!(randn(Xoshiro(99), ComplexF64, cache.N))
    coordinates = similar(ψ)
    DiSLOUTrajectories._solve_coordinates!(coordinates, cache, ψ)
    coordinates ./= sqrt(real(dot(coordinates, cache.G, coordinates)))

    @test _test_survival_probability(
        coordinates, cache.Λ, cache.G, 0.0
    ) ≈ 1.0 atol = 1.0e-10
    for time in (0.0, 0.3, 1.0, 2.5)
        @test _test_survival_probability(
            coordinates, cache.Λ, cache.G, time
        ) ≈
            direct_survival(H_eff, ψ, time) atol = 1.0e-9
    end

    metric_coordinates = similar(coordinates)
    for time in (0.1, 0.7, 1.6)
        evolved = DiSLOUTrajectories._phase_evolve!(similar(coordinates), cache.Λ, coordinates, time)
        survival_rate = _test_survival_and_rate!(
            metric_coordinates, evolved, cache.Λ, cache.G
        )
        rate = _test_jump_rate(evolved, cache.M)
        physical = exp(-im * time * H_eff) * ψ
        direct_rate = sum(norm(C * physical)^2 for C in cache.C)
        @test survival_rate.S ≈ _test_survival_probability(
            coordinates, cache.Λ, cache.G, time
        ) atol = 1.0e-10
        @test survival_rate.R ≈ rate atol = 1.0e-10
        @test rate ≈ direct_rate atol = 1.0e-8
        step = 1.0e-6
        derivative = (
            direct_survival(H_eff, ψ, time + step) -
                direct_survival(H_eff, ψ, time - step)
        ) / (2step)
        @test rate ≈ -derivative rtol = 1.0e-4
    end
end

@testset "long-horizon Lambda dark state remains a no-jump trajectory" begin
    coupling = 1.0
    decay = 1.0
    H = QuantumObject(
        ComplexF64[
            0 0 coupling
            0 0 coupling
            coupling coupling 0
        ]
    )
    C1 = zeros(ComplexF64, 3, 3)
    C2 = zeros(ComplexF64, 3, 3)
    C1[1, 3] = sqrt(decay / 2)
    C2[2, 3] = sqrt(decay / 2)
    cache = DiSLOUTrajectories._DiagonalCache(H, QuantumObject.([C1, C2]))
    state = ComplexF64[1, 0, 0]
    coordinates = cache.Vfac \ state
    coordinates ./= sqrt(real(dot(coordinates, cache.G, coordinates)))

    result = try
        DiSLOUTrajectories._find_first_passage!(
            DiSLOUTrajectories._FirstPassageBuffers(cache), cache, coordinates, 0.25, 100.0;
            method = :log_survival, survival_rtol = 1.0e-10, time_rtol = 1.0e-12,
            time_atol = 0.0
        )
    catch err
        err
    end

    @test !(result isa Exception)
    if !(result isa Exception)
        @test !result.jumped
        @test result.time == 100.0
        @test _test_survival(cache, coordinates, result.time) ≈ 0.5 atol = 1.0e-12
    end
end

@testset "first-passage quadratics tolerate only numerical negativity" begin
    N = 2
    coefficient = 16N + 8
    γ = coefficient * eps(Float64) / (1 - coefficient * eps(Float64))
    state = ComplexF64[0, 1]

    for scale in (1.0e-200, 1.0e200)
        roundoff = Diagonal(ComplexF64[scale, -0.5γ * scale])
        material = Diagonal(ComplexF64[scale, -2γ * scale])
        @test _test_jump_rate(state, [roundoff]) == 0.0
        @test_throws DomainError _test_jump_rate(state, [material])
        @test _test_survival_probability(
            state, zeros(ComplexF64, N), roundoff, 0.0
        ) == 0.0
        @test_throws DomainError _test_survival_probability(
            state, zeros(ComplexF64, N), material, 0.0
        )

        scratch = similar(state)
        @test _test_survival_and_rate!(
            scratch, state, zeros(ComplexF64, N), roundoff
        ).S == 0.0
        @test_throws DomainError _test_survival_and_rate!(
            scratch, state, zeros(ComplexF64, N), material
        )
    end
end

@testset "scalar first passage solves analytic and stiff laws" begin
    Γ = 0.7
    exponential_survival = time -> exp(-Γ * time)
    exponential_rate = time -> Γ * exp(-Γ * time)
    for threshold in (0.9, 0.5, 0.1, 0.01)
        result = DiSLOUTrajectories._find_scalar_first_passage(
            exponential_survival, exponential_rate, threshold, 100.0;
            survival_rtol = 1.0e-12, time_rtol = 1.0e-12
        )
        @test result.jumped
        @test result.converged
        @test result.tau ≈ -log(threshold) / Γ rtol = 1.0e-6
        @test result.iterations >= 1
        @test result.bracket_width >= 0.0
    end

    nojump = DiSLOUTrajectories._find_scalar_first_passage(
        exponential_survival, exponential_rate, 0.99, 0.001
    )
    @test !nojump.jumped
    @test nojump.converged
    @test nojump.tau == 0.001
    @test nojump.iterations == 0

    stiff_survival = time -> exp(-time^2)
    stiff_rate = time -> 2time * exp(-time^2)
    linear = DiSLOUTrajectories._find_scalar_first_passage(
        stiff_survival, stiff_rate, 0.2, 10.0;
        survival_rtol = 1.0e-12, time_rtol = 1.0e-12, root_variant = :survival
    )
    logarithmic = DiSLOUTrajectories._find_scalar_first_passage(
        stiff_survival, stiff_rate, 0.2, 10.0;
        survival_rtol = 1.0e-12, time_rtol = 1.0e-12,
        root_variant = :log_survival
    )
    predictor_named = DiSLOUTrajectories._find_scalar_first_passage(
        stiff_survival, stiff_rate, 0.2, 10.0;
        survival_rtol = 1.0e-12, time_rtol = 1.0e-12,
        root_variant = :log_survival_predictor
    )
    @test stiff_survival(linear.tau) ≈ 0.2 atol = 1.0e-8
    @test logarithmic.tau ≈ linear.tau atol = 1.0e-10
    @test predictor_named.tau ≈ logarithmic.tau atol = 1.0e-12
    @test logarithmic.iterations <= linear.iterations
end

@testset "diagonal first passage brackets jump and no-jump cases" begin
    H, c_ops = random_system(N = 6, seed = 12)
    cache = DiSLOUTrajectories._DiagonalCache(H, c_ops)
    ψ = normalize!(randn(Xoshiro(77), ComplexF64, cache.N))
    coordinates = similar(ψ)
    DiSLOUTrajectories._solve_coordinates!(coordinates, cache, ψ)
    coordinates ./= sqrt(real(dot(coordinates, cache.G, coordinates)))
    evolved = similar(coordinates)
    scratch = similar(coordinates)

    for remaining in (0.02, 0.4, 1.7)
        endpoint = _test_survival_probability(
            coordinates, cache.Λ, cache.G, remaining
        )
        nojump = DiSLOUTrajectories._find_diagonal_first_passage!(
            evolved, scratch, coordinates, cache.Λ, cache.G,
            0.5endpoint, remaining
        )
        @test !nojump.jumped
        @test nojump.converged
        @test nojump.tau == remaining
        @test nojump.iterations == 0

        threshold = 0.5 * (1 + endpoint)
        jumped = DiSLOUTrajectories._find_diagonal_first_passage!(
            evolved, scratch, coordinates, cache.Λ, cache.G,
            threshold, remaining; survival_rtol = 1.0e-12, time_rtol = 1.0e-12
        )
        @test jumped.jumped
        @test jumped.converged
        @test _test_survival_probability(
            coordinates, cache.Λ, cache.G, jumped.tau
        ) ≈ threshold atol = 1.0e-9
        @test jumped.iterations >= 1
    end
end

@testset "first passage handles tiny thresholds plateaus and scale changes" begin
    survival = time -> exp(-time)
    rate = time -> exp(-time)
    threshold = 1.0e-250
    tiny = DiSLOUTrajectories._find_scalar_first_passage(
        survival, rate, threshold, 600.0;
        survival_rtol = 1.0e-12, time_rtol = 1.0e-12,
        root_variant = :log_survival
    )
    @test tiny.jumped
    @test tiny.tau ≈ -log(threshold) rtol = 1.0e-12
    @test abs(log(survival(tiny.tau)) - log(threshold)) <= 1.0e-12

    zero = DiSLOUTrajectories._find_scalar_first_passage(
        survival, rate, 0.0, 1_000.0;
        survival_rtol = 1.0e-12, time_rtol = 1.0e-12,
        root_variant = :log_survival
    )
    @test !zero.jumped
    @test zero.tau == 1_000.0

    time_rtol = 1.0e-8
    plateau = DiSLOUTrajectories._find_scalar_first_passage(
        _ -> 1.0, _ -> 0.0, 1.0, 1.0; time_rtol
    )
    @test plateau.jumped
    @test 0.0 < plateau.bracket_width <= time_rtol
    @test 0.0 < plateau.tau <= time_rtol

    function scaled_result(scale)
        scaled_survival = time -> exp(-(time / scale)^2)
        scaled_rate = time -> 2time / scale^2 * scaled_survival(time)
        DiSLOUTrajectories._find_scalar_first_passage(
            scaled_survival, scaled_rate, 0.2, 4scale;
            survival_rtol = 1.0e-12, time_rtol = 1.0e-12
        )
    end
    base = scaled_result(1.0)
    small = scaled_result(1.0e-9)
    large = scaled_result(1.0e9)
    @test small.tau / 1.0e-9 ≈ base.tau rtol = 1.0e-10
    @test large.tau / 1.0e9 ≈ base.tau rtol = 1.0e-10
end

@testset "first-passage convergence errors retain numerical payloads" begin
    Γ = 0.7
    survival = time -> exp(-Γ * time)
    rate = time -> Γ * exp(-Γ * time)
    scalar_error = try
        DiSLOUTrajectories._find_scalar_first_passage(
            survival, rate, 0.2, 10.0;
            survival_rtol = 1.0e-15, time_rtol = 1.0e-15, maxiter = 1
        )
        nothing
    catch error
        error
    end
    @test scalar_error isa DiSLOUTrajectories.FirstPassageConvergenceError
    @test scalar_error.iterations == 1
    @test isfinite(scalar_error.residual)
    @test scalar_error.bracket_width > 0.0
    @test 0.0 < scalar_error.time < 10.0

    H, c_ops = random_system(N = 6, seed = 12)
    cache = DiSLOUTrajectories._DiagonalCache(H, c_ops)
    ψ = normalize!(randn(Xoshiro(77), ComplexF64, cache.N))
    coordinates = similar(ψ)
    DiSLOUTrajectories._solve_coordinates!(coordinates, cache, ψ)
    coordinates ./= sqrt(real(dot(coordinates, cache.G, coordinates)))
    remaining = 1.7
    endpoint = _test_survival_probability(
        coordinates, cache.Λ, cache.G, remaining
    )
    diagonal_error = try
        DiSLOUTrajectories._find_diagonal_first_passage!(
            similar(coordinates), similar(coordinates), coordinates,
            cache.Λ, cache.G, 0.5 * (1 + endpoint), remaining;
            survival_rtol = 1.0e-15, time_rtol = 1.0e-15,
            maxiter = 1, root_variant = :log_survival
        )
        nothing
    catch error
        error
    end
    @test diagonal_error isa DiSLOUTrajectories.FirstPassageConvergenceError
    @test diagonal_error.iterations == 1
    @test isfinite(diagonal_error.residual)
    @test diagonal_error.bracket_width > 0.0
    @test 0.0 < diagonal_error.time < remaining
end

@testset "small rates cannot satisfy convergence through residual alone" begin
    Γ = 1.0e-9
    expected = 1.0e6
    threshold = exp(-Γ * expected)
    remaining = 2expected
    survival = time -> exp(-Γ * time)
    rate = time -> Γ * exp(-Γ * time)
    scalar = DiSLOUTrajectories._find_scalar_first_passage(
        survival, rate, threshold, remaining;
        survival_rtol = 1.0e-6, time_rtol = 1.0e-12
    )
    @test scalar.tau ≈ expected rtol = 5.0e-12

    eigenvalues = ComplexF64[-0.5im * Γ]
    metric = reshape(ComplexF64[1], 1, 1)
    coordinates = ComplexF64[1]
    diagonal = DiSLOUTrajectories._find_diagonal_first_passage!(
        similar(coordinates), similar(coordinates), coordinates,
        eigenvalues, metric, threshold, remaining;
        survival_rtol = 1.0e-6, time_rtol = 1.0e-12
    )
    @test diagonal.tau ≈ expected rtol = 5.0e-12
end

struct _CountingIdentityMetric <: AbstractMatrix{ComplexF64}
    n::Int
    multiplies::Base.RefValue{Int}
end

Base.size(metric::_CountingIdentityMetric) = (metric.n, metric.n)
Base.getindex(metric::_CountingIdentityMetric, i::Int, j::Int) =
    i == j ? 1.0 + 0im : 0.0 + 0im

function LinearAlgebra.mul!(
        out::AbstractVector{ComplexF64},
        metric::_CountingIdentityMetric, state::AbstractVector{ComplexF64}
    )
    metric.multiplies[] += 1
    return copyto!(out, state)
end

struct _CountingDenseMetric <: AbstractMatrix{ComplexF64}
    data::Matrix{ComplexF64}
    multiplies::Base.RefValue{Int}
end

Base.size(metric::_CountingDenseMetric) = size(metric.data)
Base.getindex(metric::_CountingDenseMetric, i::Int, j::Int) = metric.data[i, j]

function LinearAlgebra.mul!(
        out::AbstractVector{ComplexF64},
        metric::_CountingDenseMetric, state::AbstractVector{ComplexF64}
    )
    metric.multiplies[] += 1
    return mul!(out, metric.data, state)
end

@testset "bounded identity metric bypasses full Gram products" begin
    N = 64
    identity = Matrix{ComplexF64}(I, N, N)
    @test DiSLOUTrajectories._identity_metric_error(identity) == 0
    identity[1, 2] = eps(Float64)
    @test DiSLOUTrajectories._identity_metric_error(identity) == eps(Float64)

    energies = collect(range(-3.0, 3.0; length = N))
    rates = collect(range(0.05, 2.0; length = N))
    eigenvalues = ComplexF64.(energies .- 0.5im .* rates)
    coordinates = normalize!(randn(Xoshiro(0x20260831), ComplexF64, N))
    survival(time) = sum(
        abs2(coordinates[j]) * exp(-rates[j] * time)
            for j in eachindex(coordinates)
    )
    remaining = 1.0
    threshold = (1 + survival(remaining)) / 2
    lo, hi = 0.0, remaining
    for _ in 1:100
        mid = (lo + hi) / 2
        survival(mid) > threshold ? (lo = mid) : (hi = mid)
    end
    reference_time = (lo + hi) / 2

    counter = Ref(0)
    metric = _CountingIdentityMetric(N, counter)
    evolved = similar(coordinates)
    scratch = similar(coordinates)
    for method in (:survival, :log_survival, :log_survival_predictor)
        counter[] = 0
        predictor = method === :log_survival_predictor ?
            DiSLOUTrajectories._FirstPassagePredictor(N; m = 8) : nothing
        result = DiSLOUTrajectories._find_diagonal_first_passage!(
            evolved, scratch, coordinates, eigenvalues, metric,
            threshold, remaining; survival_rtol = 1.0e-10,
            time_rtol = 1.0e-12, time_atol = 0.0, root_variant = method,
            rg = predictor, Gnorm = 1.0, metric_error = 0.0
        )
        @test counter[] == 0
        @test result.tau ≈ reference_time atol = 1.0e-11 rtol = 0
        @test abs(log(survival(result.tau)) - log(threshold)) <= 1.0e-10
    end

    counter[] = 0
    nojump = DiSLOUTrajectories._find_diagonal_first_passage!(
        evolved, scratch, coordinates, eigenvalues, metric,
        survival(remaining) / 2, remaining; root_variant = :log_survival,
        Gnorm = 1.0, metric_error = 1.0e-12
    )
    @test !nojump.jumped
    @test counter[] == 0

    counter[] = 0
    approximate = DiSLOUTrajectories._find_diagonal_first_passage!(
        evolved, scratch, coordinates, eigenvalues, metric,
        threshold, remaining; survival_rtol = 1.0e-10,
        time_rtol = 1.0e-12, root_variant = :log_survival,
        Gnorm = 1.0, metric_error = 1.0e-12
    )
    @test counter[] == 1
    @test approximate.tau ≈ reference_time atol = 1.0e-11 rtol = 0
    @test abs(log(survival(approximate.tau)) - log(threshold)) <= 1.0e-10

    counter[] = 0
    DiSLOUTrajectories._find_diagonal_first_passage!(
        evolved, scratch, coordinates, eigenvalues, metric,
        threshold, remaining; root_variant = :log_survival,
        Gnorm = 1.0, metric_error = 1.0e-6
    )
    @test counter[] > 0
end

@testset "guarded metric shortcut handles a nonidentity Gram matrix" begin
    metric_perturbation = 1.0e-12
    dense_metric = ComplexF64[
        1 metric_perturbation;
        metric_perturbation 1
    ]
    metric_error = DiSLOUTrajectories._identity_metric_error(dense_metric)
    @test metric_error == metric_perturbation

    counter = Ref(0)
    metric = _CountingDenseMetric(dense_metric, counter)
    eigenvalues = ComplexF64[0.3 - 0.1im, -0.7 - 0.9im]
    coordinates = ComplexF64[0.6 + 0.2im, -0.3 + 0.65im]
    coordinates ./= sqrt(real(dot(coordinates, dense_metric, coordinates)))
    evolved = similar(coordinates)
    scratch = similar(coordinates)
    Gnorm = opnorm(dense_metric, Inf)

    probe_time = 0.41
    DiSLOUTrajectories._phase_evolve!(evolved, eigenvalues, coordinates, probe_time)
    estimate = DiSLOUTrajectories._survival_and_rate_identity!(
        evolved, eigenvalues, metric_error
    )
    exact = _test_survival_and_rate!(
        scratch, evolved, eigenvalues, dense_metric, Gnorm
    )

    inside_threshold = estimate.S + estimate.error / 2
    counter[] = 0
    inside = DiSLOUTrajectories._certified_full_survival_and_rate!(
        scratch, evolved, eigenvalues, metric, Gnorm, metric_error,
        inside_threshold
    )
    @test counter[] == 1
    @test inside.S == exact.S
    @test inside.R == exact.R

    outside_threshold = estimate.S + 2estimate.error
    counter[] = 0
    outside = DiSLOUTrajectories._certified_full_survival_and_rate!(
        scratch, evolved, eigenvalues, metric, Gnorm, metric_error,
        outside_threshold
    )
    @test counter[] == 0
    @test (outside.S > outside_threshold) == (exact.S > outside_threshold)

    target_time = 0.37
    threshold = _test_survival_probability(
        coordinates, eigenvalues, dense_metric, target_time
    )
    remaining = 1.2
    for method in (:survival, :log_survival, :log_survival_predictor)
        dense_predictor = method === :log_survival_predictor ?
            DiSLOUTrajectories._FirstPassagePredictor(2; m = 2) : nothing
        counter[] = 0
        dense = DiSLOUTrajectories._find_diagonal_first_passage!(
            evolved, scratch, coordinates, eigenvalues, metric,
            threshold, remaining; survival_rtol = 1.0e-10,
            time_rtol = 1.0e-12, root_variant = method,
            rg = dense_predictor, Gnorm, metric_error = Inf
        )
        dense_products = counter[]

        fast_predictor = method === :log_survival_predictor ?
            DiSLOUTrajectories._FirstPassagePredictor(2; m = 2) : nothing
        counter[] = 0
        fast = DiSLOUTrajectories._find_diagonal_first_passage!(
            evolved, scratch, coordinates, eigenvalues, metric,
            threshold, remaining; survival_rtol = 1.0e-10,
            time_rtol = 1.0e-12, root_variant = method,
            rg = fast_predictor, Gnorm, metric_error
        )
        fast_products = counter[]

        @test dense.jumped && fast.jumped
        @test fast.tau ≈ dense.tau atol = 1.0e-11 rtol = 0
        @test abs(
            _test_survival_probability(
                coordinates, eigenvalues, dense_metric, fast.tau
            ) - threshold
        ) <= 1.0e-10
        @test fast_products < dense_products
    end
end

function _first_passage_kernel_allocations()
    H, c_ops = random_system(N = 6, seed = 12)
    cache = DiSLOUTrajectories._DiagonalCache(H, c_ops)
    coordinates = ones(ComplexF64, cache.N)
    coordinates ./= sqrt(real(dot(coordinates, cache.G, coordinates)))
    evolved = similar(coordinates)
    scratch = similar(coordinates)
    DiSLOUTrajectories._phase_evolve!(evolved, cache.Λ, coordinates, 0.4)
    _test_survival_and_rate!(scratch, evolved, cache.Λ, cache.G)
    return @allocated begin
        DiSLOUTrajectories._phase_evolve!(evolved, cache.Λ, coordinates, 0.4)
        _test_survival_and_rate!(scratch, evolved, cache.Λ, cache.G)
    end
end

@testset "first-passage inner kernels allocate no memory" begin
    @test _first_passage_kernel_allocations() == 0
end
