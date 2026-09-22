@testset "Layer II diagonal primitives retain exact cache and reusable buffers" begin
    H, c_ops = random_system(N = 5, seed = 29)
    cache = DiSLOUTrajectories._DiagonalCache(H, c_ops)
    buffers = DiSLOUTrajectories._FirstPassageBuffers(cache)
    ψ = normalize!(randn(Xoshiro(30), ComplexF64, cache.N))
    c = cache.Vfac \ ψ
    H_eff = DiSLOUTrajectories._effective_hamiltonian(cache.H, cache.C)

    @test cache isa DiSLOUTrajectories._DiagonalCache
    @test length(buffers.Dc) == cache.N
    @test DiSLOUTrajectories._survival_probability!(
        buffers.Dc, buffers.Gc, c, cache.Λ, cache.G, 0.4, cache.Gnorm
    ) ≈
        direct_survival(H_eff, ψ, 0.4) atol = 1.0e-12

    err = DiSLOUTrajectories.FirstPassageConvergenceError(0.4, 7, 1.0e-8, 1.0e-9)
    @test (err.time, err.iterations, err.residual, err.bracket_width) ==
        (0.4, 7, 1.0e-8, 1.0e-9)
end

@testset "observable storage controls Layer II caches" begin
    H, c_ops = random_system(N = 6, seed = 211)
    observable = num(6)
    dense_cache = DiSLOUTrajectories._DiagonalCache(
        H, c_ops; e_ops = [observable], observable_storage = :dense
    )
    sparse_cache = DiSLOUTrajectories._DiagonalCache(
        H, c_ops; e_ops = [observable], observable_storage = :sparse
    )

    @test dense_cache.observable_storage === :dense
    @test dense_cache.Z isa Vector{Matrix{ComplexF64}}
    @test length(dense_cache.ZV) == 1
    @test sparse_cache.observable_storage === :sparse
    @test sparse_cache.Z isa Vector{SparseMatrixCSC{ComplexF64, Int}}
    @test isempty(sparse_cache.ZV)
    @test sparse_cache.Z[1] == sparse(observable.data)

    dense_reduced = DiSLOUTrajectories._get_reduced!(dense_cache, [1, 3])
    sparse_reduced = DiSLOUTrajectories._get_reduced!(sparse_cache, [1, 3])
    @test length(dense_reduced.Z_I) == 1
    @test isempty(sparse_reduced.Z_I)
end

@testset "observable validation preserves selected format" begin
    H = num(4)
    dimensions = H.dimensions
    dense = QuantumObject(Matrix(num(4).data); dims = dimensions)
    sparse_op = QuantumObject(sparse(num(4).data); dims = dimensions)
    wrapped = Diagonal(ComplexF64[0, 1, 2, 3])
    wrapped_qobj = QuantumObject(wrapped; dims = dimensions)

    @test only(
        DiSLOUTrajectories._validated_observables(
            [sparse_op], 4, dimensions, :dense
        )
    ) isa Matrix{ComplexF64}
    @test only(
        DiSLOUTrajectories._validated_observables(
            [dense], 4, dimensions, :sparse
        )
    ) isa SparseMatrixCSC{ComplexF64, Int}
    @test only(
        DiSLOUTrajectories._validated_observables(
            [wrapped], 4, dimensions, :dense
        )
    ) isa Matrix{ComplexF64}
    @test only(
        DiSLOUTrajectories._validated_observables(
            [wrapped], 4, dimensions, :sparse
        )
    ) isa SparseMatrixCSC{ComplexF64, Int}
    @test_throws ArgumentError DiSLOUTrajectories._validated_observables(
        [wrapped_qobj], 4, dimensions, :dense
    )
    @test_throws ArgumentError DiSLOUTrajectories._validated_concrete_observable_storage(:auto)
    @test_throws ArgumentError DiSLOUTrajectories._validated_concrete_observable_storage(:packed)
end

@testset "Layer II cache identities" begin
    H, c_ops = random_system(N = 6, seed = 2)
    observable = QuantumObject(Matrix(Diagonal(ComplexF64.(0:5))))
    cache = DiSLOUTrajectories._DiagonalCache(H, c_ops; e_ops = [observable])
    Hmat = Matrix{ComplexF64}(H.data)
    collapse = Matrix{ComplexF64}[Matrix{ComplexF64}(op.data) for op in c_ops]
    effective = Hmat - (im / 2) * sum(C' * C for C in collapse)

    @test cache.condition_limit == DiSLOUTrajectories._DEFAULT_CONDITION_LIMIT
    @test all(j -> isapprox(norm(cache.V[:, j]), 1.0; atol = 1.0e-12), 1:cache.N)
    @test all(j -> isapprox(real(cache.G[j, j]), 1.0; atol = 1.0e-10), 1:cache.N)
    @test effective * cache.V ≈ cache.V * Diagonal(cache.Λ)
    coordinates = similar(cache.V)
    ldiv!(coordinates, cache.Vfac, cache.V)
    @test coordinates ≈ I
    @test cache.G ≈ cache.V' * cache.V
    @test all(cache.A[μ] ≈ collapse[μ] * cache.V for μ in 1:cache.Nc)
    @test all(cache.M[μ] ≈ cache.A[μ]' * cache.A[μ] for μ in 1:cache.Nc)
    @test all(cache.Γ .>= -1.0e-10)
    @test cache.Γ ≈ -2 .* imag.(cache.Λ)
    @test cache.ZV[1] ≈ cache.V' * Matrix(observable.data) * cache.V
end

@testset "Layer II raw-matrix cache matches the quantum-object boundary" begin
    H, c_ops = random_system(N = 6, seed = 1)
    e_ops = [num(6)]
    from_objects = DiSLOUTrajectories._DiagonalCache(H, c_ops; e_ops)
    Hmat = Matrix{ComplexF64}(H.data)
    collapse = Matrix{ComplexF64}[Matrix{ComplexF64}(op.data) for op in c_ops]
    observables = Matrix{ComplexF64}[Matrix{ComplexF64}(op.data) for op in e_ops]
    from_matrices = DiSLOUTrajectories._diagonal_cache_from_matrices(Hmat, collapse; Z = observables)

    @test from_objects.Λ ≈ from_matrices.Λ rtol = 1.0e-10
    @test from_objects.V ≈ from_matrices.V rtol = 1.0e-10
    @test from_objects.G ≈ from_matrices.G rtol = 1.0e-10
    @test Matrix(from_objects.Vfac) ≈ Matrix(from_matrices.Vfac) rtol = 1.0e-10
    @test all(
        isapprox(from_objects.A[μ], from_matrices.A[μ]; rtol = 1.0e-10)
            for μ in eachindex(collapse)
    )
    @test all(
        isapprox(from_objects.M[μ], from_matrices.M[μ]; rtol = 1.0e-10)
            for μ in eachindex(collapse)
    )
    @test from_objects.ZV[1] ≈ from_matrices.ZV[1] rtol = 1.0e-10
    @test from_objects.normal == from_matrices.normal
    @test from_objects.κV ≈ from_matrices.κV rtol = 1.0e-10

    @test_throws ArgumentError DiSLOUTrajectories._diagonal_cache_from_matrices(
        ones(ComplexF64, 2, 3), Matrix{ComplexF64}[]
    )
    @test_throws ArgumentError DiSLOUTrajectories._diagonal_cache_from_matrices(
        Matrix{ComplexF64}(I, 2, 2), [ones(ComplexF64, 3, 3)]
    )
    @test_throws ArgumentError DiSLOUTrajectories._diagonal_cache_from_matrices(
        ComplexF64[NaN 0; 0 1], Matrix{ComplexF64}[]
    )
end

@testset "Layer II Hermitian validation is scale aware" begin
    for scale in (1.0e-100, 1.0e100)
        accepted_delta = 32eps(Float64) * scale
        rejected_delta = 128eps(Float64) * scale
        accepted = ComplexF64[scale accepted_delta; 0 -scale]
        rejected = ComplexF64[scale rejected_delta; 0 -scale]
        cache = DiSLOUTrajectories._DiagonalCache(
            QuantumObject(accepted), QuantumObject[];
            e_ops = [QuantumObject(accepted)]
        )
        expected = (accepted + accepted') / 2

        @test cache.H == expected
        @test cache.Z[1] == accepted
        @test_throws ArgumentError DiSLOUTrajectories._DiagonalCache(
            QuantumObject(rejected), QuantumObject[]
        )
        observable_cache = DiSLOUTrajectories._DiagonalCache(
            QuantumObject(Matrix(Diagonal(ComplexF64[scale, -scale]))),
            QuantumObject[]; e_ops = [QuantumObject(rejected)]
        )
        @test observable_cache.Z[1] == rejected
    end

    extreme = ComplexF64[
        floatmax(Float64) floatmax(Float64);
        0.0 floatmax(Float64)
    ]
    @test all(isfinite, extreme)
    @test_throws ArgumentError DiSLOUTrajectories._checked_matrix(extreme, "H"; hermitian = true)
end

# Mutations caught: absolute-only comparisons, nontransitive grouping, or
# accepting invalid relative-plus-absolute tolerances.
@testset "Layer II degeneracy clusters are scale aware and transitive" begin
    chain = ComplexF64[0, 0.6 + 0.6im, 1.2 + 1.2im, 4 + 3im]
    @test DiSLOUTrajectories.degeneracy_clusters(chain; rtol = 0.0, atol = 1.0) ==
        [[1, 2, 3], [4]]

    large = ComplexF64[1.0e8 + 2.0e8im, 1.0e8 + 1 + 2.0e8im, -1.0e8im]
    @test DiSLOUTrajectories.degeneracy_clusters(large; rtol = 1.0e-8, atol = 0.0) ==
        [[1, 2], [3]]

    @test_throws ArgumentError DiSLOUTrajectories.degeneracy_clusters(
        chain; rtol = 0.0, atol = 0.0
    )
    @test_throws ArgumentError DiSLOUTrajectories.degeneracy_clusters(
        chain; rtol = -1.0, atol = 1.0
    )
    @test_throws ArgumentError DiSLOUTrajectories.degeneracy_clusters(
        chain; rtol = NaN, atol = 1.0
    )
    @test_throws ArgumentError DiSLOUTrajectories.degeneracy_clusters(
        chain; rtol = 1.0, atol = Inf
    )
end

@testset "Layer II cache conditioning and norm scales" begin
    N = 8
    a = destroy(N)
    normal_cache = DiSLOUTrajectories._DiagonalCache(0 * a, [a, 0.5 * a^2])
    @test normal_cache.normal
    @test normal_cache.metric_error == 0
    @test DiSLOUTrajectories._use_identity_metric(normal_cache.metric_error, 1.0e-10)
    @test isapprox(normal_cache.κV, 1.0; atol = 1.0e-6)
    @test opnorm(normal_cache.G - I) < 1.0e-8

    zero_tol_cache = DiSLOUTrajectories._DiagonalCache(0 * a, [a]; normal_tol = 0)
    @test !zero_tol_cache.normal
    @test zero_tol_cache.metric_error == 0
    @test DiSLOUTrajectories._use_identity_metric(zero_tol_cache.metric_error, 1.0e-10)

    rng = Xoshiro(118)
    Q = Matrix(qr(randn(rng, ComplexF64, N, N)).Q)
    energies = collect(range(-2.0, 2.0; length = N))
    rates = collect(range(0.1, 0.8; length = N))
    dense_normal = DiSLOUTrajectories._DiagonalCache(
        QuantumObject(Q * Diagonal(energies) * Q'),
        [QuantumObject(Q * Diagonal(sqrt.(rates)) * Q')]
    )
    @test dense_normal.normal
    @test 0 < dense_normal.metric_error < 1.0e-10
    @test DiSLOUTrajectories._use_identity_metric(dense_normal.metric_error, 1.0e-10)
    @test !DiSLOUTrajectories._use_identity_metric(
        dense_normal.metric_error, dense_normal.metric_error
    )

    V = ComplexF64[
        -inv(sqrt(2)) -2inv(sqrt(5));
        -inv(sqrt(2)) -inv(sqrt(5))
    ]
    Vfac = lu(V; check = false)
    metric_error = DiSLOUTrajectories._identity_metric_error(V' * V)
    κV, normal = DiSLOUTrajectories._validate_eigenbasis(
        V, Vfac, metric_error; condition_limit = 7.2, normal_tol = 1.0e-9
    )
    @test κV ≈ 4 + sqrt(10)
    @test !normal
    @test_throws ArgumentError DiSLOUTrajectories._validate_eigenbasis(
        V, Vfac, metric_error; condition_limit = 7.1, normal_tol = 1.0e-9
    )
    @test_throws MethodError DiSLOUTrajectories._validate_eigenbasis(
        V; condition_limit = 7.2, normal_tol = 1.0e-9
    )
    singular = ComplexF64[1 1; 1 1]
    @test_throws ArgumentError DiSLOUTrajectories._validate_eigenbasis(
        singular, lu(singular; check = false),
        DiSLOUTrajectories._identity_metric_error(singular' * singular);
        condition_limit = 1.0e9, normal_tol = 1.0e-9
    )
    identity = Matrix{ComplexF64}(I, 2, 2)
    identity_fac = lu(identity; check = false)
    for invalid_limit in (0.0, 0.99, Inf, NaN, big(10)^1000)
        @test_throws ArgumentError DiSLOUTrajectories._validate_eigenbasis(
            identity, identity_fac, 0.0;
            condition_limit = invalid_limit, normal_tol = 1.0e-9
        )
    end
    @test_throws ArgumentError DiSLOUTrajectories._validate_eigenbasis(
        identity, identity_fac, 0.0; condition_limit = 2.0,
        normal_tol = -eps(Float64)
    )

    H, c_ops = random_system(N = 4, seed = 8)
    cache = DiSLOUTrajectories._DiagonalCache(H, c_ops)
    reduced = DiSLOUTrajectories._get_reduced!(cache, [1, 3])
    @test cache.Gnorm ≈ opnorm(cache.G, Inf)
    @test cache.Mnorms ≈ [opnorm(M, Inf) for M in cache.M]
    @test reduced.Gnorm ≈ opnorm(reduced.G_I, Inf)
    @test reduced.Knorms ≈ [opnorm(K, Inf) for K in reduced.K_I]
end

struct _FixedFloatRNG <: AbstractRNG
    value::Float64
end
Random.rand(rng::_FixedFloatRNG) = rng.value

@testset "Layer II jump primitives preserve physical weights and states" begin
    H, c_ops = random_system(N = 6, seed = 12)
    cache = DiSLOUTrajectories._DiagonalCache(H, c_ops)
    rng = Xoshiro(3)
    coordinates = randn(rng, ComplexF64, cache.N)
    state = cache.V * coordinates
    @test all(
        isapprox(
            real(dot(coordinates, cache.M[μ], coordinates)),
            norm(cache.C[μ] * state)^2; atol = 1.0e-9
        ) for μ in 1:cache.Nc
    )

    reduced = DiSLOUTrajectories._get_reduced!(cache, [1, 3])
    local_coordinates = randn(rng, ComplexF64, 2)
    local_state = reduced.V_I * local_coordinates
    @test all(
        isapprox(
            real(dot(local_coordinates, reduced.K_I[μ], local_coordinates)),
            norm(cache.C[μ] * local_state)^2; atol = 1.0e-9
        ) for μ in 1:cache.Nc
    )

    zero = _FixedFloatRNG(0.0)
    upper = _FixedFloatRNG(nextfloat(1.0))
    @test DiSLOUTrajectories._sample_channel([0.0, 2.0, 0.0], zero) == 2
    @test DiSLOUTrajectories._sample_channel([0.0, 2.0, 0.0], upper) == 2
    @test DiSLOUTrajectories._sample_channel([2.0, 0.0], upper) == 1
    @test_throws DomainError DiSLOUTrajectories._sample_channel([1.0, -eps(Float64)], zero)
    @test_throws DomainError DiSLOUTrajectories._sample_channel([1.0, NaN], zero)
    @test_throws DomainError DiSLOUTrajectories._sample_channel([1.0, Inf], zero)
    @test_throws DomainError DiSLOUTrajectories._sample_channel([0.0, 0.0], zero)

    post_coordinates = similar(coordinates)
    buffers = DiSLOUTrajectories._WorkBuffers(cache)
    channel = DiSLOUTrajectories._apply_exact_jump_coordinates!(
        post_coordinates, buffers, cache, coordinates, Xoshiro(5)
    )
    post_state = cache.V * post_coordinates
    reference = cache.C[channel] * state
    reference ./= norm(reference)
    @test buffers.ψ ≈ reference atol = 1.0e-10
    @test post_state ≈ buffers.ψ atol = 1.0e-10
    @test norm(post_state) ≈ 1.0 atol = 1.0e-10
    @test abs(dot(reference, post_state)) ≈ 1.0 atol = 1.0e-9
end
