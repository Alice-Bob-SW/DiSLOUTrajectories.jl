# The eigensystem identities themselves live in test_layer2_primitives.jl;
# what this owns is the backend dispatch: with CUDA off, preparation must
# report :lapack and hand back the documented field types.
@testset "CPU LAPACK preparation reports the LAPACK backend" begin
    was_enabled = backend_info().cuda_enabled
    try
        SM._disable_cuda_diagonalization!()
        rng = Xoshiro(91)
        N = 8
        X = randn(rng, ComplexF64, N, N)
        H = Matrix(Hermitian(X))
        C = [randn(rng, ComplexF64, N, N) / sqrt(N)]
        Z = [randn(rng, ComplexF64, N, N)]
        result = SM._prepare_diagonal_cache_data(H, C, Z)
        @test result.backend === :lapack
        @test result.V isa Matrix{ComplexF64}
        @test result.Λ isa Vector{ComplexF64}
        @test result.Vfac isa LU{ComplexF64, Matrix{ComplexF64}, Vector{Int}}
        @test backend_info().cuda_enabled === false
    finally
        was_enabled ? SM._enable_cuda_diagonalization!() :
            SM._disable_cuda_diagonalization!()
    end
end

@testset "CUDA preparation fallback without an extension method" begin
    was_enabled = backend_info().cuda_enabled
    try
        rng = Xoshiro(92)
        N = 8
        X = randn(rng, ComplexF64, N, N)
        H = Matrix(Hermitian(X))
        C = [randn(rng, ComplexF64, N, N) / sqrt(N)]
        SM._enable_cuda_diagonalization!()
        result = @test_logs (:warn, r"CUDA preparation failed.*retrying with LAPACK") begin
            SM._prepare_diagonal_cache_data(H, C, Matrix{ComplexF64}[])
        end
        H_eff = SM._effective_hamiltonian(H, C)
        @test result.backend === :lapack
        @test norm(
            H_eff * result.V -
                result.V * Diagonal(result.Λ)
        ) /
            (norm(H_eff) * norm(result.V)) < 1.0e-12
        @test backend_info().cuda_enabled === false
    finally
        was_enabled ? SM._enable_cuda_diagonalization!() :
            SM._disable_cuda_diagonalization!()
    end
end

const _DISLOU_WHOLE_PREP_CALLS = Ref(0)
const _DISLOU_WHOLE_PREP_RESULT = Ref{Any}(nothing)

function SM._cuda_prepare_diagonal_data(
        ::Matrix{ComplexF64},
        C::Vector{Matrix{ComplexF64}}, Z::Vector{Matrix{ComplexF64}}
    )
    _DISLOU_WHOLE_PREP_CALLS[] += 1
    V = ComplexF64[
        -inv(sqrt(2)) -2inv(sqrt(5));
        -inv(sqrt(2)) -inv(sqrt(5))
    ]
    A = [Cμ * V for Cμ in C]
    result = (;
        backend = :cuda,
        V,
        Λ = ComplexF64[1, 2 - 0.5im],
        Γ = Float64[0, 1],
        Vfac = lu(V),
        G = V' * V,
        Gnorm = opnorm(V' * V, Inf),
        A,
        M = [Aμ' * Aμ for Aμ in A],
        Mnorms = [opnorm(Aμ' * Aμ, Inf) for Aμ in A],
        ZV = [V' * Ze * V for Ze in Z],
        metric_error = SM._identity_metric_error(V' * V),
    )
    _DISLOU_WHOLE_PREP_RESULT[] = result
    return result
end

@testset "CUDA hook owns the whole preparation result" begin
    was_enabled = backend_info().cuda_enabled
    H = ComplexF64[1 0; 0 2]
    C = [ComplexF64[0 0; 0 1]]
    Z = [ComplexF64[3 0; 0 4]]
    try
        _DISLOU_WHOLE_PREP_CALLS[] = 0
        _DISLOU_WHOLE_PREP_RESULT[] = nothing
        SM._enable_cuda_diagonalization!()
        cache = SM._diagonal_cache_from_matrices(H, C; Z)
        @test _DISLOU_WHOLE_PREP_CALLS[] == 1
        if _DISLOU_WHOLE_PREP_RESULT[] !== nothing
            prepared = _DISLOU_WHOLE_PREP_RESULT[]
            @test cache.backend === :cuda
            @test cache.Vfac === prepared.Vfac
            @test cache.G === prepared.G
            @test cache.A === prepared.A
            @test cache.M === prepared.M
            @test cache.ZV === prepared.ZV
            @test cache.κV ≈ cond(prepared.V, 1)
            @test_throws ArgumentError SM._diagonal_cache_from_matrices(
                H, C; Z, condition_limit = 7.1
            )
        end
    finally
        Base.delete_method(
            which(
                SM._cuda_prepare_diagonal_data,
                Tuple{
                    Matrix{ComplexF64}, Vector{Matrix{ComplexF64}},
                    Vector{Matrix{ComplexF64}},
                }
            )
        )
        was_enabled ? SM._enable_cuda_diagonalization!() :
            SM._disable_cuda_diagonalization!()
    end
end

const _DISLOU_CUDA_MOCK_CALLS = Ref(0)
const _DISLOU_CUDA_MOCK_FAIL_AT = Ref(typemax(Int))

function SM._cuda_prepare_diagonal_data(
        H::Matrix{ComplexF64},
        C::Vector{Matrix{ComplexF64}}, Z::Vector{Matrix{ComplexF64}}
    )
    _DISLOU_CUDA_MOCK_CALLS[] += 1
    _DISLOU_CUDA_MOCK_CALLS[] == _DISLOU_CUDA_MOCK_FAIL_AT[] &&
        error("mock CUDA preparation failure")
    return merge(SM._cpu_prepare_diagonal_data(H, C, Z), (; backend = :cuda))
end

@testset "solutions report the actual backend and warn once after CUDA fallback" begin
    was_enabled = backend_info().cuda_enabled
    d = 3
    a = destroy(d)
    common = (;
        gauge_set = ComplexF64[0 0.4], ntraj = 2,
        seed = 51, ensemblealg = :serial,
    )
    try
        _DISLOU_CUDA_MOCK_CALLS[] = 0
        _DISLOU_CUDA_MOCK_FAIL_AT[] = typemax(Int)
        SM._enable_cuda_diagonalization!()
        cuda = dislou_solve(0.1 * num(d), fock(d, 1), [0.0, 0.1], [a]; common...)
        @test cuda.eigensystem_backend === :cuda
        @test _DISLOU_CUDA_MOCK_CALLS[] == size(common.gauge_set, 2)

        outcomes = @test_logs (:warn, r"disabling CUDA.*retrying with LAPACK") begin
            _DISLOU_CUDA_MOCK_CALLS[] = 0
            _DISLOU_CUDA_MOCK_FAIL_AT[] = 1
            SM._enable_cuda_diagonalization!()
            first_fallback = dislou_solve(
                0.1 * num(d), fock(d, 1), [0.0, 0.1], [a]; common...
            )
            first_disabled = !backend_info().cuda_enabled

            _DISLOU_CUDA_MOCK_CALLS[] = 0
            _DISLOU_CUDA_MOCK_FAIL_AT[] = 1
            SM._enable_cuda_diagonalization!()
            second_fallback = dislou_solve(
                0.1 * num(d), fock(d, 1), [0.0, 0.1], [a]; common...
            )
            second_disabled = !backend_info().cuda_enabled

            _DISLOU_CUDA_MOCK_CALLS[] = 0
            _DISLOU_CUDA_MOCK_FAIL_AT[] = 2
            SM._enable_cuda_diagonalization!()
            mixed = dislou_solve(
                0.1 * num(d), fock(d, 1), [0.0, 0.1], [a]; common...
            )
            (;
                first_fallback, first_disabled, second_fallback,
                second_disabled, mixed,
                mixed_calls = _DISLOU_CUDA_MOCK_CALLS[],
            )
        end
        @test outcomes.first_fallback.eigensystem_backend === :lapack
        @test outcomes.second_fallback.eigensystem_backend === :lapack
        @test outcomes.first_disabled && outcomes.second_disabled
        @test outcomes.first_fallback.col_times ==
            outcomes.second_fallback.col_times
        @test outcomes.mixed_calls == 2
        @test outcomes.mixed.eigensystem_backend === :mixed
        @test backend_info().cuda_enabled === false
    finally
        Base.delete_method(
            which(
                SM._cuda_prepare_diagonal_data,
                Tuple{
                    Matrix{ComplexF64}, Vector{Matrix{ComplexF64}},
                    Vector{Matrix{ComplexF64}},
                }
            )
        )
        was_enabled ? SM._enable_cuda_diagonalization!() :
            SM._disable_cuda_diagonalization!()
    end
end

@testset "CUDA interruption propagates without disabling the backend" begin
    function SM._cuda_prepare_diagonal_data(
            ::Matrix{ComplexF64},
            ::Vector{Matrix{ComplexF64}}, ::Vector{Matrix{ComplexF64}}
        )
        throw(InterruptException())
    end
    was_enabled = backend_info().cuda_enabled
    try
        SM._enable_cuda_diagonalization!()
        interrupted = try
            SM._prepare_diagonal_cache_data(
                Matrix{ComplexF64}(I, 2, 2),
                Matrix{ComplexF64}[], Matrix{ComplexF64}[]
            )
            false
        catch err
            err isa InterruptException
        end
        @test interrupted
        @test backend_info().cuda_enabled
    finally
        Base.delete_method(
            which(
                SM._cuda_prepare_diagonal_data,
                Tuple{
                    Matrix{ComplexF64}, Vector{Matrix{ComplexF64}},
                    Vector{Matrix{ComplexF64}},
                }
            )
        )
        was_enabled ? SM._enable_cuda_diagonalization!() :
            SM._disable_cuda_diagonalization!()
    end
end
