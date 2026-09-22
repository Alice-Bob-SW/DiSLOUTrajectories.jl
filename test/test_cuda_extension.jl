if Sys.islinux() || Sys.iswindows()
    using CUDA

    @testset "optional CUDA full preparation extension" begin
        @test Base.get_extension(DiSLOUTrajectories, :DiSLOUTrajectoriesCUDAExt) !== nothing
        @test backend_info().cuda_enabled == CUDA.functional()

        if CUDA.functional()
            was_enabled = backend_info().cuda_enabled
            try
                CUDA.allowscalar(false)
                rng = Xoshiro(92)
                N = 8
                X = randn(rng, ComplexF64, N, N)
                H = Matrix(Hermitian(X))
                C = [randn(rng, ComplexF64, N, N) / sqrt(N)]
                Z = [randn(rng, ComplexF64, N, N)]
                H_before, C_before, Z_before = copy(H), copy.(C), copy.(Z)
                H_eff = DiSLOUTrajectories._effective_hamiltonian(H, C)

                DiSLOUTrajectories._enable_cuda_diagonalization!()
                cache = DiSLOUTrajectories._diagonal_cache_from_matrices(H, C; Z)

                @test H == H_before
                @test C == C_before
                @test Z == Z_before
                @test cache.backend === :cuda
                @test cache.V isa Matrix{ComplexF64}
                @test cache.Λ isa Vector{ComplexF64}
                @test cache.Γ isa Vector{Float64}
                @test cache.Vfac isa LU{ComplexF64, Matrix{ComplexF64}, Vector{Int}}
                @test cache.G isa Matrix{ComplexF64}
                @test cache.A isa Vector{Matrix{ComplexF64}}
                @test cache.M isa Vector{Matrix{ComplexF64}}
                @test cache.ZV isa Vector{Matrix{ComplexF64}}
                sparse_cache = DiSLOUTrajectories._diagonal_cache_from_matrices(
                    H, C; Z = sparse.(Z), observable_storage = :sparse
                )
                @test sparse_cache.backend === :cuda
                @test sparse_cache.observable_storage === :sparse
                @test sparse_cache.Z isa Vector{SparseMatrixCSC{ComplexF64, Int}}
                @test isempty(sparse_cache.ZV)
                @test norm(
                    H_eff * cache.V -
                        cache.V * Diagonal(cache.Λ)
                ) /
                    (norm(H_eff) * norm(cache.V)) < 1.0e-10
                @test all(j -> isapprox(norm(cache.V[:, j]), 1; atol = 1.0e-12), 1:N)
                @test all(
                    j -> begin
                        v = cache.V[:, j]
                        pivot = v[argmax(abs.(v))]
                        real(pivot) > 0 && abs(imag(pivot)) < 1.0e-12
                    end, 1:N
                )
                coordinates = similar(cache.V)
                ldiv!(coordinates, cache.Vfac, cache.V)
                @test coordinates ≈ I atol = 1.0e-12
                @test cache.G ≈ cache.V' * cache.V
                @test cache.A[1] ≈ C[1] * cache.V
                @test cache.M[1] ≈ cache.A[1]' * cache.A[1]
                @test cache.ZV[1] ≈ cache.V' * Z[1] * cache.V
                @test cache.Gnorm ≈ opnorm(cache.G, Inf)
                @test cache.Mnorms ≈ [opnorm(M, Inf) for M in cache.M]
                @test cache.κV ≈ cond(cache.V, 1) rtol = 1.0e-10
                @test cache.metric_error ≈ DiSLOUTrajectories._identity_metric_error(cache.G)
            finally
                was_enabled ? DiSLOUTrajectories._enable_cuda_diagonalization!() :
                    DiSLOUTrajectories._disable_cuda_diagonalization!()
            end
        end
    end
else
    @testset "optional CUDA full preparation extension" begin
        @test_skip "CUDA runtime activation is unsupported on this platform"
    end
end
