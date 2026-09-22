@testset "initial state normalization is scale invariant" begin
    for scale in (1.0, 1.0e-200, 1.0e200, nextfloat(0.0), floatmax(Float64), big"1e400", big"1e-400"),
            direction in (ComplexF64[1, 0], ComplexF64[1 + im, -1 + im])
        @testset "scale=$scale, direction=$direction" begin
            psi0 = scale .* direction
            original = copy(psi0)
            expected = normalize(direction)
            sol = dislou_solve(
                zeros(ComplexF64, 2, 2), psi0, [0.0, 1.0], Matrix{ComplexF64}[];
                gauge_set = zeros(ComplexF64, 0, 1), ntraj = 1,
                ensemblealg = :serial, save_final_states = true
            )
            @test sol.final_states[:, 1] ≈ expected atol = 1.0e-14
            @test sol.states[end].data ≈ expected * expected' atol = 1.0e-14
            @test psi0 == original
        end
    end
end

@testset "observable storage resolves from Layer III" begin
    @test DiSLOUTrajectories._resolved_observable_storage(:auto, false) === :sparse
    @test DiSLOUTrajectories._resolved_observable_storage(:auto, true) === :dense
    @test DiSLOUTrajectories._resolved_observable_storage(:sparse, true) === :sparse
    @test DiSLOUTrajectories._resolved_observable_storage(:dense, false) === :dense
    for invalid in (:packed, "sparse", nothing, missing, true)
        @test_throws ArgumentError DiSLOUTrajectories._resolved_observable_storage(invalid, false)
    end
end

@testset "public boundary validates finite model and solve inputs" begin
    d = 3
    a = destroy(d)
    H, psi0, tlist, c_ops = 0.1 * num(d), fock(d, 1), [0.0, 0.1], [a]
    common = (;
        gauge_set = zeros(ComplexF64, 1, 1), ntraj = 1,
        ensemblealg = :serial,
    )

    for bad_times in (Float64[], [0.1, 0.0], [0.0, 0.0], [0.0, NaN], "0.1")
        @test_throws ArgumentError dislou_solve(H, psi0, bad_times, c_ops; common...)
    end
    for bad_saveat in (Float64[], [-0.1], [0.2], [0.0, 0.0], [NaN], true, "0.1")
        @test_throws ArgumentError dislou_solve(
            H, psi0, tlist, c_ops; common..., saveat = bad_saveat
        )
    end
    for (keyword, value) in (
            (:survival_rtol, 0.0), (:survival_rtol, Inf),
            (:time_rtol, 0.0), (:time_atol, -eps(Float64)),
            (:first_passage_maxiter, 0),
            (:residual_tolerance, 0.0), (:residual_tolerance, -eps(Float64)),
            (:residual_tolerance, Inf), (:residual_tolerance, NaN),
            (:condition_limit, 0.99), (:condition_limit, Inf),
            (:condition_limit, big(10)^1000),
            (:hysteresis, -0.1), (:hysteresis, 0.0),
            (:hysteresis, nextfloat(1.0)), (:hysteresis, Inf),
            (:hysteresis, NaN),
        )
        options = NamedTuple{(keyword,)}((value,))
        @test_throws ArgumentError dislou_solve(
            H, psi0, tlist, c_ops; common..., options...
        )
    end
    @test_throws ArgumentError dislou_solve(
        H, psi0, tlist, c_ops;
        common..., first_passage_method = :unsupported
    )

    nonnormal_H = ComplexF64[0 1; 1 0]
    nonnormal_C = ComplexF64[0 1; 0 0]
    nonnormal_common = (;
        gauge_set = zeros(ComplexF64, 1, 1), ntraj = 1, ensemblealg = :serial,
    )
    @test_throws ArgumentError dislou_solve(
        nonnormal_H, ComplexF64[1, 0], [0.0], [nonnormal_C];
        nonnormal_common..., condition_limit = 2.0
    )
    @test dislou_solve(
        nonnormal_H, ComplexF64[1, 0], [0.0], [nonnormal_C];
        nonnormal_common..., condition_limit = 2.1
    ) isa DiSLOUSolution
    @test_throws ArgumentError dislou_solve(
        H, psi0, tlist, c_ops;
        gauge_set = common.gauge_set, ntraj = 0, ensemblealg = :serial
    )
    @test_throws ArgumentError dislou_solve(
        H, psi0, tlist, c_ops;
        gauge_set = common.gauge_set, ntraj = 1, ensemblealg = :invalid
    )
    @test_throws ArgumentError dislou_solve(
        H, psi0, tlist, c_ops;
        common..., seed = -1
    )
    @test_throws ArgumentError dislou_solve(
        H, psi0, tlist, c_ops;
        common..., seed = big(typemax(UInt64)) + 1
    )
    @test_throws ArgumentError dislou_solve(H, 0 * psi0, tlist, c_ops; common...)
    @test_throws ArgumentError dislou_solve(
        H, ComplexF64[1, NaN, 0], tlist, c_ops;
        common...
    )
    @test_throws ArgumentError dislou_solve(
        ComplexF64[0 1; 0 0], ComplexF64[1, 0], tlist,
        [zeros(ComplexF64, 2, 2)];
        gauge_set = zeros(ComplexF64, 1, 1), ntraj = 1,
        ensemblealg = :serial
    )
    @test_throws DimensionMismatch dislou_solve(
        H, psi0, tlist, c_ops;
        gauge_set = zeros(ComplexF64, 2, 1), ntraj = 1,
        ensemblealg = :serial
    )
    @test_throws ArgumentError dislou_solve(
        H, psi0, tlist, c_ops;
        gauge_set = reshape(ComplexF64[Inf], 1, 1), ntraj = 1,
        ensemblealg = :serial
    )
    # Every rejection must name the offending keyword: a bare ArgumentError
    # leaves the caller guessing which option was wrong.
    for (options, keyword) in (
            ((; layer3_sizes = 1), "layer3_sizes"),
            ((; layer3 = true), "layer3_sizes"),
            ((; layer3 = true, layer3_sizes = 0), "layer3_sizes"),
            ((; layer3 = true, layer3_sizes = 4), "layer3_sizes"),
            ((; layer3 = true, layer3_sizes = true), "layer3_sizes"),
            ((; layer3 = true, layer3_sizes = 1.5), "layer3_sizes"),
            ((; layer3 = true, layer3_sizes = Int[]), "layer3_sizes"),
            ((; layer3 = true, layer3_sizes = [0]), "layer3_sizes"),
            ((; layer3 = true, layer3_sizes = [1.0]), "layer3_sizes"),
            ((; layer3 = true, layer3_sizes = [1, 2]), "layer3_sizes"),
            ((; hysteresis = 0.0), "hysteresis"),
            ((; residual_tolerance = 0.0), "residual_tolerance"),
        )
        caught = try
            dislou_solve(H, psi0, tlist, c_ops; common..., options...)
            nothing
        catch error
            error
        end
        @test caught isa ArgumentError
        @test caught isa Exception && occursin(keyword, sprint(showerror, caught))
    end
    @test_throws ArgumentError dislou_solve(
        H, psi0, tlist, c_ops;
        gauge_set = common.gauge_set, ntraj = 1, ensemblealg = :distributed,
        layer3 = true
    )
end

@testset "matrix inputs and exact Layer III delegation remain available" begin
    H = ComplexF64[0 0; 0 0.2]
    psi0 = ComplexF64[0, 1]
    c_ops = [ComplexF64[0 0.5; 0 0]]
    tlist = [0.0, 0.2]
    common = (;
        e_ops = [ComplexF64[0 0; 0 1]],
        gauge_set = zeros(ComplexF64, 1, 1), ntraj = 8,
        seed = 9, ensemblealg = :serial,
    )
    dense = dislou_solve(
        H, psi0, tlist, c_ops;
        common..., observable_storage = :dense
    )
    sparse = dislou_solve(
        H, psi0, tlist, c_ops;
        common..., observable_storage = :sparse
    )
    reduced = dislou_solve(
        H, psi0, tlist, c_ops; common...,
        layer3 = true, layer3_sizes = 2
    )

    @test dense isa DiSLOUSolution
    @test reduced.layer3_diagnostics.sizes == [2]
    @test size(dense.expect) == (1, length(tlist))
    @test dense.expect[1, 1] ≈ 1
    @test dense.expect == sparse.expect
    @test dense.col_times == sparse.col_times
    @test dense.col_which == sparse.col_which
    @test dense.col_gauge == sparse.col_gauge
    @test reduced.col_times == dense.col_times
    @test reduced.col_which == dense.col_which
    @test reduced.col_gauge == dense.col_gauge
    @test all(state -> state.type == Operator(), dense.states)
end

@testset "wrapped dense observables remain accepted at the public boundary" begin
    H = ComplexF64[0 0; 0 0.2]
    psi0 = ComplexF64[0, 1]
    tlist = [0.0, 0.2]
    observable = Diagonal(ComplexF64[1, 2])
    common = (;
        e_ops = [observable], gauge_set = zeros(ComplexF64, 0, 1),
        ntraj = 3, seed = 8, ensemblealg = :serial, saveat = tlist,
    )

    dense_cache = DiSLOUTrajectories._DiagonalCache(
        QuantumObject(H), QuantumObject[];
        e_ops = [observable], observable_storage = :dense
    )
    sparse_cache = DiSLOUTrajectories._DiagonalCache(
        QuantumObject(H), QuantumObject[];
        e_ops = [observable], observable_storage = :sparse
    )
    dense = dislou_solve(
        H, psi0, tlist, QuantumObject[];
        common..., observable_storage = :dense
    )
    sparse_sol = dislou_solve(
        H, psi0, tlist, QuantumObject[];
        common..., observable_storage = :sparse
    )

    @test dense_cache.Z isa Vector{Matrix{ComplexF64}}
    @test dense_cache.Z[1] == Matrix(observable)
    @test sparse_cache.Z isa Vector{SparseMatrixCSC{ComplexF64, Int}}
    @test sparse_cache.Z[1] == sparse(observable)
    @test isempty(sparse_cache.ZV)
    @test dense.expect ≈ fill(2.0, 1, length(tlist)) atol = 1.0e-12
    @test dense.expect == sparse_sol.expect
end

@testset "Layer III sizes validate before eigensystem preparation" begin
    function DiSLOUTrajectories._cuda_prepare_diagonal_data(
            ::Matrix{ComplexF64},
            ::Vector{Matrix{ComplexF64}}, ::Vector{Matrix{ComplexF64}}
        )
        throw(InterruptException())
    end
    try
        DiSLOUTrajectories._enable_cuda_diagonalization!()
        error = try
            dislou_solve(
                zeros(ComplexF64, 2, 2),
                ComplexF64[1, 0],
                [0.0],
                Matrix{ComplexF64}[];
                gauge_set = zeros(ComplexF64, 0, 1),
                ntraj = 1,
                ensemblealg = :serial,
                layer3 = true,
                layer3_sizes = 0,
            )
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("layer3_sizes", sprint(showerror, error))
    finally
        Base.delete_method(
            which(
                DiSLOUTrajectories._cuda_prepare_diagonal_data,
                Tuple{
                    Matrix{ComplexF64}, Vector{Matrix{ComplexF64}},
                    Vector{Matrix{ComplexF64}},
                }
            )
        )
        DiSLOUTrajectories._disable_cuda_diagonalization!()
    end
end

struct FakeDeviceMatrix <: AbstractMatrix{ComplexF64}
    data::Matrix{ComplexF64}
end
Base.size(A::FakeDeviceMatrix) = size(A.data)
Base.getindex(A::FakeDeviceMatrix, I...) = getindex(A.data, I...)

@testset "public boundary rejects unsupported operator inputs" begin
    h_sparse = tensor(num(2), qeye(2))
    dimensions = h_sparse.dimensions
    h_dense = QuantumObject(Matrix(h_sparse.data); dims = dimensions)
    c_dense = QuantumObject(
        ComplexF64[0 1 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0]; dims = dimensions
    )
    ψ0 = basis(2, 0) ⊗ basis(2, 0)
    common = (;
        gauge_set = zeros(ComplexF64, 1, 1), ntraj = 1, seed = 9,
        ensemblealg = :serial,
    )
    empty_gauge = (;
        gauge_set = zeros(ComplexF64, 0, 1), ntraj = 1, ensemblealg = :serial,
    )

    # A device-resident array type must be refused rather than silently copied.
    fake_device_h = QuantumObject(
        FakeDeviceMatrix(Matrix(h_dense.data)); dims = dimensions
    )
    @test_throws ArgumentError dislou_solve(
        fake_device_h, ψ0, [0.0], [c_dense]; common...
    )
    @test_throws ArgumentError dislou_solve(
        QuantumObject(ones(ComplexF64, 2, 3)), basis(2, 0), [0.0],
        QuantumObject[]; empty_gauge...
    )
    @test_throws ArgumentError dislou_solve(
        h_dense, ψ0, [0.0],
        [QuantumObject(Matrix{ComplexF64}(I, 3, 3))]; common...
    )
    @test_throws ArgumentError dislou_solve(
        h_dense, basis(3, 0), [0.0], [c_dense]; common...
    )
    @test_throws MethodError dislou_solve(
        h_dense, ψ0, [0.0], [c_dense]; cache = :unsupported, common...
    )

    # A non-Hermitian observable is a legitimate request, not an error.
    nonhermitian = dislou_solve(
        h_dense, ψ0, [0.0], [c_dense]; e_ops = [c_dense], common...
    )
    @test nonhermitian.expect !== nothing
end

@testset "sparse QuantumObject inputs match dense at the public boundary" begin
    h_sparse = tensor(num(2), qeye(2))
    dimensions = h_sparse.dimensions
    sparse_c = QuantumObject(
        sparse(ComplexF64[0 1 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0]); dims = dimensions
    )
    sparse_z = QuantumObject(
        sparse(Matrix(Diagonal(ComplexF64[0, 1, 1, 2]))); dims = dimensions
    )
    ψ0 = basis(2, 0) ⊗ basis(2, 0)
    common = (;
        e_ops = [sparse_z], gauge_set = zeros(ComplexF64, 1, 1), ntraj = 1,
        seed = 9, ensemblealg = :serial,
    )
    dense = dislou_solve(
        h_sparse, ψ0, [0.0, 0.01], [sparse_c];
        common..., observable_storage = :dense
    )
    sparse_solution = dislou_solve(
        h_sparse, ψ0, [0.0, 0.01], [sparse_c];
        common..., observable_storage = :sparse
    )

    @test dense isa DiSLOUSolution
    @test sparse_solution isa DiSLOUSolution
    @test dense.expect == sparse_solution.expect
    @test dense.col_times == sparse_solution.col_times
    @test dense.col_which == sparse_solution.col_which
    @test dense.col_gauge == sparse_solution.col_gauge
end
