using Test
using DiSLOUTrajectories
using QuantumToolbox
using LinearAlgebra
using SparseArrays

@testset "manual gauge matrices preserve shape, sign, and order" begin
    Z = ComplexF64[
        -0.3 + 0.2im  0.0;
        0.1          0.4im
    ]
    data = DiSLOUTrajectories._validated_gauge_data(Z, 2)
    @test data.shifts == Z
    @test data.shifts !== Z
    @test data.method === :manual
    @test_throws DimensionMismatch DiSLOUTrajectories._validated_gauge_data(zeros(ComplexF64, 1, 2), 2)
    @test_throws ArgumentError DiSLOUTrajectories._validated_gauge_data(reshape(ComplexF64[NaN], 1, 1), 1)
    @test_throws ArgumentError DiSLOUTrajectories._validated_gauge_data(zeros(ComplexF64, 1, 0), 1)
end

@testset "named gauge results validate and copy their full contract" begin
    shifts = ComplexF64[0.1 0.2; -0.3 0.4]
    centers = ComplexF64[1 2; 3 4]
    samples = [ComplexF64[1, 0], ComplexF64[0, 1]]
    diagnostics = (; samples, nested = [ComplexF64[2, 3]])
    result = (;
        shifts, method = :trajectories, centers,
        weights = [0.25, 0.75], diagnostics,
    )
    data = DiSLOUTrajectories._validated_gauge_data(result, 2)

    @test propertynames(result) == (:shifts, :method, :centers, :weights, :diagnostics)
    @test data.method === :trajectories
    @test data.shifts == shifts && data.centers == centers && data.weights == result.weights
    @test data.shifts !== shifts && data.centers !== centers && data.weights !== result.weights
    @test data.diagnostics.samples !== samples
    @test data.diagnostics.samples[1] !== samples[1]
    @test data.diagnostics.nested !== diagnostics.nested
    @test data.diagnostics.nested[1] !== diagnostics.nested[1]

    shifts[1, 1] = 99
    samples[1][1] = 99
    diagnostics.nested[1][1] = 99
    @test data.shifts[1, 1] != 99
    @test data.diagnostics.samples[1][1] != 99
    @test data.diagnostics.nested[1][1] != 99

    for method in (:manual, :trajectories, :semiclassical)
        @test DiSLOUTrajectories._validated_gauge_data(
            (;
                shifts = zeros(ComplexF64, 2, 1),
                method,
                centers = zeros(ComplexF64, 0, 1),
                weights = [1.0],
                diagnostics = (;),
            ), 2
        ).method === method
    end
    @test_throws DimensionMismatch DiSLOUTrajectories._validated_gauge_data(
        (;
            shifts = zeros(ComplexF64, 2, 2),
            method = :manual,
            centers = zeros(ComplexF64, 0, 2),
            weights = [1.0],
            diagnostics = (;),
        ), 2
    )
end

@testset "QuantumObject shifted channels preserve sparse storage and dimensions" begin
    dims = (2,)
    H = QuantumObject(sparse(ComplexF64[0 1; 1 0]); dims)
    C = [QuantumObject(sparse(ComplexF64[0 1; 0 0]); dims)]
    Hs, Cs = DiSLOUTrajectories._shifted_problem(H, C, [(1, 0.2 + 0.1im)])
    @test Hs.data isa SparseMatrixCSC
    @test Cs[1].data isa SparseMatrixCSC
    @test Hs.dimensions == H.dimensions
    @test Cs[1].dimensions == C[1].dimensions
end
