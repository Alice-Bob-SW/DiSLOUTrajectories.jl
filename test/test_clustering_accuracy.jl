@testset "in-memory DBSCAN clustering retains physical centers and noise" begin
    points = ComplexF64[
        1.0 + 1.0im 1.04 + 0.98im 0.97 + 1.02im -1.0 - 1.0im -0.98 - 1.04im -1.03 - 0.97im 5.0 + 5.0im
        0.5 + 0.5im 0.52 + 0.49im 0.48 + 0.51im -0.5 - 0.5im -0.49 - 0.52im -0.52 - 0.48im 5.0 + 5.0im
    ]
    result = SM._cluster_terminal_means(
        points;
        cluster_scales = [0.2, 0.2], dbscan_radius = 0.5,
        min_neighbors = 2, min_weight = 0.0
    )
    @test result.counts == [3, 3]
    @test result.weights == [3 / 7, 3 / 7]
    @test result.labels == [2, 2, 2, 1, 1, 1, 0]
    @test result.centers[:, 1] ≈ ComplexF64[
        -1.0033333333333334 - 1.0033333333333334im,
        -0.5033333333333333 - 0.5im,
    ]
    @test result.centers[:, 2] ≈ ComplexF64[
        1.0033333333333334 + 1im,
        0.5 + 0.5im,
    ]
end
