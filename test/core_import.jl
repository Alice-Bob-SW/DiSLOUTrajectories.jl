using Test
using DiSLOUTrajectories

@testset "Core-only import" begin
    @test Base.get_extension(DiSLOUTrajectories, :DiSLOUTrajectoriesClusteringExt) === nothing
    @test Base.get_extension(DiSLOUTrajectories, :DiSLOUTrajectoriesQuantumCumulantsExt) === nothing
    loaded = Set(pkg.name for pkg in keys(Base.loaded_modules))
    @test "Clustering" ∉ loaded
    @test "Distances" ∉ loaded
    @test "QuantumCumulants" ∉ loaded
    @test "ModelingToolkitBase" ∉ loaded
    trajectory_error = try
        DiSLOUTrajectories._cluster_terminal_means(
            zeros(ComplexF64, 1, 0);
            cluster_scales = [1.0], dbscan_radius = 1.5,
            min_neighbors = 1, min_weight = 0.0,
        )
        nothing
    catch caught
        caught
    end
    @test trajectory_error isa ArgumentError
    @test occursin("Clustering", sprint(showerror, trajectory_error))
end
