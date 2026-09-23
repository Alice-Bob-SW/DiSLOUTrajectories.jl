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
    for (method, hint) in (
            :trajectories => "using Clustering",
            :semiclassical => "using QuantumCumulants",
            :unknown => "Unsupported gauge discovery method :unknown",
        )
        err = try
            discover_gauges(nothing, []; method)
        catch caught
            caught
        end
        @test err isa MethodError
        @test occursin(hint, sprint(showerror, err))
    end
end
