if haskey(ENV, "DISLOU_APPLE_ACCELERATE_ENV")
    push!(LOAD_PATH, ENV["DISLOU_APPLE_ACCELERATE_ENV"])
    using AppleAccelerate, LinearAlgebra
    @assert any(lib -> occursin("Accelerate", lib.libname), BLAS.get_config().loaded_libs)
    @info "CI BLAS backend" config = BLAS.get_config()
end

include("testsetup.jl")

# Optional-dependency and multi-process suites are owned by the dedicated
# `extension`, `cuda`, and `distributed` CI jobs, which include their files
# directly. Running them here too tripled this job's wall time.
@testset "DiSLOUTrajectories" begin
    include("test_backend.jl")
    include("test_public_api.jl")
    include("test_layer2_primitives.jl")
    include("test_first_passage.jl")
    include("test_layer1.jl")
    include("test_layer3.jl")
    include("test_recording_trajectory.jl")
    include("test_result_contract.jl")
    include("test_solver_layers12.jl")
    include("test_results.jl")
    include("test_ensemble_indexing.jl")
    include("test_parallel.jl")
    include("test_closed_system.jl")
    include("test_integration.jl")
    include("test_edge_cases.jl")
    include("test_gauge_input.jl")
    include("test_gauge_discovery_trajectories.jl")
    include("test_clustering_accuracy.jl")
end
