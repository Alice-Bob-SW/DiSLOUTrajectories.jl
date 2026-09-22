@testset "solution retains aggregate and optional trajectory data separately" begin
    d = 4
    a = destroy(d)
    tlist = [1.0, 1.1, 1.2]
    sol = dislou_solve(
        0.1 * num(d), fock(d, 2), tlist, [0.4 * a];
        e_ops = [num(d), a], gauge_set = zeros(ComplexF64, 1, 1),
        ntraj = 6, seed = 41, ensemblealg = :serial, saveat = tlist,
        save_trajectories = true, save_final_states = true
    )

    @test sol isa DiSLOUSolution
    @test sol isa QuantumToolbox.TimeEvolutionMultiTrajSol
    @test sol.times == tlist
    @test sol.times_states == tlist
    @test sol.states isa Vector{<:QuantumObject}
    @test size(sol.states) == (length(tlist),)
    @test all(state -> state.type == Operator(), sol.states)
    @test size(sol.expect) == (2, length(tlist))
    @test size(sol.expect_std) == size(sol.expect)
    @test size(sol.expect_sem) == size(sol.expect)
    @test sol.trajectory_states isa Matrix{QuantumObject}
    @test size(sol.trajectory_states) == (sol.ntraj, length(tlist))
    @test all(state -> state.type == Ket(), sol.trajectory_states)
    @test sol.trajectory_expect isa Array{ComplexF64, 3}
    @test size(sol.trajectory_expect) == (2, sol.ntraj, length(tlist))
    @test size(sol.final_states) == (d, sol.ntraj)
    @test all(
        isapprox(norm(view(sol.final_states, :, i)), 1.0; atol = 1.0e-12)
            for i in 1:sol.ntraj
    )
    @test sol.final_density ≈ sol.final_states * sol.final_states' / sol.ntraj atol = 1.0e-12
    @test sol.final_density ≈ sol.final_density' atol = 1.0e-12
    @test real(tr(sol.final_density)) ≈ 1.0 atol = 1.0e-12
    @test expect_mean(sol, 2) == sol.expect[2, :]
    @test expect_sem(sol, 2) == sol.expect_sem[2, :]
    @test expect_mean(sol) == sol.expect[1, :]
    @test expect_sem(sol) == sol.expect_sem[1, :]
    @test QuantumToolbox.average_states(sol) == sol.states
    @test QuantumToolbox.average_expect(sol) == sol.expect
    @test all(
        isapprox(real(tr(state.data)), 1.0; atol = 1.0e-12) for state in sol.states
    )
    @test all(
        isapprox(state.data, state.data'; atol = 1.0e-12) for state in sol.states
    )
end

@testset "aggregate save defaults and slim diagnostics" begin
    d = 3
    a = destroy(d)
    tlist = [0.0, 0.1, 0.2]
    common = (;
        gauge_set = zeros(ComplexF64, 1, 1), ntraj = 2,
        seed = 3, ensemblealg = :serial,
    )
    states_only = dislou_solve(0 * a, fock(d, 1), tlist, [a]; common...)
    observed = dislou_solve(
        0 * a, fock(d, 1), tlist, [a];
        common..., e_ops = [num(d)]
    )

    @test states_only.expect === nothing
    @test states_only.expect_std === nothing
    @test states_only.expect_sem === nothing
    @test states_only.times_states == [last(tlist)]
    @test length(states_only.states) == 1
    @test observed.times_states == [last(tlist)]
    @test length(observed.states) == 1
    @test states_only.trajectory_states === nothing
    @test states_only.trajectory_expect === nothing
    @test states_only.final_states === nothing
    @test states_only.final_density === nothing
    @test keys(observed.gauge_diagnostics) ==
        (:count, :method, :status, :switches, :residence_time, :jumps_by_gauge)
    @test fieldnames(DiSLOUTrajectories._GaugeDiagnostics) == (
        :ngauges, :switches, :residence_time, :jumps_by_gauge,
        :accepted_projections, :fallback_segments,
        :fallback_residence_time, :fallback_jumps, :maximum_residual,
    )
    @test observed.gauge_diagnostics.count == 1
    @test sum(observed.gauge_diagnostics.residence_time) ≈
        observed.ntraj * (last(tlist) - first(tlist)) atol = 1.0e-12
end

# Mutation caught: charging full-basis fallback segments to the private Ng+1
# slot drops their physical gauge activity when the public result keeps 1:Ng.
@testset "Layer III fallback retains physical gauge diagnostics" begin
    d = 4
    a = destroy(d)
    tlist = [0.0, 0.4]
    sol = dislou_solve(
        0.3 * (a + a'), fock(d, 3), tlist, [sqrt(10.0) * a];
        gauge_set = ComplexF64[0 1], ntraj = 8, seed = 91,
        ensemblealg = :serial, layer3 = true, layer3_sizes = 1,
        residual_tolerance = eps(Float64)
    )
    logged_gauges = reduce(vcat, sol.col_gauge; init = Int[])

    @test sol.njumps_total > 0
    @test length(logged_gauges) == sol.njumps_total
    @test sol.gauge_diagnostics.jumps_by_gauge ==
        [count(==(gauge), logged_gauges) for gauge in 1:2]
    @test sum(sol.gauge_diagnostics.jumps_by_gauge) == sol.njumps_total
    @test sum(sol.gauge_diagnostics.residence_time) ≈
        sol.ntraj * (last(tlist) - first(tlist)) atol = 1.0e-12

end

# Mutation caught: representing a saved zero-observable axis as `nothing`
# violates the documented Ne×ntraj×Nt trajectory-expectation shape.
@testset "saved trajectories retain an empty expectation axis" begin
    d = 2
    a = destroy(d)
    tlist = [0.0, 0.1]
    sol = dislou_solve(
        0 * a, fock(d, 1), tlist, [a];
        gauge_set = zeros(ComplexF64, 1, 1), ntraj = 3,
        seed = 12, ensemblealg = :serial, save_trajectories = true
    )

    @test sol.expect === nothing
    @test sol.trajectory_expect isa Array{ComplexF64, 3}
    @test sol.trajectory_expect === nothing ||
        size(sol.trajectory_expect) == (0, sol.ntraj, length(tlist))
end
