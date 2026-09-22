@testset "empty collapse channels use one gauge and closed evolution" begin
    d = 4
    H = 0.3 * num(d)
    psi0 = normalize(fock(d, 0) + fock(d, 2))
    tlist = [2.0, 2.2, 2.5]
    sol = dislou_solve(
        H, psi0, tlist, QuantumObject[];
        gauge_set = zeros(ComplexF64, 0, 1),
        e_ops = [num(d)], ntraj = 3, seed = 8, ensemblealg = :serial,
        saveat = tlist, save_final_states = true
    )
    exact_states = [
        exp(-im * Matrix{ComplexF64}(H.data) * (time - first(tlist))) * psi0.data
            for time in tlist
    ]

    @test sol.gauge_diagnostics.count == 1
    @test sol.gauge_diagnostics.method === :manual
    @test sol.gauge_diagnostics.status === :manual
    @test sol.gauge_diagnostics.switches == 0
    @test sol.gauge_diagnostics.jumps_by_gauge == [0]
    @test sol.njumps_total == 0
    @test isempty(sol.jumps_by_channel)
    @test sol.expect ≈ ones(1, length(tlist)) atol = 1.0e-12
    @test all(
        isapprox(
            sol.states[k].data,
            exact_states[k] * exact_states[k]'; atol = 1.0e-12
        )
            for k in eachindex(tlist)
    )
    @test size(sol.final_states) == (d, sol.ntraj)
    @test sol.final_density ≈ sol.states[end].data atol = 1.0e-12
end

@testset "one-time solve records the initial state" begin
    d = 3
    psi0 = normalize(fock(d, 0) + im * fock(d, 1))
    sol = dislou_solve(
        0 * num(d), psi0, [7.0], QuantumObject[];
        gauge_set = zeros(ComplexF64, 0, 1),
        ntraj = 2, seed = 4, ensemblealg = :serial,
        save_trajectories = true, save_final_states = true
    )

    @test sol.times == [7.0]
    @test sol.times_states == [7.0]
    @test sol.states[1].data ≈ psi0.data * psi0.data' atol = 1.0e-14
    @test all(isempty, sol.col_times)
    @test sol.gauge_diagnostics.residence_time == [0.0]
    @test all(
        isapprox(sol.trajectory_states[i, 1].data, psi0.data; atol = 1.0e-14)
            for i in 1:sol.ntraj
    )
    @test all(
        isapprox(sol.final_states[:, i], psi0.data; atol = 1.0e-14)
            for i in 1:sol.ntraj
    )
end

@testset "empty channels reject multiple manual gauges" begin
    error = try
        dislou_solve(
            0 * num(2), fock(2, 0), [0.0], QuantumObject[];
            gauge_set = zeros(ComplexF64, 0, 2), ntraj = 1,
            ensemblealg = :serial
        )
        nothing
    catch caught
        caught
    end
    @test error isa ArgumentError
    @test occursin("exactly one gauge", sprint(showerror, error))
end

@testset "one-dimensional closed system uses an explicit empty gauge" begin
    tlist = [0.0, 0.2]
    outcome = try
        dislou_solve(
            num(1), fock(1, 0), tlist, QuantumObject[];
            gauge_set = zeros(ComplexF64, 0, 1),
            ntraj = 2, seed = 5, ensemblealg = :serial
        )
    catch caught
        caught
    end

    @test outcome isa DiSLOUSolution
    if outcome isa DiSLOUSolution
        @test outcome.gauge_diagnostics.count == 1
        @test outcome.gauge_diagnostics.method === :manual
        @test outcome.gauge_diagnostics.status === :manual
        @test outcome.gauge_diagnostics.residence_time == [0.4]
        @test outcome.gauge_diagnostics.jumps_by_gauge == [0]
        @test outcome.njumps_total == 0
        @test all(state -> state.data == ones(ComplexF64, 1, 1), outcome.states)
    end
end
