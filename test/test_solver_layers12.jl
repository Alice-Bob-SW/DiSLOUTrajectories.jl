isdefined(Main, :DrivenKerrParams) ||
    include(joinpath(@__DIR__, "fixtures", "driven_kerr_model.jl"))

@testset "paper-scale manual Kerr gauges prepare under the default spectral tolerance" begin
    model = driven_kerr_model(
        DrivenKerrParams(
            N = 190, κ = 1.0, Δ = 13.0,
            K = 0.2, ε = 22.217 + 0im
        )
    )
    shifts = reshape(-sqrt(model.κ) .* ComplexF64[model.αlow, model.αhigh], 1, :)

    solution = dislou_solve(
        model.H, model.ψ0, [0.0], model.c_ops;
        gauge_set = shifts, ntraj = 1, ensemblealg = :serial
    )

    @test solution isa DiSLOUSolution
    @test solution.gauge_diagnostics.count == 2
end

@testset "manual gauge columns keep their input order" begin
    d = 3
    a = destroy(d)
    H, psi0, tlist, c_ops = 0 * a, fock(d, 0), [0.0, 0.25], [a]
    forward = dislou_solve(
        H, psi0, tlist, c_ops;
        gauge_set = ComplexF64[0 2], ntraj = 3, seed = 7, ensemblealg = :serial
    )
    reversed = dislou_solve(
        H, psi0, tlist, c_ops;
        gauge_set = ComplexF64[2 0], ntraj = 3, seed = 7, ensemblealg = :serial
    )

    @test forward.gauge_diagnostics.residence_time == [0.75, 0.0]
    @test reversed.gauge_diagnostics.residence_time == [0.0, 0.75]
end
