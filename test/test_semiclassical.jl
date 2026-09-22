using Test
using DiSLOUTrajectories
using QuantumCumulants

@testset "extension activation needs only QuantumCumulants" begin
    @test Base.get_extension(DiSLOUTrajectories, :DiSLOUTrajectoriesQuantumCumulantsExt) !== nothing
    @test !isdefined(Main, :ModelingToolkitBase)
end

include("fixtures/two_mode_diamond.jl")

@testset "semiclassical named result passes through the matching numerical solve" begin
    limits = (8.0, 8.0)
    result = discover_gauges(
        two_mode_diamond_hamiltonian,
        two_mode_diamond_collapse_operators;
        method = :semiclassical,
        limits,
    )
    fixture = two_mode_diamond_numerical_fixture(Tuple(Int.(limits) .+ 1))
    solution = dislou_solve(
        fixture.H, fixture.psi0, [0.0], fixture.c_ops;
        gauge_set = result, ntraj = 1, seed = 5, ensemblealg = :serial
    )

    @test solution isa DiSLOUSolution
    @test solution.gauge_diagnostics.method === :semiclassical
    @test solution.gauge_diagnostics.count == size(result.shifts, 2)
end

function captured_argument_error(f)
    try
        f()
        @test false
        return ""
    catch err
        @test err isa ArgumentError
        return sprint(showerror, err)
    end
end

@variables Δ::Real η::Real γ::Real

# Paper: H (Eq. 1).
parameterized_hamiltonian(a, b) =
    Δ * adjoint(a) * a + η * (a + adjoint(a)) + 0.7 * adjoint(b) * b
# Paper: C_μ (Eq. 1).
parameterized_collapse_operators(a, b) = [sqrt(γ) * a, sqrt(0.6) * b]
const PARAMETERIZED_VALUES = (Δ => 1.0, η => 0.2, γ => 0.4)

# Paper: H (Eq. 7).
kerr_hamiltonian(a) = -13.0 * adjoint(a) * a +
    0.1 * adjoint(a)^2 * a^2 + 22.217im * (adjoint(a) - a)
# Paper: C = √κ a, with κ = 1 (Eq. 7).
kerr_collapse_operators(a) = [a]

# Paper: H (Eq. 1).
three_mode_hamiltonian(a, b, c) =
    0.3 * adjoint(a) * a + 0.5 * adjoint(b) * b + 0.7 * adjoint(c) * c
# Paper: C_μ (Eq. 1).
three_mode_collapse_operators(a, b, c) =
    [sqrt(0.2) * a, sqrt(0.4) * b, sqrt(0.6) * c]

@testset "semiclassical discovery supports one and three modes" begin
    kerr = discover_gauges(
        kerr_hamiltonian,
        kerr_collapse_operators;
        method = :semiclassical,
        limits = (189.0,),
    )
    expected_centers = reshape(
        ComplexF64[
            0.07266261691929639 + 1.7953859931208265im,
            1.7414229459815616 - 8.62240298901377im,
        ], 1, :
    )
    expected_saddle = reshape(
        ComplexF64[
            1.1116019736702265 + 6.939543439607197im,
        ], 1, :
    )

    @test kerr.centers ≈ expected_centers atol = 1.0e-7
    @test kerr.shifts ≈ -expected_centers atol = 1.0e-7
    @test kerr.diagnostics.rejected ≈ expected_saddle atol = 1.0e-7

    three_mode = discover_gauges(
        three_mode_hamiltonian,
        three_mode_collapse_operators;
        method = :semiclassical,
        limits = (1.0e-24, 1.0e-24, 1.0e-24),
    )

    @test size(three_mode.centers) == (3, 1)
    @test size(three_mode.shifts) == (3, 1)
    @test maximum(abs, three_mode.centers) < 1.0e-9
    @test maximum(abs, three_mode.shifts) < 1.0e-9
    @test size(three_mode.diagnostics.roots, 1) == 3
    @test size(three_mode.diagnostics.rejected) == (3, 0)
end

@testset "semiclassical discovery substitutes physical collapse parameters" begin
    result = discover_gauges(
        parameterized_hamiltonian,
        parameterized_collapse_operators;
        method = :semiclassical,
        limits = (4.0, 4.0),
        parameters = PARAMETERIZED_VALUES,
    )

    @test all(isfinite, result.shifts)
    @test result.shifts[1, :] ≈ -sqrt(0.4) .* result.centers[1, :]
    @test result.shifts[2, :] ≈ -sqrt(0.6) .* result.centers[2, :]
end

@testset "semiclassical discovery validates public symbolic inputs" begin
    for limits in (
            nothing, 1.0, (), ("bad",), (0.0, 1.0),
            (-1.0, 1.0), (Inf, 1.0), (1.0, NaN),
        )
        message = captured_argument_error() do
            discover_gauges(
                parameterized_hamiltonian,
                parameterized_collapse_operators;
                method = :semiclassical,
                limits,
                parameters = PARAMETERIZED_VALUES,
            )
        end
        @test occursin("occupation limits", message)
    end

    incomplete = captured_argument_error() do
        discover_gauges(
            parameterized_hamiltonian,
            parameterized_collapse_operators;
            method = :semiclassical,
            limits = (4.0, 4.0),
            parameters = (Δ => 1.0, η => 0.2),
        )
    end
    @test occursin("constructor parameter", incomplete)

    duplicate = captured_argument_error() do
        discover_gauges(
            parameterized_hamiltonian,
            parameterized_collapse_operators;
            method = :semiclassical,
            limits = (4.0, 4.0),
            parameters = (Δ => 1.0, Δ => 2.0, η => 0.2, γ => 0.4),
        )
    end
    @test occursin("constructor parameter", duplicate)

    nonfinite = captured_argument_error() do
        discover_gauges(
            parameterized_hamiltonian,
            parameterized_collapse_operators;
            method = :semiclassical,
            limits = (4.0, 4.0),
            parameters = (Δ => 1.0, η => NaN, γ => 0.4),
        )
    end
    @test occursin("constructor parameter", nonfinite)
end

# This fails if the public dispatcher does not expose the extension result, or
# if discovery returns unscaled targets rather than generator-preserving shifts.
@testset "semiclassical discovery returns a reusable named gauge set" begin
    cfg = TWO_MODE_DIAMOND
    result = discover_gauges(
        two_mode_diamond_hamiltonian,
        two_mode_diamond_collapse_operators;
        method = :semiclassical,
        limits = (cfg.Na - 1, cfg.Nb - 1),
    )

    @test propertynames(result) == (:shifts, :method, :centers, :weights, :diagnostics)
    @test result.method === :semiclassical
    @test size(result.centers) == (2, 3)
    @test size(result.shifts) == (2, 3)
    @test length(result.weights) == 3
    @test size(result.shifts, 2) == size(result.centers, 2) == length(result.weights)
    @test all(isfinite, result.shifts)
    @test all(result.diagnostics.stable)
    @test result.shifts == -reshape(sqrt.([cfg.κa, cfg.κb]), :, 1) .* result.centers
    @test size(result.diagnostics.roots, 1) == 2
    @test length(result.diagnostics.stability) == size(result.diagnostics.roots, 2)
    @test size(result.diagnostics.rejected, 1) == 2
    @test DiSLOUTrajectories._validated_gauge_data(result, size(result.shifts, 1)).shifts == result.shifts
end
