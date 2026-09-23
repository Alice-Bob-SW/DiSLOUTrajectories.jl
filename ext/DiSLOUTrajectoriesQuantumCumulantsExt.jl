module DiSLOUTrajectoriesQuantumCumulantsExt

import ForwardDiff
import ModelingToolkitBase
import QuantumCumulants
import SimpleNonlinearSolve
import DiSLOUTrajectories
using LinearAlgebra

const MTK = ModelingToolkitBase
const QC = QuantumCumulants
const SNS = SimpleNonlinearSolve

const _SEMICLASSICAL_NUMERICAL_POLICY = (
    starts_per_coordinate = 5,
    max_iterations = 80,
    residual_tolerance = 1.0e-10,
    stability_cutoff = -1.0e-8,
    occupation_slack = 1.0e-8,
    deduplication_tolerance = 1.0e-7,
)

function _parameter_pairs(values, stage)
    values isa Union{Tuple, AbstractVector, AbstractDict} ||
        throw(ArgumentError("$stage parameters must be a concrete collection of key => value pairs"))
    pairs = collect(values)
    all(pair -> pair isa Pair, pairs) ||
        throw(ArgumentError("$stage parameters must contain only key => value pairs"))
    keys = first.(pairs)
    length(unique(keys)) == length(keys) ||
        throw(ArgumentError("$stage parameter keys must be unique"))
    all(pair -> last(pair) isa Number && isfinite(last(pair)), pairs) ||
        throw(ArgumentError("$stage parameter values must be finite numbers"))
    return pairs
end

function _require_complete_parameters(pairs, declared, stage)
    supplied = Set(first.(pairs))
    expected = Set(declared)
    supplied == expected && return pairs
    missing = [parameter for parameter in declared if parameter ∉ supplied]
    extra = [parameter for parameter in supplied if parameter ∉ expected]
    throw(
        ArgumentError(
            "$stage parameter keys must exactly match the declared mean-field parameters; " *
                "missing=$(missing), extra=$(extra)"
        )
    )
end

function _semiclassical_limits(limits)
    limits isa Union{Tuple, AbstractVector} || throw(
        ArgumentError(
            "occupation limits must be a concrete collection of bounds"
        )
    )
    isempty(limits) && throw(
        ArgumentError(
            "occupation limits must contain at least one bound"
        )
    )
    bounds = try
        Float64.(collect(limits))
    catch
        throw(ArgumentError("occupation limits must be finite positive numbers, got $limits"))
    end
    all(bound -> isfinite(bound) && bound > 0, bounds) ||
        throw(ArgumentError("occupation limits must be finite and positive, got $limits"))
    return bounds
end

# Paper: α_j from x = (Re α₁, Im α₁, …) (Appendix A.1).
function _complex_amplitudes(z::AbstractVector{<:Real})
    iseven(length(z)) || throw(
        ArgumentError(
            "phase-space coordinates must contain one real-imaginary pair per mode"
        )
    )
    return [complex(z[index], z[index + 1]) for index in 1:2:length(z)]
end

# Paper: R(x) = (Re F₁, Im F₁, …) (Eq. A.3).
function _semiclassical_residual(drift, z::AbstractVector{<:Real})
    amplitudes = _complex_amplitudes(z)
    values = drift(amplitudes)
    length(values) == length(amplitudes) || throw(
        ArgumentError(
            "mean-field evaluation stage returned $(length(values)) drifts; " *
                "expected $(length(amplitudes))"
        )
    )
    return collect(Iterators.flatten((real(value), imag(value)) for value in values))
end

# Paper: x^(g) with R(x^(g)) = 0 (Eqs. A.3–A.4), J = ∂R/∂x by forward-mode AD.
function _semiclassical_root(residual, seed::Vector{Float64})
    policy = _SEMICLASSICAL_NUMERICAL_POLICY
    problem = SNS.NonlinearProblem((z, _) -> residual(z), seed)
    solution = SNS.solve(
        problem, SNS.SimpleTrustRegion();
        abstol = policy.residual_tolerance, maxiters = policy.max_iterations
    )
    return norm(residual(solution.u)) <= policy.residual_tolerance ? solution.u : nothing
end

# Paper: α^(g) and max_ℓ Re λ_ℓ[J(x^(g))] (Eq. A.5).
function _semiclassical_point(residual, z::Vector{Float64})
    value = residual(z)
    jacobian = ForwardDiff.jacobian(residual, z)
    all(isfinite, jacobian) || return nothing
    eigenvalues = try
        eigvals(jacobian)
    catch err
        err isa LinearAlgebra.LAPACKException || rethrow()
        return nothing
    end
    all(isfinite, eigenvalues) || return nothing
    max_real_eigenvalue = Float64(maximum(real, eigenvalues))
    return (;
        amplitudes = _complex_amplitudes(z),
        stable = max_real_eigenvalue < _SEMICLASSICAL_NUMERICAL_POLICY.stability_cutoff,
        max_real_eigenvalue, residual = Float64(norm(value)),
    )
end

# Paper: candidate fixed points α^(g) (Appendix A.1).
function _phase_space_candidates(drift, bounds)
    policy = _SEMICLASSICAL_NUMERICAL_POLICY
    residual = z -> _semiclassical_residual(drift, z)
    seed_axes = [
        range(
            -sqrt(bounds[(index + 1) ÷ 2]),
            sqrt(bounds[(index + 1) ÷ 2]); length = policy.starts_per_coordinate
        )
            for index in 1:(2 * length(bounds))
    ]
    roots = Vector{Vector{Float64}}()
    # TODO: Cartesian seeding scales as starts_per_coordinate^(2 * nmodes),
    # replace with configurable multistart sampling when larger systems need it.
    for seed in Iterators.product(seed_axes...)
        root = _semiclassical_root(residual, Float64[seed...])
        root === nothing && continue
        amplitudes = _complex_amplitudes(root)
        all(
            mode -> abs2(amplitudes[mode]) <=
                bounds[mode] + policy.occupation_slack, eachindex(bounds)
        ) || continue
        any(other -> norm(root - other) <= policy.deduplication_tolerance, roots) &&
            continue
        push!(roots, root)
    end

    points = NamedTuple[]
    for root in roots
        point = _semiclassical_point(residual, root)
        point === nothing || push!(points, point)
    end
    return sort!(
        points; by = point -> Tuple(
            Iterators.flatten(
                (abs2(amplitude), real(amplitude), imag(amplitude))
                    for amplitude in point.amplitudes
            )
        )
    )
end

# Paper: columns α^(g) of mean-field amplitudes (Appendix A.1).
_point_matrix(points, nmodes) = isempty(points) ? zeros(ComplexF64, nmodes, 0) :
    hcat((point.amplitudes for point in points)...)

# Paper: ζ_μ^(g) = -C_{μ,sc}(α^(g), α^(g)*) (Eq. A.6).
function _discover_gauges_semiclassical(
        hamiltonian::Function, collapse_operators::Function;
        limits, parameters = ()
    )
    bounds = _semiclassical_limits(limits)
    defaults = _parameter_pairs(parameters, "constructor")
    nmodes = length(bounds)

    space = QC.ProductSpace(
        ntuple(
            mode -> QC.FockSpace(Symbol(:mode_, mode)), nmodes
        )
    )
    modes = [QC.Destroy(space, Symbol(:a_, mode), mode) for mode in 1:nmodes]
    H = hamiltonian(modes...)
    H isa QC.QField || throw(
        ArgumentError(
            "Hamiltonian evaluation stage must return a QuantumCumulants operator"
        )
    )
    raw_c_ops = collapse_operators(modes...)
    raw_c_ops isa Union{Tuple, AbstractVector} || throw(
        ArgumentError(
            "collapse-operator evaluation stage must return a concrete collection"
        )
    )
    c_ops = collect(raw_c_ops)
    all(c_op -> c_op isa QC.QField, c_ops) || throw(
        ArgumentError(
            "collapse-operator evaluation stage must return only QuantumCumulants operators"
        )
    )
    equations = QC.meanfield(modes, H, c_ops; order = 1)
    length(equations) == nmodes || throw(
        ArgumentError(
            "mean-field derivation stage produced $(length(equations)) equations; " *
                "expected $nmodes"
        )
    )

    system = MTK.System(equations; name = :semiclassical_meanfield)
    compiled = MTK.mtkcompile(system)
    declared = collect(MTK.parameters(compiled))
    _require_complete_parameters(defaults, declared, "constructor")
    parameter_map = Dict(defaults)
    initial = merge(QC.initial_values(equations, zeros(ComplexF64, nmodes)), parameter_map)
    problem = MTK.ODEProblem(
        compiled, initial, (0.0, 1.0); build_initializeprob = false
    )
    evaluated = MTK.remake(problem; p = parameter_map)
    points = _phase_space_candidates(
        state -> evaluated.f(state, evaluated.p, 0.0), bounds
    )
    stable = filter(point -> point.stable, points)
    isempty(stable) && throw(ArgumentError("semiclassical discovery found no stable fixed points"))

    centers = _point_matrix(stable, nmodes)
    shifts = Matrix{ComplexF64}(undef, length(c_ops), length(stable))
    for (gauge, point) in enumerate(stable)
        raw_values = collapse_operators(point.amplitudes...)
        raw_values isa Union{Tuple, AbstractVector} || throw(
            ArgumentError(
                "collapse-operator evaluation stage must return a concrete collection"
            )
        )
        values = [
            MTK.Symbolics.evaluate(
                MTK.substitute(value, parameter_map), Dict()
            ) for value in raw_values
        ]
        length(values) == length(c_ops) || throw(
            ArgumentError(
                "collapse-operator evaluation stage returned $(length(values)) values; expected $(length(c_ops))"
            )
        )
        all(value -> value isa Number && isfinite(value), values) || throw(
            ArgumentError(
                "collapse-operator evaluation stage must return finite physical amplitudes"
            )
        )
        shifts[:, gauge] .= -ComplexF64[
            ComplexF64(MTK.Symbolics.value(real(value)), MTK.Symbolics.value(imag(value)))
                for value in values
        ]
    end
    roots = _point_matrix(points, nmodes)
    rejected = _point_matrix(filter(point -> !point.stable, points), nmodes)
    diagnostics = (
        roots = copy(roots),
        stability = [
            (;
                stable = point.stable,
                max_real_eigenvalue = point.max_real_eigenvalue,
                residual = point.residual,
            ) for point in points
        ],
        rejected = copy(rejected),
        stable = trues(length(stable)),
    )
    ngauges = length(stable)
    return (;
        shifts = copy(shifts), method = :semiclassical, centers = copy(centers),
        weights = fill(1.0 / ngauges, ngauges), diagnostics,
    )
end

end
