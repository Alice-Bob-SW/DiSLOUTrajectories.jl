module DiSLOUTrajectoriesClusteringExt

import Clustering
import DiSLOUTrajectories
using Distances
using Statistics

const CF = ComplexF64

# Paper: rescaled (Re ᾱ_j^(r), Im ᾱ_j^(r)) (Appendix A.2).
function _scaled_features(points::AbstractMatrix{<:Complex}, scales::AbstractVector)
    nmodes, npoints = size(points)
    features = Matrix{Float64}(undef, 2nmodes, npoints)
    @inbounds for point in 1:npoints, mode in 1:nmodes
        z = points[mode, point] / scales[mode]
        features[2mode - 1, point] = real(z)
        features[2mode, point] = imag(z)
    end
    return features
end

_center_sort_key(center) = Tuple(Iterators.flatten((real(z), imag(z)) for z in center))

# Paper: clusters 𝓘_g and centers of ᾱ_j^(r) (Appendix A.2).
function _cluster_terminal_means(
        points::AbstractMatrix{<:Complex};
        cluster_scales::AbstractVector, dbscan_radius::Real,
        min_neighbors::Int, min_weight::Real
    )
    nmodes, npoints = size(points)
    length(cluster_scales) == nmodes || throw(DimensionMismatch("need one clustering scale per mode"))
    all(scale -> isfinite(scale) && scale > 0, cluster_scales) || throw(ArgumentError("cluster_scales must be finite and positive"))
    isfinite(dbscan_radius) && dbscan_radius > 0 || throw(ArgumentError("dbscan_radius must be finite and positive"))
    min_neighbors >= 1 || throw(ArgumentError("min_neighbors must be positive"))
    0 <= min_weight < 1 || throw(ArgumentError("min_weight must lie in [0, 1)"))
    all(isfinite, points) || throw(ArgumentError("terminal means must be finite"))
    npoints == 0 && return (; centers = zeros(CF, nmodes, 0), weights = Float64[], counts = Int[], labels = Int[])
    features = _scaled_features(points, cluster_scales)
    result = Clustering.dbscan(features, dbscan_radius; min_neighbors)
    labels = copy(Clustering.assignments(result))
    for (label, cluster) in enumerate(result.clusters), point in cluster.boundary_indices
        any(
            core -> evaluate(Euclidean(), @view(features[:, point]), @view(features[:, core])) <= dbscan_radius,
            cluster.core_indices
        ) || (labels[point] = 0)
    end
    records = NamedTuple[]
    for old in sort!(unique(filter(>(0), labels)))
        members = findall(==(old), labels)
        weight = length(members) / npoints
        weight < min_weight && continue
        center = CF.(vec(mean(@view(points[:, members]); dims = 2)))
        push!(records, (; old, count = length(members), weight, center))
    end
    sort!(records; by = record -> (-record.weight, _center_sort_key(record.center)))
    remap = Dict(record.old => index for (index, record) in enumerate(records))
    final_labels = [get(remap, label, 0) for label in labels]
    centers = isempty(records) ? zeros(CF, nmodes, 0) : hcat((copy(record.center) for record in records)...)
    return (; centers, weights = Float64[record.weight for record in records], counts = Int[record.count for record in records], labels = final_labels)
end

end
