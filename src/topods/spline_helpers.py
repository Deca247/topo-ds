from __future__ import annotations

import numpy as np

_EPS = 1e-12


def score_path(path, w_array, support_matrix, eps=1e-12):
    """Score a candidate path by evidence, straightness, and curvature."""
    path = list(path)

    if len(path) < 2:
        return 0.0

    evidence = sum(
        support_matrix[u, v]
        for u, v in zip(path[:-1], path[1:])
    )
    length = sum(
        np.linalg.norm(w_array[v] - w_array[u])
        for u, v in zip(path[:-1], path[1:])
    )

    direct = np.linalg.norm(w_array[path[-1]] - w_array[path[0]])
    straightness = direct / (length + eps)
    curvatures = []

    if len(path) >= 3:
        points = w_array[path]
        for p0, p1, p2 in zip(points[:-2], points[1:-1], points[2:]):
            v1 = p1 - p0
            v2 = p2 - p1
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)

            if n1 < eps or n2 < eps:
                continue

            cosine = np.dot(v1, v2) / (n1 * n2)
            theta = np.arccos(np.clip(cosine, -1.0, 1.0))
            curvatures.append((theta / np.pi) ** 2)

    mean_curvature = np.mean(curvatures) if curvatures else 0.0
    return evidence * straightness / (1.0 + mean_curvature)


def _polyline_geometry(spline, n_samples: int) -> dict:
    """Sample a spline and build an arc-length polyline approximation."""
    n_samples = max(int(n_samples), 50)
    s_grid = np.linspace(0.0, 1.0, n_samples + 1, dtype=np.float64)
    points = np.asarray(spline(s_grid), dtype=np.float64)

    segment_start = points[:-1]
    segment_vec = points[1:] - points[:-1]
    segment_len = np.linalg.norm(segment_vec, axis=1)
    segment_len_sq = np.maximum(segment_len * segment_len, _EPS)
    arc_start = np.concatenate([[0.0], np.cumsum(segment_len[:-1])])
    total_length = float(np.sum(segment_len))

    if total_length <= _EPS:
        total_length = 1.0

    return {
        "s_grid": s_grid,
        "curve_points": points,
        "segment_start": segment_start,
        "segment_vec": segment_vec,
        "segment_len": segment_len,
        "segment_len_sq": segment_len_sq,
        "arc_start": arc_start,
        "total_length": total_length,
    }


def _project_to_polyline(X: np.ndarray, path_model: dict, chunk_size: int = 512):
    """Return projected arc length, transverse distance and projected points."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]

    A = np.asarray(path_model["segment_start"], dtype=np.float64)
    V = np.asarray(path_model["segment_vec"], dtype=np.float64)
    len_sq = np.asarray(path_model["segment_len_sq"], dtype=np.float64)
    seg_len = np.asarray(path_model["segment_len"], dtype=np.float64)
    arc_start = np.asarray(path_model["arc_start"], dtype=np.float64)

    n = X.shape[0]
    arc = np.empty(n, dtype=np.float64)
    distance = np.empty(n, dtype=np.float64)
    projection = np.empty_like(X, dtype=np.float64)

    for start in range(0, n, max(int(chunk_size), 1)):
        stop = min(start + max(int(chunk_size), 1), n)
        Xc = X[start:stop]

        diff = Xc[:, None, :] - A[None, :, :]
        t = np.einsum("nkd,kd->nk", diff, V) / len_sq[None, :]
        t = np.clip(t, 0.0, 1.0)
        proj = A[None, :, :] + t[:, :, None] * V[None, :, :]
        d2 = np.sum((Xc[:, None, :] - proj) ** 2, axis=2)

        best = np.argmin(d2, axis=1)
        rows = np.arange(Xc.shape[0])
        best_t = t[rows, best]

        projection[start:stop] = proj[rows, best]
        distance[start:stop] = np.sqrt(np.maximum(d2[rows, best], 0.0))
        arc[start:stop] = arc_start[best] + best_t * seg_len[best]

    return arc, distance, projection


def _prototype_arc_coordinates(anchors: np.ndarray, total_length: float, is_cycle: bool) -> np.ndarray:
    """Place ordered prototypes on the spline arc-length coordinate."""
    anchors = np.asarray(anchors, dtype=np.float64)
    p = anchors.shape[0]

    if p <= 1:
        return np.zeros(p, dtype=np.float64)

    if is_cycle:
        edges = np.linalg.norm(
            np.roll(anchors, -1, axis=0) - anchors,
            axis=1,
        )
        perimeter = float(np.sum(edges))
        if perimeter <= _EPS:
            return np.linspace(0.0, total_length, p, endpoint=False)
        cumulative = np.concatenate([[0.0], np.cumsum(edges[:-1])])
        return cumulative * (total_length / perimeter)

    edges = np.linalg.norm(np.diff(anchors, axis=0), axis=1)
    path_length = float(np.sum(edges))
    if path_length <= _EPS:
        return np.linspace(0.0, total_length, p)
    cumulative = np.concatenate([[0.0], np.cumsum(edges)])
    return cumulative * (total_length / path_length)


def _neighbor_indices(j: int, p: int, radius: int, is_cycle: bool) -> np.ndarray:
    indices = [j]
    for offset in range(1, radius + 1):
        if is_cycle:
            indices.extend([(j - offset) % p, (j + offset) % p])
        else:
            if j - offset >= 0:
                indices.append(j - offset)
            if j + offset < p:
                indices.append(j + offset)
    return np.unique(np.asarray(indices, dtype=np.int64))


def fit_spline_coordinate_tube_model(data_pos: np.ndarray, w_array: np.ndarray, splines, all_paths, all_is_cycle,
                                     transverse_scale: float = 1.5, min_radius: float = 0.01,
                                     min_samples: int = 5, neighbor_pool: int = 1, gamma_margin: float = 1.0,
                                     gamma_beta: float = 10.0, projection_samples: int = 400,
                                     data_path_assignment: np.ndarray | None = None):
    """Fit a spline-coordinate tube in position space.

    Each prototype receives one transverse radius.  Data are assigned to global
    prototype Voronoi cells, but their dispersion is measured by distance to
    the corresponding full spline, not by a tangent-line covariance ellipse.

    Longitudinal localization is parameter-free: neighboring local A_{k,i}(x)
    matrices are blended into M_k(x) by a compact C2 partition along arc length.
    """
    data_pos = np.asarray(data_pos, dtype=np.float64)
    w_array = np.asarray(w_array, dtype=np.float64)
    dim = int(data_pos.shape[1])

    transverse_scale = max(float(transverse_scale), _EPS)
    min_samples = max(int(min_samples), 1)
    neighbor_pool = max(int(neighbor_pool), 0)
    projection_samples = max(int(projection_samples), 50)

    data_span = max(float(np.max(np.ptp(data_pos, axis=0))), _EPS)
    min_radius_abs = max(float(min_radius) * data_span, _EPS)

    sq_dist = np.sum((data_pos[:, None, :] - w_array[None, :, :]) ** 2, axis=2)
    global_assignment = np.argmin(sq_dist, axis=1)

    if data_path_assignment is not None:
        data_path_assignment = np.asarray(data_path_assignment, dtype=np.int64).reshape(-1)
        if data_path_assignment.shape[0] != data_pos.shape[0]:
            raise ValueError("data_path_assignment must contain one path id per sample.")

    path_models = []
    for p_id, path in enumerate(all_paths):
        path = np.asarray(path, dtype=np.int64).ravel()
        anchors = w_array[path]
        p = len(path)
        is_cycle = bool(all_is_cycle[p_id])

        geometry = _polyline_geometry(splines[p_id], projection_samples)
        prototype_arc = _prototype_arc_coordinates(anchors, geometry["total_length"], is_cycle)

        path_model = {
            "prototype_indices": path.copy(),
            "centers": anchors.copy(),
            "prototype_arc": prototype_arc,
            "is_cycle": is_cycle,
            **geometry,
        }

        radius_n = np.empty(p, dtype=np.float64)
        counts = np.empty(p, dtype=np.int64)

        for j, global_idx in enumerate(path):
            path_mask = (np.ones(data_pos.shape[0], dtype=bool) if data_path_assignment is None
                         else data_path_assignment == p_id)
            strict_mask = path_mask & (global_assignment == int(global_idx))
            X_cell = data_pos[strict_mask]

            if X_cell.shape[0] < min_samples and p > 1:
                local_neighbors = _neighbor_indices(j, p, neighbor_pool, is_cycle)
                pooled_global = np.unique(path[local_neighbors])
                pooled_mask = path_mask & np.isin(global_assignment, pooled_global)
                X_cell = data_pos[pooled_mask]

            counts[j] = X_cell.shape[0]
            if X_cell.shape[0] == 0:
                transverse_rms = min_radius_abs
            else:
                _, transverse_distance, _ = _project_to_polyline(X_cell, path_model)
                if dim > 1:
                    transverse_rms = float(np.sqrt(np.mean(transverse_distance * transverse_distance) / (dim - 1)))
                else:
                    transverse_rms = float(np.sqrt(np.mean(transverse_distance * transverse_distance)))

            radius_n[j] = transverse_scale * max(transverse_rms, min_radius_abs)

        path_model["radius_n"] = radius_n
        path_model["sample_counts"] = counts
        path_models.append(path_model)

    return {
        "type": "spline_coordinate_tube",
        "dim": dim,
        "transverse_scale": float(transverse_scale),
        "min_radius": float(min_radius),
        "min_samples": int(min_samples),
        "neighbor_pool": int(neighbor_pool),
        "projection_samples": int(projection_samples),
        "gamma_margin": float(gamma_margin),
        "gamma_beta": float(gamma_beta),
        "paths": path_models,
    }


def _smootherstep(t: np.ndarray) -> np.ndarray:
    """C2 transition from zero to one on [0, 1]."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def phase_partition(arc: np.ndarray, prototype_arc: np.ndarray, total_length: float, is_cycle: bool) -> np.ndarray:
    """Compact C2 partition of unity over ordered spline prototypes.

    At any point, at most two neighboring local A_{k,i}(x) matrices are active.
    """
    arc = np.asarray(arc, dtype=np.float64).reshape(-1)
    knots = np.asarray(prototype_arc, dtype=np.float64).reshape(-1)
    n = arc.shape[0]
    p = knots.shape[0]

    if p == 0:
        raise ValueError("A path must contain at least one prototype.")
    if p == 1:
        return np.ones((n, 1), dtype=np.float64)

    phi = np.zeros((n, p), dtype=np.float64)

    if not is_cycle:
        a = np.clip(arc, knots[0], knots[-1])
        right = np.searchsorted(knots, a, side="right")
        right = np.clip(right, 1, p - 1)
        left = right - 1

        denom = np.maximum(knots[right] - knots[left], _EPS)
        t = (a - knots[left]) / denom
        w_right = _smootherstep(t)

        rows = np.arange(n)
        phi[rows, left] = 1.0 - w_right
        phi[rows, right] = w_right

        below = arc <= knots[0]
        above = arc >= knots[-1]
        phi[below] = 0.0
        phi[below, 0] = 1.0
        phi[above] = 0.0
        phi[above, -1] = 1.0
        return phi

    length = max(float(total_length), _EPS)
    a = np.mod(arc, length)
    right = np.searchsorted(knots, a, side="right")
    left = right - 1

    wrap_right = right == p
    right_wrapped = np.where(wrap_right, 0, right)
    left_wrapped = np.where(left < 0, p - 1, left)

    k_left = knots[left_wrapped].copy()
    k_right = knots[right_wrapped].copy()

    before_first = left < 0
    k_left[before_first] -= length
    k_right[wrap_right] += length

    a_unwrapped = a.copy()
    a_unwrapped[before_first] += 0.0

    denom = np.maximum(k_right - k_left, _EPS)
    t = (a_unwrapped - k_left) / denom
    w_right = _smootherstep(t)

    rows = np.arange(n)
    phi[rows, left_wrapped] = 1.0 - w_right
    phi[rows, right_wrapped] += w_right
    return phi


def spline_weights_and_gate(X: np.ndarray, p_id: int, tube_model: dict,
                            projected_points: np.ndarray | None = None,
                            transverse_distances: np.ndarray | None = None,
                            gamma_margin: float | None = None, gamma_beta: float | None = None,
                            boundary_lambda: np.ndarray | None = None):
    """Return spline-phase weights and the boundary-aware support gate.

    The inner tube gate is
        G(x) = sigmoid(-beta * (d_Gamma(x) - delta)),
    and the returned gate is
        Gamma(x) = (1 - lambda_partial(x)) * G(x).

    When ``boundary_lambda`` is not supplied, ``lambda_partial`` is computed
    automatically from the active and second-closest spline distances using
    the polyline models stored in ``tube_model``. ``projected_points`` and
    ``transverse_distances`` can be supplied from the Lyapunov projection to
    avoid recomputing active-spline quantities already available.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]

    path_model = tube_model["paths"][int(p_id)]

    if projected_points is None:
        arc, distance, projection = _project_to_polyline(X, path_model)
    else:
        projection = np.asarray(projected_points, dtype=np.float64)
        if projection.ndim == 1:
            projection = projection[None, :]
        if projection.shape != X.shape:
            raise ValueError("projected_points must have the same shape as X.")
        arc, _, _ = _project_to_polyline(projection, path_model)

        if transverse_distances is None:
            distance = np.linalg.norm(X - projection, axis=1)
        else:
            distance = np.asarray(transverse_distances, dtype=np.float64).reshape(-1)
            if distance.shape[0] != X.shape[0]:
                raise ValueError("transverse_distances must contain one value per point.")
            distance = np.abs(distance)

    phi = phase_partition(arc, np.asarray(path_model["prototype_arc"], dtype=np.float64),
                          float(path_model["total_length"]), bool(path_model["is_cycle"]))

    radius_n = np.asarray(path_model["radius_n"], dtype=np.float64)
    local_radius_sq = phi @ np.maximum(radius_n * radius_n, _EPS)
    local_radius = np.sqrt(np.maximum(local_radius_sq, _EPS))
    q_perp = distance * distance / np.maximum(local_radius_sq, _EPS)

    margin = float(tube_model.get("gamma_margin", 1.0)) if gamma_margin is None else float(gamma_margin)
    beta = float(tube_model.get("gamma_beta", 10.0)) if gamma_beta is None else float(gamma_beta)
    beta = max(beta, 0.0)

    # Inner data-support gate G(x).
    support_logit = np.clip(beta * (q_perp - margin), -60.0, 60.0)
    support_gate = 1.0 / (1.0 + np.exp(support_logit))

    # Boundary factor lambda_partial(x) = exp(-z_kj(x)^2), where
    # z_kj = (d_j^2 - d_k^2) / d_k^2 and j is the second-closest spline.
    # For a single spline there is no switching boundary, so lambda_partial=0.
    if boundary_lambda is None:
        path_models = tube_model.get("paths", [])
        if len(path_models) <= 1:
            lambda_partial = np.zeros(X.shape[0], dtype=np.float64)
            second_distance = np.full(X.shape[0], np.inf, dtype=np.float64)
            second_path = np.full(X.shape[0], -1, dtype=np.int64)
            boundary_z = np.full(X.shape[0], np.inf, dtype=np.float64)
        else:
            second_distance = np.full(X.shape[0], np.inf, dtype=np.float64)
            second_path = np.full(X.shape[0], -1, dtype=np.int64)
            for other_id, other_model in enumerate(path_models):
                if other_id == int(p_id):
                    continue
                _, other_distance, _ = _project_to_polyline(X, other_model)
                better = other_distance < second_distance
                second_distance[better] = other_distance[better]
                second_path[better] = int(other_id)

            active_d2 = distance * distance
            second_d2 = second_distance * second_distance
            boundary_q = np.maximum(second_d2 - active_d2, 0.0)
            boundary_z = boundary_q / np.maximum(active_d2, _EPS)
            lambda_partial = np.exp(-np.minimum(boundary_z * boundary_z, 700.0))
    else:
        lambda_partial = np.asarray(boundary_lambda, dtype=np.float64).reshape(-1)
        if lambda_partial.shape[0] != X.shape[0]:
            raise ValueError("boundary_lambda must contain one value per point.")
        lambda_partial = np.clip(lambda_partial, 0.0, 1.0)
        second_distance = np.full(X.shape[0], np.nan, dtype=np.float64)
        second_path = np.full(X.shape[0], -1, dtype=np.int64)
        boundary_z = np.full(X.shape[0], np.nan, dtype=np.float64)

    gamma = (1.0 - lambda_partial) * support_gate
    gamma = np.clip(gamma, 0.0, 1.0)

    diagnostics = {
        "arc": arc,
        "projection": projection,
        "transverse_distance": distance,
        "local_radius": local_radius,
        "q_perp": q_perp,
        "support_gate": support_gate,
        "lambda_partial": lambda_partial,
        "boundary_z": boundary_z,
        "second_distance": second_distance,
        "second_path": second_path,
    }
    return phi, gamma, diagnostics
