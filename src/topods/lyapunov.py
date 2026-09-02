from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar


_EPS = 1e-20


def build_limit_cycle_flags(components, spline_nodes) -> List[bool]:
    """Return one cycle flag per flattened spline/path."""
    flags: List[bool] = []
    for (_, is_cycle), paths in zip(components, spline_nodes):
        if is_cycle:
            flags.append(True)
        else:
            flags.extend([False] * len(paths))
    return flags


def _metadata_entry(spline_metadata, k: int) -> dict:
    if spline_metadata is None or k < 0 or k >= len(spline_metadata):
        return {}
    entry = spline_metadata[k]
    return entry if isinstance(entry, dict) else {}


def _pair_is_transparent(k: int, j: int, spline_metadata) -> bool:
    """Return whether the categorical topology permits k<->j switching."""
    if k == j:
        return False
    mk = _metadata_entry(spline_metadata, k)
    mj = _metadata_entry(spline_metadata, j)
    if mk.get("type") != "path" or mj.get("type") != "path":
        return False
    return (
        j in set(map(int, mk.get("transparent_neighbors", [])))
        or k in set(map(int, mj.get("transparent_neighbors", [])))
    )


def _prefer_on_distance_tie(candidate: int, current: int, spline_metadata) -> bool:
    """Prefer the downstream primitive at an exact merge-junction tie."""
    if current < 0 or candidate == current:
        return current < 0
    mc = _metadata_entry(spline_metadata, candidate)
    mb = _metadata_entry(spline_metadata, current)
    if current in set(map(int, mc.get("predecessor_ids", []))):
        return True
    if candidate in set(map(int, mb.get("predecessor_ids", []))):
        return False
    # Deterministic fallback only; ordinary non-tied points are unaffected.
    return int(candidate) < int(current)


def _phase_hermite_value_derivative(t: float, cs, dcs, meta: dict):
    """Return global phase s(t) and ds/dt for one path primitive.

    ``phase_slope_start/end`` are derivatives ds/d(arc length).  Multiplying
    by the endpoint spline speed converts them to Hermite derivatives ds/dt,
    which makes the longitudinal derivative agree across a geometric junction.
    """
    if not meta or meta.get("phase_mode") != "global_hermite":
        return float(t), 1.0

    s0 = float(meta.get("phase_start", 0.0))
    s1 = float(meta.get("phase_end", 1.0))
    slope0 = float(meta.get("phase_slope_start", max(s1 - s0, 1e-12)))
    slope1 = float(meta.get("phase_slope_end", max(s1 - s0, 1e-12)))
    tau0 = np.asarray(dcs(0.0), dtype=np.float64)
    tau1 = np.asarray(dcs(1.0), dtype=np.float64)
    m0 = slope0 * float(np.linalg.norm(tau0))
    m1 = slope1 * float(np.linalg.norm(tau1))

    t = float(t)
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    value = h00 * s0 + h10 * m0 + h01 * s1 + h11 * m1

    dh00 = 6.0 * t2 - 6.0 * t
    dh10 = 3.0 * t2 - 4.0 * t + 1.0
    dh01 = -6.0 * t2 + 6.0 * t
    dh11 = 3.0 * t2 - 2.0 * t
    deriv = dh00 * s0 + dh10 * m0 + dh01 * s1 + dh11 * m1
    return float(value), float(deriv)


def _phase_endpoint_data(cs, dcs, meta: dict):
    """Return (s0, ds/dt|0, s1, ds/dt|1) for endpoint extrapolation."""
    if not meta or meta.get("phase_mode") != "global_hermite":
        return 0.0, 1.0, 1.0, 1.0
    s0 = float(meta.get("phase_start", 0.0))
    s1 = float(meta.get("phase_end", 1.0))
    slope0 = float(meta.get("phase_slope_start", max(s1 - s0, 1e-12)))
    slope1 = float(meta.get("phase_slope_end", max(s1 - s0, 1e-12)))
    m0 = slope0 * float(np.linalg.norm(dcs(0.0)))
    m1 = slope1 * float(np.linalg.norm(dcs(1.0)))
    return s0, m0, s1, m1


def closest_spline_phase(
    x: np.ndarray,
    cs,
    Cg: np.ndarray,
    S_COARSE: np.ndarray,
    coarse_i0: int | None = None,
) -> float:
    """Refine the closest-point phase around the best coarse sample."""
    if coarse_i0 is None:
        d2 = np.sum((Cg - x) ** 2, axis=1)
        coarse_i0 = int(np.argmin(d2))

    s_lo = float(S_COARSE[max(coarse_i0 - 1, 0)])
    s_hi = float(S_COARSE[min(coarse_i0 + 1, len(S_COARSE) - 1)])
    if s_hi <= s_lo + 1e-15:
        return float(S_COARSE[coarse_i0])

    result = minimize_scalar(
        lambda s: float(np.sum((x - cs(float(s))) ** 2)),
        bounds=(s_lo, s_hi),
        method="bounded",
    )
    return float(result.x)


def closest_spline_phases(
    X: np.ndarray,
    cs,
    S_COARSE: np.ndarray,
    coarse_i0: np.ndarray,
    max_iter: int = 24,
) -> np.ndarray:
    """Vectorized bounded refinement of closest phases for many points.

    Each point uses the same local interval around its best coarse sample as
    :func:`closest_spline_phase`. A fixed-iteration golden-section search lets
    the B-spline evaluate all phases in one batch instead of invoking a scalar
    optimizer for every sample.
    """
    X = np.asarray(X, dtype=np.float64)
    S_COARSE = np.asarray(S_COARSE, dtype=np.float64)
    coarse_i0 = np.asarray(coarse_i0, dtype=np.int64).reshape(-1)
    if X.ndim != 2 or coarse_i0.shape[0] != X.shape[0]:
        raise ValueError("coarse_i0 must contain one index per sample")

    lo_idx = np.maximum(coarse_i0 - 1, 0)
    hi_idx = np.minimum(coarse_i0 + 1, len(S_COARSE) - 1)
    lo = S_COARSE[lo_idx].copy()
    hi = S_COARSE[hi_idx].copy()

    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    left = hi - inv_phi * (hi - lo)
    right = lo + inv_phi * (hi - lo)

    def _distance_sq(phases):
        delta = X - np.asarray(cs(phases), dtype=np.float64)
        return np.einsum("ni,ni->n", delta, delta, optimize=True)

    f_left = _distance_sq(left)
    f_right = _distance_sq(right)

    for _ in range(max(int(max_iter), 1)):
        choose_left = f_left <= f_right
        old_left = left
        old_right = right
        old_f_left = f_left
        old_f_right = f_right

        lo = np.where(choose_left, lo, old_left)
        hi = np.where(choose_left, old_right, hi)
        candidate = np.where(
            choose_left,
            hi - inv_phi * (hi - lo),
            lo + inv_phi * (hi - lo),
        )
        f_candidate = _distance_sq(candidate)

        left = np.where(choose_left, candidate, old_right)
        right = np.where(choose_left, old_left, candidate)
        f_left = np.where(choose_left, f_candidate, old_f_right)
        f_right = np.where(choose_left, old_f_left, f_candidate)

    return np.where(f_left <= f_right, left, right)


def distance_gradient_D(
    x: np.ndarray,
    cs,
    dcs,
    ddcs,
    s: float,
    is_cycle: bool,
) -> Tuple[float, np.ndarray]:
    """Return ``d^2`` and the gradient of ``D=0.5 d^2``."""
    C = cs(s)
    u = x - C
    dist2 = float(np.dot(u, u))

    if is_cycle:
        tau = dcs(s)
        kappa = ddcs(s)
        tau_norm2 = float(np.dot(tau, tau))
        denom = tau_norm2 - float(np.dot(u, kappa))
        grad_D = (tau_norm2 / denom) * u if abs(denom) > 1e-12 else u
    else:
        grad_D = u

    return dist2, np.asarray(grad_D, dtype=np.float64)


def boundary_weight_and_alpha(
    k: int,
    x: np.ndarray,
    active_dist2: float,
    coarse_best: np.ndarray,
    coarse_i0: np.ndarray,
    spline_cache,
    limit_cycle_flags: Sequence[bool],
    S_COARSE: np.ndarray,
    alpha: float,
    enable_boundary_progress_gate: bool = True,
    boundary_progress_gate_scale: float = 0.5,
    boundary_progress_gate_eps: float = 1e-12,
    phase_cache: dict | None = None,
    spline_metadata=None,
) -> Tuple[float, float]:
    """Return ``lambda_partial`` and ``alpha_partial``.

    The scalar Lyapunov function does not use these state-dependent weights.
    They are used only to construct the boundary-aware descent direction

        g = alpha_partial grad(D)
            + (1-alpha_partial) grad(R).

    Consequently, no derivative of ``lambda_partial`` enters the direction.
    """
    n_splines = len(spline_cache)
    if (not enable_boundary_progress_gate) or n_splines <= 1:
        return 0.0, float(alpha)

    # First identify the *actual* second-closest spline, independent of topology.
    # Only after that geometric competitor is known do we apply the categorical
    # protected/transparent pair mask.  This avoids a protected sibling hiding a
    # legitimate parent-child junction surface.
    second_dist2 = np.inf
    second_idx = -1
    for j in range(n_splines):
        if j == k:
            continue

        cs_j, dcs_j, ddcs_j, Cg_j = spline_cache[j]
        if phase_cache is not None and j in phase_cache:
            s_j = float(phase_cache[j])
        else:
            s_j = closest_spline_phase(
                x, cs_j, Cg_j, S_COARSE, int(coarse_i0[j])
            )
            if phase_cache is not None:
                phase_cache[j] = s_j

        dist2_j, _ = distance_gradient_D(
            x, cs_j, dcs_j, ddcs_j, s_j, bool(limit_cycle_flags[j])
        )
        dist2_j = float(dist2_j)
        if dist2_j < second_dist2:
            second_dist2 = dist2_j
            second_idx = int(j)

    if second_idx >= 0 and _pair_is_transparent(k, second_idx, spline_metadata):
        return 0.0, float(alpha)

    gap = max(second_dist2 - float(active_dist2), 0.0)
    rho = max(float(boundary_progress_gate_scale), 1e-20)
    eps_gate = max(float(boundary_progress_gate_eps), 1e-20)

    z = gap / (float(active_dist2) + eps_gate)
    scaled_z = z / rho
    lambda_partial = float(np.exp(-min(scaled_z * scaled_z, 700.0)))
    alpha_partial = float(alpha) + (1.0 - float(alpha)) * lambda_partial
    return lambda_partial, alpha_partial


def path_energy_direction(
    x: np.ndarray,
    k: int,
    cs,
    dcs,
    ddcs,
    C: np.ndarray,
    u: np.ndarray,
    dist: float,
    tau: np.ndarray,
    tau_norm: float,
    tau_hat: np.ndarray,
    s_star: float,
    coarse_best: np.ndarray,
    coarse_i0: np.ndarray,
    spline_cache,
    limit_cycle_flags: Sequence[bool],
    S_COARSE: np.ndarray,
    alpha: float,
    enable_boundary_progress_gate: bool = True,
    boundary_progress_gate_scale: float = 0.5,
    boundary_progress_gate_eps: float = 1e-12,
    phase_cache: dict | None = None,
    spline_metadata=None,
):
    """Return the nominal Lyapunov energy and boundary-aware direction."""
    kappa = ddcs(s_star)
    tau_norm2 = float(np.dot(tau, tau))
    utau = float(np.dot(u, tau))
    denom = tau_norm2 - float(np.dot(u, kappa))

    x0 = cs(0.0)
    tau0 = dcs(0.0)
    tau0_norm2 = float(np.dot(tau0, tau0)) + _EPS

    x1 = cs(1.0)
    tau1 = dcs(1.0)
    tau1_norm2 = float(np.dot(tau1, tau1)) + _EPS

    # ``s_star`` is the local spline parameter t in [0,1].  Junction-aware
    # paths map it to a global source-to-goal phase s(t), while the geometric
    # closest-point projection remains unchanged.
    meta = _metadata_entry(spline_metadata, k)
    s0_global, dsdt0, s1_global, dsdt1 = _phase_endpoint_data(cs, dcs, meta)

    t_ext0 = float(np.dot(x - x0, tau0) / tau0_norm2)
    t_ext1 = 1.0 + float(np.dot(x - x1, tau1) / tau1_norm2)

    s_eps = 1e-3
    ortho_tol = 1e-2 * max(1.0, dist * tau_norm)

    if (s_star <= s_eps) and (t_ext0 < 0.0):
        s_eff = s0_global + dsdt0 * t_ext0
        grad_s = dsdt0 * tau0 / tau0_norm2
        C_eff = x0
    elif (s_star >= 1.0 - s_eps) and (t_ext1 > 1.0):
        s_eff = s1_global + dsdt1 * (t_ext1 - 1.0)
        grad_s = dsdt1 * tau1 / tau1_norm2
        C_eff = x1
    else:
        s_eff, dsdt = _phase_hermite_value_derivative(s_star, cs, dcs, meta)
        C_eff = C
        if denom > 1e-8 and abs(utau) < ortho_tol:
            grad_t = tau / denom
        else:
            grad_t = tau_hat / (tau_norm + _EPS)
        grad_s = dsdt * grad_t

    u_eff = x - C_eff
    dist_eff = float(np.linalg.norm(u_eff))
    phase_dist = 1.0 - s_eff

    D_val = 0.5 * float(np.dot(u_eff, u_eff))
    R_val = 0.5 * float(phase_dist * phase_dist)

    grad_D = u_eff
    grad_R = -phase_dist * grad_s

    lambda_partial, alpha_partial = boundary_weight_and_alpha(
        k=k,
        x=x,
        active_dist2=float(np.dot(u_eff, u_eff)),
        coarse_best=coarse_best,
        coarse_i0=coarse_i0,
        spline_cache=spline_cache,
        limit_cycle_flags=limit_cycle_flags,
        S_COARSE=S_COARSE,
        alpha=alpha,
        enable_boundary_progress_gate=enable_boundary_progress_gate,
        boundary_progress_gate_scale=boundary_progress_gate_scale,
        boundary_progress_gate_eps=boundary_progress_gate_eps,
        phase_cache=phase_cache,
        spline_metadata=spline_metadata,
    )

    # Simple scalar Lyapunov function.
    V_val = float(alpha) * D_val + (1.0 - float(alpha)) * R_val
    grad_V = float(alpha) * grad_D + (1.0 - float(alpha)) * grad_R

    # Boundary awareness appears only in the descent direction.
    direction = alpha_partial * grad_D + (1.0 - alpha_partial) * grad_R

    return (
        C_eff,
        dist_eff,
        phase_dist,
        V_val,
        np.asarray(direction, dtype=np.float64),
        np.asarray(grad_V, dtype=np.float64),
        lambda_partial,
    )


def cycle_energy_direction(
    x: np.ndarray,
    k: int,
    cs,
    dcs,
    ddcs,
    C: np.ndarray,
    u: np.ndarray,
    dist: float,
    tau: np.ndarray,
    tau_hat: np.ndarray,
    s_star: float,
    coarse_best: np.ndarray,
    coarse_i0: np.ndarray,
    spline_cache,
    limit_cycle_flags: Sequence[bool],
    S_COARSE: np.ndarray,
    alpha: float,
    enable_boundary_progress_gate: bool = True,
    boundary_progress_gate_scale: float = 0.5,
    boundary_progress_gate_eps: float = 1e-12,
    phase_cache: dict | None = None,
    spline_metadata=None,
):
    """Return cycle energy, boundary-aware attracting/circulating direction."""
    kappa = ddcs(s_star)
    tau_norm2 = float(np.dot(tau, tau))
    denom = tau_norm2 - float(np.dot(u, kappa))
    grad_D = (tau_norm2 / denom) * u if abs(denom) > 1e-12 else u

    D_val = 0.5 * float(dist * dist)
    lambda_partial, alpha_partial = boundary_weight_and_alpha(
        k=k,
        x=x,
        active_dist2=float(dist * dist),
        coarse_best=coarse_best,
        coarse_i0=coarse_i0,
        spline_cache=spline_cache,
        limit_cycle_flags=limit_cycle_flags,
        S_COARSE=S_COARSE,
        alpha=alpha,
        enable_boundary_progress_gate=enable_boundary_progress_gate,
        boundary_progress_gate_scale=boundary_progress_gate_scale,
        boundary_progress_gate_eps=boundary_progress_gate_eps,
        phase_cache=phase_cache,
        spline_metadata=spline_metadata,
    )

    # V = alpha D. The tangential term is orthogonal to grad(V).
    V_val = float(alpha) * D_val
    grad_V = float(alpha) * grad_D
    circulation = -tau_hat
    direction = alpha_partial * grad_D + (1.0 - alpha_partial) * circulation

    return (
        C,
        float(dist),
        0.0,
        float(V_val),
        np.asarray(direction, dtype=np.float64),
        np.asarray(grad_V, dtype=np.float64),
        lambda_partial,
    )


def compute_energy_direction_and_aux_one(
    x: np.ndarray,
    spline_cache,
    spline_nodes,
    components,
    alpha: float,
    dim: int,
    enable_boundary_progress_gate: bool = True,
    boundary_progress_gate_scale: float = 0.5,
    boundary_progress_gate_eps: float = 1e-12,
    S_COARSE: np.ndarray | None = None,
    top_k: int = 3,
    _limit_cycle_flags: Sequence[bool] | None = None,
    _coarse_best: np.ndarray | None = None,
    _coarse_i0: np.ndarray | None = None,
    _precomputed_phases: np.ndarray | None = None,
    spline_metadata=None,
    return_phase: bool = False,
):
    """Return projection, distances, V, direction, id, grad(V), and lambda."""
    x = np.asarray(x, dtype=np.float64)
    if S_COARSE is None:
        S_COARSE = np.linspace(0.0, 1.0, 100)
    else:
        S_COARSE = np.asarray(S_COARSE, dtype=np.float64)

    limit_cycle_flags = (
        build_limit_cycle_flags(components, spline_nodes)
        if _limit_cycle_flags is None
        else _limit_cycle_flags
    )
    n_splines = len(spline_cache)
    if n_splines == 0:
        raise ValueError("spline_cache is empty")

    # Boundary correctness requires the true closest spline. Evaluate all
    # fitted splines when boundary awareness is enabled.
    if enable_boundary_progress_gate and n_splines > 1:
        top_k = n_splines
    else:
        top_k = max(1, min(int(top_k), n_splines))

    if _coarse_best is None or _coarse_i0 is None:
        coarse_best = np.empty(n_splines, dtype=np.float64)
        coarse_i0 = np.empty(n_splines, dtype=np.int64)
        for idx, entry in enumerate(spline_cache):
            _, _, _, Cg = entry
            d2 = np.sum((Cg - x) ** 2, axis=1)
            i0 = int(np.argmin(d2))
            coarse_best[idx] = d2[i0]
            coarse_i0[idx] = i0
    else:
        coarse_best = np.asarray(_coarse_best, dtype=np.float64)
        coarse_i0 = np.asarray(_coarse_i0, dtype=np.int64)

    cand_idx = np.argpartition(coarse_best, top_k - 1)[:top_k]
    if _precomputed_phases is None:
        phase_cache: dict[int, float] = {}
    else:
        precomputed_phases = np.asarray(_precomputed_phases, dtype=np.float64)
        if precomputed_phases.shape != (n_splines,):
            raise ValueError("_precomputed_phases must contain one phase per spline")
        phase_cache = {
            k: float(precomputed_phases[k]) for k in range(n_splines)
        }

    best_dist = np.inf
    best_phase_dist = 1.0
    best_V = 0.0
    best_direction = np.zeros(dim, dtype=np.float64)
    best_grad_V = np.zeros(dim, dtype=np.float64)
    best_lambda = 0.0
    best_idx = -1
    best_proj = None
    best_phase = 0.0

    for raw_k in cand_idx:
        k = int(raw_k)
        cs, dcs, ddcs, Cg = spline_cache[k]
        is_cycle = bool(limit_cycle_flags[k])

        if k in phase_cache:
            s_star = phase_cache[k]
        else:
            s_star = closest_spline_phase(
                x, cs, Cg, S_COARSE, int(coarse_i0[k])
            )
            phase_cache[k] = s_star

        C = cs(s_star)
        u = x - C
        dist = float(np.linalg.norm(u))
        tau = dcs(s_star)
        tau_norm = float(np.linalg.norm(tau))
        tau_hat = tau / (tau_norm + _EPS)

        if is_cycle:
            result = cycle_energy_direction(
                x=x,
                k=k,
                cs=cs,
                dcs=dcs,
                ddcs=ddcs,
                C=C,
                u=u,
                dist=dist,
                tau=tau,
                tau_hat=tau_hat,
                s_star=s_star,
                coarse_best=coarse_best,
                coarse_i0=coarse_i0,
                spline_cache=spline_cache,
                limit_cycle_flags=limit_cycle_flags,
                S_COARSE=S_COARSE,
                alpha=alpha,
                enable_boundary_progress_gate=enable_boundary_progress_gate,
                boundary_progress_gate_scale=boundary_progress_gate_scale,
                boundary_progress_gate_eps=boundary_progress_gate_eps,
                phase_cache=phase_cache,
                spline_metadata=spline_metadata,
            )
        else:
            result = path_energy_direction(
                x=x,
                k=k,
                cs=cs,
                dcs=dcs,
                ddcs=ddcs,
                C=C,
                u=u,
                dist=dist,
                tau=tau,
                tau_norm=tau_norm,
                tau_hat=tau_hat,
                s_star=s_star,
                coarse_best=coarse_best,
                coarse_i0=coarse_i0,
                spline_cache=spline_cache,
                limit_cycle_flags=limit_cycle_flags,
                S_COARSE=S_COARSE,
                alpha=alpha,
                enable_boundary_progress_gate=enable_boundary_progress_gate,
                boundary_progress_gate_scale=boundary_progress_gate_scale,
                boundary_progress_gate_eps=boundary_progress_gate_eps,
                phase_cache=phase_cache,
                spline_metadata=spline_metadata,
            )

        proj, dist_val, progress_val, V_val, direction, grad_V, lam = result
        tie_tol = 1e-10 * max(1.0, abs(float(best_dist)) if np.isfinite(best_dist) else 1.0, abs(float(dist_val)))
        better = dist_val < best_dist - tie_tol
        tied = np.isfinite(best_dist) and abs(dist_val - best_dist) <= tie_tol
        if better or (tied and _prefer_on_distance_tie(k, best_idx, spline_metadata)):
            best_proj = proj
            best_dist = dist_val
            best_phase_dist = progress_val
            best_V = V_val
            best_direction = direction
            best_grad_V = grad_V
            best_lambda = lam
            best_idx = k
            best_phase = s_star

    result = (
        best_proj,
        best_dist,
        best_phase_dist,
        best_V,
        best_direction,
        best_idx,
        best_grad_V,
        best_lambda,
    )
    if return_phase:
        return result + (best_phase,)
    return result


def compute_energy_direction_and_aux_multi(
    grid_points: np.ndarray,
    spline_cache,
    spline_nodes,
    components,
    alpha: float,
    dim: int,
    enable_boundary_progress_gate: bool = True,
    boundary_progress_gate_scale: float = 0.5,
    boundary_progress_gate_eps: float = 1e-12,
    S_COARSE: np.ndarray | None = None,
    top_k: int = 3,
    spline_metadata=None,
    return_phases: bool = False,
):
    """Batched evaluation of the nominal energy and boundary-aware direction."""
    grid_points = np.asarray(grid_points, dtype=np.float64)
    if S_COARSE is None:
        S_COARSE = np.linspace(0.0, 1.0, 100)
    else:
        S_COARSE = np.asarray(S_COARSE, dtype=np.float64)

    limit_cycle_flags = build_limit_cycle_flags(components, spline_nodes)
    n_points = grid_points.shape[0]
    n_splines = len(spline_cache)
    if n_splines == 0:
        raise ValueError("spline_cache is empty")

    coarse_best = np.empty((n_points, n_splines), dtype=np.float64)
    coarse_i0 = np.empty((n_points, n_splines), dtype=np.int64)
    for k, entry in enumerate(spline_cache):
        _, _, _, Cg = entry
        diff = Cg[None, :, :] - grid_points[:, None, :]
        d2 = np.sum(diff * diff, axis=2)
        nearest = np.argmin(d2, axis=1)
        coarse_i0[:, k] = nearest
        coarse_best[:, k] = d2[np.arange(n_points), nearest]

    refined_phases = None
    if return_phases:
        refined_phases = np.empty((n_points, n_splines), dtype=np.float64)
        for k, entry in enumerate(spline_cache):
            cs = entry[0]
            refined_phases[:, k] = closest_spline_phases(
                grid_points, cs, S_COARSE, coarse_i0[:, k]
            )

    projections = np.empty((n_points, dim), dtype=np.float64)
    distances = np.empty(n_points, dtype=np.float64)
    phase_distances = np.empty(n_points, dtype=np.float64)
    energies = np.empty(n_points, dtype=np.float64)
    directions = np.empty((n_points, dim), dtype=np.float64)
    spline_ids = np.empty(n_points, dtype=np.int64)
    lyapunov_gradients = np.empty((n_points, dim), dtype=np.float64)
    boundary_lambdas = np.empty(n_points, dtype=np.float64)
    phases = np.empty(n_points, dtype=np.float64) if return_phases else None

    for point_idx, x in enumerate(grid_points):
        result = compute_energy_direction_and_aux_one(
            x=x,
            spline_cache=spline_cache,
            spline_nodes=spline_nodes,
            components=components,
            alpha=alpha,
            dim=dim,
            enable_boundary_progress_gate=enable_boundary_progress_gate,
            boundary_progress_gate_scale=boundary_progress_gate_scale,
            boundary_progress_gate_eps=boundary_progress_gate_eps,
            S_COARSE=S_COARSE,
            top_k=top_k,
            _limit_cycle_flags=limit_cycle_flags,
            _coarse_best=coarse_best[point_idx],
            _coarse_i0=coarse_i0[point_idx],
            _precomputed_phases=(
                refined_phases[point_idx] if refined_phases is not None else None
            ),
            spline_metadata=spline_metadata,
            return_phase=return_phases,
        )
        projections[point_idx] = result[0]
        distances[point_idx] = result[1]
        phase_distances[point_idx] = result[2]
        energies[point_idx] = result[3]
        directions[point_idx] = result[4]
        spline_ids[point_idx] = result[5]
        lyapunov_gradients[point_idx] = result[6]
        boundary_lambdas[point_idx] = result[7]
        if return_phases:
            phases[point_idx] = result[8]

    result = (
        projections,
        distances,
        phase_distances,
        energies,
        directions,
        spline_ids,
        lyapunov_gradients,
        boundary_lambdas,
    )
    if return_phases:
        return result + (phases,)
    return result


def tangent_projectors(
    X: np.ndarray,
    spline_cache,
    spline_id: int,
    S_COARSE: np.ndarray | None = None,
    phases: np.ndarray | None = None,
):
    """Return tangent vectors and normal/tangent projectors."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]

    if S_COARSE is None:
        S_COARSE = np.linspace(0.0, 1.0, 100)
    else:
        S_COARSE = np.asarray(S_COARSE, dtype=np.float64)

    cs, dcs, _, Cg = spline_cache[spline_id]
    dim = X.shape[1]
    tangents = np.zeros((X.shape[0], dim), dtype=np.float64)
    P_t = np.zeros((X.shape[0], dim, dim), dtype=np.float64)
    P_n = np.zeros((X.shape[0], dim, dim), dtype=np.float64)
    I = np.eye(dim, dtype=np.float64)

    if phases is not None:
        phases = np.asarray(phases, dtype=np.float64).reshape(-1)
        if phases.shape[0] != X.shape[0]:
            raise ValueError("phases must contain one value per sample")
        tau = np.asarray(dcs(phases), dtype=np.float64).reshape(X.shape[0], dim)
        tau_hat = tau / (np.linalg.norm(tau, axis=1, keepdims=True) + _EPS)
        P_t = np.einsum("ni,nj->nij", tau_hat, tau_hat, optimize=True)
        P_n = I[None, :, :] - P_t
        return tau_hat, P_n, P_t

    for n, x in enumerate(X):
        d2 = np.sum((Cg - x) ** 2, axis=1)
        i0 = int(np.argmin(d2))
        s_star = closest_spline_phase(x, cs, Cg, S_COARSE, i0)
        tau = dcs(s_star)
        tau_hat = tau / (np.linalg.norm(tau) + _EPS)
        tangents[n] = tau_hat
        P_t[n] = np.outer(tau_hat, tau_hat)
        P_n[n] = I - P_t[n]

    return tangents, P_n, P_t


def block_modulation_matrices(
    Phi: np.ndarray,
    modulation,
    P_n: np.ndarray,
    P_t: np.ndarray,
) -> np.ndarray:
    """Blend tangent-normal block modulation parameters."""
    Phi = np.asarray(Phi, dtype=np.float64)
    P_n = np.asarray(P_n, dtype=np.float64)
    P_t = np.asarray(P_t, dtype=np.float64)

    B_local = np.asarray(modulation["B"], dtype=np.float64)
    b_local = np.asarray(modulation["b"], dtype=np.float64)
    n_samples, n_local = Phi.shape
    dim = P_n.shape[1]

    B_mix = np.einsum("np,pij->nij", Phi, B_local)
    b_mix = Phi @ b_local
    normal = np.einsum("nij,njk,nkl->nil", P_n, B_mix, P_n)
    return normal + b_mix[:, None, None] * P_t


class LyapunovField:
    """Own all Lyapunov/energy-side state used by inference.
    """

    def __init__(
        self,
        w_array,
        splines,
        spline_nodes,
        components,
        tube_model=None,
        alpha: float = 0.8,
        enable_boundary_progress_gate: bool = True,
        boundary_progress_gate_scale: float | None = None,
        boundary_progress_gate_eps: float = 1e-12,
        enable_H_gate: bool = True,
        beta_H: float = 1000.0,
        goals=None,
        coarse_samples: int = 100,
    ):
        self.w_array = np.asarray(w_array, dtype=np.float64)
        self.splines = list(splines)
        self.spline_nodes = spline_nodes
        self.components = components
        self.tube_model = {} if tube_model is None else tube_model

        self.dim = int(self.w_array.shape[1])
        self.alpha = float(alpha)
        self.enable_boundary_progress_gate = bool(enable_boundary_progress_gate)
        if boundary_progress_gate_scale is None:
            boundary_progress_gate_scale = self.tube_model.get(
                "boundary_progress_gate_scale", 0.5
            )
        self.boundary_progress_gate_scale = float(boundary_progress_gate_scale)
        self.boundary_progress_gate_eps = float(boundary_progress_gate_eps)
        self.enable_H_gate = bool(enable_H_gate)
        self.beta_H = float(beta_H)

        self.S_COARSE = np.linspace(0.0, 1.0, int(coarse_samples))
        self.spline_cache = []
        for cs in self.splines:
            Cg = cs(self.S_COARSE)
            dcs = cs.derivative()
            ddcs = dcs.derivative()
            self.spline_cache.append((cs, dcs, ddcs, Cg))

        self.limit_cycle_flags = build_limit_cycle_flags(
            self.components, self.spline_nodes
        )
        self.spline_metadata = self.tube_model.get(
            "spline_primitive_metadata", None
        )
        if not isinstance(self.spline_metadata, list):
            self.spline_metadata = None

        self.goal_points = self._build_goal_points(goals)

    def _flatten_paths(self):
        paths = []
        is_cycle = []
        for (_, component_is_cycle), component_paths in zip(
            self.components, self.spline_nodes
        ):
            if component_is_cycle:
                paths.append(component_paths[0])
                is_cycle.append(True)
            else:
                for path in component_paths:
                    paths.append(path)
                    is_cycle.append(False)
        return paths, is_cycle

    def _build_goal_points(self, goals):
        """Return only true terminal task goals, never merge junctions."""
        centers = []

        if goals is not None:
            explicit = np.asarray(goals, dtype=np.float64)
            if explicit.size > 0:
                if explicit.ndim == 1:
                    explicit = explicit[None, :]
                centers.append(explicit)

        # Junction-aware models explicitly identify which path primitive really
        # terminates at a goal.  Upstream primitives ending at merge prototypes
        # therefore do not create zeros of H.
        if self.spline_metadata is not None:
            for meta in self.spline_metadata:
                if not isinstance(meta, dict):
                    continue
                if meta.get("type") != "path" or not meta.get(
                    "terminal_is_goal", False
                ):
                    continue
                nodes = np.asarray(meta.get("nodes", ()), dtype=int).ravel()
                if nodes.size:
                    centers.append(self.w_array[int(nodes[-1])][None, :])
        elif not centers:
            # Backward compatibility for pre-junction models: every non-cycle
            # path endpoint was historically treated as a goal.
            paths, cycle_flags = self._flatten_paths()
            for path, is_cycle in zip(paths, cycle_flags):
                if is_cycle:
                    continue
                path = np.asarray(path, dtype=int).ravel()
                if path.size:
                    centers.append(self.w_array[int(path[-1])][None, :])

        if not centers:
            return np.zeros((0, self.dim), dtype=np.float64)

        points = np.vstack(centers).astype(np.float64, copy=False)
        # Stable de-duplication without imposing a scale-dependent tolerance.
        _, unique_idx = np.unique(points, axis=0, return_index=True)
        return points[np.sort(unique_idx)]

    def evaluate_one(self, x, top_k: int = 3, return_phase: bool = False):
        return compute_energy_direction_and_aux_one(
            x=np.asarray(x, dtype=np.float64),
            spline_cache=self.spline_cache,
            spline_nodes=self.spline_nodes,
            components=self.components,
            alpha=self.alpha,
            dim=self.dim,
            enable_boundary_progress_gate=self.enable_boundary_progress_gate,
            boundary_progress_gate_scale=self.boundary_progress_gate_scale,
            boundary_progress_gate_eps=self.boundary_progress_gate_eps,
            S_COARSE=self.S_COARSE,
            top_k=top_k,
            spline_metadata=self.spline_metadata,
            return_phase=return_phase,
        )

    def evaluate_multi(self, x, top_k: int = 3):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must have shape (n_samples, {self.dim})")
        return compute_energy_direction_and_aux_multi(
            grid_points=x,
            spline_cache=self.spline_cache,
            spline_nodes=self.spline_nodes,
            components=self.components,
            alpha=self.alpha,
            dim=self.dim,
            enable_boundary_progress_gate=self.enable_boundary_progress_gate,
            boundary_progress_gate_scale=self.boundary_progress_gate_scale,
            boundary_progress_gate_eps=self.boundary_progress_gate_eps,
            S_COARSE=self.S_COARSE,
            top_k=top_k,
            spline_metadata=self.spline_metadata,
        )

    def H_gate_one(self, x, spline_id: int) -> float:
        """Terminal goal gate H(x); cycles and junction endpoints are untouched."""
        if not self.enable_H_gate:
            return 1.0
        spline_id = int(spline_id)
        if self.limit_cycle_flags[spline_id]:
            return 1.0
        if self.goal_points.shape[0] == 0 or self.beta_H <= 0.0:
            return 1.0

        x = np.asarray(x, dtype=np.float64)
        delta = self.goal_points - x[None, :]
        min_dist2 = float(np.min(np.sum(delta * delta, axis=1)))
        return float(1.0 - np.exp(-self.beta_H * min_dist2))

    def H_gate_multi(self, x, spline_id: int) -> np.ndarray:
        """Vectorized terminal goal gate with shape ``(n_samples, 1)``."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must have shape (n_samples, {self.dim})")
        spline_id = int(spline_id)
        if (
            not self.enable_H_gate
            or self.limit_cycle_flags[spline_id]
            or self.goal_points.shape[0] == 0
            or self.beta_H <= 0.0
        ):
            return np.ones((len(x), 1), dtype=np.float64)

        delta = x[:, None, :] - self.goal_points[None, :, :]
        min_dist2 = np.min(np.sum(delta * delta, axis=2), axis=1, keepdims=True)
        return 1.0 - np.exp(-self.beta_H * min_dist2)
