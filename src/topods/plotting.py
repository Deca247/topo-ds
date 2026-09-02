import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .spline_helpers import spline_weights_and_gate
from .lyapunov import block_modulation_matrices, tangent_projectors

mpl.rcParams["savefig.dpi"] = 600
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.pad_inches"] = 0.02
mpl.rcParams["figure.dpi"] = 120      # display dpi only


def _plot_rollouts_3d(data_pos, goal, w_array, C, ds_inf,
                      box_constraint_margin, background_mode,
                      trajectories, surface_plot):
    """Plot demonstrations and the inferred dynamical system in 3D."""
    fig = plt.figure(figsize=(7.0, 6.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    # Keep the 3D field sparse: Matplotlib expands every quiver arrow into
    # several line segments, and dense fields obscure the trajectories anyway.
    h = 6
    data_min = np.min(data_pos, axis=0)
    data_max = np.max(data_pos, axis=0)
    data_range = data_max - data_min
    fallback_range = max(float(np.max(data_range)), 1.0)
    margins = np.where(
        data_range > 0.0,
        data_range * box_constraint_margin,
        fallback_range * max(box_constraint_margin, 0.05),
    )
    starts = data_min - margins
    ends = data_max + margins

    x_vals = np.linspace(starts[0], ends[0], h)
    y_vals = np.linspace(starts[1], ends[1], h)
    z_vals = np.linspace(starts[2], ends[2], h)
    xg, yg, zg = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
    grid_points = np.column_stack((xg.ravel(), yg.ravel(), zg.ravel()))

    vectors, aux = ds_inf.velocity_multi(grid_points, return_aux=True)
    grid_V = np.asarray(aux[3])
    gradients = np.asarray(aux[6])
    grid_dV = np.einsum("ij,ij->i", gradients, vectors)

    magnitude = np.linalg.norm(vectors, axis=1)
    background_values = None
    background_label = None
    if background_mode == "mag":
        background_values = magnitude
        background_label = "|f(x)|"
    elif background_mode == "V":
        background_values = grid_V
        background_label = "V(x)"
    elif background_mode == "logV":
        background_values = np.log(grid_V + 1e-8)
        background_label = "log V(x)"
    elif background_mode == "dV":
        background_values = grid_dV
        background_label = "dV(x)"
    elif background_mode == "pos":
        background_values = grid_dV

    # Matplotlib has no 3D streamplot or volumetric contourf equivalent. A
    # translucent scalar-colored point cloud plus quiver arrows conveys the
    # same scalar background and vector-field information in three dimensions.
    if background_values is not None:
        if background_mode == "pos":
            point_colors = np.where(
                grid_dV > 0.0, "red",
                np.where(grid_dV < 0.0, "green", "white"),
            )
            ax.scatter(*grid_points.T, c=point_colors, s=8, alpha=0.12,
                       depthshade=False, zorder=0)
        else:
            scalar_cmap = plt.cm.Blues \
                if background_mode in {"V", "logV"} else "viridis"
            if background_mode == "mag":
                scalar_cmap = mpl.colors.LinearSegmentedColormap.from_list(
                    "velocity_gray_to_blue", ("gray", "#0067C5")
                )
            elif background_mode == "dV":
                scalar_cmap = "coolwarm"
            scalar_field = ax.scatter(
                *grid_points.T, c=background_values, s=8, alpha=0.14,
                cmap=scalar_cmap, depthshade=False, zorder=0,
            )
            fig.colorbar(scalar_field, ax=ax, shrink=0.65, pad=0.1).set_label(
                background_label
            )

    nonzero_vectors = magnitude > np.finfo(float).eps
    if np.any(nonzero_vectors):
        quiver_length = 0.08 * max(float(np.max(ends - starts)), 1e-12)
        # White remains legible on scalar-colored samples; gray works better
        # against an otherwise empty 3D field.
        vector_color = "white" if background_mode in {"V", "logV", "dV"} \
            else "gray"
        ax.quiver(
            *grid_points[nonzero_vectors].T,
            *vectors[nonzero_vectors].T,
            length=quiver_length, normalize=True, color=vector_color,
            alpha=0.55, linewidth=0.55,
        )

    ax.scatter(
        *data_pos.T, alpha=0.3, s=7, color="black", linewidths=0,
        depthshade=False, zorder=2, label="Data",
    )

    if trajectories is not None:
        for i, traj in enumerate(trajectories):
            ax.plot(
                *np.asarray(traj)[:, :3].T, color="red", alpha=0.5, lw=1.5,
                zorder=6, label="Rollouts" if i == 0 else None,
            )

    rows, columns = np.where(np.asarray(C) > 0)
    for source, target in zip(rows, columns):
        start = w_array[source]
        direction = w_array[target] - start
        if np.linalg.norm(direction) <= 1e-8:
            continue
        ax.quiver(
            *(start + 0.08 * direction), *(0.84 * direction),
            color="green", linewidth=1.5, arrow_length_ratio=0.12,
        )

    ax.scatter(
        *w_array.T, s=42, c="green", edgecolors="black", linewidths=0.8,
        depthshade=False, zorder=7, label="Prototypes",
    )
    ax.scatter(
        *goal[:3], marker="X", s=180, color="black", linewidths=2.5,
        depthshade=False, label="Goal", zorder=10,
    )

    _finish_3d_axes(ax, starts, ends)
    ax.legend(loc="upper left", markerscale=0.8)

    # A scalar function over a 3D domain would require a fourth dimension for
    # a true surface plot; its values are already represented by the colored
    # samples in the main axes.
    if surface_plot and background_mode == "None":
        scalar_field = ax.scatter(
            *grid_points.T, c=grid_V, s=8, alpha=0.14,
            cmap=plt.cm.Blues, depthshade=False, zorder=0,
        )
        fig.colorbar(scalar_field, ax=ax, shrink=0.65, pad=0.1).set_label("V(x)")


def plot_rollouts(data_pos, goal, data_vel, w_array, C, ds_inf,
                  box_constraint_margin,
                  background_mode="None",
                  trajectories=None,
                  surface_plot=False):

    data_pos = np.asarray(data_pos)
    w_array = np.asarray(w_array)
    goal = np.asarray(goal)
    if data_pos.ndim != 2 or data_pos.shape[1] not in (2, 3):
        raise ValueError(
            "plot_rollouts supports data with shape (n_samples, 2) "
            "or (n_samples, 3)"
        )
    if w_array.ndim != 2 or w_array.shape[1] != data_pos.shape[1]:
        raise ValueError("w_array must have the same dimensionality as data_pos")
    if goal.ndim != 1 or goal.shape[0] < data_pos.shape[1]:
        raise ValueError("goal must contain one coordinate per data dimension")

    if data_pos.shape[1] == 3:
        _plot_rollouts_3d(
            data_pos, goal, w_array, C, ds_inf, box_constraint_margin,
            background_mode, trajectories, surface_plot,
        )
        return [plt.figure(n) for n in plt.get_fignums()]

    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    # ---- grid ----
    h = 50
    x_min, x_max = np.min(data_pos[:,0]), np.max(data_pos[:,0])
    y_min, y_max = np.min(data_pos[:,1]), np.max(data_pos[:,1])

    margin_x = (x_max - x_min) * box_constraint_margin
    margin_y = (y_max - y_min) * box_constraint_margin

    x_start, x_end = x_min - margin_x, x_max + margin_x
    y_start, y_end = y_min - margin_y, y_max + margin_y

    x_vals = np.linspace(x_start, x_end, h)
    y_vals = np.linspace(y_start, y_end, h)
    xg, yg = np.meshgrid(x_vals, y_vals)
    grid_points = np.stack([xg.ravel(), yg.ravel()], axis=1)

    vectors, aux = ds_inf.velocity_multi(grid_points, return_aux=True)
    grid_V = np.asarray(aux[3])
    grid_dV = np.einsum("ij,ij->i", np.asarray(aux[6]), vectors)

    u = vectors[:,0].reshape(h,h)
    v = vectors[:,1].reshape(h,h)
    grid_V  = grid_V.reshape(h,h)
    grid_dV = grid_dV.reshape(h,h)

    # ---- background ----
    if background_mode == "mag":
        mag = np.sqrt(u**2 + v**2)
        speed_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "velocity_gray_to_blue", ("gray", "#0067C5")
        )
        ax.contourf(
            xg, yg, mag, levels=30, cmap=speed_cmap,
            norm=mpl.colors.PowerNorm(gamma=0.4), alpha=0.7, zorder=0,
        )

    elif background_mode == "V":
        _draw_lyapunov_fill(ax, xg, yg, grid_V)
        ax.contour(
            xg, yg, grid_V, levels=30, colors="gray",
            alpha=0.8, linewidths=1.0, zorder=1,
        )

    elif background_mode == "logV":
        log_V = np.log(grid_V + 1e-8)
        shifted_log_V = log_V - np.nanmin(log_V)
        ax.contourf(
            xg, yg, shifted_log_V, levels=30, cmap=plt.cm.Blues,
            norm=mpl.colors.PowerNorm(gamma=0.35), alpha=0.7, zorder=0,
        )
        ax.contour(
            xg, yg, log_V, levels=30, colors="gray",
            alpha=0.8, linewidths=1.0, zorder=1,
        )

    elif background_mode == "dV":
        limit = max(float(np.nanmax(np.abs(grid_dV))), 1e-12)
        ax.contourf(
            xg, yg, grid_dV, levels=30, cmap="coolwarm",
            norm=mpl.colors.Normalize(vmin=-limit, vmax=limit),
            alpha=0.7, zorder=0,
        )

    elif background_mode == "pos":
        ax.contourf(
            xg, yg, np.sign(grid_dV), levels=[-1.5, -0.5, 0.5, 1.5],
            colors=["green", "white", "red"], alpha=0.5, zorder=0,
        )

    # ---- vector field ----
    stream_color = "white" if background_mode in {"mag", "V", "logV", "dV"} \
        else "gray"
    stream = ax.streamplot(
        xg, yg, u, v, color=stream_color, density=0.7,
        linewidth=1.0, arrowsize=0.9, integration_direction="both",
        zorder=2,
    )
    stream.lines.set_alpha(0.6)
    stream.arrows.set_alpha(0.6)

    # ---- demonstrations ----
    demonstrations = _draw_demonstrations(ax, data_pos, zorder=3)
    demonstrations.set_label("Data")

    # ---- rollouts ----
    if trajectories is not None:
        for i, traj in enumerate(trajectories):
            ax.plot(
                traj[:, 0], traj[:, 1], color="red", alpha=0.5, lw=1.5,
                zorder=6, label="Rollouts" if i == 0 else None,
            )

    # ---- prototypes + topology ----
    _draw_topology_edges(ax, w_array, C)
    prototypes = ax.scatter(
        w_array[:, 0], w_array[:, 1], s=110, c="green",
        edgecolors="black", linewidths=1.2, zorder=10,
    )
    prototypes.set_label("Prototypes")

    # ---- goal ----
    ax.scatter(
        goal[0], goal[1], marker="X", s=180, color="black",
        linewidths=2.5, label="Goal", zorder=11,
    )

    _finish_overview_axes(ax, (x_start, x_end, y_start, y_end))
    ax.legend(loc="upper left", markerscale=0.8)
    fig.tight_layout()

    # ---- optional surface ----
    if surface_plot:
        fig2, ax2 = plt.subplots(subplot_kw={"projection":"3d"})
        ax2.plot_surface(xg, yg, grid_V,
                         cmap=plt.cm.Blues, alpha=0.7)
        ax2.set_title("Lyapunov Surface")

    return [plt.figure(n) for n in plt.get_fignums()]


_SPLINE_COLORS = (
    "#1f4e79",  # muted navy
    "#a7591e",  # muted orange
    "#2f6f4e",  # muted green
    "#5b4b8a",  # muted purple
    "#8a3b3b",  # muted red
    "#6b6b6b",  # dark gray
)

def _draw_demonstrations(ax, data_pos, zorder=1):
    return ax.scatter(
        data_pos[:, 0], data_pos[:, 1],
        s=8, c="black", alpha=0.15, linewidths=0, zorder=zorder,
    )


def _draw_prototypes(ax, prototypes, zorder=10):
    return ax.scatter(
        prototypes[:, 0], prototypes[:, 1],
        s=190, c="green", edgecolors="black", linewidths=1.4,
        zorder=zorder,
    )


def _draw_splines(ax, splines, s_plot, colored=True, linewidth=4.0,
                  zorder=6):
    artists = []
    for index, spline in enumerate(splines):
        points = spline(s_plot)
        color = _SPLINE_COLORS[index % len(_SPLINE_COLORS)] \
            if colored else "black"
        line, = ax.plot(
            points[:, 0], points[:, 1], color=color,
            linewidth=linewidth, zorder=zorder,
        )
        artists.append(line)
    return artists


def _draw_topology_edges(ax, prototypes, connectivity):
    """Draw directed topology with the styling used by the topology panel."""
    import matplotlib.patheffects as pe

    artists = []
    rows, columns = np.where(connectivity > 0)
    for source, target in zip(rows, columns):
        p1, p2 = prototypes[source], prototypes[target]
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-8:
            continue
        annotation = ax.annotate(
            "",
            xy=p2,
            xytext=p1,
            arrowprops=dict(
                arrowstyle="-|>", lw=0.6, color="green",
                mutation_scale=16, shrinkA=11, shrinkB=9,
            ),
            zorder=8,
        )
        annotation.arrow_patch.set_path_effects([
            pe.Stroke(linewidth=4.0, foreground="black"),
            pe.Normal(),
        ])
        artists.append(annotation)
    return artists


def _draw_voronoi(ax, prototypes):
    """Draw the prototype Voronoi diagram used by the topology panel."""
    from scipy.spatial import Voronoi

    voronoi = Voronoi(prototypes)
    center = prototypes.mean(axis=0)
    artists = []
    for (p1, p2), (v1, v2) in zip(
        voronoi.ridge_points, voronoi.ridge_vertices
    ):
        if v1 >= 0 and v2 >= 0:
            start, end = voronoi.vertices[v1], voronoi.vertices[v2]
        else:
            start = voronoi.vertices[v1 if v1 >= 0 else v2]
            tangent = prototypes[p2] - prototypes[p1]
            tangent = tangent / np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = (prototypes[p1] + prototypes[p2]) / 2
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            end = start + direction * 1000
        line, = ax.plot(
            [start[0], end[0]], [start[1], end[1]],
            color="black", linewidth=1.5, zorder=3,
        )
        artists.append(line)
    return artists


def _draw_lyapunov_fill(ax, xx, yy, values):
    from matplotlib.colors import PowerNorm

    # gamma = 0.25 for angle / contractiveness
    return ax.contourf(
        xx, yy, values, levels=30, cmap=plt.cm.Blues,
        norm=PowerNorm(gamma=0.35), alpha=0.7, zorder=0,
    )


def _finish_overview_axes(ax, bounds, title=""):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    x_min, x_max, y_min, y_max = bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.margins(x=0.0, y=0.0)


def _bounds_3d(data_pos, prototypes):
    """Return compact three-dimensional plotting bounds."""
    data_pos = np.asarray(data_pos)
    prototypes = np.asarray(prototypes)
    bounds_data = np.vstack((data_pos, prototypes))
    data_min = np.min(bounds_data, axis=0)
    data_max = np.max(bounds_data, axis=0)
    data_range = data_max - data_min
    fallback_range = max(float(np.max(data_range)), 1.0)
    margin = np.where(data_range > 0.0, 0.1 * data_range, 0.05 * fallback_range)
    starts = data_min - margin
    ends = data_max + margin
    return starts, ends


def _finish_3d_axes(ax, starts, ends):
    """Apply the shared compact 3D axes treatment."""
    ax.set(xlim=(starts[0], ends[0]), ylim=(starts[1], ends[1]),
           zlim=(starts[2], ends[2]))
    ax.set_xlabel("X(m)", labelpad=12)
    ax.set_ylabel("Y(m)", labelpad=12)
    ax.set_zlabel("Z(m)", labelpad=12)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(mpl.ticker.MaxNLocator(
            nbins=4, min_n_ticks=3, prune="both",
            steps=(1, 2, 2.5, 5, 10),
        ))
        axis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    ax.set_box_aspect(ends - starts)
    ax.grid(True)


def _render_results_3d(data_pos, prototypes, connectivity, starts, ends,
                       trajectories=None, goal_prototype=None, show=True):
    """Render data, topology, rollouts, and the goal prototype in 3D."""
    data_pos = np.asarray(data_pos)
    prototypes = np.asarray(prototypes)

    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        *data_pos.T, s=7, c="black", alpha=0.3, linewidths=0,
        depthshade=False, label="Data",
    )

    rows, columns = np.where(np.asarray(connectivity) > 0)
    for source, target in zip(rows, columns):
        start = prototypes[source]
        direction = prototypes[target] - start
        if np.linalg.norm(direction) <= 1e-8:
            continue
        ax.quiver(
            *(start + 0.08 * direction), *(0.84 * direction),
            color="green", linewidth=1.5, arrow_length_ratio=0.12,
        )

    if trajectories is not None:
        for index, trajectory in enumerate(trajectories):
            trajectory = np.asarray(trajectory)
            ax.plot(
                *trajectory[:, :3].T, color="red", alpha=0.5, lw=1.5,
                zorder=6, label="Rollouts" if index == 0 else None,
            )

    ax.scatter(
        *prototypes.T, s=65, c="green", edgecolors="black", linewidths=0.8,
        depthshade=False, label="Prototypes",
    )
    if goal_prototype is not None:
        ax.scatter(
            *np.asarray(goal_prototype).T, s=180, c="black", marker="X",
            linewidths=2.5, depthshade=False, zorder=10, label="Goal",
        )
    _finish_3d_axes(ax, starts, ends)
    ax.legend(loc="upper left", markerscale=0.8)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def _plot_results_3d(topods, trajectories=None):
    """Plot a compact overview of a fitted three-dimensional TopoDS."""
    data_pos = np.asarray(topods.data_pos)
    prototypes = np.asarray(topods.w_array)
    bounds_data = data_pos
    if trajectories is not None and len(trajectories) > 0:
        bounds_data = np.vstack(
            [data_pos]
            + [np.asarray(trajectory) for trajectory in trajectories]
        )
    starts, ends = _bounds_3d(bounds_data, prototypes)

    goal_prototype = None
    goal_indices = np.asarray(
        getattr(topods, "goal_prototype_indices", []), dtype=np.int64
    )
    if goal_indices.size > 0:
        goal_prototype = prototypes[goal_indices]

    return _render_results_3d(
        data_pos, prototypes, topods.C, starts, ends,
        trajectories=trajectories, goal_prototype=goal_prototype,
    )


def plot_results(topods, color_lyapunov_splines=True, trajectories=None):
    """Plot TopoDS diagnostics or a compact three-dimensional overview.

    Parameters
    ----------
    topods : TopoDS
        Fitted model whose diagnostic state is visualized.
    color_lyapunov_splines : bool, optional
        Use the dedicated spline plot's principal color sequence in the
        Lyapunov plot. Enabled by default; pass ``False`` for black splines.
    trajectories : sequence of array-like, optional
        Reproductions to draw in red in the spline-based Lyapunov plot or the
        compact 3D overview.
    """
    self = topods
    if not isinstance(color_lyapunov_splines, (bool, np.bool_)):
        raise TypeError("color_lyapunov_splines must be True or False.")
    if self.w_array.shape[1] == 3:
        return _plot_results_3d(self, trajectories=trajectories)
    if self.w_array.shape[1] != 2:
        return

    # -------------------------------------------------
    # flatten paths (align with splines)
    # -------------------------------------------------
    all_paths = []
    all_is_cycle = []
    for (_, is_cycle), paths in zip(self.components, self.spline_nodes):
        if is_cycle:
            all_paths.append(paths[0])
            all_is_cycle.append(True)
        else:
            all_paths.extend(paths)
            all_is_cycle.extend([False] * len(paths))

    grid_res = 120  # increased for smoother tube contours
    margin = 0.2
    x0, y0 = np.min(self.data_pos, axis=0)
    x1, y1 = np.max(self.data_pos, axis=0)
    dx = (x1 - x0 + 2*margin) / (grid_res - 1)
    dy = (y1 - y0 + 2*margin) / (grid_res - 1)
    xvals = x0 - margin + np.arange(grid_res, dtype=np.float64) * dx
    yvals = y0 - margin + np.arange(grid_res, dtype=np.float64) * dy

    xx, yy = np.meshgrid(xvals, yvals)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # Compute V, boundary-aware direction, grad(V), and lambda_partial
    (proj_points, dist_vals, phase_distances, V_vals, directions, closest_idx,
     grad_V_vals, boundary_lambda) = \
        self.compute_energy_direction_and_aux_multi(
            grid_points, self.spline_nodes, self.components)
    V_grid = V_vals.reshape(xx.shape)

    flow = np.zeros_like(grid_points)

    for p_id, path in enumerate(all_paths):
        modulation = self.modulations[p_id]

        mask = closest_idx == p_id
        if np.sum(mask) == 0:
            continue

        Xp = grid_points[mask]
        g = directions[mask]

        # Spline-coordinate partition and transverse support gate. Reuse the
        # Lyapunov projection and distance already evaluated on this grid.
        Phi, returned_gate, diagnostics = spline_weights_and_gate(
            Xp, p_id, self.tube_model, projected_points=proj_points[mask],
            transverse_distances=dist_vals[mask], gamma_margin=self.gamma_margin,
            gamma_beta=self.gamma_beta)
        support_gate = np.asarray(
            diagnostics.get("support_gate", returned_gate), dtype=np.float64)
        gamma_p = (
            (1.0 - boundary_lambda[mask]) * support_gate
        )[:, None]

        eta_p = np.ones((Xp.shape[0], 1))
        H_p = self._H_gate(
            Xp, path=path, is_cycle=bool(all_is_cycle[p_id]))

        g_norm = np.linalg.norm(g, axis=1, keepdims=True)
        g_hat = g / (g_norm + 1e-8)

        _, P_n, P_t = tangent_projectors(
            Xp, spline_cache=self.spline_cache, spline_id=p_id,
            S_COARSE=self.S_COARSE)
        M_k = block_modulation_matrices(Phi, modulation, P_n, P_t)
        f_learned = -np.einsum("nab,nb->na", M_k, g_hat)

        f_hat = H_p * eta_p * (
            -(1.0 - gamma_p) * self.normed_gradient_scale * g_hat
            + gamma_p * f_learned)

        idx = np.where(mask)[0]
        flow[idx] = f_hat

    U = flow[:, 0].reshape(xx.shape)
    V = flow[:, 1].reshape(xx.shape)

    grid_bounds = (
        float(xvals[0]), float(xvals[-1]),
        float(yvals[0]), float(yvals[-1]),
    )
    s_plot = np.linspace(0.0, 1.0, 300)

    # ============================================================
    # Individual diagnostic plots
    # ============================================================

    sequential_maps = [
        plt.cm.Greens,
        plt.cm.Blues,
        plt.cm.Oranges,
        plt.cm.Purples,
        plt.cm.Reds,
        plt.cm.Greys,
        plt.cm.YlGnBu,
        plt.cm.PuRd,
    ]

    # ------------------------------------------------------------
    # 1) Topology
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    _draw_demonstrations(ax, self.data_pos)
    _draw_topology_edges(ax, self.w_array, self.C)
    _draw_prototypes(ax, self.w_array)
    _draw_voronoi(ax, self.w_array)
    _finish_overview_axes(ax, grid_bounds)
    fig.tight_layout()

    # ------------------------------------------------------------
    # 1.1) Splines
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    _draw_prototypes(ax, self.w_array)
    _draw_splines(ax, self.splines, s_plot)
    _finish_overview_axes(ax, grid_bounds)
    fig.tight_layout()

    # ------------------------------------------------------------
    # 2) Spline-based Lyapunov
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    _draw_lyapunov_fill(ax, xx, yy, V_grid)
    ax.contour(
        xx,
        yy,
        V_grid,
        levels=30,
        colors="gray",
        alpha=0.8,
        linewidths=1.0,
        zorder=1,
    )
    #demonstrations = _draw_demonstrations(ax, self.data_pos, zorder=3)
    #demonstrations.set_alpha(0.55)
    #demonstrations.set_label("Data")
    _draw_splines(
        ax,
        self.splines,
        s_plot,
        colored=bool(color_lyapunov_splines),
        linewidth=4,
    )
    #ax.streamplot(xx, yy, U, V, color="white", density=1.8, zorder=2, integration_direction="both")
    ax.streamplot(xx, yy, U, V, color="white", density=0.8, linewidth=1.25, arrowsize=1.25, zorder=2, integration_direction="both")
    if trajectories is not None:
        for index, trajectory in enumerate(trajectories):
            trajectory = np.asarray(trajectory)
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color="#d62728",
                linewidth=1.5,
                alpha=0.8,
                zorder=8,
                label="Rollouts" if index == 0 else None,
            )
    _draw_prototypes(ax, self.w_array)
    _finish_overview_axes(ax, grid_bounds)
    ax.legend(loc="upper left")
    fig.tight_layout()

    # ------------------------------------------------------------
    # 3) Learned field, colored by velocity magnitude
    # ------------------------------------------------------------
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm
    from scipy.ndimage import gaussian_filter, maximum_filter

    speed = np.hypot(U, V)
    speed_excess = np.nan_to_num(
        np.maximum(speed - 1.0, 0.0),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    # Expand only the color footprint—not the learned field—so neighboring
    # off-spline streamlines reveal the local region of faster motion too.
    color_excess = maximum_filter(speed_excess, size=9, mode="nearest")
    color_excess = gaussian_filter(color_excess, sigma=1.4, mode="nearest")
    color_excess = np.maximum(color_excess, speed_excess)
    color_ceiling = max(float(np.max(color_excess)), 1e-12)
    speed_norm = PowerNorm(
        gamma=0.4,
        vmin=0.0,
        vmax=color_ceiling,
        clip=True,
    )
    speed_cmap = LinearSegmentedColormap.from_list(
        "velocity_gray_to_blue",
        ("gray", "#0067C5"),
    )

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.streamplot(
        xx,
        yy,
        U,
        V,
        color=color_excess,
        cmap=speed_cmap,
        norm=speed_norm,
        density=1.0,
        minlength=0.02,
        maxlength=8.0,
        integration_direction="both",
        zorder=2,
    )
    _draw_splines(
        ax, self.splines, s_plot, colored=False, linewidth=2.6
    )
    _draw_prototypes(ax, self.w_array)
    _finish_overview_axes(ax, grid_bounds)
    fig.tight_layout()
    # ------------------------------------------------------------
    # 4) Per-spline, prototype-stepped active support tube
    # ------------------------------------------------------------
    # Inside the active Gamma >= 0.5 tube, color each grid point by the
    # strongest local prototype activation. Each spline uses its own
    # sequential color family, ordered dark-to-light from source to terminal
    # prototype. The dashed black contour preserves the outer tube boundary.
    from matplotlib.colors import BoundaryNorm, ListedColormap

    n_grid = grid_points.shape[0]
    gamma_threshold = 0.5

    path_region_offsets = []
    region_colors = []
    next_region = 1

    for p_id, path in enumerate(all_paths):
        n_path_prototypes = len(path)
        shades = np.linspace(0.90, 0.35, max(n_path_prototypes, 1))
        colors_p = sequential_maps[p_id % len(sequential_maps)](shades)
        colors_p = colors_p[:n_path_prototypes]
        path_region_offsets.append(next_region)
        region_colors.extend(tuple(color) for color in colors_p)
        next_region += n_path_prototypes

    active_gamma = np.zeros(n_grid, dtype=np.float64)
    active_region = np.zeros(n_grid, dtype=np.int64)

    for p_id, path in enumerate(all_paths):
        mask = closest_idx == p_id
        if not np.any(mask):
            continue

        phi_p, returned_gate, diagnostics = spline_weights_and_gate(
            grid_points[mask], p_id, self.tube_model,
            projected_points=proj_points[mask],
            transverse_distances=dist_vals[mask],
            gamma_margin=self.gamma_margin,
            gamma_beta=self.gamma_beta)
        support_gate = np.asarray(
            diagnostics.get("support_gate", returned_gate), dtype=np.float64)
        gamma_p = (1.0 - boundary_lambda[mask]) * support_gate

        local_idx = np.argmax(phi_p, axis=1)

        active_gamma[mask] = gamma_p
        active_region[mask] = path_region_offsets[p_id] + local_idx

    active_gamma_grid = active_gamma.reshape(xx.shape)
    tube_map = np.where(active_gamma >= gamma_threshold, active_region, 0).reshape(xx.shape)

    tube_cmap = ListedColormap(
        [(1.0, 1.0, 1.0, 1.0)]
        + region_colors
    )
    n_regions = len(region_colors)
    tube_levels = np.arange(-0.5, n_regions + 1.5, 1.0)
    tube_norm = BoundaryNorm(tube_levels, tube_cmap.N)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.contourf(xx, yy, tube_map, levels=tube_levels, cmap=tube_cmap, norm=tube_norm, alpha=0.70,
                antialiased=False, zorder=0)

    # Preserve the outer Gamma = 0.5 tube boundary.
    if (
        np.nanmin(active_gamma_grid)
        <= gamma_threshold
        <= np.nanmax(active_gamma_grid)
    ):
        ax.contour(xx, yy, active_gamma_grid, levels=[gamma_threshold], colors="black", linewidths=2.0,
                   linestyles="--", zorder=5)

    ax.streamplot(xx, yy, U, V, color="gray", density=0.8, linewidth=1.25, arrowsize=1.25, zorder=2)

    _draw_splines(
        ax, self.splines, s_plot, colored=False, linewidth=2.6
    )
    _draw_prototypes(ax, self.w_array)
    _finish_overview_axes(ax, grid_bounds)
    fig.tight_layout()
    plt.show()

    return
