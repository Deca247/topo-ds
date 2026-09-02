"""Animation helpers for visualizing the stages of TopoDS training.

"""


_CANVAS = "#FFFFFF"
_PANEL = "#FFFFFF"
_INK = "#1A1A1A"
_MUTED = "#666666"
_BORDER = "#D0D0D0"
_DATA = "#000000"
_PROTOTYPE = "#008000"
_PROTOTYPE_EDGE = "#000000"
_PROTOTYPE_SIZE = 190
_PROTOTYPE_LINEWIDTH = 1.4
_DATA_SIZE = 8
_DATA_ALPHA = 0.15
_CANDIDATE = "#9A9A9A"
_ACCENT = "#008000"
_ACTIVE = "#2D9CDB"
_FOCUS = "#F4A261"
_REMOVE = "#E76F51"
_WHITE = "#FFFFFF"
_SERIES_COLORS = (
    "#2F6F4E",
    "#1F4E79",
    "#A7591E",
    "#5B4B8A",
    "#8A3B3B",
    "#6B6B6B",
)


def _create_animation_figure(plt, window_title, stage_label):
    """Create the shared canvas and stage overline used by all animations."""
    fig, ax = plt.subplots(figsize=(7.2, 7.2), facecolor=_CANVAS)
    ax.set_facecolor(_PANEL)
    fig.text(
        0.065,
        0.962,
        stage_label.upper(),
        color=_MUTED,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="top",
    )
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None:
        manager.set_window_title(window_title)
    return fig, ax


def _play_frames(fig, update, frame_count, interval, plt):
    """Render every frame before returning to the TopoDS pipeline."""
    plt.show(block=False)

    if getattr(fig.canvas, "required_interactive_framework", None) is None:
        update(frame_count - 1)
        fig.canvas.draw()
        plt.close(fig)
        return

    frame_seconds = max(float(interval) / 1000.0, 0.001)
    for frame in range(frame_count):
        if not plt.fignum_exists(fig.number):
            return
        update(frame)
        fig.canvas.draw()
        fig.canvas.start_event_loop(frame_seconds)

    plt.close(fig)


def _style_animation_axes(ax, data_xy, np):
    """Apply the shared frameless, equal-aspect animation layout."""
    span = max(np.ptp(data_xy[:, 0]), np.ptp(data_xy[:, 1]), 1e-8)
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(
        np.min(data_xy[:, 0]) - 0.1 * span,
        np.max(data_xy[:, 0]) + 0.1 * span,
    )
    ax.set_ylim(
        np.min(data_xy[:, 1]) - 0.1 * span,
        np.max(data_xy[:, 1]) + 0.1 * span,
    )


def _set_animation_title(ax, title):
    ax.set_title(
        title,
        loc="left",
        pad=18,
        color=_INK,
        fontsize=15,
        fontweight="semibold",
    )


def _style_legend(ax, handles=None, labels=None):
    """Draw a quiet, consistent legend that does not compete with the data."""
    legend = ax.legend(
        handles=handles,
        labels=labels,
        loc="upper right",
        frameon=True,
        facecolor=_WHITE,
        edgecolor=_BORDER,
        framealpha=0.94,
        borderpad=0.7,
        labelspacing=0.55,
        handletextpad=0.55,
        fontsize=9,
        markerscale=0.85,
    )
    legend.get_frame().set_linewidth(0.8)
    for text in legend.get_texts():
        text.set_color(_INK)
    return legend


def _annotate_prototypes(ax, prototype_xy):
    """Place unobtrusive node identifiers inside the prototype markers."""
    for index, point in enumerate(prototype_xy):
        ax.annotate(
            str(index),
            point,
            ha="center",
            va="center",
            color=_WHITE,
            fontsize=7.2,
            fontweight="bold",
            zorder=8,
        )


def animate_neural_gas(data_pos, prototype_history, interval=50, repeat=False, show=True):
    """Animate Neural Gas prototype updates.

    Parameters
    ----------
    data_pos : array-like
        Demonstration positions used to train the Neural Gas model.
    prototype_history : sequence of array-like
        Prototype positions captured over the training iterations.
    interval : int, optional
        Delay between frames in milliseconds.
    repeat : bool, optional
        Whether playback restarts after the final frame.
    show : bool, optional
        Whether to display the figure immediately.

    Returns
    -------
    matplotlib.animation.FuncAnimation or None
        An animation object when ``show=False``; otherwise playback completes
        synchronously and the function returns ``None``.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    data_pos = np.asarray(data_pos)
    history = [np.asarray(prototypes) for prototypes in prototype_history]
    if data_pos.ndim != 2 or data_pos.shape[0] == 0:
        raise ValueError("data_pos must be a non-empty two-dimensional array.")
    if not history:
        raise ValueError("prototype_history must contain at least one frame.")

    def coordinates(values):
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("Each prototype frame must be a two-dimensional array.")
        if values.shape[1] == 1:
            return np.column_stack(
                (values[:, 0], np.zeros(values.shape[0], dtype=values.dtype))
            )
        return values[:, :2]

    n_display = min(data_pos.shape[0], 5000)
    display_idx = np.linspace(0, data_pos.shape[0] - 1, n_display, dtype=np.int64)
    data_xy = coordinates(data_pos[display_idx])
    history_xy = [coordinates(prototypes) for prototypes in history]

    fig, ax = _create_animation_figure(
        plt, "Neural Gas training", "TopoDS · Stage 1 of 4"
    )

    ax.scatter(
        data_xy[:, 0],
        data_xy[:, 1],
        s=_DATA_SIZE,
        alpha=_DATA_ALPHA,
        color=_DATA,
        linewidths=0,
        label="Demonstrations",
        zorder=1,
    )
    prototype_artist = ax.scatter(
        history_xy[0][:, 0],
        history_xy[0][:, 1],
        s=_PROTOTYPE_SIZE,
        marker="o",
        edgecolors=_PROTOTYPE_EDGE,
        color=_PROTOTYPE,
        linewidths=_PROTOTYPE_LINEWIDTH,
        label="Prototypes",
        zorder=3,
    )
    _style_legend(ax)
    _style_animation_axes(ax, data_xy, np)

    playback = {"active": False, "complete": False}

    def update(frame):
        prototype_artist.set_offsets(history_xy[frame])
        if frame == len(history_xy) - 1:
            _set_animation_title(ax, "Neural Gas · pruning")
            if playback["active"]:
                playback["complete"] = True
                fig.canvas.stop_event_loop()
        else:
            _set_animation_title(ax, f"Neural Gas · epoch {frame}")
        return prototype_artist,

    update(0)
    fig.tight_layout(rect=(0.035, 0.025, 0.98, 0.94))
    animation = FuncAnimation(
        fig,
        update,
        frames=len(history_xy),
        interval=interval,
        repeat=repeat,
        blit=False,
    )

    if not show:
        return animation

    plt.show(block=False)
    if getattr(fig.canvas, "required_interactive_framework", None) is None:
        update(len(history_xy) - 1)
        fig.canvas.draw()
        plt.close(fig)
        return None

    def stop_on_close(_event):
        fig.canvas.stop_event_loop()

    fig.canvas.mpl_connect("close_event", stop_on_close)
    playback["active"] = True
    fig.canvas.draw()
    if not playback["complete"] and plt.fignum_exists(fig.number):
        fig.canvas.start_event_loop(0)
    if plt.fignum_exists(fig.number):
        plt.close(fig)
    return None


def animate_topology(data_pos, prototypes, topology_history, interval=100, repeat=False, show=True):
    """Animate construction and orientation of the prototype topology.

    Parameters
    ----------
    data_pos : array-like
        Demonstration positions and their prototype assignments.
    prototypes : array-like
        Final prototype positions.
    topology_history : sequence of tuple
        Compact events recorded during construction. The first event contains
        all candidate edges; later events remove or orient one edge at a time.
    interval : int, optional
        Delay between frames in milliseconds.
    repeat : bool, optional
        Whether playback restarts after the final frame.
    show : bool, optional
        Whether to display the figure immediately.

    Returns
    -------
    matplotlib.animation.FuncAnimation or None
        An animation object when ``show=False``; otherwise playback completes
        synchronously and the function returns ``None``.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.animation import FuncAnimation
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    data_pos = np.asarray(data_pos)
    prototypes = np.asarray(prototypes)
    events = list(topology_history)
    if data_pos.ndim != 2 or data_pos.shape[0] == 0:
        raise ValueError("data_pos must be a non-empty two-dimensional array.")
    if prototypes.ndim != 2 or prototypes.shape[0] == 0:
        raise ValueError("prototypes must be a non-empty two-dimensional array.")
    if not events or events[0][0] != "candidates":
        raise ValueError("topology_history must start with a candidates event.")

    def coordinates(values):
        if values.shape[1] == 1:
            return np.column_stack(
                (values[:, 0], np.zeros(values.shape[0], dtype=values.dtype))
            )
        return values[:, :2]

    data_xy = coordinates(data_pos)
    prototype_xy = coordinates(prototypes)

    candidate_edges = {
        tuple(sorted((int(u), int(v)))) for u, v in events[0][1]
    }
    directed_edges = []
    frames = [
        (tuple(sorted(candidate_edges)), tuple(directed_edges), events[0])
    ]

    for event in events[1:]:
        event_type, u, v, _ = event
        candidate_edges.discard(tuple(sorted((int(u), int(v)))))
        if event_type == "orient":
            directed_edges.append((int(u), int(v)))
        frames.append(
            (tuple(sorted(candidate_edges)), tuple(directed_edges), event)
        )

    frames.append(
        (tuple(), tuple(directed_edges), ("complete",))
    )

    fig, ax = _create_animation_figure(
        plt, "TopoDS topology construction", "TopoDS · Stage 2 of 4"
    )

    n_display = min(data_xy.shape[0], 5000)
    display_idx = np.linspace(0, data_xy.shape[0] - 1, n_display, dtype=np.int64)
    data_artist = ax.scatter(
        data_xy[display_idx, 0],
        data_xy[display_idx, 1],
        s=_DATA_SIZE,
        alpha=_DATA_ALPHA,
        color=_DATA,
        linewidths=0,
        label="Demonstrations",
        zorder=1,
    )
    prototype_artist = ax.scatter(
        prototype_xy[:, 0],
        prototype_xy[:, 1],
        s=_PROTOTYPE_SIZE,
        marker="o",
        edgecolors=_PROTOTYPE_EDGE,
        color=_PROTOTYPE,
        linewidths=_PROTOTYPE_LINEWIDTH,
        label="Prototypes",
        zorder=6,
    )
    _annotate_prototypes(ax, prototype_xy)

    edge_key = Line2D(
        [0], [0], color=_ACCENT, linewidth=1.0, label="Oriented edge"
    )
    edge_key.set_path_effects(
        [pe.Stroke(linewidth=3.2, foreground=_PROTOTYPE_EDGE), pe.Normal()]
    )
    _style_legend(ax, handles=[data_artist, prototype_artist, edge_key])
    _style_animation_axes(ax, data_xy, np)

    edge_artists = []

    def update(frame_index):
        for artist in edge_artists:
            artist.remove()
        edge_artists.clear()

        candidates, directed, event = frames[frame_index]
        for u, v in candidates:
            line, = ax.plot(
                prototype_xy[[u, v], 0],
                prototype_xy[[u, v], 1],
                color=_CANDIDATE,
                linestyle=(0, (3, 3)),
                linewidth=1.35,
                solid_capstyle="round",
                zorder=2,
            )
            edge_artists.append(line)

        for u, v in directed:
            arrow = FancyArrowPatch(
                prototype_xy[u],
                prototype_xy[v],
                arrowstyle="-|>",
                mutation_scale=16,
                color=_ACCENT,
                linewidth=0.8,
                shrinkA=10,
                shrinkB=10,
                zorder=4,
            )
            arrow.set_path_effects(
                [
                    pe.Stroke(linewidth=3.8, foreground=_PROTOTYPE_EDGE),
                    pe.Normal(),
                ]
            )
            ax.add_patch(arrow)
            edge_artists.append(arrow)

        event_type = event[0]
        if event_type == "candidates":
            _set_animation_title(ax, "Topology · candidate edges")
        elif event_type == "remove":
            _, u, v, support = event
            line, = ax.plot(
                prototype_xy[[u, v], 0],
                prototype_xy[[u, v], 1],
                color=_REMOVE,
                linewidth=3.0,
                solid_capstyle="round",
                zorder=5,
            )
            line.set_path_effects(
                [
                    pe.Stroke(linewidth=4.5, foreground=_PROTOTYPE_EDGE),
                    pe.Normal(),
                ]
            )
            edge_artists.append(line)
            _set_animation_title(
                ax, f"Topology · removed {u}—{v}  ·  support {support:.2f}"
            )
        elif event_type == "orient":
            _, u, v, support = event
            _set_animation_title(
                ax, f"Topology · oriented {u}→{v}  ·  support {support:.2f}"
            )
        else:
            _set_animation_title(ax, "Topology · complete")

        return tuple(edge_artists)

    update(0)
    fig.tight_layout(rect=(0.035, 0.025, 0.98, 0.94))
    if show:
        _play_frames(fig, update, len(frames), interval, plt)
        return None

    return FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval,
        repeat=repeat,
        blit=False,
    )


def animate_primitives_and_splines(data_pos, prototypes, connectivity, components, spline_nodes, splines,
                                   interval=100, spline_frames=15, repeat=False, show=True):
    """Animate the selected path/cycle primitives and their fitted splines.

    Parameters
    ----------
    data_pos : array-like
        Demonstration positions.
    prototypes : array-like
        Prototype positions used by the selected primitives.
    connectivity : array-like
        Directed prototype connectivity matrix.
    components, spline_nodes : sequence
        Component metadata and selected prototype paths returned by TopoDS.
    splines : sequence
        Fitted spline objects aligned with the flattened selected primitives.
    interval : int, optional
        Delay between frames in milliseconds.
    spline_frames : int, optional
        Number of progressive drawing frames per fitted spline.
    repeat : bool, optional
        Whether playback restarts after the final frame.
    show : bool, optional
        Whether to display the figure immediately.

    Returns
    -------
    matplotlib.animation.FuncAnimation or None
        An animation object when ``show=False``; otherwise playback completes
        synchronously and the function returns ``None``.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.animation import FuncAnimation
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    data_pos = np.asarray(data_pos)
    prototypes = np.asarray(prototypes)
    connectivity = np.asarray(connectivity)
    spline_frames = max(int(spline_frames), 1)

    if data_pos.ndim != 2 or data_pos.shape[0] == 0:
        raise ValueError("data_pos must be a non-empty two-dimensional array.")
    if prototypes.ndim != 2 or prototypes.shape[0] == 0:
        raise ValueError("prototypes must be a non-empty two-dimensional array.")
    if connectivity.shape != (len(prototypes), len(prototypes)):
        raise ValueError("connectivity must be square and aligned with prototypes.")

    def coordinates(values):
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("Animation coordinates must be two-dimensional arrays.")
        if values.shape[1] == 1:
            return np.column_stack((values[:, 0], np.zeros(values.shape[0], dtype=values.dtype)))
        return values[:, :2]

    data_xy = coordinates(data_pos)
    prototype_xy = coordinates(prototypes)

    primitives = []
    for (_, is_cycle), paths in zip(components, spline_nodes):
        selected_paths = paths[:1] if is_cycle else paths
        for path in selected_paths:
            primitives.append({"nodes": np.asarray(path, dtype=np.int64), "is_cycle": bool(is_cycle)})

    if len(primitives) != len(splines):
        raise ValueError("Selected primitives and fitted splines are not aligned.")

    s_plot = np.linspace(0.0, 1.0, 120)
    spline_xy = [coordinates(spline(s_plot)) for spline in splines]
    frames = [("start", None, 0)]
    for primitive_index in range(len(primitives)):
        frames.append(("primitive", primitive_index, 0))
        frames.extend(("spline", primitive_index, step) for step in range(1, spline_frames + 1))
    frames.append(("complete", None, 0))

    fig, ax = _create_animation_figure(
        plt, "TopoDS primitive and spline fitting", "TopoDS · Stage 3 of 4"
    )

    n_display = min(data_xy.shape[0], 5000)
    display_idx = np.linspace(0, data_xy.shape[0] - 1, n_display, dtype=np.int64)
    data_artist = ax.scatter(
        data_xy[display_idx, 0],
        data_xy[display_idx, 1],
        s=_DATA_SIZE,
        alpha=_DATA_ALPHA,
        color=_DATA,
        linewidths=0,
        label="Demonstrations",
        zorder=1,
    )

    rows, columns = np.where(connectivity > 0)
    for source, target in zip(rows, columns):
        arrow = FancyArrowPatch(
            prototype_xy[source],
            prototype_xy[target],
            arrowstyle="-|>",
            mutation_scale=11,
            color=_ACCENT,
            linewidth=0.65,
            alpha=0.62,
            shrinkA=10,
            shrinkB=10,
            zorder=2,
        )
        arrow.set_path_effects(
            [
                pe.Stroke(linewidth=3.0, foreground=_PROTOTYPE_EDGE),
                pe.Normal(),
            ]
        )
        ax.add_patch(arrow)

    prototype_artist = ax.scatter(
        prototype_xy[:, 0],
        prototype_xy[:, 1],
        s=_PROTOTYPE_SIZE,
        marker="o",
        edgecolors=_PROTOTYPE_EDGE,
        color=_PROTOTYPE,
        linewidths=_PROTOTYPE_LINEWIDTH,
        label="Prototypes",
        zorder=7,
    )
    _annotate_prototypes(ax, prototype_xy)

    spline_key = Line2D(
        [0], [0], color=_SERIES_COLORS[0], linewidth=3, label="Fitted primitive"
    )
    _style_legend(ax, handles=[data_artist, prototype_artist, spline_key])
    _style_animation_axes(ax, data_xy, np)
    dynamic_artists = []

    def draw_primitive(primitive_index):
        primitive = primitives[primitive_index]
        nodes = primitive["nodes"]
        path_xy = prototype_xy[nodes]
        if primitive["is_cycle"] and len(path_xy) > 0:
            path_xy = np.vstack((path_xy, path_xy[0]))
        line, = ax.plot(
            path_xy[:, 0],
            path_xy[:, 1],
            color=_SERIES_COLORS[primitive_index % len(_SERIES_COLORS)],
            linestyle=(0, (3, 2)),
            linewidth=2.2,
            alpha=0.9,
            dash_capstyle="round",
            zorder=4,
        )
        dynamic_artists.append(line)

    def draw_spline(primitive_index, fraction=1.0):
        points = spline_xy[primitive_index]
        endpoint = max(2, int(np.ceil(fraction * len(points))))
        endpoint = min(endpoint, len(points))
        line, = ax.plot(
            points[:endpoint, 0],
            points[:endpoint, 1],
            color=_SERIES_COLORS[primitive_index % len(_SERIES_COLORS)],
            linewidth=4.0,
            solid_capstyle="round",
            zorder=5,
        )
        dynamic_artists.append(line)

    def update(frame_index):
        for artist in dynamic_artists:
            artist.remove()
        dynamic_artists.clear()

        phase, primitive_index, step = frames[frame_index]
        completed_count = len(primitives) if phase == "complete" else 0 if primitive_index is None else primitive_index
        for completed_index in range(completed_count):
            draw_spline(completed_index)

        if phase == "start":
            _set_animation_title(ax, "Primitives · directed topology")
        elif phase == "primitive":
            draw_primitive(primitive_index)
            kind = "cycle" if primitives[primitive_index]["is_cycle"] else "path"
            _set_animation_title(
                ax, f"Primitives · selected {kind} {primitive_index + 1}"
            )
        elif phase == "spline":
            draw_primitive(primitive_index)
            draw_spline(primitive_index, step / spline_frames)
            percentage = round(100 * step / spline_frames)
            _set_animation_title(
                ax,
                f"Spline fit · primitive {primitive_index + 1}  ·  {percentage}%",
            )
        else:
            _set_animation_title(ax, "Primitives & splines · complete")

        return tuple(dynamic_artists)

    update(0)
    fig.tight_layout(rect=(0.035, 0.025, 0.98, 0.94))
    if show:
        _play_frames(fig, update, len(frames), interval, plt)
        return None

    return FuncAnimation(fig, update, frames=len(frames), interval=interval, repeat=repeat, blit=False)


def animate_modulation_learning(topods, interval=35, matrix_frames=6, grid_resolution=26,
                                repeat=False, show=True):
    """Incrementally reveal the learned modulation vector field.

    Parameters
    ----------
    topods : TopoDS
        Fitted model containing splines, tube geometry, and modulations.
    interval : int, optional
        Delay between frames in milliseconds.
    matrix_frames : int, optional
        Number of blending frames used to add each local tangent-block contribution.
    grid_resolution : int, optional
        Number of vector-field samples along each displayed dimension.
    repeat : bool, optional
        Whether playback restarts after the final frame.
    show : bool, optional
        Whether to display the figure immediately.

    Returns
    -------
    matplotlib.animation.FuncAnimation or None
        An animation object when ``show=False``; otherwise playback completes
        synchronously and the function returns ``None``.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from .spline_helpers import spline_weights_and_gate
    from .lyapunov import block_modulation_matrices, tangent_projectors

    data_pos = np.asarray(topods.data_pos, dtype=np.float64)
    prototypes = np.asarray(topods.w_array, dtype=np.float64)
    matrix_frames = max(int(matrix_frames), 1)
    grid_resolution = max(int(grid_resolution), 2)

    if data_pos.ndim != 2 or data_pos.shape[0] == 0:
        raise ValueError("data_pos must be a non-empty two-dimensional array.")
    if prototypes.ndim != 2 or prototypes.shape[0] == 0:
        raise ValueError("prototypes must be a non-empty two-dimensional array.")

    def coordinates(values):
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("Animation coordinates must be two-dimensional arrays.")
        if values.shape[1] == 1:
            return np.column_stack((values[:, 0], np.zeros(values.shape[0], dtype=values.dtype)))
        return values[:, :2]

    data_xy = coordinates(data_pos)
    prototype_xy = coordinates(prototypes)

    paths = []
    path_is_cycle = []
    for (_, is_cycle), component_paths in zip(topods.components, topods.spline_nodes):
        selected_paths = component_paths[:1] if is_cycle else component_paths
        for path in selected_paths:
            paths.append(np.asarray(path, dtype=np.int64))
            path_is_cycle.append(bool(is_cycle))

    if len(paths) != len(topods.splines):
        raise ValueError("Selected paths and fitted splines are not aligned.")

    span = max(np.ptp(data_xy[:, 0]), np.ptp(data_xy[:, 1]), 1e-8)
    x_values = np.linspace(np.min(data_xy[:, 0]) - 0.1 * span,
                           np.max(data_xy[:, 0]) + 0.1 * span, grid_resolution)
    if data_pos.shape[1] == 1:
        y_values = np.array([0.0])
    else:
        y_values = np.linspace(np.min(data_xy[:, 1]) - 0.1 * span,
                               np.max(data_xy[:, 1]) + 0.1 * span, grid_resolution)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid_points = np.tile(np.mean(data_pos, axis=0), (x_grid.size, 1))
    grid_points[:, 0] = x_grid.ravel()
    if data_pos.shape[1] > 1:
        grid_points[:, 1] = y_grid.ravel()

    (projected, distances, _, _, directions, closest_path, _,
     boundary_lambda) = topods.compute_energy_direction_and_aux_multi(
        grid_points, topods.spline_nodes, topods.components)
    directions = np.asarray(directions, dtype=np.float64)
    direction_norm = np.linalg.norm(directions, axis=1, keepdims=True)
    direction_hat = directions / np.maximum(direction_norm, 1e-12)

    baseline = np.zeros_like(grid_points, dtype=np.float64)
    entries = []
    for path_index, path in enumerate(paths):
        modulation = topods.modulations[path_index]
        mask = closest_path == path_index
        if not np.any(mask):
            continue

        path_points = grid_points[mask]
        path_directions = direction_hat[mask]
        phi, returned_gate, diagnostics = spline_weights_and_gate(
            path_points, path_index, topods.tube_model,
            projected_points=np.asarray(projected[mask], dtype=np.float64),
            transverse_distances=np.asarray(distances[mask], dtype=np.float64),
            gamma_margin=topods.gamma_margin, gamma_beta=topods.gamma_beta)
        support_gate = np.asarray(
            diagnostics.get("support_gate", returned_gate), dtype=np.float64)
        gamma = (
            (1.0 - boundary_lambda[mask]) * support_gate
        )[:, None]

        H = topods._H_gate(path_points, path=path, is_cycle=path_is_cycle[path_index])
        baseline[mask] = H * (
            -(1.0 - gamma) * topods.normed_gradient_scale * path_directions
        )

        _, P_n, P_t = tangent_projectors(
            path_points,
            spline_cache=topods.spline_cache,
            spline_id=path_index,
            S_COARSE=topods.S_COARSE,
        )
        if phi.shape[1] != len(path):
            raise ValueError(
                f"Path {path_index} has {len(path)} prototypes but "
                f"{phi.shape[1]} spline weights."
            )
        for local_index, prototype_index in enumerate(path):
            local_phi = np.zeros_like(phi)
            local_phi[:, local_index] = phi[:, local_index]
            M_ki = block_modulation_matrices(
                local_phi, modulation, P_n, P_t
            )
            local_field = -np.einsum(
                "nij,nj->ni", M_ki, path_directions, optimize=True
            )
            contribution = np.zeros_like(grid_points, dtype=np.float64)
            contribution[mask] = H * gamma * local_field
            entries.append({"path_index": path_index, "prototype_index": int(prototype_index),
                            "contribution": contribution,
                            "active_mask": np.linalg.norm(contribution, axis=1) > 1e-12})

    frames = [("start", None, baseline.copy())]
    accumulated = baseline.copy()
    for entry_index in range(len(entries)):
        contribution = entries[entry_index]["contribution"]
        frames.extend(("matrix", entry_index, accumulated + (step / matrix_frames) * contribution)
                      for step in range(1, matrix_frames + 1))
        accumulated = accumulated + contribution
    frames.append(("complete", None, accumulated))

    fig, ax = _create_animation_figure(
        plt, "TopoDS modulation learning", "TopoDS · Stage 4 of 4"
    )

    n_display = min(data_xy.shape[0], 5000)
    display_idx = np.linspace(0, data_xy.shape[0] - 1, n_display, dtype=np.int64)
    data_artist = ax.scatter(
        data_xy[display_idx, 0],
        data_xy[display_idx, 1],
        s=_DATA_SIZE,
        alpha=_DATA_ALPHA,
        color=_DATA,
        linewidths=0,
        label="Demonstrations",
        zorder=1,
    )

    s_plot = np.linspace(0.0, 1.0, 120)
    for path_index, spline in enumerate(topods.splines):
        points = coordinates(spline(s_plot))
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=_SERIES_COLORS[path_index % len(_SERIES_COLORS)],
            linewidth=3.0,
            alpha=0.9,
            solid_capstyle="round",
            zorder=2,
        )

    prototype_artist = ax.scatter(
        prototype_xy[:, 0],
        prototype_xy[:, 1],
        s=_PROTOTYPE_SIZE,
        marker="o",
        edgecolors=_PROTOTYPE_EDGE,
        color=_PROTOTYPE,
        linewidths=_PROTOTYPE_LINEWIDTH,
        label="Prototypes",
        zorder=6,
    )
    _annotate_prototypes(ax, prototype_xy)
    field_key = Line2D(
        [0],
        [0],
        color=_ACCENT,
        marker=r"$\rightarrow$",
        linestyle="none",
        markersize=13,
        label="Learned field",
    )
    _style_legend(ax, handles=[data_artist, prototype_artist, field_key])
    _style_animation_axes(ax, data_xy, np)

    def display_vectors(field):
        field_xy = coordinates(field)
        norms = np.linalg.norm(field_xy, axis=1, keepdims=True)
        return field_xy / np.maximum(norms, 1e-12)

    initial_vectors = display_vectors(frames[0][2])
    vector_artist = ax.quiver(
        x_grid,
        y_grid,
        initial_vectors[:, 0].reshape(x_grid.shape),
        initial_vectors[:, 1].reshape(y_grid.shape),
        angles="xy",
        scale_units="xy",
        scale=20.0 / span,
        color=_CANDIDATE,
        width=0.0037,
        headwidth=3.5,
        headlength=4.5,
        headaxislength=4.0,
        alpha=0.9,
        zorder=3,
    )

    def update(frame_index):
        phase, entry_index, field = frames[frame_index]
        vectors = display_vectors(field)
        vector_artist.set_UVC(
            vectors[:, 0].reshape(x_grid.shape),
            vectors[:, 1].reshape(y_grid.shape),
        )

        arrow_colors = np.tile(to_rgba(_CANDIDATE), (len(grid_points), 1))
        prototype_edges = np.tile(
            to_rgba(_PROTOTYPE_EDGE),
            (len(prototypes), 1),
        )
        prototype_widths = np.full(
            len(prototypes), _PROTOTYPE_LINEWIDTH
        )

        if phase == "start":
            _set_animation_title(ax, "Modulation · baseline field")
        elif phase == "matrix":
            entry = entries[entry_index]
            arrow_colors[entry["active_mask"]] = to_rgba(_ACTIVE)
            prototype_index = entry["prototype_index"]
            prototype_edges[prototype_index] = to_rgba(_FOCUS)
            prototype_widths[prototype_index] = 3.6
            _set_animation_title(
                ax,
                "Modulation · "
                f"path {entry['path_index'] + 1}  ·  prototype {entry['prototype_index']}",
            )
        else:
            arrow_colors[:] = to_rgba(_ACCENT)
            _set_animation_title(ax, "Modulation · learned field")

        vector_artist.set_color(arrow_colors)
        prototype_artist.set_edgecolors(prototype_edges)
        prototype_artist.set_linewidths(prototype_widths)
        return vector_artist, prototype_artist

    update(0)
    fig.tight_layout(rect=(0.035, 0.025, 0.98, 0.94))
    if show:
        _play_frames(fig, update, len(frames), interval, plt)
        return None

    return FuncAnimation(fig, update, frames=len(frames), interval=interval, repeat=repeat, blit=False)
