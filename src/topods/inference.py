import time

import numpy as np
from .spline_helpers import spline_weights_and_gate

from .lyapunov import (
    LyapunovField,
    block_modulation_matrices,
    tangent_projectors,
)


class DSInference:
    """Real-time inference for learned tangent-block dynamics.

    Set ``normalize_velocity=True`` to return constant-speed vectors whose
    norm is ``normed_gradient_scale``. Zero velocity remains zero.
    Set ``live_visualization=True`` to update the model-only state-space view
    automatically from scalar inference calls.
    """

    def __init__(self, w_array, splines, spline_nodes, components,
                 modulations, tube_model,
                 integration_dt=0.01, normed_gradient_scale=0.0001,
                 alpha=0.8, gamma_margin=1.0, gamma_beta=10.0,
                 enable_boundary_progress_gate=True,
                 boundary_progress_gate_scale=None,
                 boundary_progress_gate_eps=1e-12,
                 enable_H_gate=True,
                 beta_H=1000.0,
                 goals=None,
                 normalize_velocity=False,
                 live_visualization=False):
        self.w_array = np.asarray(w_array, dtype=np.float64)
        self.modulations = modulations
        self.tube_model = tube_model
        self.integration_dt = float(integration_dt)
        self.normed_gradient_scale = float(normed_gradient_scale)
        self.normalize_velocity = bool(normalize_velocity)
        self.gamma_margin = float(gamma_margin)
        self.gamma_beta = float(gamma_beta)
        self.dim = self.w_array.shape[1]

        self.lyapunov = LyapunovField(
            w_array=self.w_array,
            splines=splines,
            spline_nodes=spline_nodes,
            components=components,
            tube_model=tube_model,
            alpha=alpha,
            enable_boundary_progress_gate=enable_boundary_progress_gate,
            boundary_progress_gate_scale=boundary_progress_gate_scale,
            boundary_progress_gate_eps=boundary_progress_gate_eps,
            enable_H_gate=enable_H_gate,
            beta_H=beta_H,
            goals=goals,
        )
        self.live_visualization = bool(live_visualization)
        self._live_plot = None

    @staticmethod
    def _validate_visualization_axes(projection_axes, axis_count):
        projection_axes = tuple(projection_axes)
        if len(projection_axes) != 2 or any(
            isinstance(axis, (bool, np.bool_))
            or not isinstance(axis, (int, np.integer))
            for axis in projection_axes
        ):
            raise ValueError(
                "projection_axes must contain exactly two integer indices"
            )
        if projection_axes[0] == projection_axes[1]:
            raise ValueError(
                "projection_axes must select two different components"
            )
        if any(axis < 0 or axis >= axis_count for axis in projection_axes):
            raise ValueError(
                f"projection_axes indices must be between 0 and "
                f"{axis_count - 1}"
            )
        return projection_axes

    @staticmethod
    def _visualization_projection(
        points,
        projection_axes=(0, 1),
        dimensionality_reduction="pca",
        random_projection_seed=0,
    ):
        dim = points.shape[1]
        if dim < 2:
            raise ValueError("Live visualization requires at least two dimensions")

        if dimensionality_reduction not in {
            "pca",
            "random_projection",
            "coordinates",
        }:
            raise ValueError(
                "dimensionality_reduction must be 'pca', "
                "'random_projection', or 'coordinates'"
            )

        projection_axes = DSInference._validate_visualization_axes(
            projection_axes,
            dim,
        )
        selected_axes = list(projection_axes)
        if dimensionality_reduction == "coordinates":
            # Preserve the original values on the selected axes and hold all
            # hidden dimensions at their data mean when lifting the 2D grid.
            center = np.mean(points, axis=0)
            center[selected_axes] = 0.0
            return center, np.eye(dim)[selected_axes]

        if dimensionality_reduction == "random_projection":
            if (
                isinstance(random_projection_seed, (bool, np.bool_))
                or not isinstance(random_projection_seed, (int, np.integer))
            ):
                raise ValueError("random_projection_seed must be an integer")

            center = np.mean(points, axis=0)
            generator = np.random.default_rng(random_projection_seed)
            directions = generator.normal(size=(dim, 2))
            orthonormal_directions, _ = np.linalg.qr(directions)
            basis = orthonormal_directions.T
            for component in basis:
                dominant_axis = np.argmax(np.abs(component))
                if component[dominant_axis] < 0.0:
                    component *= -1.0
            return center, basis

        if dim == 2:
            return np.zeros(2), np.eye(2)[selected_axes]

        center = np.mean(points, axis=0)
        _, _, basis = np.linalg.svd(points - center, full_matrices=True)
        basis = basis[selected_axes].copy()
        for component in basis:
            dominant_axis = np.argmax(np.abs(component))
            if component[dominant_axis] < 0.0:
                component *= -1.0
        return center, basis

    def _setup_live_visualization(
        self,
        projection_axes=(0, 1),
        dimensionality_reduction="pca",
        random_projection_seed=0,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.colors import PowerNorm

        spline_parameter = np.linspace(0.0, 1.0, 300)
        spline_points = [
            np.asarray(spline(spline_parameter), dtype=np.float64)
            for spline in self.lyapunov.splines
        ]
        extent_points = np.vstack([self.w_array, *spline_points])
        center, basis = self._visualization_projection(
            extent_points,
            projection_axes=projection_axes,
            dimensionality_reduction=dimensionality_reduction,
            random_projection_seed=random_projection_seed,
        )
        project = lambda points: (
            np.asarray(points, dtype=np.float64) - center
        ) @ basis.T

        extent_2d = project(extent_points)
        span = max(np.ptp(extent_2d[:, 0]), np.ptp(extent_2d[:, 1]), 1e-8)
        # Square bounds preserve equal geometric scaling while allowing the
        # video axes to fill a fixed square canvas without whitespace.
        midpoint = 0.5 * (
            np.min(extent_2d, axis=0) + np.max(extent_2d, axis=0)
        )
        half_extent = 0.65 * span
        lower = midpoint - half_extent
        upper = midpoint + half_extent

        x_values = np.linspace(lower[0], upper[0], 70)
        y_values = np.linspace(lower[1], upper[1], 70)
        xx, yy = np.meshgrid(x_values, y_values)
        grid_2d = np.column_stack((xx.ravel(), yy.ravel()))
        grid_state = center + grid_2d @ basis
        state_flow, aux = self.velocity_multi(grid_state, return_aux=True)
        energy = np.asarray(aux[3]).reshape(xx.shape)
        flow = state_flow @ basis.T

        plt.ion()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.contourf(
            xx, yy, energy, levels=30, cmap=plt.cm.Blues,
            norm=PowerNorm(gamma=0.15), alpha=0.7, zorder=0, vmax = np.quantile(energy, 0.95)
        )
        ax.streamplot(
            xx, yy,
            np.nan_to_num(flow[:, 0].reshape(xx.shape)),
            np.nan_to_num(flow[:, 1].reshape(xx.shape)),
            color="lightgray", density=0.8, linewidth=2, zorder=1, broken_streamlines=True
        )
        for points in reversed(spline_points):
            points_2d = project(points)
            ax.plot(
                points_2d[:, 0], points_2d[:, 1],
                lw=3, color="black", zorder=3,
            )
        prototypes = project(self.w_array)
        ax.scatter(
            prototypes[:, 0], prototypes[:, 1], c="green", s=120,
            edgecolors="black", zorder=4,
        )
        current_artist = ax.scatter(
            [], [], c="red", s=110, edgecolors="black", linewidths=1.2,
            zorder=10, label="Current position",
        )
        use_blit = fig.canvas.supports_blit
        current_artist.set_animated(use_blit)

        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best", markerscale=0.8)
        plt.tight_layout()

        self._live_plot = {
            "plt": plt,
            "fig": fig,
            "ax": ax,
            "artist": current_artist,
            "project": project,
            "use_blit": use_blit,
            "background": None,
            "last_draw": 0.0,
        }
        fig.canvas.draw()
        if use_blit:
            fig.canvas.mpl_connect("draw_event", self._cache_live_background)
            self._cache_live_background()
        fig.canvas.flush_events()

    def _cache_live_background(self, _event=None):
        plot = self._live_plot
        if plot is None:
            return
        plot["background"] = plot["fig"].canvas.copy_from_bbox(
            plot["ax"].bbox
        )

    def _refresh_live_plot(self, state):
        if not self.live_visualization:
            return

        state = np.asarray(state, dtype=np.float64).reshape(-1)
        if state.shape != (self.dim,):
            raise ValueError(f"state must have shape ({self.dim},)")

        plot = self._live_plot
        if plot is None:
            self._setup_live_visualization()
            plot = self._live_plot
        if not plot["plt"].fignum_exists(plot["fig"].number):
            return
        now = time.perf_counter()
        if now - plot["last_draw"] < 1.0 / 30.0:
            return

        point = plot["project"](state[None, :])[0]
        plot["artist"].set_offsets(point[None, :])

        canvas = plot["fig"].canvas
        if plot["use_blit"] and plot["background"] is not None:
            canvas.restore_region(plot["background"])
            plot["ax"].draw_artist(plot["artist"])
            canvas.blit(plot["ax"].bbox)
        else:
            canvas.draw_idle()
        canvas.flush_events()
        plot["last_draw"] = now

    def _velocity_and_projection(self, x):
        x = np.asarray(x, dtype=np.float64)
        field = self.lyapunov

        (proj_point, transverse_distance, _, _, direction, spline_id,
         _, boundary_lambda, phase) = field.evaluate_one(
            x, return_phase=True
        )
        spline_id = int(spline_id)

        phi, gamma, _ = spline_weights_and_gate(
            x[None, :],
            spline_id,
            self.tube_model,
            projected_points=np.asarray(proj_point, dtype=np.float64)[None, :],
            transverse_distances=np.asarray(
                [transverse_distance], dtype=np.float64
            ),
            gamma_margin=self.gamma_margin,
            gamma_beta=self.gamma_beta,
            boundary_lambda=np.asarray([boundary_lambda]),
        )
        gamma = float(gamma[0])
        direction_hat = direction / (np.linalg.norm(direction) + 1e-8)
        _, P_n, P_t = tangent_projectors(
            x[None, :],
            spline_cache=field.spline_cache,
            spline_id=spline_id,
            S_COARSE=field.S_COARSE,
            phases=np.asarray([phase], dtype=np.float64),
        )
        modulation = block_modulation_matrices(
            phi, self.modulations[spline_id], P_n, P_t
        )[0]
        learned = -(modulation @ direction_hat)
        velocity = field.H_gate_one(x, spline_id) * (
            -(1.0 - gamma) * self.normed_gradient_scale * direction_hat
            + gamma * learned
        )
        if self.normalize_velocity:
            norm = np.linalg.norm(velocity)
            if norm > 0.0:
                velocity *= self.normed_gradient_scale / norm
        return velocity, proj_point

    def velocity(self, x, visualize=True):
        """Return the translational velocity at ``x``."""
        if visualize:
            self._refresh_live_plot(x)
        x_dot, _ = self._velocity_and_projection(x)
        return x_dot

    def velocity_multi(self, x, return_aux=False):
        """Evaluate velocities in batches and optionally return Lyapunov data."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must have shape (n_samples, {self.dim})")

        aux = self.lyapunov.evaluate_multi(x)
        (projected_points, transverse_distances, _, _, directions,
         spline_ids, _, boundary_lambdas) = aux
        velocities = np.zeros_like(x)

        for raw_spline_id in np.unique(spline_ids):
            spline_id = int(raw_spline_id)
            mask = spline_ids == spline_id
            points = x[mask]
            direction = directions[mask]

            phi, gamma, _ = spline_weights_and_gate(
                points,
                spline_id,
                self.tube_model,
                projected_points=projected_points[mask],
                transverse_distances=transverse_distances[mask],
                gamma_margin=self.gamma_margin,
                gamma_beta=self.gamma_beta,
                boundary_lambda=boundary_lambdas[mask],
            )
            gamma = gamma[:, None]

            direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
            direction_hat = direction / (direction_norm + 1e-8)

            _, P_n, P_t = tangent_projectors(
                points,
                spline_cache=self.lyapunov.spline_cache,
                spline_id=spline_id,
                S_COARSE=self.lyapunov.S_COARSE,
            )
            modulation = block_modulation_matrices(
                phi,
                self.modulations[spline_id],
                P_n,
                P_t,
            )
            learned = -np.einsum("nab,nb->na", modulation, direction_hat)

            H = self.lyapunov.H_gate_multi(points, spline_id)
            velocities[mask] = H * (
                -(1.0 - gamma) * self.normed_gradient_scale * direction_hat
                + gamma * learned
            )

        if self.normalize_velocity:
            norms = np.linalg.norm(velocities, axis=1, keepdims=True)
            np.divide(velocities, norms, out=velocities, where=norms > 0.0)
            velocities *= self.normed_gradient_scale

        if return_aux:
            return velocities, aux
        return velocities

    def step(self, x, step_scale=1.0, visualize=True):
        """Return one translational Euler increment with a stable array type."""
        return (
            float(step_scale)
            * self.integration_dt
            * self.velocity(x, visualize=visualize)
        )

    def step_with_projection(self, x, step_scale=1.0, visualize=True):
        """Return one translational increment and its spline projection."""
        if visualize:
            self._refresh_live_plot(x)
        x_dot, projected_point = self._velocity_and_projection(x)
        x_delta = float(step_scale) * self.integration_dt * x_dot
        return x_delta, projected_point
