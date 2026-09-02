import time
import tqdm
import numpy as np
import cvxpy as cp
import networkx as nx
from copy import deepcopy
from scipy.spatial.distance import cdist
from scipy.interpolate import make_interp_spline
from .ng import NeuralGas
from .spline_helpers import (
    fit_spline_coordinate_tube_model,
    score_path,
    spline_weights_and_gate,
)

from .lyapunov import (
    compute_energy_direction_and_aux_multi as lyapunov_compute_energy_direction_and_aux_multi,
    tangent_projectors as lyapunov_tangent_projectors,
)

class TopoDS():
    def __init__(self, data_pos, data_vel, n_trajectories, goals, batch_size=1, num_protos=10, 
                 eps_start=1, eps_end=0.001, lmb_start=5, lmb_end=0.01, t_max=5, alpha=0.8, topo_prob=0.3,
                 normed_gradient_scale=1.0, gamma_margin=1.0, gamma_beta=10.0,
                 enable_boundary_progress_gate=True, boundary_progress_gate_scale=0.5,
                 enable_H_gate=True, beta_H=10000,
                 tube_transverse_scale=1.5, tube_min_radius=0.01,
                 tube_min_samples=5, tube_neighbor_pool=1,
                 tube_projection_samples=400, animate=False):
        self.data_pos = np.array(data_pos, dtype=np.float32)
        self.data_vel = np.array(data_vel, dtype=np.float32)
        self.data_pos_vel = np.hstack((self.data_pos, self.data_vel))
        self.n_trajectories = n_trajectories
        self.goals = goals
        self.goal_prototype_indices = []
        self.animate = bool(animate)
        self.n_points_per_trajectory = int(self.data_pos.shape[0] / self.n_trajectories)

        # NG parameters
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.lmb_start = lmb_start
        self.lmb_end = lmb_end
        self.t_max = t_max
        self.eps = eps_start
        self.lmb = lmb_start
        self.batch_size = batch_size
        self.num_protos = num_protos
        self.alpha = alpha
        self.topo_prob = topo_prob
        self.normed_gradient_scale = normed_gradient_scale
        self.gamma_margin = float(gamma_margin)
        self.gamma_beta = float(gamma_beta)
        self.enable_boundary_progress_gate = enable_boundary_progress_gate
        self.boundary_progress_gate_scale = boundary_progress_gate_scale
        self.boundary_progress_gate_eps = 1e-12
        self.enable_H_gate = enable_H_gate
        self.beta_H = beta_H

        # Spline-coordinate tube controls. Longitudinal localization is
        # determined directly by neighboring prototype arc coordinates; only
        # the transverse data dispersion needs to be estimated.
        self.tube_transverse_scale = float(tube_transverse_scale)
        self.tube_min_radius = float(tube_min_radius)
        self.tube_min_samples = int(tube_min_samples)
        self.tube_neighbor_pool = int(tube_neighbor_pool)
        self.tube_projection_samples = int(tube_projection_samples)

        # dimensionality of input space
        self.dim = self.data_pos.shape[1]

        # initialize NeuralGas
        self.ng = NeuralGas(data_pos=self.data_pos, data_vel=self.data_vel, n_trajectories=self.n_trajectories, 
                            batch_size=self.batch_size, num_protos=self.num_protos, eps_start=self.eps_start, eps_end=self.eps_end,
                            lmb_start=self.lmb_start, lmb_end=self.lmb_end, t_max=self.t_max,
                            record_history=self.animate)

    def fit_spline(self, path_points, limit_cycle=False, iters=1, num_support_points=None):

        X0 = path_points.astype(np.float64)
        n, d = X0.shape

        if num_support_points is None:
            num_support_points = n * 10
        num_support_points = max(int(num_support_points), n)

        def _remove_consecutive_duplicates(X, eps=1e-12):
            if len(X) <= 1:
                return X
            keep = np.ones(len(X), dtype=bool)
            keep[1:] = np.linalg.norm(np.diff(X, axis=0), axis=1) > eps
            return X[keep]

        def _normalized_t(X):
            t = np.zeros(len(X), dtype=np.float64)
            t[1:] = np.cumsum(np.linalg.norm(X[1:] - X[:-1], axis=1))
            if t[-1] <= 1e-12:
                return np.linspace(0.0, 1.0, len(X))
            return t / t[-1]

        def _augment_knots(t_base, n_support):
            if n_support <= len(t_base):
                return t_base.copy()

            n_extra = n_support - len(t_base)
            n_intervals = len(t_base) - 1
            extras = np.full(n_intervals, n_extra // n_intervals, dtype=int)
            extras[:n_extra % n_intervals] += 1

            t_aug = [t_base[0]]
            for i in range(n_intervals):
                ti, tj = t_base[i], t_base[i + 1]
                if extras[i] > 0:
                    t_aug.extend(np.linspace(ti, tj, extras[i] + 2)[1:-1].tolist())
                t_aug.append(tj)

            return np.asarray(t_aug, dtype=np.float64)

        def _path_bspline(t, X):
            # Use cubic splines whenever possible. Junction splitting can create
            # short two- or three-prototype primitives, for which a lower-order
            # interpolating B-spline is required.
            degree = min(3, len(X) - 1)
            if degree < 3:
                return make_interp_spline(t, X, k=degree, axis=0)

            # Cubic B-spline with endpoint derivatives clamped to first/last graph edge.
            v0 = (X[1] - X[0]) / max(t[1] - t[0], 1e-12)
            v1 = (X[-1] - X[-2]) / max(t[-1] - t[-2], 1e-12)

            return make_interp_spline(t, X, k=3, axis=0, bc_type=([(1, v0)], [(1, v1)]))

        def _cycle_bspline(t, X):
            return make_interp_spline(t, X, k=3, axis=0, bc_type="periodic")

        def _build_spline(X, periodic=False):
            t_base = _normalized_t(X)
            Xtmp = X.copy()

            for _ in range(iters):
                cs = _cycle_bspline(t_base, Xtmp) if periodic else _path_bspline(t_base, Xtmp)
                Xtmp = cs(t_base)

            cs_base = _cycle_bspline(t_base, Xtmp) if periodic else _path_bspline(t_base, Xtmp)
            t_dense = _augment_knots(t_base, num_support_points)
            Xdense = cs_base(t_dense)

            cs_final = _cycle_bspline(t_dense, Xdense) if periodic else _path_bspline(t_dense, Xdense)
            return Xdense, cs_final, t_dense, t_base

        # -------------------------------------------------
        # LIMIT CYCLE CASE
        # -------------------------------------------------
        if limit_cycle:
            X = _remove_consecutive_duplicates(X0.copy())

            appended = False
            if np.linalg.norm(X[0] - X[-1]) > 1e-12:
                X = np.vstack([X, X[0]])
                appended = True

            Xs, cs_final, t, t_base = _build_spline(X, periodic=True)

            proto_point = Xs[0]
            data_idx = np.argmin(np.linalg.norm(self.data_pos - proto_point, axis=1))
            local_vel = self.data_vel[data_idx]
            spline_tangent = cs_final.derivative()(t[0])

            if np.dot(local_vel, spline_tangent) < 0:
                X = X[::-1]
                if appended:
                    X[-1] = X[0]
                Xs, cs_final, t, t_base = _build_spline(X, periodic=True)

            if appended:
                Xs = Xs[:-1]

            return Xs, cs_final

        # -------------------------------------------------
        # PATH CASE
        # -------------------------------------------------
        X = _remove_consecutive_duplicates(X0.copy())

        if len(X) < 2:
            raise ValueError("Path spline needs at least two distinct points.")

        Xs, cs_final, t, t_base = _build_spline(X, periodic=False)

        Xs[0] = X[0]
        Xs[-1] = X[-1]
        cs_final = _path_bspline(t, Xs)

        return Xs, cs_final

    def create_topology(self):
        """
        Voronoi-neighbor topology + probability edge killing.

        1) Assign data to prototype Voronoi cells.
        2) Compute directed transition probabilities P.
        3) Build undirected Voronoi/Delaunay neighbor graph E0.
        4) Remove weak edges only if no endpoint becomes isolated.
        5) Directly orient each surviving edge by transition dominance.
        """

        from scipy.spatial import Delaunay

        prototypes = self.w_array
        data = self.data_pos
        m = prototypes.shape[0]
        min_prob = self.topo_prob

        # -------------------------------------------------
        # Correct goal prototypes if dedicated goals are given
        # -------------------------------------------------
        if self.goals is not None:
            for g in self.goals:
                jg = np.argmin(np.linalg.norm(self.w_array - g, axis=1))
                self.w_array[jg] = g.copy()

        # -------------------------------------------------
        # Assign data points to nearest prototypes
        # -------------------------------------------------
        dists = cdist(data, prototypes)
        assignments = np.argmin(dists, axis=1)

        # -------------------------------------------------
        # Count directed transitions
        # -------------------------------------------------
        n_per_traj = data.shape[0] // self.n_trajectories
        traj_slices = [slice(i * n_per_traj, (i + 1) * n_per_traj) for i in range(self.n_trajectories - 1)]
        traj_slices.append(slice((self.n_trajectories - 1) * n_per_traj, data.shape[0]))

        T = np.zeros((m, m), dtype=float)

        for s in traj_slices:
            traj_assign = assignments[s]
            for t in range(len(traj_assign) - 1):
                u, v = traj_assign[t], traj_assign[t + 1]
                if u != v:
                    T[u, v] += 1.0

        # -------------------------------------------------
        # Row-normalized transition probabilities
        # -------------------------------------------------
        row_sums = T.sum(axis=1)
        P = np.zeros_like(T)
        nz = row_sums > 0
        P[nz] = T[nz] / row_sums[nz, None]

        # -------------------------------------------------
        # Build undirected Voronoi/Delaunay neighbor graph E0
        # -------------------------------------------------
        G_keep = nx.Graph()
        G_keep.add_nodes_from(range(m))

        if m <= 1:
            pass

        elif m <= self.dim + 1:
            # Delaunay impossible/degenerate: complete fallback
            for u in range(m):
                for v in range(u + 1, m):
                    G_keep.add_edge(u, v)

        else:
            try:
                tri = Delaunay(prototypes)

                for simplex in tri.simplices:
                    simplex = list(simplex)
                    for a in range(len(simplex)):
                        for b in range(a + 1, len(simplex)):
                            G_keep.add_edge(int(simplex[a]), int(simplex[b]))

            except Exception:
                # Degenerate fallback: nearest-neighbor graph
                D = cdist(prototypes, prototypes)
                np.fill_diagonal(D, np.inf)

                for u in range(m):
                    v = int(np.argmin(D[u]))
                    G_keep.add_edge(u, v)

        # -------------------------------------------------
        # Edge killing + orientation in one pass
        # -------------------------------------------------
        if self.animate:
            self.topology_history = [("candidates", tuple(G_keep.edges()))]

        deg = dict(G_keep.degree())

        C_new = np.zeros((m, m), dtype=np.int8)
        support_matrix = np.zeros((m, m), dtype=float)

        for u, v in list(G_keep.edges()):
            support = max(P[u, v], P[v, u])

            # Kill weak edge only if both endpoints keep another incident edge
            if support < min_prob:
                if deg[u] > 1 and deg[v] > 1:
                    G_keep.remove_edge(u, v)
                    deg[u] -= 1
                    deg[v] -= 1
                    if self.animate:
                        self.topology_history.append(("remove", u, v, support))
                    continue

            # Otherwise keep and orient by transition dominance
            if P[u, v] >= P[v, u]:
                C_new[u, v] = 1
                support_matrix[u, v] = P[u, v]
                direction = (u, v)
            else:
                C_new[v, u] = 1
                support_matrix[v, u] = P[v, u]
                direction = (v, u)
            if self.animate:
                self.topology_history.append(("orient", *direction, support_matrix[direction]))

        self.C[:] = C_new
        self.support_matrix = support_matrix

        # -------------------------------------------------
        # Build directed graph with Euclidean edge weights
        # -------------------------------------------------
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(m))

        for i in range(m):
            for j in range(m):
                if i != j and self.C[i, j]:
                    w = np.linalg.norm(self.w_array[j] - self.w_array[i])
                    self.G.add_edge(i, j, weight=w)

        # -------------------------------------------------
        # Junction prototypes
        # -------------------------------------------------
        # A path junction is a merge node: at least two incoming directed edges
        # and at least one outgoing edge. Pure sinks are deliberately excluded,
        # so the existing path/goal and limit-cycle handling remains separate.
        self.junction_prototypes = {
            int(n) for n in self.G.nodes()
            if self.G.in_degree(n) >= 2 and self.G.out_degree(n) >= 1
        }

        return self.w_array, self.C, self.support_matrix, self.G

    def compute_energy_direction_and_aux_multi(
            self, grid_points, spline_nodes, components, top_k=3,
            return_phases=False):
        """Evaluate V, its gradient, and the boundary-aware direction."""
        return lyapunov_compute_energy_direction_and_aux_multi(
            grid_points=np.asarray(grid_points, dtype=np.float64),
            spline_cache=self.spline_cache,
            spline_nodes=spline_nodes,
            components=components,
            alpha=self.alpha,
            dim=self.data_pos.shape[1],
            enable_boundary_progress_gate=self.enable_boundary_progress_gate,
            boundary_progress_gate_scale=self.boundary_progress_gate_scale,
            boundary_progress_gate_eps=self.boundary_progress_gate_eps,
            S_COARSE=self.S_COARSE,
            top_k=top_k,
            spline_metadata=getattr(self, "spline_primitive_metadata", None),
            return_phases=return_phases,
        )

    def _H_gate(self, X, path=None, is_cycle=False, spline_id=None):
        """Goal gate H(x), excluding nonterminal junction endpoints.

        A path primitive that ends at a merge junction is *not* a task goal.
        Therefore its endpoint must not make H vanish.  When junction-aware
        primitive metadata is available, only primitives marked
        ``terminal_is_goal`` contribute their terminal prototype.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]

        if (not self.enable_H_gate) or bool(is_cycle):
            return np.ones((X.shape[0], 1), dtype=np.float64)

        beta_H = max(float(self.beta_H), 0.0)
        if beta_H <= 0.0:
            return np.ones((X.shape[0], 1), dtype=np.float64)

        centers = []

        goals = self.goals
        if goals is not None:
            goal_array = np.asarray(goals, dtype=np.float64)
            if goal_array.ndim == 1 and goal_array.size == self.dim:
                centers.append(goal_array[None, :])
            elif (goal_array.ndim == 2 and goal_array.shape[1] == self.dim
                  and goal_array.shape[0] > 0):
                centers.append(goal_array)

        add_path_endpoint = False
        metadata = getattr(self, "spline_primitive_metadata", None)
        if spline_id is not None and metadata is not None and spline_id < len(metadata):
            add_path_endpoint = bool(metadata[spline_id].get("terminal_is_goal", False))
        elif metadata is None:
            # Backward-compatible behavior for models without junction metadata.
            add_path_endpoint = path is not None

        if add_path_endpoint and path is not None:
            path_arr = np.asarray(path, dtype=int).ravel()
            if path_arr.size > 0:
                centers.append(
                    np.asarray(self.w_array[path_arr[-1]], dtype=np.float64)[None, :]
                )

        if not centers:
            return np.ones((X.shape[0], 1), dtype=np.float64)

        Wg = np.vstack(centers)
        d2_min = np.min(
            np.sum((X[:, None, :] - Wg[None, :, :]) ** 2, axis=2),
            axis=1, keepdims=True
        )
        H = 1.0 - np.exp(-beta_H * d2_min)
        return np.clip(H, 0.0, 1.0)

    def fit_spline_lyapunov(self):
        """
        Fit topology-informed spline primitives.

        Path components and limit-cycle components remain strictly separated:
        path extraction is attempted first only for sink-like components, while
        components without a valid path structure fall back to cycle extraction.

        For path components, candidate source-to-sink routes are scored and
        processed from highest to lowest goodness. A graph junction is a prototype
        with at least two incoming edges and at least one outgoing edge. Once a
        high-scoring route has claimed the downstream suffix of a junction, a
        later lower-scoring route may merge into that junction and reuses the
        already selected downstream suffix. This gives the higher-quality route
        priority over the downstream continuation.

        Selected routes are finally split at junction prototypes. Consequently,
        a Y topology A->C, B->C is represented by three spline primitives.
        """

        splines = []
        spline_nodes = []
        components_out = []
        primitive_metadata = []
        selected_route_records = []

        k_shortest_paths = 20

        # --------------------------------------------------
        # explicit goals, if available
        # --------------------------------------------------
        goals = self.goals
        has_goals = goals is not None and len(goals) > 0

        proto_goal_idx = []
        if has_goals:
            goals_arr = np.asarray(goals, dtype=float)
            if goals_arr.ndim == 1:
                goals_arr = goals_arr[None, :]

            proto_goal_idx = [
                int(np.argmin(np.linalg.norm(self.w_array - goal, axis=1)))
                for goal in goals_arr
            ]

        all_junctions = set(getattr(self, "junction_prototypes", set()))
        components = list(nx.components.weakly_connected_components(self.G))

        def _select_priority_routes(scored_paths, junction_nodes):
            """Select routes in decreasing score order with merge priority.

            The first accepted route through a junction owns its downstream
            suffix. Later routes may introduce a disjoint upstream prefix and
            attach to that established suffix at the junction.
            """
            ordered = sorted(
                scored_paths,
                key=lambda item: (-float(item[0]), len(item[1]), tuple(item[1]))
            )

            selected = []
            selected_sources = set()
            used_nonterminal_nodes = set()

            # junction -> tuple of nodes from junction to terminal sink, fixed by
            # the first (therefore highest-priority) accepted route through it.
            junction_suffix = {}

            seen_candidates = set()
            for score, candidate in ordered:
                candidate = list(map(int, candidate))
                signature = tuple(candidate)
                if signature in seen_candidates:
                    continue
                seen_candidates.add(signature)

                source = candidate[0]
                if source in selected_sources:
                    continue

                # Earliest already-established junction encountered by this route.
                join_idx = None
                join_node = None
                for idx, node in enumerate(candidate[1:], start=1):
                    if node in junction_nodes and node in junction_suffix:
                        join_idx = idx
                        join_node = node
                        break

                if join_node is not None:
                    # Keep this candidate's new upstream branch, then reuse the
                    # high-priority downstream suffix already assigned to the junction.
                    route = candidate[:join_idx + 1] + list(junction_suffix[join_node][1:])

                    # The new branch may only touch the selected network at the
                    # junction itself. This preserves distinct upstream branches.
                    new_prefix = set(candidate[:join_idx])
                    if new_prefix & used_nonterminal_nodes:
                        continue
                else:
                    route = candidate

                    # Preserve the old non-overlap rule when there is no legitimate
                    # merge: only a common terminal sink is allowed.
                    nonterminal = set(route[:-1])
                    if nonterminal & used_nonterminal_nodes:
                        continue

                route_sig = tuple(route)
                if any(tuple(rec[1]) == route_sig for rec in selected):
                    continue

                selected.append((float(score), route, join_node))
                selected_sources.add(source)
                used_nonterminal_nodes.update(route[:-1])

                # Register downstream suffixes only once. Because candidates are
                # processed by decreasing score, the first registration has priority.
                for idx, node in enumerate(route[:-1]):
                    if node in junction_nodes and node not in junction_suffix:
                        junction_suffix[node] = tuple(route[idx:])

            return selected, junction_suffix

        def _split_routes_at_junctions(selected_routes, junction_nodes):
            """Split selected source-to-sink routes into unique spline primitives."""
            primitives = []
            seen = set()

            for _, route, _ in selected_routes:
                route = list(route)
                cut_indices = [0]
                cut_indices.extend(
                    idx for idx, node in enumerate(route[1:-1], start=1)
                    if node in junction_nodes
                )
                cut_indices.append(len(route) - 1)

                # Remove accidental duplicate cuts while retaining route order.
                cut_indices = list(dict.fromkeys(cut_indices))

                for a, b in zip(cut_indices[:-1], cut_indices[1:]):
                    primitive = route[a:b + 1]
                    if len(primitive) < 2:
                        continue
                    sig = tuple(primitive)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    primitives.append(primitive)

            return primitives

        for component_id, component in enumerate(components):
            subgraph = self.G.subgraph(component).copy()
            junction_nodes = all_junctions.intersection(component)

            # --------------------------------------------------
            # 1) Try path extraction first
            # --------------------------------------------------
            sink_candidates = []

            # Use explicit goal only if it is actually sink-like.
            # This prevents forcing a limit cycle into a path just because a goal was given.
            if has_goals:
                for gidx in proto_goal_idx:
                    if gidx in component and subgraph.out_degree(gidx) == 0:
                        sink_candidates.append(gidx)

            # If no explicit valid sink, infer terminal nodes from topology.
            if not sink_candidates:
                sink_candidates = [
                    n for n in subgraph.nodes()
                    if subgraph.in_degree(n) > 0 and subgraph.out_degree(n) == 0
                ]

            scored_paths = []

            for sink_idx in sink_candidates:
                sources_idx = [
                    n for n in subgraph.nodes()
                    if subgraph.in_degree(n) == 0 and n != sink_idx
                ]

                # If there are terminal sinks but no pure sources, try all other nodes.
                # This still only happens for sink-like components.
                if not sources_idx:
                    sources_idx = [n for n in subgraph.nodes() if n != sink_idx]

                for source_idx in sources_idx:
                    try:
                        paths_gen = nx.shortest_simple_paths(
                            subgraph, source_idx, sink_idx, weight="weight"
                        )

                        for _ in range(k_shortest_paths):
                            path = next(paths_gen)
                            if len(path) >= 2:
                                scored_paths.append((
                                    score_path(path, self.w_array, self.support_matrix),
                                    path,
                                ))

                    except (nx.NetworkXNoPath, StopIteration):
                        continue

            if scored_paths:
                selected_routes, junction_suffix = _select_priority_routes(
                    scored_paths, junction_nodes
                )

                if selected_routes:
                    primitives = _split_routes_at_junctions(
                        selected_routes, junction_nodes
                    )

                    for primitive in primitives:
                        path_points = self.w_array[primitive]
                        coords, spline = self.fit_spline(path_points, iters=1)
                        splines.append(spline)
                        primitive_metadata.append({
                            "type": "path",
                            "component_id": component_id,
                            "nodes": tuple(primitive),
                            "starts_at_junction": primitive[0] in junction_nodes,
                            "ends_at_junction": primitive[-1] in junction_nodes,
                            "terminal_is_goal": primitive[-1] in sink_candidates,
                        })

                    spline_nodes.append(primitives)
                    components_out.append((component, False))
                    selected_route_records.append({
                        "component_id": component_id,
                        "routes": [
                            {
                                "score": score,
                                "nodes": tuple(route),
                                "joined_at": join_node,
                            }
                            for score, route, join_node in selected_routes
                        ],
                        "junction_suffix": dict(junction_suffix),
                    })
                    continue

            # --------------------------------------------------
            # 2) Fall back to cycle extraction
            # --------------------------------------------------
            # This block is intentionally unchanged in spirit: cycles are not
            # subjected to junction route splitting or path-priority logic.
            cycles = list(nx.simple_cycles(subgraph))

            if cycles:
                cycle_supports = []

                for c in cycles:
                    support = sum(
                        self.support_matrix[u, v]
                        for u, v in zip(c, c[1:] + [c[0]])
                    )
                    cycle_supports.append((support, c))

                ordered = max(cycle_supports, key=lambda x: x[0])[1]
                path_points = self.w_array[ordered]
                coords, spline = self.fit_spline(
                    path_points, limit_cycle=True, iters=1
                )

                splines.append(spline)
                spline_nodes.append([ordered])
                components_out.append((component, True))
                primitive_metadata.append({
                    "type": "cycle",
                    "component_id": component_id,
                    "nodes": tuple(ordered),
                    "starts_at_junction": False,
                    "ends_at_junction": False,
                    "terminal_is_goal": False,
                })

        # --------------------------------------------------
        # Junction-aware global phase and categorical boundary topology
        # --------------------------------------------------
        # Phase is global along each selected source-to-goal route: the first
        # (highest-goodness) route establishes the downstream phase values.
        # Lower-priority branches start at phase 0 and inherit the already
        # established phase at the junction where they merge.  Thus a junction
        # is an ordinary intermediate phase value; only a true terminal goal has
        # phase 1.
        path_meta_ids_by_component = {}
        for spline_id, meta in enumerate(primitive_metadata):
            meta["spline_id"] = int(spline_id)
            meta["predecessor_ids"] = []
            meta["successor_ids"] = []
            meta["transparent_neighbors"] = []
            if meta["type"] == "path":
                path_meta_ids_by_component.setdefault(meta["component_id"], []).append(spline_id)

        route_record_by_component = {
            int(rec["component_id"]): rec for rec in selected_route_records
        }

        # Node phases are established route-by-route in decreasing score order.
        for component_id, spline_ids in path_meta_ids_by_component.items():
            rec = route_record_by_component.get(component_id)
            node_phase = {}
            if rec is not None:
                routes = rec.get("routes", [])
                if routes:
                    terminal = int(routes[0]["nodes"][-1])
                    node_phase[terminal] = 1.0

                for route_rec in routes:
                    route = list(map(int, route_rec["nodes"]))
                    if len(route) < 2:
                        continue

                    # The first phase value already fixed by a higher-priority
                    # route is exactly where this branch joins the established
                    # downstream network.  For the first route this is the sink.
                    assigned = [i for i, node in enumerate(route[1:], start=1)
                                if node in node_phase]
                    target_idx = assigned[0] if assigned else len(route) - 1
                    target_node = route[target_idx]
                    target_phase = float(node_phase.get(target_node, 1.0))

                    prefix = route[:target_idx + 1]
                    points = np.asarray(self.w_array[prefix], dtype=np.float64)
                    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
                    cum = np.concatenate(([0.0], np.cumsum(seg)))
                    total = float(cum[-1])
                    if total <= 1e-12:
                        values = np.linspace(0.0, target_phase, len(prefix))
                    else:
                        values = target_phase * cum / total

                    for node, value in zip(prefix, values):
                        if node not in node_phase:
                            node_phase[int(node)] = float(value)

            # Fallback for any path component lacking a route record.
            if not node_phase:
                component_nodes = []
                for sid in spline_ids:
                    component_nodes.extend(primitive_metadata[sid]["nodes"])
                if component_nodes:
                    node_phase[int(component_nodes[0])] = 0.0
                    node_phase[int(component_nodes[-1])] = 1.0

            # Assign phase interval and geometric arc length to every primitive.
            for sid in spline_ids:
                meta = primitive_metadata[sid]
                nodes = meta["nodes"]
                s0 = float(node_phase.get(int(nodes[0]), 0.0))
                s1 = float(node_phase.get(
                    int(nodes[-1]), 1.0 if meta.get("terminal_is_goal", False) else s0
                ))
                if s1 < s0:
                    s0, s1 = s1, s0
                meta["phase_start"] = s0
                meta["phase_end"] = s1

                sample_t = np.linspace(0.0, 1.0, 300)
                sample_x = np.asarray(splines[sid](sample_t), dtype=np.float64)
                arc_length = float(np.sum(np.linalg.norm(np.diff(sample_x, axis=0), axis=1)))
                meta["arc_length"] = max(arc_length, 1e-12)
                dcs_meta = splines[sid].derivative()
                meta["endpoint_speed_start"] = max(
                    float(np.linalg.norm(dcs_meta(0.0))), 1e-12
                )
                meta["endpoint_speed_end"] = max(
                    float(np.linalg.norm(dcs_meta(1.0))), 1e-12
                )

            # Direct parent-child spline pairs are transparent to boundary
            # protection. Siblings that merely share a merge endpoint are not.
            junctions_here = set(self.junction_prototypes).intersection(
                set().union(*(set(primitive_metadata[sid]["nodes"]) for sid in spline_ids))
            ) if spline_ids else set()
            for i in spline_ids:
                mi = primitive_metadata[i]
                end_i = int(mi["nodes"][-1])
                if end_i not in junctions_here:
                    continue
                for j in spline_ids:
                    if i == j:
                        continue
                    mj = primitive_metadata[j]
                    if int(mj["nodes"][0]) == end_i:
                        mi["successor_ids"].append(int(j))
                        mj["predecessor_ids"].append(int(i))

            # Pick one common positive ds/d(arc length) at every merge node.
            # The additional endpoint-speed bound keeps the one-interval cubic
            # Hermite phase monotone (positive progression) on every incident
            # primitive while matching the longitudinal derivative at the merge.
            for junction in junctions_here:
                incident_bounds = []
                for sid in spline_ids:
                    meta = primitive_metadata[sid]
                    at_start = int(meta["nodes"][0]) == junction
                    at_end = int(meta["nodes"][-1]) == junction
                    if not (at_start or at_end):
                        continue
                    ds = float(meta["phase_end"] - meta["phase_start"])
                    L = float(meta["arc_length"])
                    if ds <= 1e-12 or L <= 1e-12:
                        continue
                    speed = float(
                        meta["endpoint_speed_start"] if at_start
                        else meta["endpoint_speed_end"]
                    )
                    incident_bounds.append(ds / L)
                    incident_bounds.append(1.5 * ds / max(speed, 1e-12))
                if not incident_bounds:
                    continue
                common_slope = 0.75 * min(incident_bounds)
                for sid in spline_ids:
                    meta = primitive_metadata[sid]
                    if int(meta["nodes"][0]) == junction:
                        meta["phase_slope_start"] = float(common_slope)
                    if int(meta["nodes"][-1]) == junction:
                        meta["phase_slope_end"] = float(common_slope)

            # Non-junction source/goal endpoints use a monotone positive slope;
            # junction endpoint values above overwrite these defaults.
            for sid in spline_ids:
                meta = primitive_metadata[sid]
                ds = max(
                    float(meta["phase_end"]) - float(meta["phase_start"]),
                    1e-12,
                )
                avg = ds / float(meta["arc_length"])
                slope_start = min(
                    avg, 1.25 * ds / float(meta["endpoint_speed_start"])
                )
                slope_end = min(
                    avg, 1.25 * ds / float(meta["endpoint_speed_end"])
                )
                meta.setdefault("phase_slope_start", max(slope_start, 1e-12))
                meta.setdefault("phase_slope_end", max(slope_end, 1e-12))
                meta["phase_mode"] = "global_hermite"

        transparent_pairs = set()
        for meta in primitive_metadata:
            if meta["type"] != "path":
                continue
            sid = int(meta["spline_id"])
            for succ in meta.get("successor_ids", []):
                pair = tuple(sorted((sid, int(succ))))
                transparent_pairs.add(pair)
                meta["transparent_neighbors"].append(int(succ))
                primitive_metadata[int(succ)]["transparent_neighbors"].append(sid)

        # De-duplicate list-valued metadata while preserving deterministic order.
        for meta in primitive_metadata:
            for key in ("predecessor_ids", "successor_ids", "transparent_neighbors"):
                meta[key] = sorted(set(map(int, meta.get(key, []))))

        self.transparent_spline_pairs = sorted(transparent_pairs)

        S_COARSE = np.linspace(0.0, 1.0, 100)

        spline_cache = []
        for spline in splines:
            Cg = spline(S_COARSE)
            dcs = spline.derivative()
            ddcs = dcs.derivative()
            spline_cache.append((spline, dcs, ddcs, Cg))

        # Expose the topology decisions for the next junction-aware stages
        # (global phase assignment, categorical boundary gating, and H gating).
        self.spline_primitive_metadata = primitive_metadata
        self.goal_prototype_indices = sorted({
            int(meta["nodes"][-1])
            for meta in primitive_metadata
            if meta.get("terminal_is_goal", False) and meta.get("nodes")
        })
        self.selected_route_records = selected_route_records

        return splines, spline_nodes, components_out, spline_cache, S_COARSE

    def _tangent_projectors(self, X, spline_id, phases=None):
        return lyapunov_tangent_projectors(
            X=X, spline_cache=self.spline_cache, spline_id=spline_id,
            S_COARSE=self.S_COARSE, phases=phases)

    def learn_modulations(self, eps=1e-6):
        """Learn tangent-normal block modulation factors.

        The scalar Lyapunov function is
            V = alpha D + (1-alpha) R,
        while the fifth output of the Lyapunov routine is the boundary-aware
        direction. To retain the analytical decrease proof, every learned
        modulation is constrained to preserve the tangent-normal decomposition:
            A_i(x) = P_n(x) B_i P_n(x) + b_i P_t(x),
        with B_i positive definite and b_i positive.
        """
        pd_floor = max(float(eps), 1e-12)

        (proj_points, dist_vals, _, _, g_vals, closest_path,
         _, boundary_lambda, closest_phase) = \
            self.compute_energy_direction_and_aux_multi(
                self.data_pos, self.spline_nodes, self.components,
                return_phases=True,
            )

        all_paths = []
        all_is_cycle = []
        for (_, is_cycle), paths in zip(self.components, self.spline_nodes):
            if is_cycle:
                all_paths.append(paths[0])
                all_is_cycle.append(True)
            else:
                all_paths.extend(paths)
                all_is_cycle.extend([False] * len(paths))
        self.all_paths = all_paths
        self.all_is_cycle = all_is_cycle

        data_pos64 = self.data_pos.astype(np.float64, copy=False)
        data_vel64 = self.data_vel.astype(np.float64, copy=False)
        g_vals64 = g_vals.astype(np.float64, copy=False)
        w_array64 = self.w_array.astype(np.float64, copy=False)

        tube_model = fit_spline_coordinate_tube_model(
            data_pos=data_pos64,
            w_array=w_array64,
            splines=self.splines,
            all_paths=all_paths,
            all_is_cycle=all_is_cycle,
            transverse_scale=self.tube_transverse_scale,
            min_radius=self.tube_min_radius,
            min_samples=self.tube_min_samples,
            neighbor_pool=self.tube_neighbor_pool,
            gamma_margin=self.gamma_margin,
            gamma_beta=self.gamma_beta,
            projection_samples=self.tube_projection_samples,
            data_path_assignment=np.asarray(closest_path, dtype=np.int64),
        )
        tube_model["boundary_progress_gate_scale"] = float(
            self.boundary_progress_gate_scale)
        tube_model["boundary_progress_gate_eps"] = float(
            self.boundary_progress_gate_eps)
        tube_model["enable_boundary_progress_gate"] = bool(
            self.enable_boundary_progress_gate)
        tube_model["spline_primitive_metadata"] = deepcopy(
            getattr(self, "spline_primitive_metadata", [])
        )
        tube_model["junction_prototypes"] = sorted(
            map(int, getattr(self, "junction_prototypes", set()))
        )
        tube_model["transparent_spline_pairs"] = [
            tuple(map(int, pair))
            for pair in getattr(self, "transparent_spline_pairs", [])
        ]

        learned = {}
        path_data = {}

        for p_id, path in enumerate(all_paths):
            anchors = w_array64[path]
            P, m = anchors.shape
            I_m = np.eye(m, dtype=np.float64)
            is_cycle = bool(all_is_cycle[p_id])

            mask = closest_path == p_id
            X = data_pos64[mask]
            dX = data_vel64[mask]
            g = g_vals64[mask]
            lambda_p = np.asarray(boundary_lambda[mask], dtype=np.float64)
            Ns = X.shape[0]

            empty_model = {
                "type": "tangent_block",
                "is_cycle": is_cycle,
                "B": [pd_floor * I_m.copy() for _ in range(P)],
                "b": [pd_floor for _ in range(P)],
            }
            if Ns == 0:
                learned[p_id] = empty_model
                continue

            Phi, returned_gate, diagnostics = spline_weights_and_gate(
                X,
                p_id,
                tube_model,
                projected_points=np.asarray(proj_points[mask], dtype=np.float64),
                transverse_distances=np.asarray(dist_vals[mask], dtype=np.float64),
                gamma_margin=self.gamma_margin,
                gamma_beta=self.gamma_beta,
            )
            support_gate = np.asarray(
                diagnostics.get("support_gate", returned_gate),
                dtype=np.float64,
            )
            Gamma = (1.0 - lambda_p) * support_gate

            H_vals = self._H_gate(X, path=path, is_cycle=is_cycle, spline_id=p_id)
            g_norm = np.linalg.norm(g, axis=1, keepdims=True)
            g_hat = g / (g_norm + eps)

            target_shift = (
                dX
                + H_vals * (1.0 - Gamma[:, None])
                * self.normed_gradient_scale * g_hat
            )

            _, P_n_arr, P_t_arr = self._tangent_projectors(
                X, p_id, phases=closest_phase[mask]
            )

            path_data[p_id] = {
                "P": P,
                "m": m,
                "Ns": Ns,
                "Phi": np.asarray(Phi, dtype=np.float64),
                "Gamma": Gamma[:, None],
                "H": H_vals,
                "g_hat": g_hat,
                "target_shift": target_shift,
                "P_n": P_n_arr,
                "P_t": P_t_arr,
                "is_cycle": is_cycle,
            }

        installed_solvers = set(cp.installed_solvers())
        solver_priority = [
            solver for solver in (cp.MOSEK, cp.CLARABEL, cp.SCS)
            if solver in installed_solvers
        ]

        def _solve_problem(prob):
            last_err = None
            for solver in solver_priority:
                try:
                    prob.solve(solver=solver, verbose=False, warm_start=True)
                    if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                        return
                except Exception as err:
                    last_err = err
            if last_err is not None:
                print(
                    "Warning: modulation optimization failed with installed "
                    f"solvers: {last_err}"
                )

        def _project_to_spd(A, m):
            if A is None:
                return pd_floor * np.eye(m, dtype=np.float64)
            A = np.asarray(A, dtype=np.float64)
            if A.shape != (m, m) or not np.all(np.isfinite(A)):
                return pd_floor * np.eye(m, dtype=np.float64)
            A = 0.5 * (A + A.T)
            vals, vecs = np.linalg.eigh(A)
            vals = np.maximum(vals, pd_floor)
            result = (vecs * vals) @ vecs.T
            return 0.5 * (result + result.T)

        for p_id, pdata in path_data.items():
            P = pdata["P"]
            m = pdata["m"]
            Ns = pdata["Ns"]
            I_m = np.eye(m, dtype=np.float64)

            B = [cp.Variable((m, m), symmetric=True) for _ in range(P)]
            b = [cp.Variable(nonneg=True) for _ in range(P)]
            constraints = [Bi >> pd_floor * I_m for Bi in B]
            constraints += [bi >= pd_floor for bi in b]

            # Build a single linear map from vec(B[i]) to all sample fields.
            # Constructing one CVXPY expression per sample here creates a very
            # large expression tree and makes canonicalization slow.
            P_n = pdata["P_n"]
            normal_gradient = np.einsum(
                "naj,nj->na", P_n, pdata["g_hat"], optimize=True
            )
            normal_operator = np.einsum(
                "nau,nv->nauv", P_n, normal_gradient, optimize=True
            ).reshape(Ns * m, m * m)
            tangent_gradient = np.einsum(
                "naj,nj->na", pdata["P_t"], pdata["g_hat"],
                optimize=True,
            )
            normal_operator = cp.Constant(normal_operator)
            tangent_gradient = cp.Constant(tangent_gradient)

            f_learned = 0
            for i in range(P):
                normal_field = cp.reshape(
                    normal_operator @ cp.reshape(B[i], (m * m,), order="C"),
                    (Ns, m),
                    order="C",
                )
                local_field = -(
                    normal_field
                    + b[i] * tangent_gradient
                )
                weight = cp.Constant(
                    pdata["H"] * pdata["Gamma"]
                    * pdata["Phi"][:, [i]]
                )
                f_learned += cp.multiply(weight, local_field)

            loss = cp.sum_squares(
                cp.Constant(pdata["target_shift"]) - f_learned
            )
            problem = cp.Problem(cp.Minimize(loss), constraints)
            _solve_problem(problem)

            B_value = [_project_to_spd(Bi.value, m) for Bi in B]
            b_value = [
                max(float(bi.value), pd_floor)
                if bi.value is not None and np.isfinite(float(bi.value))
                else pd_floor
                for bi in b
            ]
            learned[p_id] = {
                "type": "tangent_block",
                "is_cycle": bool(pdata["is_cycle"]),
                "B": B_value,
                "b": b_value,
            }

        return learned, tube_model


    def plot_results(self, color_lyapunov_splines=None, trajectories=None):
        """Plot 2D diagnostics or a compact 3D model overview."""
        from .plotting import plot_results
        if color_lyapunov_splines is None:
            return plot_results(self, trajectories=trajectories)
        return plot_results(
            self, color_lyapunov_splines, trajectories=trajectories
        )


    def fit(self):
        # obtain prototypes from parallelized NeuralGas
        step_start = time.perf_counter()
        self.w_array, self.C = self.ng.fit()

        neural_gas_time = time.perf_counter() - step_start
        if self.animate:
            from .animation import (animate_modulation_learning, animate_neural_gas,
                                        animate_primitives_and_splines, animate_topology)
            self.neural_gas_animation = animate_neural_gas(self.data_pos, self.ng.prototype_history)

        # create topology
        step_start = time.perf_counter()
        self.w_array, self.C, self.C_support, self.G = self.create_topology()
        topology_time = time.perf_counter() - step_start
        if self.animate:
            self.topology_animation = animate_topology(self.data_pos, self.w_array, self.topology_history)

        # fit topology-informed spline-based lyapunov function
        step_start = time.perf_counter()
        self.splines, self.spline_nodes, self.components, self.spline_cache, self.S_COARSE = \
            self.fit_spline_lyapunov()
        spline_time = time.perf_counter() - step_start
        if self.animate:
            self.spline_animation = animate_primitives_and_splines(
                self.data_pos, self.w_array, self.C, self.components, self.spline_nodes, self.splines)

        # Learn local tangent-block parameters; inference blends them into M_k(x).
        step_start = time.perf_counter()
        self.modulations, self.tube_model = self.learn_modulations()
        dynamics_time = time.perf_counter() - step_start
        if self.animate:
            self.modulation_animation = animate_modulation_learning(self)

        #self.plot_results()

        total_time = neural_gas_time + topology_time + spline_time + dynamics_time
        print(
            f"TopoDS [s]: NG={neural_gas_time:.3f}  topology={topology_time:.3f}  "
            f"spline={spline_time:.3f}  dynamics={dynamics_time:.3f}  total={total_time:.3f}"
        )
        return self.w_array, self.C, self.splines, self.spline_nodes, self.components, self.modulations, self.tube_model
