from __future__ import annotations

from typing import Tuple

import numpy as np
from numba import njit
from scipy.spatial.distance import cdist


_EPS = 1e-12


@njit(cache=True)
def _train_epoch(data_pos, data_phase, w_array, w_phase, permutation,
                 batch_size, eps, lmb, max_update):
    """Run one Neural Gas epoch; cached by Numba for compatible array types."""
    n_data, dim = data_pos.shape
    n_protos = w_array.shape[0]
    rank_weights = np.exp(-np.arange(n_protos) / lmb)
    distances = np.empty(n_protos)
    delta = np.empty((n_protos, dim))
    phase_delta = np.empty(n_protos)

    for start in range(0, n_data, batch_size):
        batch_count = min(batch_size, n_data - start)
        delta.fill(0.0)
        phase_delta.fill(0.0)

        for batch_idx in range(batch_count):
            sample_idx = permutation[start + batch_idx]
            for prototype_idx in range(n_protos):
                distance = (data_phase[sample_idx, 0] - w_phase[prototype_idx, 0]) ** 2
                for dimension in range(dim):
                    difference = data_pos[sample_idx, dimension] - w_array[prototype_idx, dimension]
                    distance += difference * difference
                distances[prototype_idx] = distance

            order = np.argsort(distances)
            for rank in range(n_protos):
                prototype_idx = order[rank]
                weight = rank_weights[rank]
                for dimension in range(dim):
                    delta[prototype_idx, dimension] += weight * (
                        data_pos[sample_idx, dimension] - w_array[prototype_idx, dimension]
                    )
                phase_delta[prototype_idx] += weight * (
                    data_phase[sample_idx, 0] - w_phase[prototype_idx, 0]
                )

        scale = eps / batch_count
        for prototype_idx in range(n_protos):
            update_norm = 0.0
            for dimension in range(dim):
                update = delta[prototype_idx, dimension] * scale
                if np.isnan(update):
                    update = 0.0
                elif np.isinf(update):
                    update = 1e10 if update > 0.0 else -1e10
                delta[prototype_idx, dimension] = update
                update_norm += update * update
            clip = min(1.0, max_update / (np.sqrt(update_norm) + _EPS))
            for dimension in range(dim):
                w_array[prototype_idx, dimension] += delta[prototype_idx, dimension] * clip
            w_phase[prototype_idx, 0] = min(1.0, max(0.0, w_phase[prototype_idx, 0]
                                                        + phase_delta[prototype_idx] * scale))


class NeuralGas:
    """Fast batched Neural Gas.

    Prototype ranking uses position augmented with normalized trajectory phase.
    Prototype updates and returned prototypes remain in position space.

    Parameters
    ----------
    data_pos, data_vel:
        Concatenated trajectory positions and velocities with shape ``(N, D)``.
    n_trajectories:
        Number of concatenated, equally sized trajectories.
    record_history:
        If ``True``, retain prototype positions after each training epoch.
    """

    def __init__(self, data_pos, data_vel, n_trajectories, batch_size=1, num_protos=10, eps_start=1,
                 eps_end=0.001, lmb_start=5, lmb_end=0.01, t_max=5, record_history=False):
        self.n_trajectories = int(n_trajectories)

        self.data_pos = np.ascontiguousarray(data_pos, dtype=np.float32)
        self.data_vel = np.ascontiguousarray(data_vel, dtype=np.float32)
        self.data_phase = np.ascontiguousarray(self._trajectory_phase(self.data_pos.shape[0], self.n_trajectories),
                                               dtype=np.float32)
        self.data_pos_vel = np.hstack((self.data_pos, self.data_vel))

        self.batch_size = int(batch_size)
        self.num_protos = int(num_protos)
        self.t_max = int(t_max)
        self.record_history = bool(record_history)

        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.lmb_start = float(lmb_start)
        self.lmb_end = float(lmb_end)

        sample_indices = self._sample_initial_prototypes(self.data_pos, self.num_protos)
        self.w_array = self.data_pos[sample_indices].astype(np.float32)
        self.w_array_phase = self.data_phase[sample_indices].astype(np.float32)
        self.prototype_history = [self.w_array.copy()] if self.record_history else []

        self.C = np.zeros((self.num_protos, self.num_protos), dtype=np.float32)
        self.dim = self.data_pos.shape[1]

    def _trajectory_phase(self, n_samples: int, n_trajectories: int) -> np.ndarray:
        """Return normalized progress for equally sized trajectories."""
        samples_per_trajectory = n_samples // n_trajectories
        phase = np.linspace(0.0, 1.0, samples_per_trajectory)
        return np.tile(phase, n_trajectories)[:, None]

    def _sample_initial_prototypes(self, data_pos: np.ndarray, n_prototypes: int) -> np.ndarray:
        """Randomly select prototype samples from the data."""
        return np.random.choice(len(data_pos), n_prototypes, replace=n_prototypes > len(data_pos))

    def adjust_params(self, t: int) -> Tuple[float, float]:
        """Return exponentially decayed learning and neighborhood rates."""
        fraction = t / self.t_max
        eps_out = self.eps_start * (self.eps_end / self.eps_start) ** fraction
        lmb_out = self.lmb_start * (self.lmb_end / self.lmb_start) ** fraction
        return eps_out, lmb_out

    def normalize_rows(self, X):
        """Normalize each row of ``X`` to unit Euclidean norm."""
        X = np.asarray(X)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    def _median_full_pairwise_distance(self, w_array: np.ndarray) -> float:
        """Return the median of the original full pairwise-distance matrix."""
        if w_array.shape[0] <= 1:
            return 0.0
        return float(np.median(cdist(w_array, w_array)))

    def prune_dead_units(self):
        """Remove prototypes that are not nearest to any training sample."""
        winners = np.argmin(cdist(self.data_pos, self.w_array), axis=1)
        keep = np.bincount(winners, minlength=self.w_array.shape[0]) > 0
        w_array = self.w_array[keep]
        self.w_array_phase = self.w_array_phase[keep]
        C = self.C[np.ix_(keep, keep)]
        return w_array, C

    def fit(self):
        """Fit Neural Gas using a disk-cached Numba training kernel."""
        max_update_scale = 0.1
        n_data = self.data_pos.shape[0]

        for t in range(1, self.t_max):
            if not (np.isfinite(self.w_array).all() and np.isfinite(self.w_array_phase).all()):
                print(f"Non-finite prototype value in epoch: {t}")
                return None

            self.eps, self.lmb = self.adjust_params(t)
            permutation = np.random.permutation(n_data).astype(np.int64)
            median_ipd = self._median_full_pairwise_distance(self.w_array)
            max_update = max(max_update_scale * median_ipd, 1e-6)
            _train_epoch(
                self.data_pos, self.data_phase, self.w_array, self.w_array_phase,
                permutation, self.batch_size, self.eps, self.lmb, max_update,
            )
            if self.record_history:
                self.prototype_history.append(self.w_array.copy())

        self.w_array, self.C = self.prune_dead_units()
        if self.record_history:
            self.prototype_history.append(self.w_array.copy())
        return self.w_array, self.C
