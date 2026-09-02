"""Fit and visualize TopoDS on the bundled recorded demonstration."""

import sys
from pathlib import Path

import numpy as np


EXPORT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = EXPORT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from topods import DSInference, TopoDS


def load_demonstrations(path):
    """Load and validate demonstrations saved by the interactive recorder."""
    with np.load(path, allow_pickle=False) as dataset:
        positions = np.asarray(dataset["positions"], dtype=float)
        velocities = np.asarray(dataset["velocities"], dtype=float)
        random_seed = int(dataset["random_seed"]) if "random_seed" in dataset.files else 0

    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError(
            f"{path} has invalid positions; expected (trajectories, points, 2)"
        )
    if velocities.shape != positions.shape:
        raise ValueError(f"{path} has invalid velocities; expected {positions.shape}")
    if not (np.isfinite(positions).all() and np.isfinite(velocities).all()):
        raise ValueError(f"{path} contains non-finite values")

    trajectory_count = positions.shape[0]
    return (
        positions.reshape(-1, positions.shape[-1]),
        velocities.reshape(-1, velocities.shape[-1]),
        trajectory_count,
        positions[:, 0],
        random_seed,
    )


def main():
    dataset_path = EXPORT_ROOT / "data" / "complex_topology.npz"
    positions, velocities, trajectory_count, starts, random_seed = (
        load_demonstrations(dataset_path)
    )
    np.random.seed(random_seed)

    prototype_count = 18
    n_epochs = 50
    model = TopoDS(
        positions,
        velocities,
        trajectory_count,
        goals=[],
        batch_size=4,
        num_protos=prototype_count,
        eps_start=0.8,
        eps_end=0.01,
        lmb_start=prototype_count / 2,
        lmb_end=0.01,
        t_max=n_epochs,
        alpha=0.3,
        gamma_margin=3.5,
        animate=True,
    )
    model.fit()

    model.plot_results()


if __name__ == "__main__":
    main()
