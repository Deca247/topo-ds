"""Fit and visualize TopoDS on a selected LASA handwriting dataset."""

import sys
from pathlib import Path

import numpy as np


EXPORT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = EXPORT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from topods import DSInference, TopoDS, load_lasa


LASA_SHAPES = (
    "Angle",
    "BendedLine",
    "CShape",
    "DoubleBendedLine",
    "GShape",
    "heee",
    "JShape",
    "JShape_2",
    "Khamesh",
    "Leaf_1",
    "Leaf_2",
    "Line",
    "LShape",
    "Multi_Models_1",
    "Multi_Models_2",
    "Multi_Models_3",
    "Multi_Models_4",
    "NShape",
    "PShape",
    "RShape",
    "Saeghe",
    "Sharpc",
    "Sine",
    "Snake",
    "Spoon",
    "Sshape",
    "Trapezoid",
    "Worm",
    "WShape",
    "Zshape",
)


def select_shape():
    print("Available LASA shapes:")
    entries = [
        f"{number:2}: {name}" for number, name in enumerate(LASA_SHAPES, start=1)
    ]
    column_width = max(map(len, entries)) + 2
    for start in range(0, len(entries), 3):
        print("  " + "".join(entry.ljust(column_width) for entry in entries[start:start + 3]).rstrip())

    while True:
        try:
            selection = int(input(f"Select a shape [1-{len(LASA_SHAPES)}]: "))
        except ValueError:
            print("Please enter a number.")
            continue

        if 1 <= selection <= len(LASA_SHAPES):
            return LASA_SHAPES[selection - 1]
        print(f"Please enter a number from 1 to {len(LASA_SHAPES)}.")


def rollouts_from_demonstration_starts(model, starts, max_steps=1200):
    """Roll out the learned dynamics once from each demonstration start."""
    dynamics = DSInference(
        model.w_array,
        model.splines,
        model.spline_nodes,
        model.components,
        model.modulations,
        model.tube_model,
        integration_dt=0.02,
        normed_gradient_scale=model.normed_gradient_scale,
        alpha=model.alpha,
        gamma_margin=model.gamma_margin,
        gamma_beta=model.gamma_beta,
        enable_boundary_progress_gate=model.enable_boundary_progress_gate,
        boundary_progress_gate_scale=model.boundary_progress_gate_scale,
        boundary_progress_gate_eps=model.boundary_progress_gate_eps,
        enable_H_gate=model.enable_H_gate,
        beta_H=model.beta_H,
        goals=model.goals,
    )

    goal = np.asarray(model.goals[0])
    rollouts = []
    for start in starts:
        trajectory = [np.asarray(start, dtype=float).copy()]
        for _ in range(max_steps):
            trajectory.append(
                trajectory[-1] + dynamics.step(trajectory[-1], visualize=False)
            )
            if np.linalg.norm(trajectory[-1] - goal) < 0.005:
                break
        rollouts.append(np.asarray(trajectory))
    return rollouts


def main():
    dataset_name = select_shape()

    np.random.seed(7)
    positions, velocities, trajectory_count, goals, _dataset = load_lasa(
        dataset_name,
        subsample_step=5,
    )

    prototype_count = 11
    n_epochs = 50
    model = TopoDS(
        positions,
        velocities,
        trajectory_count,
        goals,
        batch_size=4,
        num_protos=prototype_count,
        eps_start=0.08,
        eps_end=0.01,
        lmb_start=prototype_count / 5,
        lmb_end=0.01,
        t_max=n_epochs,
        alpha=0.9,
        topo_prob=0.5,
        gamma_margin=1.5,
        animate=True,
    )
    model.fit()

    demonstrations = positions.reshape(trajectory_count, -1, positions.shape[1])
    rollouts = rollouts_from_demonstration_starts(model, demonstrations[:, 0])
    model.plot_results(trajectories=rollouts)


if __name__ == "__main__":
    main()
