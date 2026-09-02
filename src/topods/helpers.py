import numpy as np
import tqdm
import pyLasaDataset as lasa
import os, sys
import matplotlib.pyplot as plt

def compute_rollouts(positions, goal, velocities, n_trajectories, ds_inf, subsample_step, start_idx,
                     dataset_name, dataset_sample_name, save=True, show_velo=False):
    """
    Compute point-wise predictions and trajectory rollouts for given position
    and velocities and save them to folderpath (assumed ./.../folder/).
    """
    if save:
        base_path = f"./eval/topods/{dataset_name}/{dataset_sample_name}/"
        max_folders = 4
        # make sure base path exists
        os.makedirs(base_path, exist_ok=True)
        # find the next available folder number
        i = 0
        while i <= max_folders and os.path.exists(os.path.join(base_path, str(i))):
            i += 1
        if i > max_folders:
            raise RuntimeError(f"Maximum number of folders ({max_folders}) reached for {base_path}")
        folderpath = os.path.join(base_path, str(i))
        os.makedirs(folderpath, exist_ok=True)
        folderpath = folderpath + "/"
        print("Created:", folderpath)
    dt_step = np.mean(np.linalg.norm(np.diff(positions, axis=0), axis=1) /
                 np.linalg.norm(velocities[:-1], axis=1))
    dt_step = 0.019
    subsample_step = 1
    print("EVAL DT:", dt_step)
    y_preds = []
    for i in range(positions.shape[0]):
        x = positions[i,:]
        y = velocities[i,:]
        y_pred = ds_inf.velocity(x)
        y_preds.append(y_pred)
    data_trajectory = positions.reshape(n_trajectories, -1, positions.shape[-1])
    rollouts = []
    x_deltas_collected = []
    for i in tqdm.tqdm(range(n_trajectories)):
        x_current = data_trajectory[i,start_idx,:]
        x_goal = data_trajectory[i,-1,:]
        # rollout until we reach the goal
        out_trajectory = [x_current]
        x_deltas = []
        for j in range(data_trajectory.shape[1]*6):
            x_delta = ds_inf.velocity(x_current)
            x_deltas.append(x_delta)
            x_current = x_current + (x_delta * dt_step)
            out_trajectory.append(x_current)
            if np.linalg.norm(goal - x_current) < (dt_step * 0.1):
                break
        out_trajectory = np.array(out_trajectory)[::subsample_step,:]
        rollouts.append(out_trajectory)
        x_deltas_collected.append(x_deltas)

    if show_velo == True:
        plt.figure()
        for x_deltas in x_deltas_collected:
            x_deltas = np.array(x_deltas)
            x_deltas_vels = np.linalg.norm(x_deltas, axis=1)
            plt.plot(x_deltas_vels)
        data_vels = velocities.reshape(n_trajectories, -1, velocities.shape[-1])
        for dv in data_vels:
            dv_norm = np.linalg.norm(dv, axis=1)
            plt.plot(dv_norm, marker="x")
        plt.show()
    data = np.hstack((positions, velocities))
    # save to eval folder
    if save:
        np.savez(folderpath + "data.npz", data)
        np.savez(folderpath + "ypred.npz", y_preds)
        np.savez(folderpath + "rollout.npz", np.array(rollouts, dtype=object))
        return folderpath
    else:
        return None

def load_lasa(name, subsample_step, normalize=True):
    lasa_dataset = getattr(lasa.DataSet, name)

    data = lasa_dataset
    demos = data.demos

    positions = []
    velocities = []

    for demo in demos:
        positions.append(demo.pos)
        velocities.append(demo.vel)

    n_trajectories = len(positions)

    positions = np.array(positions).swapaxes(1, 2).reshape(-1, 2)
    velocities = np.array(velocities).swapaxes(1, 2).reshape(-1, 2)

    # replace NaNs
    positions = np.nan_to_num(positions, nan=0.0)
    velocities = np.nan_to_num(velocities, nan=0.0)

    # subsample
    positions = positions[::subsample_step, :]
    velocities = velocities[::subsample_step, :]

    # compute goal before normalization
    goal = positions.reshape(n_trajectories, -1, positions.shape[1])[:, -1, :].mean(axis=0)

    if normalize:
        # compute scales
        pos_norms = np.linalg.norm(positions, axis=1)
        vel_norms = np.linalg.norm(velocities, axis=1)

        scale_pos = np.max(pos_norms)
        scale_vel = np.max(vel_norms)

        if scale_pos == 0:
            scale_pos = 1.0
        if scale_vel == 0:
            scale_vel = 1.0

        # normalize
        positions = positions / scale_pos
        velocities = velocities / scale_vel
        goal = goal / scale_pos

    return positions, velocities, n_trajectories, [goal], "lasa"
