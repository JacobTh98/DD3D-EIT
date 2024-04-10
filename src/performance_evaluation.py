import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def center_of_mass(voxel_matrix):
    x_coords, y_coords, z_coords = np.meshgrid(
        np.arange(voxel_matrix.shape[0]),
        np.arange(voxel_matrix.shape[1]),
        np.arange(voxel_matrix.shape[2]),
    )
    total_mass = np.sum(voxel_matrix)
    center_x = np.sum(x_coords * voxel_matrix) / total_mass
    center_y = np.sum(y_coords * voxel_matrix) / total_mass
    center_z = np.sum(z_coords * voxel_matrix) / total_mass

    return center_y, center_x, center_z


def compute_voxel_err(predicted_voxels, true_voxels):
    com_pred = center_of_mass(predicted_voxels)
    com_true = center_of_mass(true_voxels)

    return np.array(com_true) - np.array(com_pred)




def visualize_errors(errors, in_percent=True, voxel_val_max=32, save=False, s_path = "images/reconstruction_axis_error.pdf"):
    if in_percent:
        errors = errors / voxel_val_max * 100
    data = {"x-pos": errors[:, 0], "y-pos": errors[:, 1], "z-pos": errors[:, 2]}
    df = pd.DataFrame(data)
    plt.figure(figsize=(6, 3))
    sns.boxplot(data=df, showfliers=False)
    # plt.ylim([-1,3])
    if in_percent:
        plt.ylabel("Error (%)")
    else:
        plt.ylabel("Error (Voxel)")
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(s_path)
    plt.show()