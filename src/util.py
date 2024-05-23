import numpy as np
import json
import random
import matplotlib.pyplot as plt
from .classes import (
    BallAnomaly,
    Boundary,
)


def plot_ball(
    ball: BallAnomaly,
    boundary: Boundary,
    res: int = 50,
    elev: int = 25,
    azim: int = 10,
):
    u = np.linspace(0, 2 * np.pi, res)
    v = np.linspace(0, np.pi, res)

    x_c = ball.x + ball.d / 2 * np.outer(np.cos(u), np.sin(v))
    y_c = ball.y + ball.d / 2 * np.outer(np.sin(u), np.sin(v))
    z_c = ball.z + ball.d / 2 * np.outer(np.ones(np.size(u)), np.cos(v))

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    # ball
    ax.plot_surface(x_c, y_c, z_c, color="C0", alpha=1)

    ax.set_xlim([boundary.x_0, boundary.x_length])
    ax.set_ylim([boundary.y_0, boundary.y_length])
    ax.set_zlim([boundary.z_0, boundary.z_length])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()


def plot_voxel_c(voxelarray, elev=20, azim=10):
    """
    fc : facecolor of the voxels
    """
    # C0 -> acrylic
    # C1 -> metal
    colors = ["C0", "C1"]  # Define colors for 1 and 2 values respectively

    ax = plt.figure(figsize=(4, 4)).add_subplot(projection="3d")
    # ax.voxels(voxelarray.transpose(1, 0, 2))
    ax.voxels(
        voxelarray.transpose(1, 0, 2), facecolors=colors[int(np.max(voxelarray) - 1)]
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(azim=azim, elev=elev)
    plt.tight_layout()
    plt.show()


def plot_voxel(voxelarray, fc=0, elev=20, azim=10):
    ax = plt.figure(figsize=(4, 4)).add_subplot(projection="3d")
    ax.voxels(voxelarray.transpose(1, 0, 2), facecolors=f"C{fc}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(azim=azim, elev=elev)
    plt.tight_layout()
    plt.show()


def voxel_ball(ball, boundary, empty_gnd=0, mask=False):
    y, x, z = np.indices((boundary.x_length, boundary.y_length, boundary.z_length))
    voxel = (
        np.sqrt((x - ball.x) ** 2 + (y - ball.y) ** 2 + (z - ball.z) ** 2) < ball.d / 2
    )
    if mask:
        return voxel
    else:
        return np.where(voxel, ball.γ, empty_gnd)


def plot_reconstruction_set(true, pred, cols=4, legends=False, save_fig=None):
    if true.shape != pred.shape:
        print("true.shape != pred.shape")
        return

    rows = 2
    colors = ["C0", "C1"]  # Define colors for 1 and 2 values respectively

    sel = random.sample(range(true.shape[0]), cols)
    print("Selcted the samples =", sel)
    fig, axes = plt.subplots(
        rows, cols, figsize=(14, 5), subplot_kw={"projection": "3d"}
    )
    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            voxelarray = true[sel[j]] if i == 0 else pred[sel[j]]
            ax.voxels(voxelarray, facecolors=colors[int(np.max(voxelarray) - 1)])
            if legends:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
            ax.view_init(azim=45, elev=30)
    if not legends:
        print("Row 0 -> true γ distribution")
        print("Row 1 -> pred γ distribution")
    # plt.tight_layout()
    if save_fig is not None:
        plt.savefig(save_fig, bbox_inches="tight", pad_inches=0)
    plt.show()


def read_json_file(file_path):
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file {file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
