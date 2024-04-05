import numpy as np
import json
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

    x_c = ball.x + ball.r * np.outer(np.cos(u), np.sin(v))
    y_c = ball.y + ball.r * np.outer(np.sin(u), np.sin(v))
    z_c = ball.z + ball.r * np.outer(np.ones(np.size(u)), np.cos(v))

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
    ax.voxels(voxelarray.transpose(1, 0, 2), facecolors=colors[np.max(voxelarray) - 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(azim=azim, elev=elev)
    plt.tight_layout()
    plt.show()


def plot_voxel(voxelarray, fc = 0, elev=20, azim=10):
    ax = plt.figure(figsize=(4, 4)).add_subplot(projection="3d")
    ax.voxels(voxelarray.transpose(1, 0, 2), facecolors = f"C{fc}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(azim=azim, elev=elev)
    plt.tight_layout()
    plt.show()


def voxel_ball(ball, boundary, empty_gnd=0, mask=False):
    y, x, z = np.indices((boundary.x_length, boundary.y_length, boundary.z_length))
    voxel = np.sqrt((x - ball.x) ** 2 + (y - ball.y) ** 2 + (z - ball.z) ** 2) < ball.r
    if mask:
        return voxel
    else:
        return np.where(voxel, ball.γ, empty_gnd)


def scale_realworld_to_intdomain(coordinate, hitbox, new_min=0, new_max=32, d=3):
    y_r, x_r, z_r = coordinate
    new_min += d
    new_max -= d
    # set minus x,y to origin
    y_r += hitbox.y_max
    x_r += hitbox.x_max

    scaled_value_y = ((y_r) / (hitbox.y_max * 2)) * (new_max - new_min) + new_min
    scaled_value_y = int(round(min(max(scaled_value_y, new_min), new_max)))

    scaled_value_x = ((x_r) / (hitbox.x_max * 2)) * (new_max - new_min) + new_min
    scaled_value_x = int(round(min(max(scaled_value_x, new_min), new_max)))

    scaled_value_z = ((z_r - hitbox.z_min) / (hitbox.z_max - hitbox.z_min)) * (
        new_max - new_min
    ) + new_min
    scaled_value_z = int(round(min(max(scaled_value_z, new_min), new_max)))
    return scaled_value_y, scaled_value_x, scaled_value_z


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
