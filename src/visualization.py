import matplotlib.pyplot as plt
from .classes import TankProperties32x2, BallAnomaly, PyEIT3DMesh
import numpy as np
from sklearn.manifold import TSNE
from typing import Union


def plot_voxel(
    voxelarray, elev=20, azim=10, save_img=False, s_name="images/voxels.png"
):
    ax = plt.figure(figsize=(6, 6)).add_subplot(projection="3d")
    ax.voxels(voxelarray)
    ax.view_init(azim=azim, elev=elev)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    if save_img:
        plt.savefig(s_name, dpi=250, transparent=True)
    plt.show()
