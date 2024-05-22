import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


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

    return np.array([center_y, center_x, center_z])


def compute_voxel_error(predicted_voxels, true_voxels, mode="stack"):
    # mode stack : e.g. shape (100,32,32,32)
    # mode single: e.g. shape (32,32,32)
    if mode == "single":
        com_pred = center_of_mass(predicted_voxels)
        com_true = center_of_mass(true_voxels)
        return np.array(com_true) - np.array(com_pred)

    elif mode == "stack":
        p_err = list()
        for pred, true in zip(predicted_voxels, true_voxels):
            com_pred = center_of_mass(pred)
            com_true = center_of_mass(true)
            p_err.append(np.array(com_true) - np.array(com_pred))
        p_err = np.array(p_err)
        return p_err


def compute_volume_error(predicted_voxels, true_voxels, mode="stack"):
    # mode stack : e.g. shape (100,32,32,32)
    # mode single: e.g. shape (32,32,32)
    if mode == "single":
        ele_pred = len(np.where(predicted_voxels != 0)[0])
        ele_true = len(np.where(true_voxels != 0)[0])
        return np.array(ele_pred - ele_true)

    elif mode == "stack":
        v_err = list()
        for pred, true in zip(predicted_voxels, true_voxels):
            ele_pred = len(np.where(pred != 0)[0])
            ele_true = len(np.where(true != 0)[0])
            if ele_pred == 0:
                v_err.append(None)
            else:
                v_err.append(ele_pred - ele_true)
        v_err = np.array(v_err)
        return v_err


def visualize_errors(
    errors,
    in_percent=True,
    voxel_val_max=32,
    save=False,
    s_path="images/reconstruction_axis_error.pdf",
):
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


def plot_confusion_matrix(Y_test, Y_pred, s_path=None, labels=["acryl", "brass"]):
    cf_matrix = confusion_matrix(Y_test, Y_pred)

    sns.heatmap(
        cf_matrix / np.sum(cf_matrix),
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("True")
    plt.ylabel("Predicted")
    if s_path != None:
        plt.savefig(s_path + "/cm.png")
    plt.show()


def compute_PCA(X):
    # 1. Reshape data to (X.shape[0], 4096)
    data_reshaped = X.reshape(X.shape[0], -1)
    # 2. PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_reshaped)
    del data_reshaped
    return data_pca


def plot_PCA(data_pca, Y):
    plt.figure(figsize=(6, 4))
    plt.scatter(
        data_pca[np.where(Y == 0)[0], 0],
        data_pca[np.where(Y == 0)[0], 1],
        c="C0",
        s=1,
        label="Acryl",
    )
    plt.scatter(
        data_pca[np.where(Y == 1)[0], 0],
        data_pca[np.where(Y == 1)[0], 1],
        c="C1",
        s=1,
        label="Brass",
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    # plt.xlim([-36,100])
    plt.legend(markerscale=4)
    plt.show()


def compute_position_error(pos_test, X_pred, mode="stack"):
    """
    Compute the linalg.norm()
    """
    if mode == "single":
        return np.linalg.norm(X_pred - pos_test)

    elif mode == "stack":
        p_err = list()
        if len(pos_test.shape) == 5:
            # no coordinates, voxels are given
            pos_test = np.squeeze(gamma_test, axis=4)

            for test, v_pred in zip(pos_test, X_pred):
                if len(v_pred[v_pred == 0]) == 32**3:
                    p_err.append(None)
                else:
                    p_pred = center_of_mass(v_pred)
                    p_test = center_of_mass(test)
                    p_err.append(np.linalg.norm(p_pred - p_test))
        else:
            for test, v_pred in zip(pos_test, X_pred):
                if len(v_pred[v_pred == 0]) == 32**3:
                    p_err.append(None)
                else:
                    p_pred = center_of_mass(v_pred)
                    p_err.append(np.linalg.norm(p_pred - test))
        return np.array(p_err)
