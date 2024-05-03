from .dataprocessing import get_measured_potential
import numpy as np
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
import os
from dataclasses import dataclass
from .util import voxel_ball
from .classes import Boundary
from .dataprocessing import get_measured_potential


def z_score(X):
    X_mean = np.mean(X)
    X_std = np.std(X)
    return (X - X_mean) / X_std


class DataGenerator(Sequence):
    def __init__(
        self,
        list_IDs,
        path,
        mean_path="measurements/datameans/",
        batch_size=32,
        eit_dim=4096,
        supervised="diameter",
        shuffle=True,
        EIT_shape="vector",
    ):
        "Initialization"
        self.path = path
        self.mean_path = mean_path
        self.eit_dim = eit_dim
        self.supervised = supervised
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.EIT_shape = EIT_shape
        self.on_epoch_end()
        self.n = 0
        self.max = self.__len__()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *eit_dim)
        # Initialization
        if self.EIT_shape == "vector":
            X = np.empty((self.batch_size, self.eit_dim, 1))  # EIT signal, self.eit_dim
        elif self.EIT_shape == "matrix":
            X = np.empty((self.batch_size, 64, 64, 1))  # EIT signal, self.eit_dim

        y = np.empty((self.batch_size, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load sample
            tmp = np.load(
                "{0:s}/sample_{1:06d}.npz".format(self.path, ID), allow_pickle=True
            )
            anomaly = tmp["anomaly"].tolist()
            raw_eit = get_measured_potential(tmp, shape_type=self.EIT_shape)

            mean_eit = np.load(
                f"{self.mean_path}mean_gnd_d_{anomaly.d}_{anomaly.material}.npy",
                allow_pickle=True,
            )
            if self.EIT_shape == "vector":
                pot = np.abs(raw_eit - mean_eit)
                X[i,] = np.expand_dims(pot, axis=1)

            elif self.EIT_shape == "matrix":
                pot = np.abs(raw_eit - np.reshape(mean_eit, (64, 64)))

                X[i,] = np.expand_dims(pot, axis=2)

            if self.supervised == "diameter":
                y[i,] = anomaly.d
            elif self.supervised == "material":
                mat_dict = {"acryl": 0, "brass": 1}
                y[i,] = mat_dict[anomaly.material]

            # add normalization?
        return X, y


@dataclass
class BallAnomaly_vxl:  #  FOR VAE
    x: float
    y: float
    z: float
    d: float
    γ: float


def DataLoader(params: dict, n_earlystop=None):
    """
    Set "n_earlystop" to a integer to load data until sample number.

    """

    def scale_meas_to_vxls(anmly, new_min=0, new_max=32):
        T_d = 194
        T_r = T_d / 2
        z_min = 40
        z_max = 110

        scaled_value_x = ((anmly.x + T_r) / (T_d)) * new_max
        scaled_value_y = ((anmly.y + T_r) / (T_d)) * new_max
        scaled_value_z = ((anmly.z - z_min) / (z_max - z_min)) * new_max

        return scaled_value_x, scaled_value_y, scaled_value_z

    X = list()
    Y = list()

    samples_range = len(os.listdir(params["path"]))
    if n_earlystop:
        samples_range = n_earlystop

    for i in tqdm(range(samples_range)):
        tmp = np.load(
            "{0:s}/sample_{1:06d}.npz".format(params["path"], i), allow_pickle=True
        )
        anomaly = tmp["anomaly"].tolist()
        raw_eit = get_measured_potential(tmp, shape_type=params["EIT_shape"])

        mean_eit = np.load(
            f"{params['mean_path']}mean_gnd_d_{anomaly.d}_{anomaly.material}.npy",
            allow_pickle=True,
        )
        if params["EIT_shape"] == "vector":
            pot = np.abs(raw_eit - mean_eit)
            X.append(pot)
        elif params["EIT_shape"] == "matrix":
            pot = np.abs(raw_eit - np.reshape(mean_eit, (64, 64)))
            X.append(pot)
        if params["supervised"] == "diameter":
            Y.append(anomaly.d)
        elif params["supervised"] == "material":
            mat_dict = {"acryl": 0, "brass": 1}
            Y.append(mat_dict[anomaly.material])

        elif params["supervised"] == "anomaly":
            anmly = tmp["anomaly"].tolist()
            dia_dict = {"20": 4, "30": 6, "40": 8}

            x_vxl, y_vxl, z_vxl = scale_meas_to_vxls(anmly)
            ball = BallAnomaly_vxl(
                x=x_vxl, y=y_vxl, z=z_vxl, d=dia_dict[str(anmly.d)], γ=1
            )
            Y.append(voxel_ball(ball, Boundary()))

    X = np.array(X)
    Y = np.array(Y)
    return X, Y
