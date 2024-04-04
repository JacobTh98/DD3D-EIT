from .dataprocessing import get_measured_potential
import numpy as np
from tensorflow.keras.utils import Sequence


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
    ):
        "Initialization"
        self.path = path
        self.mean_path = mean_path
        self.eit_dim = eit_dim
        self.supervised = supervised
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
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
        X = np.empty((self.batch_size, 64, 64, 1))  # EIT signal, self.eit_dim
        y = np.empty((self.batch_size, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load sample
            tmp = np.load(
                "{0:s}/sample_{1:06d}.npz".format(self.path, ID), allow_pickle=True
            )
            anomaly = tmp["anomaly"].tolist()
            raw_eit = get_measured_potential(tmp, shape_type="matrix")
            # raw_eit = get_measured_potential(tmp, shape_type="vector")
            mean_eit = np.load(
                f"{self.mean_path}mean_gnd_d_{anomaly.d}_{anomaly.material}.npy",
                allow_pickle=True,
            )
            pot = np.abs(raw_eit - np.reshape(mean_eit, (64, 64)))
            # pot = np.abs(raw_eit - mean_eit) # vector eit data shape
            X[
                i,
            ] = np.expand_dims(pot, axis=2)
            # X[
            #    i,
            # ] = np.expand_dims(pot, axis=1)
            if self.supervised == "diameter":
                y[
                    i,
                ] = anomaly.d
            elif self.supervised == "material":
                mat_dict = {"acryl": 0, "messing": 1}
                y[
                    i,
                ] = mat_dict[anomaly.material]

            # add normalization?
        return X, y
