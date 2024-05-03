import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.datagenerator import DataLoader, z_score
from tensorflow import distribute as dist


strategy = dist.MirroredStrategy(["GPU:0", "GPU:1", "GPU:2", "GPU:3"])
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# load vae model
from src.vae import vae_model

vae = vae_model()
vae.load_weights("models/vaes/vae_4.weights.h5")
vae.summary()

params = {
    "path": "../../../data/data_DD3D-EIT/datapool/",
    "mean_path": "../../../data/data_DD3D-EIT/datameans/",
    "eit_dim": 4096,
    "EIT_shape": "matrix",
    "supervised": "anomaly",  # "diameter", "material", "anomaly"
    "batch_size": 128,
    "shuffle": True,
}


X, Y = DataLoader(params)
X = z_score(X)

X = np.expand_dims(X, axis=3)
gamma = np.expand_dims(Y, axis=4)

del Y
print(X.shape, gamma.shape)

z = vae.encoder.predict(gamma)

X_train, X_test, z_train, z_test, gamma_train, gamma_test = train_test_split(
    X, z, gamma, test_size=0.05, random_state=42
)


with strategy.scope():

    def mapper_CNN(input_shape=(64, 64, 1), latent_dim=8, kernel=3):
        mapper_input = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(8, kernel, strides=(2, 4), padding="same")(
            mapper_input
        )
        x = tf.keras.layers.Conv2D(8, kernel, strides=(2, 4), padding="same")(x)
        x = tf.keras.layers.Conv2D(16, kernel, strides=(2, 4), padding="same")(x)
        x = tf.keras.layers.Conv2D(16, kernel, strides=(2, 4), padding="same")(x)
        x = tf.keras.layers.Flatten()(x)
        mapper_output = tf.keras.layers.Dense(latent_dim, activation="linear")(x)

        return tf.keras.Model(mapper_input, mapper_output)

    mapper = mapper_CNN()
    mapper.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    mapper.summary()

history = mapper.fit(X_train, z_train, batch_size=128, epochs=500)
mapper.save_weights("models/material_mapper.weights.h5")
np.savez("models/mapper_test.npz", X_test=X_test, z_test=z_test, gamma_test=gamma_test)
