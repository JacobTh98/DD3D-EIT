import numpy as np
from sklearn.model_selection import train_test_split
from src.classes import BallAnomaly, Boundary
from src.util import voxel_ball
import os
from tensorflow.keras.optimizers import Adam
from src.vae import vae_model

boundary = Boundary()
ball = BallAnomaly(x=1, y=1, z=1, r=5, γ=1)

X_all_anomalys = list()
radius_labels = list()

###
print("\n\n\t+----------------------------------+")
print("\t| Hyperparametertuning for the VAE |")
print("\t+----------------------------------+\n\n")
print("generate data")

γ = 1  # set object geometries to 1 and empty space to 0
for r in [2, 3, 4]:  # radius
    for x in np.arange(boundary.x_0 + ball.r, boundary.x_length - ball.r, 1):
        for y in np.arange(boundary.y_0 + ball.r, boundary.y_length - ball.r, 1):
            for z in np.arange(boundary.z_0 + ball.r, boundary.z_length - ball.r, 1):
                ball = BallAnomaly(x, y, z, r, γ)
                X_all_anomalys.append(voxel_ball(ball, boundary))
                radius_labels.append(r)

X_all_anomalys = np.array(X_all_anomalys)
radius_labels = np.array(radius_labels)

print(X_all_anomalys.shape, radius_labels.shape)

X_train, X_test, r_train, r_test = train_test_split(
    X_all_anomalys, radius_labels, train_size=0.90
)

hyperparameters = {
    "beta_s": np.linspace(0, 2, 11),
    "batch_s": [50, 100, 150, 200, 250, 300, 350, 400],
    "epoch_s": [10, 50, 100, 150, 200, 250],
    "savepath": "../models/vae_hpt_1/",
}

# try:
os.mkdir(hyperparameters["savepath"])
# except BaseException:
#    print("[Errno 17] File exists")

np.savez(f"{hyperparameters['savepath']}vae_testdata.npz", X_test=X_test, r_test=r_test)

for epoch in hyperparameters["epoch_s"]:
    for batch in hyperparameters["batch_s"]:
        for beta in hyperparameters["beta_s"]:
            print(f"{epoch=}, {batch=}, {beta=}")

            vae = vae_model(input_shape=(32, 32, 32, 1), beta=beta)
            vae.compile(optimizer=Adam())  # learning_rate = learning_rate

            history = vae.fit(
                np.expand_dims(X_train, 4),
                epochs=epoch,
                batch_size=batch,
            )
            # postprocessing and savings
            path = f"{hyperparameters['savepath']}ep_{epoch}_ba_{batch}_be_{int(10*np.round(beta,1))}"
            print(f"Save path: {path}")
            vae.save_weights(f"{path}.weights.h5")

            _, _, z = vae.encoder.predict(X_test)
            pred = vae.decoder.predict(z)
            pred = np.squeeze(pred, axis=4)
            pred = np.clip(pred, a_min=0, a_max=1)

            np.savez(
                f"{path}_saves.npz",
                history=history.history,
                hyperparameters=hyperparameters,
                pred=pred,
            )
