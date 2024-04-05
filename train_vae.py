import numpy as np
from sklearn.model_selection import train_test_split
from src.classes import BallAnomaly, Boundary
from src.util import voxel_ball

from tensorflow.keras.optimizers import Adam
from src.vae import vae_model

boundary = Boundary()
ball = BallAnomaly(x=1,y=1,z=1,r=5,γ=1)

X_all_anomalys = list()
radius_labels = list()

###
print("generate data")

γ = 1  # set object geometries to 1 and empty space to 0
for r in [3, 4, 5]:  # radius
    for x in np.arange(boundary.x_0 + ball.r, boundary.x_length - ball.r, 1):
        for y in np.arange(boundary.y_0 + ball.r, boundary.y_length - ball.r, 1):
            for z in np.arange(boundary.z_0 + ball.r, boundary.z_length - ball.r, 1):
                ball = BallAnomaly(x, y, z, r, γ)
                X_all_anomalys.append(voxel_ball(ball, boundary))
                radius_labels.append(r)

X_all_anomalys = np.array(X_all_anomalys)
radius_labels = np.array(radius_labels)

X_train, X_test, r_train, r_test = train_test_split(
    X_all_anomalys, radius_labels, train_size=0.95
)

np.savez("../models/vae_testdata.npz", X_test=X_test, r_test=r_test)

print(X_all_anomalys.shape, radius_labels.shape)


vae = vae_model(input_shape=(32, 32, 32, 1), beta=0.4)

# learning_rate = 0.0001
# sgd = SGD(learning_rate = learning_rate_1, momentum = 0.9, nesterov = True)
# vae.compile(optimizer = SGD())# , metrics = ['accuracy']
vae.compile(optimizer=Adam())  # learning_rate = learning_rate

epochs = 150
batch_size = 100

# cb = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=0, start_from_epoch=47)

history = vae.fit(
    np.expand_dims(X_train, 4),
    epochs=epochs,
    batch_size=batch_size,
    # callbacks=[cb],
)

vae.save_weights("../models/vae.weights.h5")
