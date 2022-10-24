# Libraries
import tensorflow as tf
import numpy as np

# Layers
layers = tf.keras.layers

timesteps = 32
channels = 16;
x = np.random.randn(100, timesteps, channels)

binary_y = np.random.randint(0, 2, size=(x.shape[0], 1))
reg_y = np.random.randn(x.shape[0], 1)

inputs = layers.Input(shape=(timesteps, channels))
hidden = layers.LSTM(32)(inputs)
out1 = layers.Dense(1, activation="sigmoid", name="binary_out")(hidden)
out2 = layers.Dense(1, activation=None, name="reg_out")(hidden)

model = tf.keras.Model(inputs=inputs, outputs=[out1, out2])

model.compile(
    loss={
        "binary_out": "binary_crossentropy",
        "reg_out":"mse"
    }, optimizer='adam',
    metrics={"binary_out":"accuracy"}
)

model.fit(x, [binary_y, reg_y], epochs=10)