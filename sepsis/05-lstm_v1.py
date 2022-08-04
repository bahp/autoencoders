"""
Author: Bernard
Description:

"""


# Libraries
import pandas as pd
import numpy as np

from datetime import datetime
from sklearn.pipeline import Pipeline
from utils import _FEATURES, _IMPUTERS, _SCALERS, _METHODS
from utils import lstm_matrix


# --------------------------------------------------
# Load data
# --------------------------------------------------
# Define path
PATH = './objects/data.clean.csv'

# Load data
data = pd.read_csv(PATH,
    #nrows=1000,
    dtype={'PersonID': 'str'},
    parse_dates=['date_collected',
                 'date_sample'])

raw = data.copy(deep=True)

data = data.drop_duplicates()

# Show
print("\nData:")
print(data)
print("\nDtypes:")
print(data.dtypes)
print("\nOrganisms:")
print(data.micro_code.value_counts())
print("\nUnique")
print(data.nunique())

# Configuration
FEATURES = _FEATURES['set1']
LABELS = [
    'micro_code',
    'death',
    'day',
]

# Create steps
imputer = 'simp'
scaler = 'mmx'

# Format data
data[FEATURES] = _IMPUTERS.get(imputer).fit_transform(data[FEATURES])
data[FEATURES] = _SCALERS.get(scaler).fit_transform(data[FEATURES])

# Format matrix
matrix = lstm_matrix(data,
    features=FEATURES+LABELS,
    groupby='PersonID')

# Show matrix
print("\nMatrix shape: %s\n\n" % str(matrix.shape))

matrix2 = matrix.copy()
matrix = matrix[:,:,:-(len(LABELS)+1)].astype('float32')


# ---------------------------
# Show windows
# ---------------------------
# Libraries
import matplotlib.pyplot as plt

PLOT = False

if PLOT:
    # Create axes
    fig, axes = plt.subplots(10, 10)
    axes = axes.flatten()

    # Plot
    for i in range(30):
        axes[i].imshow(matrix[i,:,:])



# -------------------------------------
# Quick Keras Model
# -------------------------------------
# Create
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt

"""
# define input sequence
seq_in = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# reshape input into [samples, timesteps, features]
n_in = len(seq_in)
seq_in = seq_in.reshape((1, n_in, 1))
# prepare output sequence
seq_out = seq_in[:, 1:, :]
n_out = n_in - 1
# define encoder
visible = Input(shape=(n_in,1))
encoder = LSTM(100, activation='relu')(visible)
# define reconstruct decoder
decoder1 = RepeatVector(n_in)(encoder)
decoder1 = LSTM(100, activation='relu', return_sequences=True)(decoder1)
decoder1 = TimeDistributed(Dense(1))(decoder1)
# define predict decoder
decoder2 = RepeatVector(n_out)(encoder)
decoder2 = LSTM(100, activation='relu', return_sequences=True)(decoder2)
decoder2 = TimeDistributed(Dense(1))(decoder2)
# tie it together
model = Model(inputs=visible, outputs=[decoder1, decoder2])
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(seq_in, [seq_in,seq_out], epochs=300, verbose=0)
# demonstrate prediction
yhat = model.predict(seq_in, verbose=0)
print(yhat)

# Show model
print("\nModel:")
model.summary()
print("\nPrediction:")
print(yhat)
print(seq_in)

print(seq_in.reshape(1,-1))

pred = pd.DataFrame()
pred['in'] = seq_in.reshape(-1,1)
#pred['dec1'] = yhat[0].reshape(-1,1)

print(pred)
"""


# -----------------------------------------------------------------
# Example of LSTM autoencoder
# -----------------------------------------------------------------
# REF: https://towardsdatascience.com/using-lstm-autoencoders-on-multidimensional-time-series-data-f5a7a51b29a1

# The model begins with an Encoder: first, the input layer. The input layer is
# an LSTM layer. This is followed by another LSTM layer, of a smaller size. Then,
# I take the sequences returned from layer 2 — then feed them to a repeat vector.
# The repeat vector takes the single vector and reshapes it in a way that allows
# it to be fed to our Decoder network which is symmetrical to our Encoder. Note
# that it doesn’t necessarily have to be symmetrical, but this is standard practice.

import torch
from keras import metrics
import keras
import tensorflow as tf

LATENT_DIM = 2

# Define earl stop
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=1e-3, patience=15, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True)

# Define loss
loss = tf.keras.losses.CosineSimilarity(
    axis=-1, reduction="auto", name="cosine_similarity"
)

# Variables
samples, timesteps, features = matrix.shape

# Construct model
model = Sequential()
model.add(
    LSTM(64,
        kernel_initializer='he_uniform',
        batch_input_shape=(None, timesteps, features),
        return_sequences=True,
        name='encoder_1')
)
model.add(
    LSTM(32, activation='relu',
        kernel_initializer='he_uniform',
        return_sequences=True,
        name='encoder_2')
)
model.add(
    LSTM(LATENT_DIM,
        kernel_initializer='he_uniform',
        return_sequences=False,
        name='encoder_3')
) # return false and Repeat Vector
model.add(
    RepeatVector(timesteps,
        name='encoder_decoder_bridge')
)
model.add(
    LSTM(LATENT_DIM,
        kernel_initializer='he_uniform',
        return_sequences=True,
        name='decoder_1')
)
model.add(
    LSTM(32, activation='relu',
        kernel_initializer='he_uniform',
        return_sequences=True,
        name='decoder_2')
)
model.add(
    LSTM(64,
        kernel_initializer='he_uniform',
        return_sequences=True,
        name='decoder_3')
)
model.add(TimeDistributed(Dense(features)))
#model.compile(loss="mse", optimizer='adam')


# Compile and fill
model.compile(loss=loss, optimizer='adam')
model.build()

# Show model
print(model.summary())

# Fit model
history = model.fit(x=matrix, y=matrix,
    validation_data=(matrix, matrix),
    epochs=50, batch_size=16,
    shuffle=False, callbacks=[early_stop])


aux = model.predict(matrix)

#print(aux)


encoder = Model(inputs=model.inputs, outputs=model.layers[2].output)

encoded = encoder.predict(matrix)

print(encoded.shape)
#print(encoded)

columns = ['x', 'y']
if LATENT_DIM == 3:
    columns += ['z']

df = pd.DataFrame(data=encoded, columns=columns)

# Add labels
df['PersonID'] = matrix2[:,:,-1][:,1] # Get first in row because all ae the same
for i,lbl in enumerate(LABELS[::-1]):
    idx = -(i+2)
    df[lbl] = matrix2[:,:,idx][:,1]


# ----------------------------------
# Save
# ----------------------------------
# Libraries
from pathlib import Path
import plotly.express as px

# Create variables
now = datetime.now().strftime('%y%m%d-%H%M%S')
name = "lstm-%s-%s-%s" % (imputer, scaler, now)
path = Path('./objects/results/%s' % name)

# Create directory
path.mkdir(parents=True, exist_ok=True)

# Save information
np.save(path / 'matrix.npy', matrix)
model.save(path / 'model.h5')
df.to_csv(path / 'encoded.csv')

# Save plots
for lbl in LABELS + ['PersonID']:

    if LATENT_DIM == 2:
        # Create figure
        fig = px.scatter(df,
            x='x', y='y', color=lbl,
            hover_data=df.columns.tolist(),
            title='e')
        fig.write_html(path / ('graph.%s.html' % lbl))

    if LATENT_DIM == 3:
        # Create figure
        fig = px.scatter_3d(df,
            x='x', y='y', z='z', color=lbl,
            hover_data=df.columns.tolist(),
            title='e')
        # Save
        fig.write_html(path / ('graph.%s.html' % lbl))