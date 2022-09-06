"""
Author: Bernard
Description:

"""


# Libraries
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime
from sklearn.pipeline import Pipeline

# Own
from utils.settings import _FEATURES
from utils.settings import _IMPUTERS
from utils.settings import _SCALERS
from utils.settings import _METHODS
from utils.lstm.preprocess import create_lstm_matrix



# --------------------------------------------------
# Step 00 - Load data
# --------------------------------------------------
# Define path
PATH = Path('./objects/datasets/tidy.csv')

# Load data
data = pd.read_csv(PATH,
    #nrows=1000,
    dtype={'PersonID': 'str'},
    parse_dates=['date_collected',
                 'date_sample'])

# Keep raw copy
raw = data.copy(deep=True)

# Drop duplicates
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


# --------------------------------------------------
# Step 01 - Preprocess data
# --------------------------------------------------
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
matrix = create_lstm_matrix(data,
    features=FEATURES+LABELS,
    groupby='PersonID',
    w=5)

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

from utils.lstm.autoencoder import LSTMAutoencoder
from utils.lstm.autoencoder import LSTMClassifier

LATENT_DIM = 10

# Define early stop
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=1e-5, patience=15, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True)

# Define loss
loss = tf.keras.losses.CosineSimilarity(
    axis=-1, reduction="auto", name="cosine_similarity"
)
los = 'mse'

# Variables
samples, timesteps, features = matrix.shape


model = LSTMAutoencoder(
    timesteps=timesteps,
    features=features,
    latent_dim=LATENT_DIM,
    loss=loss
)

model = LSTMClassifier(
    timesteps=timesteps,
    features=features,
    outputs=2,
    loss=loss
)

# Show model
print(model.summary())

# Fit model
history = model.fit(x=matrix, y=matrix,
    validation_data=(matrix, matrix),
    epochs=500, batch_size=32,
    shuffle=True, callbacks=[early_stop])


aux = model.predict(matrix)

#print(aux)


encoder = model.encoder()
encoded = encoder.predict(matrix)

print(encoded.shape)
#print(encoded)

columns = ['x', 'y']
if LATENT_DIM == 3:
    columns += ['z']

df = pd.DataFrame(data=encoded,
    columns=['e%s' % i for i in range(LATENT_DIM)]) #, columns=columns)

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
            x='e0', y='e1', color=lbl,
            hover_data=df.columns.tolist(),
            title=str(path))
        fig.write_html(path / ('graph.%s.html' % lbl))

    if LATENT_DIM == 3:
        # Create figure
        fig = px.scatter_3d(df,
            x='e0', y='e1', z='e2', color=lbl,
            hover_data=df.columns.tolist(),
            title=str(path))
        # Save
        fig.write_html(path / ('graph.%s.html' % lbl))