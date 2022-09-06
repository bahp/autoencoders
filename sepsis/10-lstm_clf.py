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

# Add positive vs negative
data['Positive'] = True
data.loc[data.micro_code=='CNS', 'Positive'] = False

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
    #'micro_code',
    #'death',
    #'day',
    'Positive'
]

# Create steps
imputer = 'simp'
scaler = 'std'

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
    monitor='val_loss', min_delta=1e-8, patience=30, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True)

# Define loss
loss = tf.keras.losses.CosineSimilarity(
    axis=-1, reduction="auto", name="cosine_similarity"
)
loss = 'mse'

# Variables
samples, timesteps, features = matrix.shape

model = LSTMClassifier(
    timesteps=timesteps,
    features=features,
    outputs=1
)

# Show model
print(model.summary())

print(matrix.shape)
print(matrix2.shape)
print(matrix2[:,:,-2].shape)

y = matrix2[:, -1, -2].astype('float32')
#y = data.Positive.astype('float32')

print(y.shape, y.sum())


# Fit model
model = model.fit(x=matrix, y=y,
    validation_data=(matrix, y),
    epochs=500, batch_size=16,
    shuffle=False, callbacks=[early_stop])


y_pred = model.predict(matrix)
y_test = y

print("\nPredictions:")
print(y_pred)
print(y_test)


#print(y_test)
#print(y_pred)
#print(y_pred.shape)
#print(np.sum(y_pred))
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

report = classification_report(y_test, y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred > 0.5)

print("\nReport:")
print(report)
print("\nConfusion Matrix:")
print(cm)
print("\nHistory:")
print(model.model_.history)


plt.title('Loss')
plt.plot(model.history_.history['loss'], label='train')
plt.plot(model.history_.history['val_loss'], label='test')
plt.legend()
plt.show();