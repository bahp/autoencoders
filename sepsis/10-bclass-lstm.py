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


CREATE_MATRIX = True

if CREATE_MATRIX:

    # --------------------------------------------------
    # Step 00 - Load data
    # --------------------------------------------------
    # Define path
    PATH = Path('./objects/datasets/test-fbc-pct-crp-wbs/data.pctc.csv')

    # Load data
    data = pd.read_csv(PATH,
        nrows=1000000,
        dtype={'PersonID': 'str'},
        parse_dates=['date_collected'])

    # Keep raw copy
    raw = data.copy(deep=True)

    # Drop duplicates
    data = data.drop_duplicates()

    # --------------------------------------------------
    # Step 01 - Preprocess data
    # --------------------------------------------------
    # Configuration
    FEATURES =  [
        #'HCT',
        #'LY',
        #'MCV',
        #'PLT',
        #'RDW',
        #'WBC',
        'CRP'
    ]
    LABELS = [
        'pathogenic'
    ]

    # Show
    print("\nData:")
    print(data[['PersonID'] + FEATURES + LABELS])

    # Create steps
    imputer = 'simp'
    scaler = 'std'

    # Format data
    data[FEATURES] = _IMPUTERS.get(imputer).fit_transform(data[FEATURES])
    data[FEATURES] = _SCALERS.get(scaler).fit_transform(data[FEATURES])

    # Format matrix with shape (samples, timestamps, features)
    matrix = create_lstm_matrix(data,
        features=FEATURES+LABELS,
        groupby='PersonID',
        w=5)

    # Save
    np.save('./10.matrix.npy', matrix)

else:
    # Load matrix
    matrix = np.load('./10.matrix.npy', allow_pickle=True)

# Create X and y
X = matrix[:,:,:-1].astype('float32')
y = matrix[:,-1,-1].astype('float32')

# Show matrix
print("\nMatrix shape: %s\n\n" % str(matrix.shape))
print(X)
print(y)
print("%s out of %s" % (y.sum(), len(y)))

#matrix = matrix[:,:,:-(len(LABELS)+1)].astype('float32')
#y = matrix2[:, -1, -2].astype('float32')


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
from utils.lstm.autoencoder import BidirectionalLSTM_MLM
from utils.lstm.autoencoder import StackedLSTM_MLM

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

model = StackedLSTM_MLM(
    timesteps=timesteps,
    features=features-1,
    outputs=1
)

# Show model
print(model.summary())

"""
print(matrix.shape)
print(matrix2.shape)
print(matrix2[:,:,-2].shape)

y = matrix2[:, -1, -2].astype('float32')
#y = data.Positive.astype('float32')

print(y.shape, y.sum())
"""

# Fit model
model = model.fit(x=X, y=y,
    validation_data=(X, y),
    epochs=500, batch_size=16,
    shuffle=False, callbacks=[early_stop])


y_pred = model.predict(X)
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