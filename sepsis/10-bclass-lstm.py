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
    PATH = Path('./objects/datasets/set1/data.csv')

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
        'HCT',
        #'LY',
        #'MCV',
        #'PLT',
        #'RDW',
        'WBC',
        'CRP'
    ]
    FEATURES = [

    ]
    LABELS = [
        'pathogenic'
    ]


    def check_time_gaps(df, by, date):
        unit = pd.Timedelta(1, unit="d")
        return (df.groupby(by=by)[date].diff() > unit).sum() > 1

    def resample_01(df, **kwargs):
        return df.droplevel(0) \
            .resample(**kwargs) \
            .asfreq() \
            .ffill()

    rsmp = data.copy(deep=True) \
        .set_index(['PersonID', 'date_collected']) \
        .groupby(level=0) \
        .apply(resample_01, rule='1D') \
        .reset_index()

    # Show
    print("\nLoaded data:")
    print(data[['PersonID', 'date_collected'] + FEATURES + LABELS])
    print("\nResampled data (ffilled):")
    print(rsmp[['PersonID', 'date_collected'] + FEATURES + LABELS])
    print("Has time gaps: %s" %
        check_time_gaps(df=rsmp, by='PersonID', date='date_collected'))

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

    # Show matrix
    print("\nMatrix shape (samples, timesteps, features): %s" % str(matrix.shape))

    # Save
    np.save('./10.matrix.npy', matrix)

else:
    # Load matrix
    matrix = np.load('./10.matrix.npy', allow_pickle=True)

# The matrix when saved has an additional column in the
# features space which represents the label for that
# window. Thus, it is splitted in the following lines:
# Create X and y
X = matrix[:,:,:-1].astype('float32')
y = matrix[:,-1,-1].astype('float32')

# Show matrix
print("\nMatrix shape (samples, timesteps, features): %s" % str(X.shape))
print("\nNumber of positive samples: %s out of %s" % (int(y.sum()), len(y)))

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
from utils.lstm.autoencoder import MT_LSTM

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

"""
# --------------------------------------------
# Conv1D
# --------------------------------------------
# Libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# Model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu',
    input_shape=(timesteps, features)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy',
    optimizer='adam', metrics=['accuracy'])
"""

#model = BinaryClassifierLSTMV1(
#    timesteps=timesteps,
#    features=features-1,
#    outputs=1
#)

#model = LSTMClassifier(
#    timesteps=timesteps,
#    features=features-1,
#    outputs=1
#)

model = MT_LSTM(
    timesteps=timesteps,
    features=features-1,
    outputs=1
)

#model = StackedLSTM_MLM(
#    timesteps=timesteps,
#    features=features-1,
#    outputs=1
#)

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
    epochs=20, batch_size=32,
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