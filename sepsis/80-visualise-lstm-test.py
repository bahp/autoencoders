# Libraries
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path

# Define path
PATH = Path('./objects/results/classification/lstm')
PATH = PATH / '221012-181435'
PATH = PATH / 'matrix.-10_3.-10_3.w5.simp.std.bare'

# Load data
train = np.load(PATH / 'data/train.npy', allow_pickle=True)
test = np.load(PATH / 'data/test.npy', allow_pickle=True)

# Create X and y
X_train = train[:, :, :-1].astype('float32')
y_train = train[:, -1, -1:].astype('float32')

X_test = test[:, :, :-1].astype('float32')
y_test = test[:, -1, -1:].astype('float32')

X = np.vstack((X_train, X_test)).astype('float32')
y = np.vstack((y_train, y_test)).astype('float32')

# Load model
model = tf.keras.models.load_model(PATH / 'model.h5')

# Compute y_pred
y_pred = model.predict(X)

# Create DataFrame
df = pd.DataFrame(
    data=np.hstack((y, y_pred, y_pred > 0.5)),
    columns=['y_true', 'y_prob', 'y_pred']
)
df['set'] = 'train'
df.loc[y_train.shape[0]:, 'set'] = 'test'

# Add tp, tn, fp, fn labels
from utils.pandas.apply import cmt
df['cmt'] = df.apply(cmt, axis=1)

# --------------------------
# Plot
# --------------------------
# Libraries
import plotly.express as px

# Display
fig = px.box(df, x='cmt', y='y_prob', color='set')
fig.show()
