"""
Author: Bernard
Description:

"""


# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Specific
from pathlib import Path
from tensorflow import keras


# --------------------------------------------------
# Load data
# --------------------------------------------------
# Define path
PATH = Path('./objects/results/')
BENCH = PATH / 'lstm-simp-mmx-220804-130156'

# Load data
data = pd.read_csv(BENCH/ 'encoded.csv',
    dtype={'PersonID': 'str'},
    parse_dates=[])

# ---------------------
# Load model
# ---------------------
# Load
model = keras.models.load_model(BENCH / 'model.h5')

# Training history (using matplotlib)
#plt.plot(model.history["loss"], label="Training Loss")
#plt.plot(model.history["val_loss"], label="Validation Loss")
#plt.legend()
#plt.show()

# ---------------------
# Display
# ---------------------
# Libraries
import plotly.express as px

# Create figure
fig = px.scatter_3d(data,
    x='x', y='y', z='z', color='micro_code',
    hover_data=data.columns.tolist(),
    title='e')

# Display
fig.show()