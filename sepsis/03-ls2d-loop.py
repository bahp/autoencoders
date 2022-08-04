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

# --------------------------------------------------
# Load data
# --------------------------------------------------
# Define path
PATH = './objects/data.clean.csv'

# Load data
data = pd.read_csv(PATH,
    nrows=100000,
    parse_dates=['date_collected',
                 'date_sample'])

# Show
print("\nData:")
print(data)
print("\nDtypes:")
print(data.dtypes)
print("\nOrganisms:")
print(data.micro_code.value_counts())


# ------------------
# Compute quick PCA
# ------------------
# Configuration
FEATURES = _FEATURES['set1']

# Create steps
imputer = 'simp'
scaler = 'mmx'
method = 'pca'

# Create pipeline
pipe = Pipeline([
    ('imputer', _IMPUTERS.get(imputer)),
    ('scaler', _SCALERS.get(scaler)),
    ('method', _METHODS.get(method))
])

# Transform
data[['x', 'y']] = pipe.fit_transform(data[FEATURES])

# ---------------------
# Display
# ---------------------
# Libraries
import plotly.express as px

# Create figure
fig = px.scatter(data.dropna(how='any'),
    x='x', y='y', color='micro_code',
    hover_data=data.columns.tolist(),
    title=str(pipe))

# Enable buttons
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(label='Color Set 1',
                     method='update',
                     args=[{'color': ['red']}]
                     ),
                dict(label='Color Set 2',
                     method='update',
                     args=[{'marker.color': ['blue']}]
                     ),
            ]),
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.11, xanchor="left",
            y=1.1, yanchor="top"
        ),
    ]
)
# Display
fig.show()

# Save
now = datetime.now().strftime('%y%m%d-%H%M%S')
filename = "ls2d-%s-%s-%s-%s.html" % (
    imputer, scaler, method, now)
fig.write_html("./objects/figures/%s" % filename)
