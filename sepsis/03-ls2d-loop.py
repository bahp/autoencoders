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
from sklearn.model_selection import ParameterGrid
from utils.settings import _FEATURES
from utils.settings import _LABELS
from utils.settings import _IMPUTERS
from utils.settings import _SCALERS
from utils.settings import _METHODS

# --------------------------------------------------
# Load data
# --------------------------------------------------
# Define path
PATH = './objects/datasets/tidy.csv'

# The name of the workbench. If None a new workbench name
# with the current datetime value will be created. Otherwise
# it will use an existing wbname and will only compute
# combos (imputer, scaler, method) that have not been
# computed before.
WBNAME = 'baseline'
if WBNAME is None:
    WBNAME = datetime.now().strftime('%y%m%d-%H%M%S')

# Create path
path = Path('./objects/results/%s' % WBNAME)

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
LABELS = _LABELS['set1']


grid = [
    {
        'imputer': ['simp'],
        'scaler': ['mmx', 'std'],
        'method': ['pca', 'tsne', 'icaf']
    },
    {
        'imputer': ['simp'],
        'scaler': ['mmx'],
        'method': ['lda', 'nmf']
    }
]
grid = ParameterGrid(grid)


# Loop
for i,e in enumerate(grid):

    # Create steps
    imputer = e.get('imputer')
    scaler = e.get('scaler')
    method = e.get('method')

    # Create variables
    folder = "%s-%s-%s" % (method, imputer, scaler)

    # Logging
    print("%s/%s. Computing... %s" % (i+1, len(grid), folder))

    # Check
    if (path / folder).exists():
        continue

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
    from pathlib import Path
    import plotly.express as px

    # Create directory
    (path / folder).mkdir(parents=True, exist_ok=True)

    # Save information
    # np.save(path / 'matrix.npy', matrix)
    # model.save(path / 'model.h5')
    data.to_csv(path / folder / 'encoded.csv')

    # Save plots
    for lbl in LABELS + FEATURES:
        # Create figure
        fig = px.scatter(data,
                         x='x', y='y', color=lbl,
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

        # Save
        fig.write_html(path / folder / ('graph.%s.html' % lbl))

        """
        # Example for 3d scatter plot.
        fig = px.scatter_3d(df,
            x='x', y='y', z='z', color=lbl,
            hover_data=df.columns.tolist(),
            title='e')
        """
