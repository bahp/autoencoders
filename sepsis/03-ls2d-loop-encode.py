"""
Author: Bernard
Description:

"""

# Libraries
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from time import time
from pathlib import Path
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from utils.settings import _FEATURES
from utils.settings import _LABELS
from utils.settings import _IMPUTERS
from utils.settings import _SCALERS
from utils.settings import _METHODS

# ----------------------------------
# Methods
# ----------------------------------

def custom_metrics(est, X, y):
    """This method computes the metrics.

    .. todo: Check whether the X value received here has
             already gone through previous steps such as
             inputation of missing data or preprocessing.
             The X value is is raw data in fit.

    .. todo: Check the outcomes of encoder, encode_inputs,
             transform, ... so that they are all consistent.

    .. note: The X and y represent the values passed to the
             .fit(X, y) method below. In general, y is either
             the class (classification) or a value (regression).

    .. note: Since we received the estimator, we need to apply
             the prediction/transformation ourselves manually
             as shown below.

    .. note: The scoring function must return numbers, thus
             including any string value such as the uuid or
             an autogenerated uuid4 does not work here.

    Parameters
    ----------
    est: object
        The estimator or pipeline.
    X:  np.array (or dataframe)
        The X data in fit.
    y: np.array (or dataframe)
        The y data in fit.
    """
    # Transform
    y_embd = est.transform(X)
    m = custom_metrics_(X, y_embd, y)
    # Return
    return m


def custom_metrics_(y_true, y_pred, y, n=1000):
    """This method computes the metrics.

    .. note: computation of pairwise distances takes too
             long when datasets are big. Thus, instead
             select a random number of examples and compute
             on them.

    Parameters
    ----------
    y_true: np.array (dataframe)
        Array with original data (X).
    y_pred: np.array
        Array with transformed data (y_emb).
    y: np.array (dataframe)
        Array with the outcomes
    n: int
        The number of samples to use for distance metrics.

    Returns
    -------
    dict-like
        Dictionary with the scores.
    """
    return {}


# ----------------------------------
# Load arguments
# ----------------------------------
# Library
import argparse

# Default example (iris)
DEFAULT_YAML = './03-ls2d-loop.yaml'

# Load
parser = argparse.ArgumentParser()
parser.add_argument("--yaml", type=str, nargs='?',
                    const=DEFAULT_YAML, default=DEFAULT_YAML,
                    help="yaml configuration file")
args = parser.parse_args()



# ----------------------------------
# Set configuration
# ----------------------------------
# Library
from utils.utils import AttrDict

# Load configuration from file
with open(Path(args.yaml)) as file:
    CONFIG = AttrDict(yaml.full_load(file))

path = Path(CONFIG.outpath)

grid = [
    {
        'imputer': ['simp'],
        'scaler': ['mmx', 'std'],
        'method': ['icaf', 'iso', 'tsne', 'mds', 'spe'] #, 'umap']
    },
    {
        'imputer': ['simp'],
        'scaler': ['std'],
        'method': ['pca']
    },

    {
        'imputer': ['simp'],
        'scaler': ['mmx'],
        'method': ['lda', 'nmf']
    }
]

grid = [
    {
        'imputer': ['simp'],
        'scaler': ['std'],
        'method': ['sae']
    },
]

grid = ParameterGrid(grid)



# ----------------------------------
# Load data
# ----------------------------------
# Load data from csv file.
data = pd.read_csv(Path(CONFIG.datapath))

# .. note: This lines is ensuring that only those observations
#          which are complete (all features available) are used
#          for training the models.
data = data.dropna(how='any', subset=CONFIG.features)

# Show
print("\nData:")
print(data)
print("\nDtypes:")
print(data.dtypes)
print("\nOrganisms:")
print(data.micro_code.value_counts())

# Create X and y
X = data[CONFIG.features]
y = data[CONFIG.targets]


# -------------------------
# Loop
# -------------------------
# For each estimator
for i, e in enumerate(grid):

    # Create steps
    imputer = e.get('imputer')
    scaler = e.get('scaler')
    method = e.get('method')

    # Create variables
    folder = "%s-%s-%s" % (method, imputer, scaler)

    # Logging
    print("\n%s/%s. Computing... %s" % (i+1, len(grid), folder))

    # Get the param grid
    param_grid = CONFIG.params.get(method, {})
    param_grid = {'method__%s' % k:v for k,v in param_grid.items()}

    # Loop for each configuration
    for i,g in enumerate(ParameterGrid(param_grid)):
        hyper = 'hyper-%02d' % i

        # Check
        #if (path / folder / hyper).exists():
        #    continue

        # Create pipeline
        pipe = Pipeline([
            ('imputer', _IMPUTERS.get(imputer)),
            ('scaler', _SCALERS.get(scaler)),
            ('method', _METHODS.get(method))
        ])

        # Update pipeline parameters
        pipe.set_params(**g)

        # Transform
        t0 = time()
        encoded = pipe.fit_transform(X)
        t1 = time()

        # Add encoding
        data[['e%s' % i for i in range(encoded.shape[1])]] = encoded

        # Compute scores
        #scores = custom_metrics(pipe, X, y)

        # Create directory
        (path / folder / hyper).mkdir(parents=True, exist_ok=True)

        # Save information
        # Save any html figure
        # Save configuration file
        # np.save(path / 'matrix.npy', matrix)                      # matrix
        # model.save(path / 'model.h5')                             # model
        #df_encoded.to_csv(path / folder / hyper / 'encoded.csv')   # encoded
        #df_complete.to_csv(path / folder / hyper / 'complete.csv')  # complete
        data.to_csv(path / folder / hyper / 'data.csv')

        plt.scatter(data.e0, data.e1, s=3)
        plt.tight_layout()
        plt.savefig(path / folder / hyper / 'thumbnail.jpg')
        plt.close()