"""
Author: Bernard
Description:

"""

# Libraries
import yaml
import shutil
import pandas as pd
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

from pandas_profiling import ProfileReport


# ----------------------------------
# Constants
# ----------------------------------
# The structure of the folder will look as follows:
#
# workbench
#   |- data.csv
#   |- report.html
#   |- settings.yaml
#   |- pca-std-simp
#      |- hyper-00
#         |- data.csv
#         |- parameters.txt
#         |- thumbnail.jpg

# The names of the files.
NAME_REPORT_FILE = 'report.html'
NAME_DATA_FILE = 'data.csv'
NAME_CONFIG_FILE = 'settings.yaml'
NAME_PARAMS_FILE = 'parameters.txt'
NAME_THUMBNAIL = 'thumbnail.jpg'

# Grid.
grid = [
    {
        'imputer': ['simp', 'iimp'],
        'scaler': ['mmx', 'std'],
        'method': ['icaf', 'iso', 'tsne', 'mds', 'spe', 'sae', 'umap']
    },
    {
        'imputer': ['simp', 'iimp'],
        'scaler': ['std'],
        'method': ['pca']
    },

    {
        'imputer': ['simp', 'iimp'],
        'scaler': ['mmx'],
        'method': ['lda', 'nmf']
    }
]

"""
grid = [
    {
        'imputer': ['simp'],
        'scaler': ['std'],
        'method': ['umap']
    },
]
"""
grid = ParameterGrid(grid)


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

# Create output path
now = datetime.now().strftime('%y%m%d-%H%M%S')
path = Path('%s-%s' % (CONFIG.outpath, now))


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

# Create directory
path.mkdir(parents=True, exist_ok=True)

# Save common information
data.to_csv(path / NAME_DATA_FILE)

# Create report
profile = ProfileReport(data,
    title="Sepsis Dataset",
    explorative=False,
    minimal=True)

# Save report
profile.to_file(path / NAME_REPORT_FILE)

# Copy configuration
shutil.copyfile(Path(args.yaml), path / NAME_CONFIG_FILE)



# -------------------------
# Loop
# -------------------------
# Create X and y
X = data[CONFIG.features]
y = data[CONFIG.targets]

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

        # Create pipeline
        pipe = Pipeline([
            ('imputer', _IMPUTERS.get(imputer)),
            ('scaler', _SCALERS.get(scaler)),
            ('method', _METHODS.get(method))
        ])

        # Update pipeline parameters
        pipe.set_params(**g)

        # Transform
        encoded = pipe.fit_transform(X)

        # Add encoding
        data[['e%s' % i for i in range(encoded.shape[1])]] = encoded

        # Compute scores
        #scores = custom_metrics(pipe, X, y)

        # Create directory
        (path / folder / hyper).mkdir(parents=True, exist_ok=True)

        # Save information
        # Save any html figure
        # np.save(path / 'matrix.npy', matrix)                       # matrix
        # model.save(path / 'model.h5')                              # model
        #df_encoded.to_csv(path / folder / hyper / 'encoded.csv')    # encoded
        #df_complete.to_csv(path / folder / hyper / 'complete.csv')  # complete
        data.to_csv(path / folder / hyper / 'data.csv')

        # Write configuration
        with open(path / folder / hyper / NAME_PARAMS_FILE, 'w') as f:
            f.write(str(g))

        # Create thumbnail
        plt.scatter(data.e0, data.e1, s=3)
        plt.title('%s %s' % (folder, hyper), fontsize=20)
        plt.tight_layout()
        plt.savefig(path / folder / hyper / NAME_THUMBNAIL)
        plt.close()