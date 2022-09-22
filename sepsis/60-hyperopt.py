# example of hyperopt-sklearn for the sonar classification dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from hpsklearn import HyperoptEstimator
from hpsklearn import any_classifier
from hpsklearn import any_preprocessing
from hyperopt import tpe


"""
Author: Bernard
Description:

    This script creates a .csv file with all the scores computed
    for different dimensionality reduction algorithms and various
    hyper-parameter configurations.

    .. note: It uses GridSearch and thus the models that do not
             have the method transform fail when computing the
             scores using 'custom_metrics'.

"""

# Libraries
import yaml
import pickle
import pandas as pd
import numpy as np

from time import time
from pathlib import Path
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from imblearn.pipeline import Pipeline

# Libraries
from utils.prep import IQRTransformer

from utils.settings import _FEATURES
from utils.settings import _LABELS
from utils.settings import _IMPUTERS
from utils.settings import _SAMPLERS
from utils.settings import _SCALERS
from utils.settings import _METHODS
from utils.settings import _ALL

# ----------------------------------
# Methods
# ----------------------------------

def resample_01(df, ffill=True):
    return df.droplevel(0) \
        .resample('1D').asfreq() \
        #.ffill() # filling missing!

# Add delta values
def delta(x, features=None, periods=1):
    """Computes delta (diff between days)

    Parameters
    ----------
    x: pd.dataFrame
        The DataFrame
    features: list
        The features to compute deltas
    periods: int
        The periods.
    Returns
    -------
    """
    aux = x[features].diff(periods=periods)
    aux.columns = ['%s_d%s' % (e, periods)
        for e in aux.columns]
    return aux

def custom_metrics(est, X, y):
    """This method computes the metrics.

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
    y_pred = est.predict(X)
    y_prob = est.predict_proba(X)
    # Metrics
    m = custom_metrics_(y, y_pred, y_prob)
    # Additional (hack!)
    #hos = custom_metrics_(y_hos,
    #    est.predict(X_hos),
    #    est.predict_proba(X_hos))
    #m.update({'hos_%s'%k:v for k,v in hos.items()})
    # Return
    return m


def custom_metrics_(y, y_pred, y_prob, n=1000):
    """This method computes the metrics.

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
    # Libraries
    from scipy.stats import gmean
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_recall_fscore_support

    # Compute confusion matrix (binary only)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    # Compute precision recall and support.
    prec, recall, fscore, support = \
        precision_recall_fscore_support(y, y_pred)

    # Show classification report
    #print(classification_report(y, y_pred, zero_division=0))

    # Create dictionary
    d = {}
    #d['report'] = classification_report(y, y_pred)
    d['accuracy'] = accuracy_score(y, y_pred)
    d['roc_auc'] = roc_auc_score(y, y_prob[:,1])
    d['sens'] = recall_score(y, y_pred, pos_label=1)
    d['spec'] = recall_score(y, y_pred, pos_label=0)
    d['gmean'] = gmean([d['sens'], d['spec']])
    d['tn'] = tn
    d['tp'] = tp
    d['fp'] = fp
    d['fn'] = fn

    for i,v in enumerate(support):
        d['support%s' % i] = v

    #d['aucroc_ovo'] = roc_auc_score(y, y_prob, average='weighted', multi_class='ovo')
    #d['aucroc_ovr'] = roc_auc_score(y, y_prob, average='weighted', multi_class='ovr')

    # Return
    return d


def create_fe_dataframe_from_yaml(data, cfg):
    """Construct frame from config.

    Parameters
    ----------

    Returns
    -------
    """
    # Check is AttrDict
    #if isinstance(yaml, 'str'):
    #    # Load file

    # Get mode
    mode = cfg.get('mode', 'normal')

    # Get feature engineering config
    fe_cfg = cfg.feature_engineering.get(mode, {}).copy()

    # Features to aggregate
    features_ = fe_cfg.get('features', 'all')
    if features_ == 'all':
        fe_cfg['features'] = cfg.features

    # Add label
    fe_cfg['label'] = cfg.outcome
    fe_cfg['label_agg'] = 'max'
    fe_cfg['groupby'] = cfg.pid

    # Compute
    if mode == 'aggregation':
        return create_df_aggregated(data, **fe_cfg)
    if mode == 'delta':
        return create_df_delta(data, **fe_cfg)
    elif mode == 'normal':
        return data[cfg.features + [cfg.outcome]].copy(deep=True)

    # Return data
    return data

def create_df_delta(data, features, label, groupby=None,
                    date=None,
                    **kwargs):
    """Construct the aggregated DataFrame.

    .. note: It needs to ensure re-sample first.
    .. note: At the moment just return because it was
             implemented in 01-prepare-data.py but it
             should be moved at some point to be
             re-usable.

    Parameters
    ----------

    Returns
    -------

    """
    """
    # Columns to keep
    keep = features + [groupby, date, label]

    # Format DataFrame
    df = data[keep].copy(deep=True)
    df[date] = pd.to_datetime(df[date])
    df = df.set_index([groupby, date])

    # Re-sample
    rsmp = df \
        .groupby(level=0) \
        .apply(resample_01) \

    rsmp = rsmp.droplevel(level=1)

    print(rsmp)

    import sys
    sys.exit()

    # Compute delta
    df_1 = rsmp.groupby(groupby) \
        .apply(delta,
            periods=1,
            features=features
        )

    # Return
    return pd.concat([df_1, rsmp], axis=1)
    """

    #print(features)
    #print(label)
    df = data.copy(deep=True)
    features_delta = ['%s_d1' % e for e in features]
    df = df[features + features_delta + [label]]
    df = df.dropna(how='any')
    return df


def create_df_aggregated(data, features, methods, label,
        label_agg='max', groupby=None, **kwargs):
    """Construct the aggregated DataFrame.

    Parameters
    ----------

    Returns
    -------

    """
    # Create DataFrame
    params = {e: methods for e in features}
    params[label] = [ label_agg ]
    df = data.groupby(groupby).agg(params)
    df.columns = ['_'.join(col).strip()
        for col in df.columns.values]
    df.rename(columns={'%s_max' % label: label}, inplace=True)
    return df


# ----------------------------------
# Load arguments
# ----------------------------------
# Library
import argparse

# Create timestamp
TIMESTAMP = datetime.now().strftime('%y%m%d-%H%M%S')
FILENAME = 'gridsearch-%s' % TIMESTAMP

# Default example (iris)
DEFAULT_YAML = './04-bclass-loop.yaml'

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
import shutil
from utils.utils import AttrDict

# Load configuration from file
with open(Path(args.yaml)) as file:
    CONFIG = AttrDict(yaml.full_load(file))

# Check CONFIG is valid!

# Workbench path
WORKBENCH = Path(CONFIG.outpath) / CONFIG.mode / TIMESTAMP

# Create directory
WORKBENCH.mkdir(parents=True, exist_ok=True)

# Copy configuration
shutil.copyfile(
    Path(args.yaml),
    WORKBENCH / ('%s.yaml' % FILENAME))

# Show
print("\nCreate workbench... %s" % WORKBENCH)





# ----------------------------------
# Load data
# ----------------------------------
# Load data from csv file.
data = pd.read_csv(Path(CONFIG.datapath))

# Filter by days
data = data[(data.day >= CONFIG.filter.day.start)]
data = data[(data.day <= CONFIG.filter.day.end)]

# Filter date
data = data[data.date_collected < CONFIG.filter.date_collected.end]

# .. note: This line is ensuring that only those observations
#          which are complete (all features available) are used
#          for training the models.
data = data.dropna(how='any', subset=CONFIG.features)

# Show data
print("\nData:")
print(data)
print("\nDtypes:")
print(data.dtypes)
print("\nOrganisms:")
print(data.micro_code.value_counts())
print("\nOutcome:")
print(data[CONFIG.outcome].value_counts())

# Create binary problem
# ---------------------
data['label'] = data.pathogenic

# Create multi-class problem
# --------------------------
# We are doing this quick conversion in pandas, but when creating
# the final model with the pre-processing steps it is better to
# use a LabelEncoder from sci-kits.
data['label'] = LabelEncoder().fit_transform(data.micro_code)




# ----------------------------------
# Loop
# ----------------------------------

# Create X and y
#X = data[CONFIG.features].to_numpy()
#y = data[CONFIG.outcomes].to_numpy().ravel()

# Create the DataFrame from the YAML configuration. These
# are the following options:
#  - aggregation
#  - delta
#  - windows
#  - normal
#
# .. note: The DataFrame will only contain the features
#          used for training and the last column will be
#          the label (outcome) (REVIEW!!).
df = create_fe_dataframe_from_yaml(data, CONFIG)

print(df)

# Create X and y
X = df.iloc[:, :-1]
y = df[CONFIG.outcome]



from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV

import xgboost as xgb

# define search space
params = dict()
params['C'] = (1e-6, 100.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['degree'] = (1,5)
params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']

# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

estimator = xgb.XGBClassifier()

# define the search
search = BayesSearchCV(estimator=SVC(), search_spaces=params, n_jobs=-1, cv=cv)

# perform the search
search.fit(X, y)

# report the best result
print(search.best_score_)
print(search.best_params_)