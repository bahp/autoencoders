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

from skopt import BayesSearchCV

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
    m['score'] = est.score(X, y)
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
    strategy = cfg.get('strategy', 'normal')

    # Get feature engineering config
    fe_cfg = cfg.feature_engineering.get(strategy, {}).copy()

    # Features to aggregate
    features_ = fe_cfg.get('features', 'all')
    if features_ == 'all':
        fe_cfg['features'] = cfg.features

    # Add label
    fe_cfg['label'] = cfg.outcome
    fe_cfg['label_agg'] = 'max'
    fe_cfg['groupby'] = cfg.pid

    # Compute
    if strategy == 'aggregation':
        return create_df_aggregated(data, **fe_cfg)
    if strategy == 'delta':
        return create_df_delta(data, **fe_cfg)
    elif strategy == 'normal':
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

    # Filter data
    df = data.copy(deep=True)
    features_delta = ['%s_d1' % e for e in features]
    df = df[features + features_delta + [label]]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(how='any')

    # Return
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

    # Return
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
"""
   .. note: Possible to check whether configuration 
            is valid.
"""
# Library
import shutil
from utils.utils import AttrDict

# Load configuration from file
with open(Path(args.yaml)) as file:
    CONFIG = AttrDict(yaml.full_load(file))


# Check CONFIG is valid!

# Workbench path
WORKBENCH = Path(CONFIG.outpath) / CONFIG.strategy / TIMESTAMP

# Create directory
WORKBENCH.mkdir(parents=True, exist_ok=True)

# Copy configuration
shutil.copyfile(
    Path(args.yaml),
    WORKBENCH / ('%s.yaml' % FILENAME)
)

# Show
print("\nCreate workbench... %s" % WORKBENCH)




# ----------------------------------
# Load data
# ----------------------------------
# Load data from csv file.
data = pd.read_csv(Path(CONFIG.datapath))

# Filter by days
#data = data[(data.day >= CONFIG.filter.day.start)]
#data = data[(data.day <= CONFIG.filter.day.end)]

# Filter by date_range (excluding covid)
#data = data[data.date_collected < CONFIG.filter.date_collected.end]


"""
# ---------------------------
# Filtering
# ---------------------------
# This filtering is mostly needed for the temporal
# approaches such as LSTMs. For the static approaches
# it could be ignored. The only option that might be
# useful is day_range.

# Import own filter method
from utils.utils import filter_data

# Apply filtering
filt = filter_data(rsmp,
    day_range=(-31, 3),
    window_size_kws=dict(
        min_w=10, col=config.pid),
    window_density_kws=dict(
        day_range=D_TUPLE,
        features=FEATURES, min_density=0.75
    ))
"""

# .. note: This line is ensuring that only those observations
#          which are complete (all features available) are used
#          for training the models.
#
# This reduces the amount of samples considerably. What about
# removing only those with more than n Nan and then filling
# using either ffill or simp?
#data = data.dropna(how='any', subset=CONFIG.features)

data = data.dropna(thresh=7, subset=CONFIG.features)
data[CONFIG.features] = data \
    .groupby('PersonID')[CONFIG.features].ffill()

#if 'sex' in data:
#    data.sex = data.sex.astype(int)

# Show data
print("\nData:")
print(data[CONFIG.features])
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
#df = create_fe_dataframe_from_yaml(data, CONFIG)


def get_data_normal_from_yaml(data, cfg={}):
    """

    Parameters
    ----------
    data
    cfg

    Returns
    -------

    """
    # Columns to keep
    keep = cfg.features + [cfg.outcome] + cfg.metadata
    return data[keep].copy(deep=True)


def get_data_agg_from_yaml(data, cfg={}):
    """Creates agg DataFrame.

    Parameters
    ----------
    data
    features
    labels

    Returns
    -------

    """
    # Libraries
    from utils.sklearn.preprocessing import AggregateTransformer

    # Define functions
    aggmap = {k: ['min', 'max', 'median'] for k in cfg.features}
    aggmap[cfg.outcome] = 'max'
    aggmap['set'] = 'first'

    # Create transformer
    agg = AggregateTransformer(by='PersonID',
        aggmap=aggmap, include=list(aggmap.keys()))

    # Transform data
    data_agg = agg.fit_transform(data)

    # Rename
    rename = {
        '%s_max' % cfg.outcome: cfg.outcome,
        '%s_first' % 'set': 'set'
    }
    data_agg.rename(columns=rename, inplace=True)

    # Return
    return data_agg

def get_data_delta_from_yaml(data, cfg):
    """"""
    # Libraries
    from utils.sklearn.preprocessing import DeltaTransformer

    delta = DeltaTransformer(by='PersonID',
        date='date_collected', include=[],
        periods=[1,2], method='diff',
        resample_params={'rule': '1D'},
        function_params={'fill_method': 'ffill'})

    return delta.fit_transform(data)

def get_X_y_from_yaml(df, cfg):
    """

    Note, if aggregate the DataFrame will have only
    the PersonID, the features, the label and the
    set information.

    Note if delta the DataFrame will have only the
    ....

    Parameters
    ----------
    data
    idxs
    features
    labels
    strategy

    Returns
    -------

    """
    #if isinstance(labels, str):
    #    labels = [labels]

    if cfg.strategy == 'normal':
        #aux = get_data_normal_from_yaml(data, cfg=cfg)
        return df[cfg.features], \
               df[cfg.outcome]

    elif cfg.strategy == 'aggregation':
        #aux = get_data_agg_from_yaml(data, cfg=cfg)
        return df.iloc[:, 1:-2], \
               df.iloc[:, -2]

    elif cfg.strategy == 'delta':
        #aux = get_data_delta_from_yaml(data, cfg=CONFIG)
        pass


def get_trfm_data(data, cfg):
    """"""
    if cfg.strategy == 'normal':
        return get_data_normal_from_yaml(data, cfg)
    elif cfg.strategy == 'aggregation':
        return get_data_agg_from_yaml(data, cfg)
    else:
        return get_data_delta_from_yaml(data, cfg)

# Get transformed data
trfm = get_trfm_data(data, cfg=CONFIG)


# Get datasets
data_train = trfm[trfm.set=='train']
data_validate = trfm[trfm.set=='validate']
data_test = trfm[trfm.set=='test']

# Get splits
X_train, y_train = get_X_y_from_yaml(data_train, cfg=CONFIG)
X_validate, y_validate = get_X_y_from_yaml(data_validate, cfg=CONFIG)
X_test, y_test = get_X_y_from_yaml(data_test, cfg=CONFIG)


# Create directory
(WORKBENCH / 'data').mkdir(parents=True, exist_ok=True)

# Save data
#np.save(WORKBENCH / 'data' / 'X.npy', X)
#np.save(WORKBENCH / 'data' / 'y.npy', y)
#df.to_csv(WORKBENCH / 'data' / 'Xy.csv')
data.to_csv(WORKBENCH / 'data' / 'data.csv')
trfm.to_csv(WORKBENCH / 'data' / 'trfm.csv')


"""
# Split
X_cvs, X_hos, y_cvs, y_hos = train_test_split(
    X, y, test_size=0.0, random_state=42)

# Oversample
from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')
X, y = oversample.fit_resample(X, y)
#X, y = _ALL.get('rus').fit_resample(X, y)

# Create filter object
iqr = IQRTransformer(iqrrange=[25, 75], coefficient=1.5)
X = iqr.fit_transform(X)
"""

# Create folds
skf = StratifiedKFold(n_splits=2, shuffle=False)

# Create grid
grid = ParameterGrid(CONFIG.grid)

# Compendium of results
compendium_grid = pd.DataFrame()
compendium_best = pd.DataFrame()

# For each estimator
for i, e in enumerate(grid):

    # See pipeline steps format
    # pipe = Pipeline([
    #     ('sampler', _SAMPLERS.get(sampler)),
    #     ('imputer', _IMPUTERS.get(imputer)),
    #     ('scaler', _SCALERS.get(scaler)),
    #     ('method', _METHODS.get(method))
    # ])

    # Is re-sample working when included in the pipeline?
    # For some reason the training sample recorded in the
    # results is not balanced (see parallel graphs).

    # Create pipeline steps
    steps, acronyms = [], []
    for s in ['sampler', 'imputer', 'scaler', 'method']:
        acronym = e.get(s, 'none')
        if acronym != 'none':
            steps.append((s, _ALL.get(acronym)))
            acronyms.append(acronym)

    # Create variables
    folder = "-".join(acronyms)

    # Logging
    print("\n%s/%s. Computing... %s" % (i+1, len(grid), folder))

    # Create pipeline
    pipe = Pipeline(steps)

    # Define the search space / param grid
    method = acronyms[-1]
    strategy = CONFIG.search.strategy
    param_grid = CONFIG.search.space[strategy].get(method, {})
    param_grid = {'method__%s' % k: v for k, v in param_grid.items()}

    # Run cross validation
    if CONFIG.search.strategy == 'bayes':

        # Create bayesian search
        grid_search = BayesSearchCV(pipe, search_spaces=param_grid,
                            cv=skf, scoring=custom_metrics,
                            return_train_score=True,
                            refit='gmean', verbose=2, n_jobs=1)

    elif CONFIG.search.strategy == 'grid':

        # Create grid search
        grid_search = GridSearchCV(pipe, param_grid=param_grid,
                            cv=skf, scoring=custom_metrics,
                            return_train_score=True, verbose=2,
                            refit='gmean', n_jobs=1)

    # Fit grid search
    grid_search.fit(X_train, y_train)


    # -------------------------------
    # Log grid search results
    # -------------------------------
    # Save results as csv
    results = grid_search.cv_results_

    # Add information
    df_ = pd.DataFrame(results)
    df_.insert(1, 'estimator', steps[-1][1].__class__.__name__)
    df_.insert(0, 'folder', folder)

    # Append to total results
    compendium_grid = pd.concat([compendium_grid, df_])

    # Save grid search results
    compendium_grid.to_csv(
        WORKBENCH / ('gridsearch-%s.csv' % TIMESTAMP), index=False)


    # -------------------------------
    # Log best model results
    # -------------------------------
    # Best model
    best_m = grid_search.best_estimator_

    # Compute metrics
    m0 = custom_metrics(best_m, X=X_train, y=y_train)
    m1 = custom_metrics(best_m, X=X_validate, y=y_validate)
    m2 = custom_metrics(best_m, X=X_test, y=y_test)

    # Add prefixes
    m0 = {'trn_%s' % k: v for k, v in m0.items()}
    m1 = {'val_%s' % k: v for k, v in m1.items()}
    m2 = {'hos_%s' % k: v for k, v in m2.items()}

    # Combine information
    m = m0 | m1 | m2
    m['estimator'] = steps[-1][1].__class__.__name__
    m['folder'] = folder
    if hasattr(grid_search, 'best_params_'):
        m |= grid_search.best_params_
    if hasattr(grid_search, 'best_index_'):
        m['best_index'] = grid_search.best_index_

    # Append
    compendium_best = pd.concat([compendium_best, pd.DataFrame([m])])

    # Save
    compendium_best.to_csv(
        WORKBENCH / ('bestsearch-%s.csv' % TIMESTAMP), index=False)

    """
    # Compute predictions (best estimator)
    y_prob = grid_search.best_estimator_.predict_proba(X)
    y_pred = grid_search.best_estimator_.predict(X)
    pcols = ['p%s' % i for i in range(y_prob.shape[1])]

    # Add predictions
    df[pcols] = y_prob
    df['y_true'] = y
    df['y_pred'] = y_pred
    """

    # Create name for best algorithm.
    name = 'grid%02d-%s-idx%s' % \
        (i, folder, grid_search.best_index_)

    # Save (y_true, y_pred, y_probs)
    #path_ = WORKBENCH / 'probs'
    #path_.mkdir(parents=True, exist_ok=True)
    #df.to_csv(path_ / ('%s.csv' % name))

    # Save best model for this grid
    path_ = WORKBENCH / 'models'
    path_.mkdir(parents=True, exist_ok=True)
    pickle.dump(grid_search.best_estimator_,
        open(path_ / ('%s.pkl' % name), 'wb'))

    # Save grid search results (checkpoint)
    #path_ = WORKBENCH / 'checkpoint'
    #path_.mkdir(parents=True, exist_ok=True)
    #df_.to_csv(path_ / ('checkpoint-%s.csv' % i), index=False)



"""
# Show
columns = [e for e in compendium.columns if 'mean'in e]
print("Results:")
print(compendium[columns])

# Create directory
Path(CONFIG.outpath).mkdir(parents=True, exist_ok=True)

# Save grid search results
compendium.to_csv(WORKBENCH / ('%s.csv' % FILENAME),
    index=False
)
"""