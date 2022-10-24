# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

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
    #prec0, recall0, fscore0, support0 = \
    #    precision_recall_fscore_support(y, y_pred, pos_label=0)
    #prec1, recall1, fscore1, support1 = \
    #    precision_recall_fscore_support(y, y_pred, pos_label=1)

    # Show classification report
    #print(classification_report(y, y_pred, zero_division=0))

    # DataFrame classification report
    report = classification_report(y, y_pred, output_dict=True)
    report = pd.DataFrame(report).transpose().stack()
    report.index = ['_'.join(col) for col in report.index.values]

    # Create dictionary
    d = {}
    #d['report'] = classification_report(y, y_pred)
    d['accuracy'] = accuracy_score(y, y_pred)
    d['roc_auc'] = roc_auc_score(y, y_prob)
    d['sens'] = recall_score(y, y_pred, pos_label=1)
    d['spec'] = recall_score(y, y_pred, pos_label=0)
    d['gmean'] = gmean([d['sens'], d['spec']])
    d['tn'] = tn
    d['tp'] = tp
    d['fp'] = fp
    d['fn'] = fn

    d.update(report.to_dict())

    #for i,v in enumerate(support):
    #    d['support%s' % i] = v

    # Return
    return d

def split_X_y(m):
    """Splits the 3d matrix in X and y.

    The matrix when saved has an additional column in the
    features space which represents the label for that
    window. Thus, it is splitted in the following lines:

    Parameters
    ----------
    m: np.array
        The matrix with shape (samples, timesteps, features)
    """
    # Create X and y
    X = m[:, :, :-1].astype('float32')
    y = m[:, -1, -1].astype('float32')
    # Return
    return X, y




# Create timestamp

WORKBENCH = Path('./objects/results/classification/lstm')
WORKBENCH = WORKBENCH / '221012-110716'
WORKBENCH = WORKBENCH / 'matrix.-10_3.-5_3.w5.simp.std.crp'

"""


# ----------------------------------------------
# Show model
# ----------------------------------------------
# Libraries
import tensorflow as tf
from keras.utils.vis_utils import plot_model

# Load model
model = tf.keras.models.load_model(WORKBENCH / 'model.h5')

# Load test data
test = np.load(WORKBENCH / 'data' / 'test.npy', allow_pickle=True)

X_test, y_test = split_X_y(test)

# Make predictions
y_test_prob = model.predict(X_test)
y_test_pred = y_test_prob > 0.5

# Compute scores
#scores_test = custom_metrics_(y_test, y_test_pred, y_test_prob)

# Display scores
#print(pd.Series(scores_test))
#print(scores_test['report'])

#print(model.summary())
#plot_model(model, show_shapes=True)
"""

def extract_info_from_filename(f):
    """Extract information from filename.

    Parameters
    ----------
    f: str-like
        The filename as <matrix.-10_3.-10_3.w5.simp.std.wft>
    """
    prefix, d_tuple, p_tuple, w, imputer, scaler, features = str(f).split(".")
    d_start, d_end = d_tuple.split('_')
    p_start, p_end = p_tuple.split('_')
    return {
        'd_start': d_start,
        'd_end': d_end,
        'p_start': p_start,
        'p_end': p_end,
        'window': w,
        'imputer': imputer,
        'scaler': scaler,
        'features': features
    }

def tpr(x):
    return (x.tp) / (x.tp + x.fn)

def fpr(x):
    return (x.fp) / (x.fp + x.tn)

def ppv(x):
    return (x.tp) / (x.tp + x.fp)

def npv(x):
    return (x.tn) / (x.tn + x.fn)

def nobs(x):
    return x.tp + x.tn + x.fp + x.fn

def all(x):
    x['tpr'] = tpr(x)
    x['fpr'] = fpr(x)
    x['ppv'] = ppv(x)
    x['npv'] = npv(x)
    x['nobs'] = nobs(x)
    return x

def load_history_compendium(f):
    """This method concatenates all history csv files.

    Parameters
    ----------
    f: str-like
        The folder where the matrix.xxx sub-folders are.

    Returns
    -------
    pd.DataFrame
        All the history csv files concatenated.
    """
    # Create DataFrame
    compendium = pd.DataFrame()

    # Loop
    for f in WORKBENCH.glob('**/history.csv'):

        # Get parameters
        params = extract_info_from_filename(f.parent)

        # Load history
        history = pd.read_csv(f)
        history['f'] = f.parent
        for k, v in params.items():
            history[k] = v

        # Concatenate
        compendium = pd.concat([compendium, history])

    compendium = compendium.rename(columns={
        'Unnamed: 0': 'epoch'
    })

    # Return
    return compendium.reset_index()


# ----------------------------------------------
# Load all history results together
# ----------------------------------------------
# Define workbench
WORKBENCH = Path('./objects/results/classification/lstm')
WORKBENCH = WORKBENCH / '221011-144226'



"""
# ------------------------------
# Visualise all reports by class
# ------------------------------
# Create empty
metrics = pd.DataFrame()
for f in WORKBENCH.glob('**/trial.json'):
    print(f)
    a = pd.read_json(f).stack()
    print(a)

import sys
sys.exit()
"""

# ------------------------------
# Visualise all reports by class
# ------------------------------
# Libraries
import hiplot as hip
import tensorflow as tf

# Create empty
metrics = pd.DataFrame()

# Loop
for f in WORKBENCH.glob('**/model.h5'):

    # Load model
    model = tf.keras.models.load_model(f)

    # Load test data
    test = np.load(f.parent / 'data' / 'test.npy', allow_pickle=True)

    # Split the data and rpedictions
    X_test, y_test = split_X_y(test)
    y_test_prob = model.predict(X_test)
    y_test_pred = y_test_prob > 0.5

    # Compute scores
    scores_test = custom_metrics_(y_test, y_test_pred, y_test_prob)

    # Combine
    series = pd.Series(scores_test)
    series.name = f.parent.stem
    metrics = pd.concat([metrics, series], axis=1)

# Transpose
metrics = metrics.transpose()

# Create hiplot.
hip.Experiment.from_iterable(
    metrics.round(decimals=3) \
        .to_dict(orient='records')
).to_html(WORKBENCH / 'hiplot.results.byclass.html')




# ---------------------------
# Visualise last epoch scores
# ---------------------------
# Libraries
import hiplot as hip

# Get compendium of history files
compendium = load_history_compendium(WORKBENCH)

# Get last score and include extra metrics
last = compendium.groupby('f') \
    .agg(['last']) \
    .droplevel(1, axis=1) \
    .apply(all, axis=1)

# Create hiplot.
hip.Experiment.from_iterable(
    last.round(decimals=3) \
        .to_dict(orient='records')
).to_html(WORKBENCH / 'hiplot.results.html')

# Show
#plt.show()