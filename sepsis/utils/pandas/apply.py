# Libraries
import numpy as np

def cmt(x):
    """Confusion matrix type

    Parameters
    ----------
    x: series
        It must contain y_true and y_pred

    Returns
    -------

    """
    if bool(x.y_true) & bool(x.y_pred):
        return 'TP'
    if bool(x.y_true) & ~bool(x.y_pred):
        return 'FN'
    if ~bool(x.y_true) & bool(x.y_pred):
        return 'FP'
    if ~bool(x.y_true) & ~bool(x.y_pred):
        return 'TN'
    return np.NaN


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