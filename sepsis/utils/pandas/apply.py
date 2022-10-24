# Libraries
import numpy as np

def something(m):
    #uniques, counts = np.unique(df_, return_counts=True)
    #percentages = dict(zip(uniques, counts * 100 / len(df_)))
    #print(percentages)
    pass

def density(m):
    return 1 - sparsity(m)

def sparsity(m):
    return np.isnan(m).sum() / np.prod(m.shape)

def cmt(x):
    """Confusion matrix type

    Usage
    -----
    df.apply(cmt, axis=0)

    Requirements
    ------------
    Columns <y_true> and <y_pred>.

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

    Usage
    -----

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


def density_score(x, features=None, day_s=-5, day_e=5):
    """

    Usage
    -----

    Requirements
    ------------
    It requires a <day> column.

    Parameters
    ----------
    x
    features
    day_s
    day_e
    ratio

    Returns
    -------

    """
    if features is None:
        features = x.columns

    # Filter
    df_ = x.copy(deep=True)
    df_ = df_[(df_.day >= day_s)]
    df_ = df_[(df_.day <= day_e)]
    df_ = df_[features]

    # Return
    return density(df_.to_numpy())

    #return (~np.isnan(m)).sum() / (day_e - day_s) * len(features)





if __name__ == '__main__':

    # Libraries
    import numpy as np
    import pandas as pd

    # --------------------------------------------
    # Test cmt
    # --------------------------------------------
    # Create data
    m = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # Create DataFrame
    df = pd.DataFrame(data=m, columns=['y_true', 'y_pred'])

    # Compute
    df['cmt'] = df.apply(cmt, axis=1)

    # Show
    print("\nTest <cmt>:")
    print(df)

    # --------------------------------------------
    # Test sparsity
    # --------------------------------------------
    # Create data
    m = np.array([
        [-4, 1, 2, 3, np.nan],
        [-3, 1, 2, 3, np.nan],
        [-2, 1, 2, np.nan, np.nan],
        [-1, 1, 2, np.nan, np.nan]
    ])

    # Create DataFrame
    df = pd.DataFrame(data=m,
        columns=['day', 'id', 'f1', 'f2', 'f3'])

    # Compute
    s = density_score(df, features=['f1', 'f2', 'f3'],
        day_s=-4, day_e=-2)

    # Show
    print("\nTest density:")
    print(m)
    print("Density: %s" % s)