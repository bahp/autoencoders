

class AttrDict(dict):
    """Dictionary subclass whose entries can be accessed by attributes"""
    def __init__(self, *args, **kwargs):
        def from_nested_dict(data):
            """ Construct nested AttrDicts from nested dictionaries. """
            if not isinstance(data, dict):
                return data
            else:
                return AttrDict({key: from_nested_dict(data[key])
                                    for key in data})

        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

        for key in self.keys():
            self[key] = from_nested_dict(self[key])


# Define function
def keras_weights(counts):
    return (1 / counts) * (counts.sum() / 2.0)


def split_X_y(m):
    """Splits the 3d matrix in X and y.

    The matrix when saved has an additional column in the
    features space which represents the label for that
    window.

    Edit manually if more than one metadata is included.

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


def split_train_test(m, test_size=0.2):
    """Split 3d matrix in train and test.

    Parameters
    ----------
    m: np.array
        The matrix with shape (samples, timesteps, features)

    Returns
    -------
    train: np.array
    test: np.array

    """
    # Libraries
    from sklearn.model_selection import train_test_split

    # Create indices
    all_indices = list(range(m.shape[0]))
    train_ind, test_ind = train_test_split(all_indices, test_size=test_size)

    # Return
    return m[train_ind, :, :], m[test_ind, :, :]


def load_data_matrix(f):
    """Load matrix.

    Parameters
    ----------
    f: str
        The filename
    """
    # Libraries
    import numpy as np
    # Load matrix
    matrix = np.load(f, allow_pickle=True)
    # Split data
    train, test = split_train_test(matrix, test_size=0.2)
    # Return
    return train, train, test


def load_data_splits(f):
    """Loads the train te"""
    # Libraries
    import numpy as np

    # Load splits
    train = np.load(f / 'train.npy', allow_pickle=True)
    validate = np.load(f / 'validate.npy', allow_pickle=True)
    test = np.load(f / 'test.npy', allow_pickle=True)

    # Return
    return train, validate, test


def filter_patient_window_size(df, min_w=10, col='PersonID'):
    """Filter those patients with less than w observations

    Parameters
    ----------
    df: pd.DataFrame
        The data
    w: int
        The minimum window size accepted
    col: str
        The column with the patient id (groupby)

    Returns
    -------
    pd.DataFrame
    """
    size = df.groupby(col).size()
    pids = size[size >= min_w]
    return df[df[col].isin(pids.index)]

def filter_patient_window_density(df, col='PersonID',
                                  day_range=None,
                                  features=None, min_density=0.75):
    """Filter those patients with less than d density

    Parameters
    ----------
    df: pd.DataFrame
        The data
    day_s:
    day_e:
    features:
    min_density:

    Returns
    -------
    """
    # Libraries
    from utils.pandas.apply import density_score

    # Get days
    day_s, day_e = day_range

    # Compute
    density = df.groupby(col) \
        .apply(density_score, day_s=day_s,
               day_e=day_e, features=features) \
        .sort_values(ascending=False) \
        .rename('score')
    pids = density[density > min_density]

    # Return
    return df[df[col].isin(pids.index)]


def filter_data(data, day_range=None,
                window_size_kws={},
                window_density_kws={},
                verbose=1):
    """Filter the dataset.

    Parameters
    ----------
    data
    wsize_kws
    wdensity_kws

    Returns
    -------

    """
    def log_step(name, data):
        print("Step... %30s: %s" % (name, str(data.shape)))

    # Copy data
    data = data.copy(deep=True)
    if verbose > 0:
        log_step('raw', data)

    # filter by day
    if day_range is not None:

        ds, de = day_range
        data = data[(data.day >= ds)]
        data = data[(data.day <= de)]

        if verbose > 0:
            log_step('filter_day_range', data)

    # filter by window size
    data = filter_patient_window_size(data, **window_size_kws)
    if verbose > 0:
        log_step("filter_patient_window_size", data)

    # filter by window density
    data = filter_patient_window_density(data, **window_density_kws)
    if verbose > 0:
        log_step("filter_patient_window_density", data)

    # Return
    return data






"""
def log_confusion_matrix(epoch, logs, feo=1):
    # Libraries
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    print("\n\n")
    print(feo)
    print("\n\n")

    # Make predictions
    y_true = y_test
    y_prob = model.predict(X_test)
    y_pred = y_prob > 0.5

    # Create figure
    figure, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes = axes.flatten()

    # Plot
    ConfusionMatrixDisplay \
        .from_predictions(y_true, y_pred,
            cmap='Blues', normalize=None, ax=axes[0])
    ConfusionMatrixDisplay \
        .from_predictions(y_true, y_pred,
            cmap='Blues', normalize='all', ax=axes[1])

    # Adjust size
    plt.tight_layout()
    # Convert to image
    cm_image = plot_to_image(figure)
    # Write summary.
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)
"""
