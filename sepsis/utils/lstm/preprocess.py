# Libraries
import numpy as np

def create_lstm_matrix(data, features, groupby, w=5):
    """Constructs the 3D matrix.

    Notes
    -----
    The patients with less days of date than the window length are
    being filtered automatically by the <generate_subsequences> method
    which returns an empty array.

    Requirements
    ------------


    Parameters
    ----------
    data:
    features:
    groupby:
    w:

    Returns
    -------
    matrix: np.array
        This matrix is shape is (samples, timesteps, features)
    """

    def generate_subsequences(aux, w=5):
        """This method generates subsequences.

        Parameters
        ----------
        aux: pd.DataFrame
            The pandas DataFrame
        groupby: str
            The groupby method.
        w: int
            The window length
        """
        # Group
        for i in range(0, (aux.shape[0] - w) + 1):
            matrix.append(aux[i:i + w].to_numpy()[:, :-1])
        # Return
        return None

    # Create final 3D matrix (list of lists)
    matrix = []
    # Fill matrix
    data[features + [groupby]] \
        .groupby(by=groupby) \
        .apply(generate_subsequences, w=w)
    # Return
    return np.asarray(matrix)




if __name__ == '__main__':

    # Libraries
    import pandas as pd

    # --------------------------------------------
    # Test <create_lstm_matrix>
    # --------------------------------------------
    # Create data
    m = np.array([
        [-4, 1, 2.1, 3.1, np.nan, 0],
        [-3, 1, 2.2, 3.2, np.nan, 0],
        [-2, 1, 2.3, np.nan, np.nan, 1],
        [-1, 1, 2.4, np.nan, np.nan, 1],
        [0, 1, 2.5, np.nan, np.nan, 0],
        [-5, 2, 2.6, 3.6, 4.6, 0],
        [-4, 2, 2.7, 3.7, 4.7, 0],
    ])

    # Create DataFrame
    df = pd.DataFrame(data=m,
         columns=['day', 'id', 'f1', 'f2', 'f3', 'y'])

    matrix0 = create_lstm_matrix(data=df,
        features=['f1', 'y'],
        groupby='id', w=2)

    matrix1 = create_lstm_matrix(data=df,
        features=['f1'],
        groupby='id', w=3)

    matrix2 = create_lstm_matrix(data=df,
        features=['f1', 'f2', 'y'],
        groupby='id', w=3)

    matrix3 = create_lstm_matrix(data=df,
        features=['f1'],
        groupby='id', w=6)

    print("\nData:")
    print(df)
    print("\nf=[f1], w=2 => %s" % str(matrix0.shape))
    print(matrix0)
    print("\nf=[f1], w=3 => %s" % str(matrix1.shape))
    print(matrix1)
    print("\nf=[f1, f2], w=3 => %s" % str(matrix2.shape))
    print(matrix2)
    print("\nf=[f1], w=6 => %s" % str(matrix3.shape))
    print(matrix3)

    # Getting X and the last value of for y.
    X = matrix2[:, :, :-1].astype('float32')
    y = matrix2[:, -1, -1].astype('float32')

    print("\nSee X and y")
    print(X)
    print(y)