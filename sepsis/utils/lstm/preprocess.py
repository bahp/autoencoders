# Libraries
import numpy as np

def create_lstm_matrix(data, features, groupby, w=5):
    """Constructs the 3D matrix.

    Parameters
    ----------

    Returns
    -------
    matrix: np.array
        This matrix is shape is (samples, timestemps, features)
    """

    def generate_subsequences(aux, w=5):
        """This method generates subsequences.

        Parameters
        ----------
        aux: pd.DataFrame
            The pandas dataframe
        groupby: str
            The groupby method.
        w: int
            The window length
        """
        # Group
        for i in range(0, (aux.shape[0] - w)):
            matrix.append(aux[i:i + w].to_numpy()[:, 2:])
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