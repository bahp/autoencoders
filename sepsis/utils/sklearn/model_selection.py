def subset_split(m, fractions, names=None, shuffle=False):
    """"""
    # Create names
    if names is None:
        names = ['set%s' % i for i in len(fractions)]

    # Check same length
    if len(fractions) != len(names):
        raise Exception('error', 'length')

    # Check if shuffle
    pass




if __name__ == '__main__':

    # Libraries
    import pandas as pd
    from sklearn.datasets import load_iris

    # Data
    X, y = load_iris(return_X_y=True, as_frame=True)

    # Create DataFrame
    data = pd.concat((X, y), axis=1)

    # Show
    print(data)