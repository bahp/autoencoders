

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
    d['report'] = classification_report(y, y_pred)
    d['accuracy'] = accuracy_score(y, y_pred)
    d['roc_auc'] = roc_auc_score(y, y_prob)
    d['sens'] = recall_score(y, y_pred, pos_label=1)
    d['spec'] = recall_score(y, y_pred, pos_label=0)
    d['gmean'] = gmean([d['sens'], d['spec']])
    d['tn'] = tn
    d['tp'] = tp
    d['fp'] = fp
    d['fn'] = fn

    for i,v in enumerate(support):
        d['support%s' % i] = v

    # Return
    return d

def get_class_weights(v, format='dict', type='count'):
    """Get class weights.

    Parameters
    ----------
    v
    format
    type: str
        count
        ratio
        percent
        function


    Returns
    -------

    """
    # Libraries
    import numpy as np

    # Get unique values and counts
    uniques, counts = np.unique(v, return_counts=True)

    # Compute type
    if type == 'ratio':
        counts = counts / v.shape[0]
    elif type == 'percent':
        counts = (counts * 100) / v.shape[0]
    elif callable(type):
        counts = type(counts)

    # Format
    if format == 'dict':
        return dict(zip(uniques, counts))
    return uniques, counts




if __name__ == '__main__':

    """
    TODO: Find where did I put the metrics with all regression
          metrics that might/might not have been created or 
          included in the default sklearn library.
    """

    # Libraries
    import numpy as np
    import pandas as pd

    # ----------------------------------------------
    # Custom metrics
    # ----------------------------------------------
    # Constants
    N = 100

    # Create matrix
    y_true = np.random.randint(0, 2, (N, 1))
    y_prob = np.random.randint(0, 101, (N, 1)) / 100

    # Create dataframe
    df = pd.DataFrame(
        data=np.hstack((y_true, y_prob, y_prob > 0.5)),
        columns=['y_true', 'y_prob', 'y_pred']
    )

    # Compute metrics
    m = custom_metrics_(df.y_true, df.y_pred, df.y_prob)

    # Show
    print("\nData")
    print(df)
    print("\nMetrics")
    print(pd.Series(m))

    # -----------------------------------------------
    # Get class weights
    # -----------------------------------------------
    # Define function
    def keras_weights(counts):
        return (1 / counts) * (counts.sum() / 2.0)

    # Get weights
    d0 = get_class_weights(df.y_true)
    d1 = get_class_weights(df.y_true, type='ratio')
    d2 = get_class_weights(df.y_true, type='percent')
    d3 = get_class_weights(df.y_true, type=keras_weights)

    # Show
    print("\nWeights")
    print(d0)
    print(d1)
    print(d2)
    print(d3)
