# Libraries
import tensorflow as tf

def f1_loss(y_true, y_pred):
    """

    See https://www.kaggle.com/code/rejpalcz/best-loss-function-for-f1-score-metric/notebook

    Parameters
    ----------
    y_true
    y_pred

    Returns
    -------

    """
    import keras.backend as K
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    print(f1)
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)
