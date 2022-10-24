# Libraries
import tensorflow as tf

# See https://www.tensorflow.org/api_docs/python/tf/keras/metrics

METRICS = {
    'acc': tf.keras.metrics.BinaryAccuracy(name='acc'),
    'auc': tf.keras.metrics.AUC(name='auc'),
    'prc': tf.keras.metrics.AUC(name='prc', curve='PR'),
    'fp': tf.keras.metrics.FalsePositives(name='fp'),
    'fn': tf.keras.metrics.FalseNegatives(name='fn'),
    'tp': tf.keras.metrics.TruePositives(name='tp'),
    'tn': tf.keras.metrics.TrueNegatives(name='tn'),
    'recall': tf.keras.metrics.Recall(name='rec'),
    'prec': tf.keras.metrics.Precision(name='prec'),
    'bce': tf.keras.losses.BinaryCrossentropy(
        from_logits=True, name='binary_crossentropy'),
    #'cm': tf.keras.confusion_matrix(name='cm')
}
