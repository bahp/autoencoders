"""


Useful:

    https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    Interesting, train, test, validation and additional handling of
    imbalanced datasets.

    https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046
    Choosing the right hyperparameters for a simple LSTM
"""
# Libraries
import numpy as np
import tensorflow as tf

# Libraries
from pathlib import Path
from utils.lstm.autoencoder import MT_LSTM
from utils.lstm.autoencoder import Damien


# -----------------------------------
# Methods
# -----------------------------------
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
    return matrix[train_ind, :, :], matrix[test_ind, :, :]


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


# -----------------------------------
# Load data
# -----------------------------------
# Base
base = Path('./objects/datasets/set1/data/')
# Load matrix
#matrix = np.load('./10.matrix.npy', allow_pickle=True)
#matrix = np.load('./100.matrix.comb.npy', allow_pickle=True)
#matrix = np.load('./100.matrix.simp.dm.pw.npy', allow_pickle=True)
#matrix = np.load(base / './matrix.mmx.wft.npy', allow_pickle=True)
#matrix = np.load(base / './matrix.-8.3.5.simp.std.bare.npy', allow_pickle=True)
matrix = np.load(base / './matrix.-31.30.5.simp.std.wft.v2.npy', allow_pickle=True)


# Variables
samples, timesteps, features = matrix.shape

# Split data
train, test = split_train_test(matrix, test_size=0.2)

# Get features and labels
X_train, y_train = split_X_y(train)
X_test, y_test = split_X_y(test)


print("\n\nShapes (samples, timesteps, features)")
print("Data:  %s " % str(matrix.shape))
print("Train: %s (%s positive)" % (str(train.shape), int(y_train.sum())))
print("Test:  %s (%s positive)" % (str(test.shape), int(y_test.sum())))
print("\n\n")



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




# -----------------------------------
# Train the model
# -----------------------------------
# Define early stop
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=1e-4, patience=30, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True)

checkpoint_save = tf.keras.callbacks.ModelCheckpoint(
    filepath='./models/test2/cp-{epoch:04d}.ckpt',
    verbose=0, save_weights_only=True, save_freq='epoch')

# Define metrics
# --------------
# See https://www.tensorflow.org/api_docs/python/tf/keras/metrics
METRICS = {
    'acc': tf.keras.metrics.BinaryAccuracy(name='acc'),
    'auc': tf.keras.metrics.AUC(name='auc'),
    'prc': tf.keras.metrics.AUC(name='prc', curve='PR'),
    'fn': tf.keras.metrics.FalseNegatives(name='fn'),
    'tp': tf.keras.metrics.TruePositives(name='tp'),
    'recall': tf.keras.metrics.Recall(name='rec'),
    'prec': tf.keras.metrics.Precision(name='prec'),
    'bce': tf.keras.losses.BinaryCrossentropy(
        from_logits=True, name='binary_crossentropy'),
}


# Define optimizers
# -----------------
# See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

# Many model train better when the learning reduce is reduces gradually
# during training. For that purpose, we can use the schedulers, so that
# learning rate is reduced overtime.
N_TRAIN = X_train.shape[0]
BATCH_SIZE = 16
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
optimizer = tf.keras.optimizers.SGD(lr=1e-3)


# Define losses
# -------------
# See https://www.tensorflow.org/api_docs/python/tf/keras/losses
loss = tf.keras.losses.CosineSimilarity(
    axis=-1, reduction="auto", name="cosine_similarity"
)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Weights
# -------
# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.

tot = y_train.shape[0]
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()

weight_for_0 = (1 / neg) * (tot / 2.0)
weight_for_1 = (1 / pos) * (tot / 2.0)

class_weight = {
    0: weight_for_0,
    1: weight_for_1
}

# Create model
model = MT_LSTM(
    timesteps=timesteps,
    features=features-1,
    outputs=1
)

#model = Damien(
#    timesteps=timesteps,
#    features=features - 1,
#    outputs=1
#)

# Show model
print(model.summary())

# Compile model
model.compile(
    loss='binary_crossentropy',
    #loss=f1_loss,
    #optimizer=optimizer,
    optimizer='adamax',
    metrics=[
        METRICS.get('acc'),
        METRICS.get('prec'),
        METRICS.get('recall'),
        METRICS.get('auc'),
        METRICS.get('prc'),
    ]
)

# Fit model
model = model.fit(x=X_train, y=y_train,
    validation_data=(X_test, y_test),
    epochs=1000, batch_size=BATCH_SIZE,
    shuffle=False, callbacks=[early_stop, checkpoint_save],
    class_weight=class_weight
)



# -----------------------------------
# Evaluation
# -----------------------------------
# Libraries
import pandas as pd

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


# Make predictions
y_test_prob = model.predict(X_test)
y_test_pred = y_test_prob > 0.5

# Compute scores
scores_test = custom_metrics_(y_test, y_test_pred, y_test_prob)

# Display scores
print(pd.Series(scores_test))
print(scores_test['report'])

# Libraries
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

"""
report = classification_report(y_test, y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred > 0.5)

print("\nReport:")
print(report)
print("\nConfusion Matrix:")
print(cm)
print("\nHistory:")
print(model.model_.history)
"""

# ---------------------------------------------
#
# ---------------------------------------------
# Full report
print("\n\nShapes (samples, timesteps, features)")
print("Data:  %s " % str(matrix.shape))
print("Train: %s (%s positive)" % (str(train.shape), int(y_train.sum())))
print("Test:  %s (%s positive)" % (str(test.shape), int(y_test.sum())))
print("\n\n")

print("Evaluation (On test)")
print(pd.Series(scores_test))
print(scores_test['report'])

# ---------------------------------------------
# Graphs
# ---------------------------------------------
# Libraries
import matplotlib.pyplot as plt

# Create plot
#fig, axes = plt.subplots(1, 3)

# Edit figure
plt.title('Loss')
plt.plot(model.history_.history['loss'], label='loss - train')
plt.plot(model.history_.history['val_loss'], label='loss - test')
plt.legend()

plt.title('PRC')
plt.plot(model.history_.history['prc'], label='prc - train')
plt.plot(model.history_.history['val_prc'], label='prc - test')
plt.legend()

plt.show()

from datetime import datetime


# Create timestamp
TIMESTAMP = datetime.now().strftime('%y%m%d-%H%M%S')
FILENAME = 'gridsearch-%s' % TIMESTAMP

# Workbench path
WORKBENCH = Path('./objects/results/classification/lstm/%s' % TIMESTAMP)

# Create directory
(WORKBENCH / 'data').mkdir(parents=True, exist_ok=True)

# Save data
np.save(WORKBENCH / 'data' / 'matrix.npy', matrix)
np.save(WORKBENCH / 'data' / 'train.npy', train)
np.save(WORKBENCH / 'data' / 'test.npy', test)

# Save results

# Save model
model.save(WORKBENCH / 'model.h5')
