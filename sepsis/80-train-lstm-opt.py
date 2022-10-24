"""


Useful:

[R1] Keras debugging tips
https://keras.io/examples/keras_recipes/debugging_tips/

[R2] Interesting, train, test, validation and additional handling ofimbalanced datasets.
https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

[R3] Choosing the right hyperparameters for a simple LSTM
https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046

[R4] Hyperparameter optimization in tensorflow
https://pub.towardsai.net/keras-tuner-tutorial-hyperparameter-optimization-tensorflow-keras-computer-vision-example-c9abbdad9887

[R5] Tutorial for keras tuner
https://pyimagesearch.com/2021/06/07/easy-hyperparameter-tuning-with-keras-tuner-and-tensorflow/
"""
# Libraries
import keras_tuner
import numpy as np
import pandas as pd
import tensorflow as tf

# Libraries
from pathlib import Path
from datetime import datetime

from utils.lstm.autoencoder import MT_LSTM
from utils.lstm.autoencoder import Damien
from utils.keras.losses import f1_loss
from utils.keras.callbacks import ConfusionMatrixLogger
from utils.lstm.autoencoder import KerasTunerLSTM

from utils.sklearn.metrics import custom_metrics_
from utils.sklearn.metrics import get_class_weights

from utils.utils import keras_weights
from utils.utils import split_X_y


# -----------------------------------
# Configuration (MODEL)
# -----------------------------------

# Callbacks
# ---------

# Define optimizers
# -----------------
# See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

# Many model train better when the learning reduce is reduces gradually
# during training. For that purpose, we can use the schedulers, so that
# learning rate is reduced overtime.
#N_TRAIN = X_train.shape[0]
BATCH_SIZE = 16
#STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

#lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#  initial_learning_rate=0.001,
#  decay_steps=STEPS_PER_EPOCH*1000,
#  decay_rate=1,
#  staircase=False)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
optimizer = tf.keras.optimizers.Adamax(learning_rate=1e-3)
#optimizer = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)

# Define losses
# -------------
# See https://www.tensorflow.org/api_docs/python/tf/keras/losses
loss = tf.keras.losses.CosineSimilarity(
    axis=-1, reduction="auto", name="cosine_similarity"
)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)



# Create output folder with timestamp
OUTPATH = Path('./objects/results/classification/lstm/')
TIMESTAMP = datetime.now().strftime('%y%m%d-%H%M%S')
WORKBENCH = OUTPATH / TIMESTAMP

# Define input folder (data)
DATAPATH = Path('./objects/datasets/set1/data/221018-161523')

# List all interesting folders
files = list(DATAPATH.glob('**/*.wft'))

# Loop for al te files.
for i, f in enumerate(files):

    # -----------------------------
    # Load
    # -----------------------------
    # Load data
    train = np.load(f / 'train.npy', allow_pickle=True)
    validate = np.load(f / 'validate.npy', allow_pickle=True)
    test = np.load(f / 'test.npy', allow_pickle=True)

    # Get features and labels
    X_train, y_train = split_X_y(train)
    X_validate, y_validate = split_X_y(validate)
    X_test, y_test = split_X_y(test)


    # -----------------------------
    # Save
    # -----------------------------
    # Create directory
    (WORKBENCH / f.name / 'data').mkdir(parents=True, exist_ok=True)

    # Save data
    np.save(WORKBENCH / f.name / 'data' / 'train.npy', train)
    np.save(WORKBENCH / f.name / 'data' / 'test.npy', test)
    np.save(WORKBENCH / f.name / 'data' / 'validate.npy', validate)


    # -----------------------------
    # Callbacks
    # -----------------------------
    # Variables
    LOGS_ROOT = Path('logs/tensor/search/%s/%s' % (TIMESTAMP, f.name))

    # If val_loss doesn't improve for a number of epochs set with
    # 'patience' var training will stop to avoid over-fitting.
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=1e-4, patience=30, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True)

    # Learning rate is reduced by 'lr_factor' if val_loss stagnates
    # for a number of epochs set with 'patience/2' var.
    reduce_lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", mode="min", factor=0.3, min_lr=1e-6,
        patience=10 // 2, verbose=1)

    # Only save the weights for each iteration. It is also possible
    # to save models and/or choose the weights that correspond to
    # the max/min of a certain monitored value (monitor='accuracy',
    # mode='max').
    checkpoint_save = tf.keras.callbacks.ModelCheckpoint(
        filepath= WORKBENCH / f.name / 'ckpt' / 'cp-{epoch:04d}.ckpt',
        verbose=0, save_weights_only=True, save_freq='epoch')

    # Saves the history into a csv.
    history_logger = tf.keras.callbacks.CSVLogger(
        filename=WORKBENCH / f.name / 'logs/history.csv')

    # Saves the history into a tensor.
    tensor_logger = tf.keras.callbacks.TensorBoard(
        WORKBENCH / f.name / 'logs/tensor/'),

    # Saves the history into a tensor (tensorboard)
    tensor_logger_root = tf.keras.callbacks.TensorBoard(
        log_dir=LOGS_ROOT,
        write_images=True, # visualise model weights
        write_graph=True,  # visualise graph (large size)
        histogram_freq=1   # freq to calculate weight histograms
    )

    # Confusion matrix logger
    cm_logger = ConfusionMatrixLogger(
        log_dir=LOGS_ROOT / 'images/cm',
        X=X_test, y=y_test)


    # --------------------------------
    # Bayesian Search with Keras Tuner
    # --------------------------------
    # Show information
    print("\n\nShapes (samples, timesteps, features)")
    print("Train: %s (%s positive)" % (str(train.shape), int(y_train.sum())))
    print("Valid: %s (%s positive)" % (str(validate.shape), int(y_validate.sum())))
    print("Test:  %s (%s positive)" % (str(test.shape), int(y_test.sum())))
    print("\n\n")

    # Model template
    hypermodel = KerasTunerLSTM(
        timesteps=X_train.shape[1],
        features=X_train.shape[2],
        outputs=1 # binary
    )

    # Test model builds
    #build_model(keras_tuner.HyperParameters())

    # Create Bayesian tuner
    # Options:
    #  objective=keras_tuner.Objective('val_auc', direction='max')
    #  objective='val_accuracy'
    #  objective='val_loss
    tuner = keras_tuner.BayesianOptimization(
        hypermodel=hypermodel, # build_model
        objective='val_loss',  # val_accuracy
        max_trials=20,
        directory=WORKBENCH / f.name / 'logs',
        overwrite=True,
        project_name='kt'
    )

    # Search
    tuner.search(
        X_train, y_train,
        validation_data=(X_test, y_test),
        callbacks=[
            early_stop,
            tensor_logger_root
        ],
        epochs=100,
    )

    # Get the top n models
    #models = tuner.get_best_models(num_models=2)
    #best_model = models[0]
    # Build the model.
    # Needed for `Sequential` without specified `input_shape`.
    #best_model.build(input_shape=(None, 28, 28))
    #best_model.summary()
    #tuner.results_summary()

    # Get the top 2 hyper-parameters.
    #best_hps = tuner.get_best_hyperparameters(2)
    #best_mdls = tuner.get_best_models()[0]

    # Build the model with the best hp.
    #model = build_model(best_hps[0])

    # Get best model from hyper-parameters
    #best_hps = tuner.get_best_hyperparameters(2)
    #model = hypermodel.build(best_hps[0])


    # --------------------------------
    # Fit best model
    # --------------------------------
    # Get best model
    model = tuner.get_best_models()[0]

    # Show
    print(model.summary())

    # Get class weights
    class_weight = get_class_weights(y_train, type=keras_weights)

    # Fit the model
    # How to get the BATCH_SIZE from search?
    # How to get the shuffle from search?
    history = model.fit(x=X_train, y=y_train,
        validation_data=(X_validate, y_validate),
        epochs=1000, batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[
            early_stop,
            checkpoint_save,
            history_logger,
            tensor_logger_root,
            cm_logger
        ],
        class_weight=class_weight
    )

    # Save model
    model.save(WORKBENCH / f.name / 'model.h5')

    # Save history
    # Note that we can also save the history on each iteration
    # by using the history_logger callback function.
    pd.DataFrame.from_dict(history.history, orient='columns') \
        .to_csv(WORKBENCH / f.name / 'history.csv')
