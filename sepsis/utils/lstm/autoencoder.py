# Generic
import tensorflow as tf

# Specific
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from tensorflow.keras import optimizers

from keras.initializers import GlorotUniform
from keras.initializers import Orthogonal
from keras.initializers import Constant

# Define SEED.
SEED = 2022

"""
Notes:

  binary
        binary_crossentropy => model.add(Dense(1, activation='sigmoid')) - {0, 1}
        hinge / squared_hinge =>  model.add(Dense(1, activation='tanh')) => {-1, 1}

        multiclass
        'categorical_crossentropy' => model.add(Dense(3, activation='softmax'))
        'sparse_categorical_crossentropy'

"""

class Classifier:
    def summary(self):
        return self.model_.summary()

    def fit(self, **kwargs):
        self.history_ = self.model_.fit(**kwargs)
        return self

    def compile(self, **kwargs):
        self.model_.compile(**kwargs)

    def predict(self, data, **kwargs):
        return self.model_.predict(data, **kwargs)

    def save(self, path, **kwargs):
        self.model_.save(path, **kwargs)


class BinaryClassifierDenseV1(Classifier):
    """"""
    def __init__(self, timesteps, features, outputs):
        """"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy',
            optimizer='adam', metrics=['accuracy'])
        model.build()
        self.model_ = model


class BinaryClassifierLSTMV1(Classifier):

    def __init__(self, timesteps, features, outputs):
        """"""
        model = Sequential()
        model.add(LSTM(1,
            batch_input_shape=(None, timesteps, features),
            dropout=0.2,
            recurrent_dropout=0.2,
            return_sequences=False,
            name='encoder_1')
        )
        """
        model.add(LSTM(32,
            kernel_initializer='he_uniform',
            return_sequences=False, name='encoder_2')
        )
        model.add(LSTM(16,
            kernel_initializer='he_uniform',
            return_sequences=True, name='encoder_3')
        )
        model.add(LSTM(8,
            kernel_initializer='he_uniform',
            return_sequences=True, name='encoder_4')
        )
        model.add(LSTM(4,
            kernel_initializer='he_uniform',
            return_sequences=True, name='encoder_5')
        )
        model.add(LSTM(2,
            kernel_initializer='he_uniform',
            return_sequences=False, name='encoder_6')
        )
        """
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=[
                tf.keras.metrics.Accuracy(),
                tf.keras.metrics.AUC()
            ])
        model.build()

        self.model_ = model


class LSTMClassifier(Classifier):

    def __init__(self, timesteps, features, outputs):
        """
        binary
        binary_crossentropy => model.add(Dense(1, activation='sigmoid')) - {0, 1}
        hinge / squared_hinge =>  model.add(Dense(1, activation='tanh')) => {-1, 1}

        multiclass
        'categorical_crossentropy' => model.add(Dense(3, activation='softmax'))
        'sparse_categorical_crossentropy'

        Parameters
        ----------
        timesteps
        features
        outputs
        """
        print(timesteps, features, outputs)
        model = Sequential()
        model.add(LSTM(100,
            batch_input_shape=(None, timesteps, features),
            #input_shape=(timesteps, features),
            dropout=0.2,
            recurrent_dropout=0.2,
            return_sequences=True,
            name='encoder_1')
        )
        model.add(LSTM(32,
            kernel_initializer='he_uniform',
            return_sequences=True,
            name='encoder_2')
        )
        model.add(LSTM(16,
            kernel_initializer='he_uniform',
            return_sequences=False,
            name='encoder_3')
        )

        model.add(Dropout(0.5))
        model.add(Dense(100,
            activation='relu',
            kernel_initializer='he_uniform')
        )
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
            metrics=[
              'accuracy',
              #tf.keras.metrics.Accuracy(),
              tf.keras.metrics.AUC()
            ])
        model.build()

        self.model_ = model



class LSTMAutoencoder:
    """LSTM multidimensional time-series data.

    This autoencoder can be used for dense latent space representation of
    multidimensional time-series data.

    REF: https://towardsdatascience.com/using-lstm-autoencoders-on-multidimensional-time-series-data-f5a7a51b29a1

    The model begins with an Encoder: first, the input layer. The input layer is
    an LSTM layer. This is followed by another LSTM layer, of a smaller size. Then,
    I take the sequences returned from layer 2 — then feed them to a repeat vector.
    The repeat vector takes the single vector and reshapes it in a way that allows
    it to be fed to our Decoder network which is symmetrical to our Encoder. Note
    that it doesn’t necessarily have to be symmetrical, but this is standard practice.
    """

    def __init__(self, timesteps, features, latent_dim, loss=None):
        """"""
        if loss is None:
            # Define loss
            loss = tf.keras.losses.CosineSimilarity(
                axis=-1, reduction="auto", name="cosine_similarity"
            )
            print("Default loss: " % loss)

        # Construct model
        model = Sequential()
        model.add(
            LSTM(64,
                kernel_initializer='he_uniform',
                batch_input_shape=(None, timesteps, features),
                return_sequences=True,
                name='encoder_1')
        )
        model.add(
            LSTM(32, activation='relu',
                kernel_initializer='he_uniform',
                return_sequences=True,
                name='encoder_2')
        )
        model.add(
            LSTM(latent_dim,
                kernel_initializer='he_uniform',
                return_sequences=False,
                name='encoder_3')
        )  # return false and Repeat Vector
        model.add(
            RepeatVector(timesteps,
                name='encoder_decoder_bridge')
        )
        model.add(
            LSTM(latent_dim,
                kernel_initializer='he_uniform',
                return_sequences=True,
                name='decoder_1')
        )
        model.add(
            LSTM(32, activation='relu',
                kernel_initializer='he_uniform',
                return_sequences=True,
                name='decoder_2')
        )
        model.add(
            LSTM(64,
                kernel_initializer='he_uniform',
                return_sequences=True,
                name='decoder_3')
        )
        model.add(TimeDistributed(Dense(features)))
        model.compile(loss=loss, optimizer='adam')
        model.build()

        self.model_ = model

    def summary(self):
        return self.model_.summary()

    def fit(self, **kwargs):
        self.history_ = self.model_.fit(**kwargs)
        return self

    def predict(self, data, **kwargs):
        return self.model_.predict(data, **kwargs)

    def encoder(self):
        return Model(
            inputs=self.model_.inputs,
            outputs=self.model_.layers[2].output
        )

    def save(self, path, **kwargs):
        self.model_.save(path, **kwargs)


"""
from tensorflow_addons.layers import MultiHeadAttention

class AttentionBlock(Model):
    def __init__(self, name='AttentionBlock', num_heads=2,
            head_size=128, ff_dim=None, dropout=0, **kwargs):
        # Super
        super().__init__(name=name, **kwargs)

        if ff_dim is None:
            ff_dim = head_size

        self.attention = MultiHeadAttention(num_heads=num_heads, head_size=head_size, dropout=dropout)
        self.attention_dropout = tf.keras.layers.Dropout(dropout)
        self.attention_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ff_conv1 = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')
        # self.ff_conv2 at build()
        self.ff_dropout = tf.keras.layers.Dropout(dropout)
        self.ff_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.ff_conv2 = tf.keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1)

    def call(self, inputs):
        x = self.attention([inputs, inputs])
        x = self.attention_dropout(x)
        x = self.attention_norm(inputs + x)

        x = self.ff_conv1(x)
        x = self.ff_conv2(x)
        x = self.ff_dropout(x)

        x = self.ff_norm(inputs + x)
        return x
"""

"""
class ModelTrunk(tf.keras.Model):
    def __init__(self, name='ModelTrunk', time2vec_dim=1, num_heads=2, head_size=128, ff_dim=None, num_layers=1,
                 dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.time2vec = Time2Vec(kernel_size=time2vec_dim)
        if ff_dim is None:
            ff_dim = head_size
        self.dropout = dropout
        self.attention_layers = [
            AttentionBlock(num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in
            range(num_layers)]


    def call(self, inputs):
        time_embedding = tf.keras.layers.TimeDistributed(self.time2vec)(inputs)
        x = K.concatenate([inputs, time_embedding], -1)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)

        return K.reshape(x, (-1, x.shape[1] * x.shape[2]))  # flat vector of features out
"""
"""
class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__(trainable=True, name='Time2VecLayer')
        self.k = kernel_size

    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(name='wb', shape=(input_shape[1],), initializer='uniform', trainable=True)
        self.bb = self.add_weight(name='bb', shape=(input_shape[1],), initializer='uniform', trainable=True)
        # periodic
        self.wa = self.add_weight(name='wa', shape=(1, input_shape[1], self.k), initializer='uniform', trainable=True)
        self.ba = self.add_weight(name='ba', shape=(1, input_shape[1], self.k), initializer='uniform', trainable=True)
        super(Time2Vec, self).build(input_shape)

    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb
        dp = K.dot(inputs, self.wa) + self.ba
        wgts = K.sin(dp)  # or K.cos(.)

        ret = K.concatenate([K.expand_dims(bias, -1), wgts], -1)
        ret = K.reshape(ret, (-1, inputs.shape[1] * (self.k + 1)))
        return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * (self.k + 1))
"""

# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

class StackedLSTM_MLM(Classifier):
    """"Stacked LSTM from Machine Learning Mastery.
    
    Ref: https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
    """

    def __init__(self, timesteps, features, outputs):
        """

        Parameters
        ----------
        timesteps
        features
        outputs
        """
        model = Sequential()
        model.add(LSTM(50,
            input_shape=(timesteps, features),
            activation='relu',
            return_sequences=True,
            name='lstm_1')
        )
        model.add(LSTM(50,
            name='lstm_2')
        )
        model.add(Dense(1,
            activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
            metrics=[
                'accuracy',
                #tf.keras.metrics.Accuracy(),
                tf.keras.metrics.AUC()
            ])
        model.build()

        self.model_ = model


class BidirectionalLSTM_MLM(Classifier):
    """"Stacked LSTM from Machine Learning Mastery.

    Ref: https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
    """

    def __init__(self, timesteps, features, outputs):
        """

        Parameters
        ----------
        timesteps
        features
        outputs
        """
        # Libraries
        from keras.layers import Bidirectional

        # Create model
        model = Sequential()
        model.add(Bidirectional(
            LSTM(50, activation='relu'),
            input_shape=(timesteps, features),
            name='bidirectional_1')
        )
        model.add(Dense(1,
             activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
            metrics=[
                'accuracy',
                # tf.keras.metrics.Accuracy(),
                tf.keras.metrics.AUC()
            ])
        model.build()

        self.model_ = model


class MT_LSTM(Classifier):
    """"""
    def __init__(self, timesteps, features, outputs):
        """
        https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046


        softmax?

        I recommend you first try SGD with default parameter values. If
        it still doesn't work, divide the learning rate by 10. Do that a
        few times if necessary. If your learning rate reaches 1e-6 and it
        still doesn't work, then you have another problem.

        Parameters
        ----------
        timesteps: int
            The number of timesteps
        features: int
            The number of features
        outputs: int
            The number of outputs
        """
        # Libraries

        #opt = tf.keras.optimizers.SGD(lr=0.01)
        opt = tf.keras.optimizers.SGD(lr=1e-5)

        # Layers
        layers = tf.keras.layers

        # Define model
        inputs = layers.Input(shape=(timesteps, features))
        hidden = layers.LSTM(32)(inputs)
        # Dropout
        out1 = layers.Dense(1, activation="sigmoid", name="binary_out")(hidden)
        #out2 = layers.Dense(1, activation=None, name="reg_out")(hidden)

        # Creat model
        model = tf.keras.Model(inputs=inputs, outputs=[out1])

        #model.compile(
        #    loss={
        #        "binary_out": "binary_crossentropy",
        #    },
        #    #optimizer='adam',
        #    optimizer=opt,
        #    metrics={"binary_out":
        #                 ["accuracy", tf.keras.metrics.AUC()]}
        #)

        self.model_ = model


def get_manual_model(name, **kwargs):
    """"""
    if name=='dense':
        return None
    elif name=='LSTM_reg':
        return None
    elif name=='Damien':
        return None
    return None



class Dense_3(Classifier):
    """"""
    def __init__(self, timesteps, features, outputs):
        """
        I recommend you first try SGD with default parameter values. If
        it still doesn't work, divide the learning rate by 10. Do that a
        few times if necessary. If your learning rate reaches 1e-6 and it
        still doesn't work, then you have another problem.

        Parameters
        ----------
        timesteps: int
            The number of timesteps
        features: int
            The number of features
        outputs: int
            The number of outputs
        """
        # Libraries
        from keras.layers import Dense
        from keras.models import Sequential

        # Create model
        model = Sequential()
        model.add(Dense(2, input_dim=2, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(2, activation='sigmoid'))

        # Set model
        self.model_ = model


class Damien(Classifier):
    """"""

    def __init__(self, timesteps, features, outputs):
        """"""
        # Libraries
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Flatten
        from keras.layers import Dropout
        from keras.layers import LSTM
        from keras import regularizers
        from keras.layers import BatchNormalization
        from keras.layers import GRU
        import numpy as np

        # Create model
        model = Sequential()
        model.add(GRU(
            units=256,
            activation='tanh',
            input_shape=(timesteps, features),
            return_sequences=True))
        model.add(Dropout(rate=0.2))
        model.add(BatchNormalization())
        model.add(GRU(
            units=128,
            activation='tanh',
        ))
        model.add(Dropout(rate=0.2))
        model.add(BatchNormalization())
        model.add(Dense(
            units=12,
            activation='tanh',
            kernel_regularizer=regularizers.l2(0.01))
        )
        model.add(Dense(1,
            activation='sigmoid',
            bias_initializer=tf.keras.initializers.Constant(
                np.array([-2.65917355])
            )
        ))
        # Set model
        self.model_ = model

        return model




class LSTM_reg(Classifier):
    """"""
    def __init__(self, timesteps, features, outputs):
        """Creates the constructor.

        Parameters
        ----------
        timesteps: int
            The number of timesteps.
        features: int
            The number of features.
        outputs: int
            The number of outputs.
        """
        self.timesteps = timesteps
        self.features = features
        self.outputs = outputs

    def build(self, hp=None):
        """"""
        # Libraries
        from utils.keras.layers import Attention
        from keras.layers import BatchNormalization

        # Create model
        model = tf.keras.Sequential()
        model.add(Input(
            shape=(self.timesteps, self.features)
        ))
        model.add(LSTM(
            units=8,
            activation='relu',
            dropout=0.2,
            recurrent_dropout=0.2,
            input_shape=(self.timesteps, self.features),
            bias_regularizer='l2',
            recurrent_regularizer='l2',
            #return_sequences=True
        ))
        model.add(Dropout(rate=0.2))
        model.add(BatchNormalization())
        #model.add(Attention()) # needs return_sequences=True
        model.add(Dense(self.outputs,
            activation='sigmoid',
        ))

        return model

    def fit(self, hp, model, *args, **kwargs):
        """"""
        return model.fit(*args, shuffle=True, **kwargs)


class DenseV1(Classifier):
    """"""
    def __init__(self, timesteps, features, outputs):
        """Creates the constructor.

        Parameters
        ----------
        timesteps: int
            The number of timesteps.
        features: int
            The number of features.
        outputs: int
            The number of outputs.
        """
        self.timesteps = timesteps
        self.features = features
        self.outputs = outputs

    def build(self, hp=None):
        """"""
        # Libraries
        from utils.keras.layers import Attention
        from keras.layers import BatchNormalization
        from tensorflow.keras.layers import Flatten

        # Create model
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=(self.timesteps, self.features)))
        model.add(Dense(56, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(self.outputs, activation='sigmoid'))

        # Return
        return model

# --------------------------------------------------
# Keras Tuner
# --------------------------------------------------
# Libraries
import keras_tuner

# Specific
from utils.keras.metrics import METRICS


class KerasTunerDense(keras_tuner.HyperModel):
    pass

class KerasTunerLSTM(keras_tuner.HyperModel):
    """Creates an LSTM hypermodel.

    See R1: https://www.kaggle.com/code/iamleonie/time-series-tips-tricks-for-training-lstms
    See R2: https://towardsdatascience.com/implementation-differences-in-lstm-layers-tensorflow-vs-pytorch-77a31d742f74
    See R3: https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046
    """
    def __init__(self, timesteps, features, outputs):
        """Creates the constructor.

        Parameters
        ----------
        timesteps: int
            The number of timesteps.
        features: int
            The number of features.
        outputs: int
            The number of outputs.
        """
        self.timesteps = timesteps
        self.features = features
        self.outputs = outputs

    def build(self, hp):
        """Builds and compiles the model.

        Questions
        ---------
            Do I need Input layer and/or input_shape?
            Is it worth to stack LSTM layers?

            Is it worth to include Regularisation?
                We can add dropout for regularisation by setting the
                recurrent_dropout value. Alternatively, it is also possible
                to include a Dropout layer.

            Is it worth adding time awareness?
                The time series are not very long. If time series have
                enough samples, it is possible to use ACF and PACF (see
                R1) as an additional input variable.

        Notes
        -----
            Using L1 and L2 regularization is called ElasticNet

            Choosing the right amount of nodes and layers [R3]
                - A common rule:
                    N_h = (N_s) / (alpha * (N_i + N_o))
                - A basic rule:
                    N_h = (2/3) * (N_i + N_o)

                    where
                       N_i is the number of input neurons
                       N_o is the number of output neurons
                       N_s is the number of samples in training data
                       alpha scaling factor between [2, 10]

            Ideally, every LSTM layer should be accompanied by a Dropout layer
            where 20% os often used as a good compromise between retaining model
            accuracy and preventing overfitting.

            In general, with one LSTM layer is generally enough.

            In general, loss function and activation function are chosen together.
                - activation is 'softmax' then loss is 'binary cross-entropy'

            The correct bias is b_0 = log(pos/neg) thus...
                Dense(1, activation='sigmoid', bias_initializer=
                    tf.keras.initializers.Constant(np.log([pos/neg]))

        Options
        -------
            hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
            hp_units = hp.Choice('units', [8, 16, 32]),
            regularizers = hp.Choice('bias_regularizer',
                L1L2(l1=0.0, l2=0.0), # None
                L1L2(l1=0.1, l2=0.0), # 'l1'
                L1L2(l1=0.0, l2=0.1)  # 'l2'
            ]),

        Parameters
        ----------
        hp: dict-like
            The hyper-parameters map.

        Returns
        -------
        The model.
        """
        # Libraries
        #import tensorflow as tf
        #from tensorflow.keras.layers import LSTM
        #from tensorflow.keras.layers import Input
        #from tensorflow.keras.layers import Dense
        #from tensorflow.keras.initializers import GlorotUniform
        #from tensorflow.keras.initializers import Orthogonal

        from keras.regularizers import L1L2

        # Constants (for reproducibility)
        initializer_glorot = GlorotUniform(seed=SEED)
        initializer_orthogonal = Orthogonal(gain=1.0, seed=SEED)

        # Create model
        model = tf.keras.Sequential()
        model.add(Input(
            shape=(self.timesteps, self.features)
        ))
        model.add(LSTM(
            units=hp.Int('units', min_value=32, max_value=512, step=32),
            activation=hp.Choice('activation', ['relu', 'tanh']),
            dropout=hp.Choice('dropout', [0., 0.2]),
            recurrent_dropout=hp.Choice('recurrent_dropout', [0., 0.2]),
            input_shape=(self.timesteps, self.features),
            kernel_initializer=initializer_glorot,
            recurrent_initializer=initializer_orthogonal,
            bias_regularizer=hp.Choice('bias_regularizer',
                ['l1', 'l2']), #l1l2
            recurrent_regularizer=hp.Choice('recurrent_regularizer',
                ['l1', 'l2']), #l1l2
        ))
        model.add(Dense(self.outputs,
            activation='sigmoid',
            kernel_initializer=initializer_glorot
        ))

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        # Optimizer
        optimizer =  tf.keras.optimizers.Adamax(learning_rate=hp_learning_rate)

        # Compile model
        # See https://neptune.ai/blog/keras-metrics
        model.compile(
            loss='binary_crossentropy',  # f1_loss, binary_crossentropy
            optimizer=optimizer,
            metrics=[
                METRICS.get('acc'),
                METRICS.get('prec'),
                METRICS.get('recall'),
                METRICS.get('auc'),
                METRICS.get('prc'),
                METRICS.get('tp'),
                METRICS.get('tn'),
                METRICS.get('fp'),
                METRICS.get('fn'),
            ]
        )

        return model

    def fit(self, hp, model, *args, **kwargs):
        """"""
        return model.fit(
            *args,
            shuffle=hp.Boolean("shuffle"), # Shuffle the data in each epoch.
            **kwargs
        )



class KerasTunerGRU(keras_tuner.HyperModel):
    """Creates an GRU hypermodel.

    This is the model shared by Damien K. Ming.

    """
    def __init__(self, timesteps, features, outputs):
        """Creates the constructor.

        Parameters
        ----------
        timesteps: int
            The number of timesteps.
        features: int
            The number of features.
        outputs: int
            The number of outputs.
        """
        self.timesteps = timesteps
        self.features = features
        self.outputs = outputs

    def build(self, hp):
        """Builds and compiles the model.

        Questions
        ---------
            Do I need Input layer and/or input_shape?
            Is it worth to stack LSTM layers?

            Is it worth to include Regularisation?
                We can add dropout for regularisation by setting the
                recurrent_dropout value. Alternatively, it is also possible
                to include a Dropout layer.

            Is it worth adding time awareness?
                The time series are not very long. If time series have
                enough samples, it is possible to use ACF and PACF (see
                R1) as an additional input variable.

        Notes
        -----
            Using L1 and L2 regularization is called ElasticNet

            Choosing the right amount of nodes and layers [R3]
                - A common rule:
                    N_h = (N_s) / (alpha * (N_i + N_o))
                - A basic rule:
                    N_h = (2/3) * (N_i + N_o)

                    where
                       N_i is the number of input neurons
                       N_o is the number of output neurons
                       N_s is the number of samples in training data
                       alpha scaling factor between [2, 10]

            Ideally, every LSTM layer should be accompanied by a Dropout layer
            where 20% os often used as a good compromise between retaining model
            accuracy and preventing overfitting.

            In general, with one LSTM layer is generally enough.

            In general, loss function and activation function are chosen together.
                - activation is 'softmax' then loss is 'binary cross-entropy'

            The correct bias is b_0 = log(pos/neg) thus...
                Dense(1, activation='sigmoid', bias_initializer=
                    tf.keras.initializers.Constant(np.log([pos/neg]))

        Options
        -------
            hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
            hp_units = hp.Choice('units', [8, 16, 32]),
            regularizers = hp.Choice('bias_regularizer',
                L1L2(l1=0.0, l2=0.0), # None
                L1L2(l1=0.1, l2=0.0), # 'l1'
                L1L2(l1=0.0, l2=0.1)  # 'l2'
            ]),

        Parameters
        ----------
        hp: dict-like
            The hyper-parameters map.

        Returns
        -------
        The model.
        """

        # Libraries
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Flatten
        from keras.layers import Dropout
        from keras.layers import LSTM
        from keras import regularizers
        from keras.layers import BatchNormalization
        from keras.layers import GRU
        import numpy as np

        # Constants (for reproducibility)
        initializer_glorot = GlorotUniform(seed=SEED)
        initializer_orthogonal = Orthogonal(gain=1.0, seed=SEED)
        initializer_constant = Constant(np.array([-2.65917355]))

        # Create model
        model = Sequential()
        model.add(GRU(
            units=hp.Int('units', min_value=32, max_value=256, step=32),
            activation='tanh',
            input_shape=(self.timesteps, self.features),
            return_sequences=True))
        model.add(Dropout(rate=0.2))
        model.add(BatchNormalization())
        model.add(GRU(
            units=hp.Int('units', min_value=32, max_value=128, step=32),
            activation='tanh',
        ))
        model.add(Dropout(rate=0.2))
        model.add(BatchNormalization())
        model.add(Dense(
            units=hp.Int('units', min_value=10, max_value=20, step=2),
            activation='tanh',
            kernel_regularizer=regularizers.l2(0.01))
        )
        model.add(Dense(1,
            activation='sigmoid',
            bias_initializer=initializer_constant)
        )

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        # Optimizer
        optimizer =  tf.keras.optimizers.Adamax(learning_rate=hp_learning_rate)

        # Compile model
        # See https://neptune.ai/blog/keras-metrics
        model.compile(
            loss='binary_crossentropy',  # f1_loss, binary_crossentropy
            optimizer=optimizer,
            metrics=[
                METRICS.get('acc'),
                METRICS.get('prec'),
                METRICS.get('recall'),
                METRICS.get('auc'),
                METRICS.get('prc'),
                METRICS.get('tp'),
                METRICS.get('tn'),
                METRICS.get('fp'),
                METRICS.get('fn'),
            ]
        )

        return model

    def fit(self, hp, model, *args, **kwargs):
        """"""
        return model.fit(
            *args,
            shuffle=hp.Boolean("shuffle"), # Shuffle the data in each epoch.
            **kwargs
        )

