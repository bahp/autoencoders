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