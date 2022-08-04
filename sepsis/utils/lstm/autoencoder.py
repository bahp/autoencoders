# Generic
import tensorflow as tf

# Specific
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from tensorflow.keras import optimizers

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