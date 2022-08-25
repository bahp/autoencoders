import numpy as np

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt


class Autoencoder(object):

    def __init__(self):
        # Encoding
        input_layer = Input(shape=(28, 28, 1))
        encoding_conv_layer_1 = Conv2D(16, (3, 3), activation='relu',
                                       padding='same')(input_layer)
        encoding_pooling_layer_1 = MaxPooling2D((2, 2),

                                                padding='same')(encoding_conv_layer_1)
        encoding_conv_layer_2 = Conv2D(8, (3, 3), activation='relu',
                                       padding='same')(encoding_pooling_layer_1)
        encoding_pooling_layer_2 = MaxPooling2D((2, 2),
                                                padding='same')(encoding_conv_layer_2)
        encoding_conv_layer_3 = Conv2D(8, (3, 3), activation='relu',
                                       padding='same')(encoding_pooling_layer_2)
        code_layer = MaxPooling2D((2, 2), padding='same')(encoding_conv_layer_3)

        # Decoding
        decodging_conv_layer_1 = Conv2D(8, (3, 3), activation='relu',
                                        padding='same')(code_layer)
        decodging_upsampling_layer_1 = UpSampling2D((2, 2))(decodging_conv_layer_1)
        decodging_conv_layer_2 = Conv2D(8, (3, 3), activation='relu',
                                        padding='same')(decodging_upsampling_layer_1)
        decodging_upsampling_layer_2 = UpSampling2D((2, 2))(decodging_conv_layer_2)
        decodging_conv_layer_3 = Conv2D(16, (3, 3),
                                        activation='relu')(decodging_upsampling_layer_2)
        decodging_upsampling_layer_3 = UpSampling2D((2, 2))(decodging_conv_layer_3)
        output_layer = Conv2D(1, (3, 3), activation='sigmoid',
                              padding='same')(decodging_upsampling_layer_3)

        self._model = Model(input_layer, output_layer)
        self._model.compile(optimizer='adadelta', loss='binary_crossentropy')

    def train(self, input_train, input_test, batch_size, epochs):
        self._model.fit(input_train,
                        input_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(
                            input_test,
                            input_test))

    def getDecodedImage(self, encoded_imgs):
        decoded_image = self._model.predict(encoded_imgs)
        return decoded_image


import numpy as np

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt


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


class BinaryClassifierConvV1(Classifier):

    def __init__(self, timesteps, features, outputs):
        # Encoding
        input_layer = Input(shape=(timesteps, features, outputs))
        encoding_conv_layer_1 = Conv2D(16, (3, 3), activation='relu',
                                       padding='same')(input_layer)
        encoding_pooling_layer_1 = MaxPooling2D((2, 2),

                                                padding='same')(encoding_conv_layer_1)
        encoding_conv_layer_2 = Conv2D(8, (3, 3), activation='relu',
                                       padding='same')(encoding_pooling_layer_1)
        encoding_pooling_layer_2 = MaxPooling2D((2, 2),
                                                padding='same')(encoding_conv_layer_2)
        encoding_conv_layer_3 = Conv2D(8, (3, 3), activation='relu',
                                       padding='same')(encoding_pooling_layer_2)
        code_layer = MaxPooling2D((2, 2), padding='same')(encoding_conv_layer_3)

        # Decoding
        decodging_conv_layer_1 = Conv2D(8, (3, 3), activation='relu',
                                        padding='same')(code_layer)
        decodging_upsampling_layer_1 = UpSampling2D((2, 2))(decodging_conv_layer_1)
        decodging_conv_layer_2 = Conv2D(8, (3, 3), activation='relu',
                                        padding='same')(decodging_upsampling_layer_1)
        decodging_upsampling_layer_2 = UpSampling2D((2, 2))(decodging_conv_layer_2)
        decodging_conv_layer_3 = Conv2D(16, (3, 3),
                                        activation='relu')(decodging_upsampling_layer_2)
        decodging_upsampling_layer_3 = UpSampling2D((2, 2))(decodging_conv_layer_3)
        output_layer = Conv2D(1, (3, 3), activation='sigmoid',
                              padding='same')(decodging_upsampling_layer_3)

        self._model = Model(input_layer, output_layer)
        self._model.compile(optimizer='adadelta', loss='binary_crossentropy')

        self.model_ = Model(input_layer, output_layer)
        self.model_.compile(optimizer='adadelta',
                            loss='binary_crossentropy')

    def train(self, input_train, input_test, batch_size, epochs):
        self._model.fit(input_train,
                        input_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(
                            input_test,
                            input_test))

    def getDecodedImage(self, encoded_imgs):
        decoded_image = self._model.predict(encoded_imgs)
        return decoded_image