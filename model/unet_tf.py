from typing import Optional, Union, Callable, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.engine import input_spec

from .unet_config import config as unet_config


class ConvBlock(layers.Layer):
    def __init__(self, n_filters, initializer, activation, kernel_size=(5, 5), strides=(2, 2), padding='same'):
        super(ConvBlock, self).__init__()
        self.n_filters=n_filters
        self.initializer=initializer
        self.activation=activation
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding

        self.conv2d = layers.Conv2D(filters=n_filters,
                                      kernel_size=(5, 5),
                                      padding=padding,
                                      strides=(2, 2),
                                      kernel_initializer=initializer)

        self.batch_norm_encoder = layers.BatchNormalization(momentum=0.9, scale=True)
        self.activation_encoder = _get_activation(activation)

    def call(self, inputs):
        x = inputs
        x = self.conv2d(x)
        x = self.batch_norm_encoder(x)
        x = self.activation_encoder(x)

        #if training:
        #    x = self.dropout_1(x)
        #x = self.activation_1(x)
        #x = self.conv2d_2(x)

        #if training:
        #    x = self.dropout_2(x)

        #x = self.activation_2(x)
        return x


class UpconvBlock(layers.Layer):
    def __init__(self, n_filters, initializer, activation, dropout, skip, kernel_size=(5, 5), strides=(2, 2), padding='same'):
        super(UpconvBlock, self).__init__()
        self.n_filters=n_filters
        self.initializer=initializer
        self.activation=activation
        self.dropout=dropout
        self.skip=skip
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding

        self.concatenate_decoder = layers.Concatenate(axis=3)
        self.deconv = layers.Conv2DTranspose(
            n_filters, kernel_size=kernel_size, padding=padding, strides=strides,
            kernel_initializer=initializer)
        self.batch_norm = layers.BatchNormalization(momentum=0.9, scale=True)
        self.dropout_decoder = layers.Dropout(0.5)
        self.activation_decoder = _get_activation(activation)

    def call(self, x, x_encoder):

        if self.skip:
            x = self.concatenate_decoder([x, x_encoder])
        x = self.deconv(x)
        x = self.batch_norm(x)
        if self.dropout:
            x = self.dropout_decoder(x)
        x = self.activation_decoder(x)

        return x


class UnetConditioner(tf.keras.Model):
    def __init__(self):
        super(UnetConditioner, self).__init__()
        self.num_layers = unet_config.N_LAYERS
    
    def call(self, inputs):

        x = inputs
        encoder_layers = []
        initializer = tf.random_normal_initializer(stddev=0.02)
        # Encoder
        for i in range(self.num_layers):
            n_filters = unet_config.FILTERS_LAYER_1 * (2 ** i)
            x = ConvBlock(n_filters, initializer, unet_config.ACTIVATION_ENCODER)(x)
            encoder_layers.append(x)
        # Decoder
        for i in range(self.num_layers):
            # parameters each decoder layer
            is_final_block = i == self.num_layers - 1  # the las layer is different
            # not dropout in the first block and the last two encoder blocks
            dropout = not (i == 0 or i == self.num_layers - 1 or i == self.num_layers - 2)
            # for getting the number of filters
            encoder_layer = encoder_layers[self.num_layers - i - 1]
            skip = i > 0    # not skip in the first encoder block - the deepest
            if is_final_block:
                n_filters = 1
                activation = unet_config.ACT_LAST
            else:
                n_filters = encoder_layer.get_shape().as_list()[-1] // 2
                activation = unet_config.ACTIVATION_DECODER
            x = UpconvBlock(n_filters, initializer, activation, dropout, skip)(x, encoder_layer)

        return layers.multiply([inputs, x])


def _get_activation(name):
    if name == 'leaky_relu':
        return layers.LeakyReLU(alpha=0.2)
    return tf.keras.layers.Activation(name)