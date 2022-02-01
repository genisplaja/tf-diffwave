from tensorflow.keras.layers import (
    Input, Conv1D, Conv2D, Dense, BatchNormalization, Dropout, LeakyReLU
)
import tensorflow as tf
from .unet_config import config

class DenseControl(tf.keras.Model):
    """
    For simple dense control:
        - n_conditions = 6
        - n_neurons = [16, 64, 256]
    For complex dense control:
        - n_conditions = 1008
        - n_neurons = [16, 128, 1024]
    """
    def __init__(self, n_conditions, n_neurons):
        super(DenseControl, self).__init__()
        self.n_conditions=n_conditions
        self.n_neurons=n_neurons
        self.initializer = tf.random_normal_initializer(stddev=0.02)

    def call(self, input_conditions):
        input_dim = [config.Z_DIM] + self.n_neurons[:-1]
        dense = self.dense_block(input_conditions, self.n_neurons, input_dim, self.initializer)
        gammas = Dense(
            self.n_conditions, input_dim=self.n_neurons[-1], activation=config.ACT_G, 
            kernel_initializer=self.initializer)(dense)
        betas = Dense(
            self.n_conditions, input_dim=self.n_neurons[-1], activation=config.ACT_B,
        kernel_initializer=self.initializer)(dense)
        # both = Add()([gammas, betas])
        return input_conditions, gammas, betas

    def dense_block(x, n_neurons, input_dim, initializer, activation='relu'):
        for i, (n, d) in enumerate(zip(n_neurons, input_dim)):
            extra = i != 0
            x = Dense(n, input_dim=d, activation=activation,
                    kernel_initializer=initializer)(x)
            if extra:
                x = Dropout(0.5)(x)
                x = BatchNormalization(momentum=0.9, scale=True)(x)
        return x


class CNNControl(tf.keras.Model):
    """
    For simple dense control:
        - n_conditions = 6
        - n_filters = [16, 32, 128]
    For complex dense control:
        - n_conditions = 1008
        - n_filters = [16, 32, 64]
    """
    def __init__(self, n_conditions, n_filters):
        super(CNNControl, self).__init__()
        self.n_conditions=n_conditions
        self.n_filters=n_filters
        self.initializer = tf.random_normal_initializer(stddev=0.02)

    def call(self, input_conditions):
        '''
        input_conditions = Input(shape=(config.Z_DIM[0], config.Z_DIM[1]))
        '''
        cnn_enc = self.cnn_block(
            self.input_conditions, self.n_filters, config.Z_DIM, config.PADDING, self.initializer)
        # cnn_dec = cnn_block(
        #     input_conditions, n_filters, config.Z_DIM, config.PADDING, initializer
        # )
        gammas = Dense(
            self.n_conditions, input_dim=self.n_filters[-1], activation=config.ACT_G,
            kernel_initializer=self.initializer)(cnn_enc)
        betas = Dense(
            self.n_conditions, input_dim=self.n_filters[-1], activation=config.ACT_B,
            kernel_initializer=self.initializer)(cnn_enc)
        # both = Add()([gammas, betas])
        return input_conditions, gammas, betas
        
    # DO 2D Condition Change Here
    def cnn_block(
        x, n_filters, kernel_size, padding, initializer, activation='relu'):
        
        kernel_shape = 10
        for i, (f, p) in enumerate(zip(n_filters, padding)):
            extra = i != 0
            x = Conv1D(f, kernel_shape, padding=p, activation=activation,
                    kernel_initializer=initializer)(x)
            if extra:
                x = Dropout(0.5)(x)
                x = BatchNormalization(momentum=0.9, scale=True)(x)
        return x