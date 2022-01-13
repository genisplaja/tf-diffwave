from effortless_config import Config, setting
import tensorflow as tf
import os

class config(Config):

    groups = ['standard', 'simple_dense', 'complex_dense', 'simple_cnn',
              'complex_cnn']
    MODE = setting(default='conditioned', standard='standard')

    # checkpoints
    EARLY_STOPPING_MIN_DELTA = 1e-8
    EARLY_STOPPING_PATIENCE = 60
    REDUCE_PLATEAU_PATIENCE = 15

    # unet paramters
    INPUT_SHAPE = [512, 128, 1]  # freq = 512, time = 128
    FILTERS_LAYER_1 = 16
    N_LAYERS = 6
    BLOCKS_DROPOUT = [0, N_LAYERS-2, N_LAYERS-1]
    LR = 0.0001
    ACTIVATION_ENCODER = 'leaky_relu'
    ACTIVATION_DECODER = 'relu'
    ACT_LAST = 'sigmoid'
    LOSS = 'mean_absolute_error'

    # -------------------------------

    # control parameters
    CONTROL_TYPE = setting(
        'cnn', simple_dense='dense', complex_dense='dense',
        simple_cnn='cnn', complex_cnn='cnn'
    )
    FILM_TYPE = setting(
        'complex', simple_dense='simple', complex_dense='complex',
        simple_cnn='simple', complex_cnn='complex'
    )
    Z_DIM = [INPUT_SHAPE[1], 360+1] # f0 point for each spec frame
    ACT_G = 'linear'
    ACT_B = 'linear'
    N_CONDITIONS = setting(
        512, simple_dense=6, complex_dense=1008,
        simple_cnn=6, complex_cnn=1008
    )

    # cnn control
    N_FILTERS = setting(
        [32, 64, 256], simple_cnn=[16, 32, 64], complex_cnn=[32, 64, 256]
    )
    PADDING = ['same', 'same', 'same']
    # Dense control
    N_NEURONS = setting(
        [16, 256, 1024], simple_dense=[16, 64, 256],
        complex_dense=[16, 256, 1024]
    )
