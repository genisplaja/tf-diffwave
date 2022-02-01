from effortless_config import Config, setting

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
    N_CONDITIONS = 512
    # cnn control
    N_FILTERS = [32, 64, 256]
    PADDING = ['same', 'same', 'same']
    N_NEURONS = [16, 256, 1024]
