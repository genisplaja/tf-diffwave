import tensorflow as tf
import numpy as np

from .unet_config import config as unet_config


class Config:
    """Configuration for DiffWave implementation.
    """
    def __init__(self):
        self.sr = 22050

        self.cond_win = 1024
        self.cond_hop = 256

        # mel-scale filter bank
        self.mel = 80
        self.fmin = 0
        self.fmax = 8000

        self.eps = 1e-5

        # sample size
        self.frames = 256 * 32  # 16384
        self.batch = 8

        # leaky relu coefficient
        self.leak = 0.4

        # embdding config
        self.embedding_size = 128
        self.embedding_proj = 512
        self.embedding_layers = 2
        self.embedding_factor = 4

        # upsampler config
        self.upsample_stride = [4, 1]
        self.upsample_kernel = [32, 3]
        self.upsample_layers = 4
        # computed hop size
        self.hop = self.upsample_stride[0] ** self.upsample_layers

        # block config
        self.channels = 64
        self.kernel_size = 3
        self.dilation_rate = 2
        self.num_layers = 30
        self.num_cycles = 3

        # noise schedule
        self.iter = 20                  # 20, 40, 50
        self.noise_policy = 'linear'
        self.noise_start = 1e-4
        #self.noise_end = 0.05           # 0.02 for 200
        self.noise_end = 0.2     # TRY IT OUT FOR SOURCE SEP APPROACH

        self.beta = np.linspace(
            self.noise_start, self.noise_end, self.iter, dtype=np.float32)

        #self.noise_ratio = 16

    def window_fn(self):
        """Return window generator.
        Returns:
            Callable, window function of tf.signal
                , which corresponds to self.win_fn.
        """
        mapper = {
            'hann': tf.signal.hann_window,
            'hamming': tf.signal.hamming_window
        }
        if self.win_fn in mapper:
            return mapper[self.win_fn]
        
        raise ValueError('invalid window function: ' + self.win_fn)