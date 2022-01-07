import tensorflow as tf

class Config:
    """Configuration for dataset construction.
    """
    def __init__(self):
        # audio config
        self.sr = 22050

        # stft
        self.hop = 128
        self.win = 1024
        self.fft = self.win
        self.win_fn = 'hann'

        # path to files
        self.path_base = '/mnt/md1/genis/musdb18hq/'
        self.path_spec = '/mnt/md1/genis/musdb18hq/train/complex/'
        self.path_audio = '/mnt/md1/genis/musdb18hq/train/raw_audio/'

        # model input shape
        #self.unet_input_shape = [512, 128, 1]

        # mel-scale filter bank
        self.mel = 80
        self.fmin = 0
        self.fmax = 8000

        self.eps = 1e-5

        # sample size
        self.frames = 8192 # 16000
        self.batch = 8

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
