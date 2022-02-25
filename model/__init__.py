import math
from sklearn import gaussian_process
import tqdm
import timeit
import random
import numpy as np
import tensorflow as tf
import pdb

from .wavenet import WaveNet


class DiffWave(tf.keras.Model):
    """DiffWave: A Versatile Diffusion Model for Audio Synthesis.
    Zhifeng Kong et al., 2020.
    """
    def __init__(self, config):
        """Initializer.
        Args:
            config: Config, model configuration.
        """
        super(DiffWave, self).__init__()
        self.config = config
        self.wavenet = WaveNet(config)

    def call(self, mixture, cond=None):
        """Generate denoised audio.
        Args:
            mel: tf.Tensor, TODO
            noise: Optional[tf.Tensor], [B, T], starting noise.
        Returns:
            tuple,
                signal: tf.Tensor, [B, T], predicted output.
                ir: List[np.ndarray: [B, T]], intermediate outputs.
        """
        #alpha = 1 - self.config.beta
        #alpha_bar = np.cumprod(alpha)
        base = tf.ones([tf.shape(mixture)[0]], dtype=tf.int32)

        ir, signal = [], mixture
        for t in range(self.config.iter, 0, -1):
            # [B, T]
            eps = self.pred_noise(signal, base * t, cond)
            # [B, T], []
            #mu, sigma = self.pred_signal(signal, eps, alpha[t - 1], alpha_bar[t - 1])
            signal = tf.subtract(signal, eps)
            # [B, T]
            #signal = mu + tf.random.normal(tf.shape(signal)) * sigma
            ir.append(signal.numpy())
        # [B, T], iter x [B, T]
        return signal, ir

    def diffusion(self, vocals, accomp, noise_idx):
        """Trans to next state with diffusion process.
        Args:
            signal: tf.Tensor, [B, T], signal.
            alpha_bar: Union[float, tf.Tensor: [B]], cumprod(1 -beta).
            eps: Optional[tf.Tensor: [B, T]], noise.
        Return:
            tuple,
                noised: tf.Tensor, [B, T], noised signal.
                eps: tf.Tensor, [B, T], noise.
        """
        conditioning = lambda x, y, z : self.soft_diffusion(x, y, z)
        diff = list(map(conditioning, vocals, accomp, noise_idx))
        return np.array(diff)[:, 0, :], np.array(diff)[:, 1, :]

    def soft_diffusion(self, vocals, accomp, noise_idx):
        """Trans to next state with diffusion process.
        Args:
            signal: tf.Tensor, [B, T], signal.
            alpha_bar: Union[float, tf.Tensor: [B]], cumprod(1 -beta).
            eps: Optional[tf.Tensor: [B, T]], noise.
        Return:
            tuple,
                noised: tf.Tensor, [B, T], noised signal.
                eps: tf.Tensor, [B, T], noise.
        """
        #stddev = np.sqrt((1 - alpha_bar / alpha) / (1 - alpha_bar) * (1 - alpha))
        accomp_spec = tf.signal.stft(
            accomp,
            frame_length=self.config.cond_win,
            frame_step=self.config.cond_hop,
            fft_length=self.config.cond_win,
            window_fn=tf.signal.hann_window)
        # Getting mag and phase
        accomp_mag = tf.abs(accomp_spec)
        accomp_phase = tf.math.angle(accomp_spec)
        orig_shape = accomp_mag.shape
        # Flatenning spec
        accomp_mag = tf.reshape(accomp_mag, orig_shape[0]*orig_shape[1])
        accomp_phase = tf.reshape(accomp_phase, orig_shape[0]*orig_shape[1])
        #approximately how would the average spectrogram look at each step
        spec_average = accomp_mag.numpy()/self.config.iter
        estimate = np.zeros(orig_shape[0]*orig_shape[1])

        gaussian_function = lambda x : random.gauss(x, x/self.config.noise_ratio)
        for _ in range(noise_idx):
            #generate an estimate around the average, the standard deviation controls the amount of noise
            #estimate += np.array([random.gauss(spec_average[i],spec_average[i]/self.config.noise_ratio) \
            #    for i in range(orig_shape[0]*orig_shape[1])])
            diff = np.array(list(map(gaussian_function, spec_average)))
            estimate += diff
        # Adding a bit of noise
        estimate = np.array(estimate + 0.1*np.random.normal(0, 1/2, [len(estimate)]), dtype=np.float32)
        # Adding original phase
        on_exp = tf.complex(tf.zeros(accomp_phase.shape), accomp_phase)
        on_est = tf.complex(estimate, tf.zeros(estimate.shape, dtype=tf.float32))
        pred_spec = tf.multiply(on_est, tf.exp(on_exp))
        # Getting sampled accomp spec
        pred_spec = tf.reshape(pred_spec, orig_shape)
        # Reconvert to time domain
        pred_spec = tf.signal.inverse_stft(
            pred_spec,
            frame_length=self.config.cond_win,
            frame_step=self.config.cond_hop,
            window_fn=tf.signal.inverse_stft_window_fn(
                self.config.cond_hop,
                forward_window_fn=tf.signal.hann_window))
        # Adding noise (accomp) to vocals
        noised_voc = tf.add(vocals, pred_spec)
        return noised_voc, pred_spec

    def fast_diffusion(self, vocals, accomp, noise_idx):
        accomp_spec = tf.signal.stft(
            accomp,
            frame_length=self.config.cond_win,
            frame_step=self.config.cond_hop,
            fft_length=self.config.cond_win,
            window_fn=tf.signal.hann_window)
        accomp_mag = tf.abs(accomp_spec)
        accomp_phase = tf.math.angle(accomp_spec)
        orig_shape = accomp_mag.shape
        # Flatenning spec
        accomp_mag = tf.reshape(accomp_mag, orig_shape[0]*orig_shape[1])
        accomp_phase = tf.reshape(accomp_phase, orig_shape[0]*orig_shape[1])

        #sum_amplitude = tf.math.reduce_sum(accomp_mag)
        mean_amplitude = tf.math.reduce_mean(accomp_mag)
        #noise = tf.random.normal(accomp_mag.shape, 0, self.config.noise_std*mean_amplitude/N)

        N = self.config.iter * 1.0
        noise_idx = noise_idx.numpy() * 1.0

        pred_spec = tf.math.minimum(accomp_mag,tf.maximum(0,noise_idx*accomp_mag/N + tf.math.abs(tf.random.normal(accomp_mag.shape,0, self.config.noise_std*mean_amplitude/N))))
        #spec_nextstep_independent = tf.math.minimum(accomp_mag,tf.maximum(0,(noise_idx+1)*accomp_mag/N + tf.math.abs(tf.random.normal(accomp_mag.shape,0, self.config.noise_std*mean_amplitude/N))))
        #spec_nextstep = tf.math.minimum(accomp_mag,tf.maximum(0,pred_spec + tf.math.abs(tf.random.normal(accomp_mag.shape,0, self.config.noise_std*mean_amplitude/N))))

        on_exp = tf.complex(tf.zeros(accomp_phase.shape), accomp_phase)
        pred_spec = tf.complex(pred_spec, tf.zeros(pred_spec.shape, dtype=tf.float32))
        pred_spec = tf.multiply(pred_spec, tf.exp(on_exp))
        # Getting sampled accomp spec
        pred_spec = tf.reshape(pred_spec, orig_shape)
        # Reconvert to time domain
        pred_spec = tf.signal.inverse_stft(
            pred_spec,
            frame_length=self.config.cond_win,
            frame_step=self.config.cond_hop,
            window_fn=tf.signal.inverse_stft_window_fn(
                self.config.cond_hop,
                forward_window_fn=tf.signal.hann_window))
        # Adding noise (accomp) to vocals
        noised_voc = tf.add(vocals, pred_spec)
        return noised_voc, pred_spec

    def pred_noise(self, signal, timestep, cond=None):
        """Predict noise from signal.
        Args:
            signal: tf.Tensor, [B, T], noised signal.
            timestep: tf.Tensor, [B], timesteps of current markov chain.
            mel: tf.Tensor, [B, T // hop, M], conditional mel-spectrogram.
        Returns:
            tf.Tensor, [B, T], predicted noise.
        """
        return self.wavenet(signal, timestep, cond)

    def write(self, path, optim=None):
        """Write checkpoint with `tf.train.Checkpoint`.
        Args:
            path: str, path to write.
            optim: Optional[tf.keras.optimizers.Optimizer]
                , optional optimizer.
        """
        kwargs = {'model': self}
        if optim is not None:
            kwargs['optim'] = optim
        ckpt = tf.train.Checkpoint(**kwargs)
        ckpt.save(path)

    def restore(self, path, optim=None):
        """Restore checkpoint with `tf.train.Checkpoint`.
        Args:
            path: str, path to restore.
            optim: Optional[tf.keras.optimizers.Optimizer]
                , optional optimizer.
        """
        kwargs = {'model': self}
        if optim is not None:
            kwargs['optim'] = optim
        ckpt = tf.train.Checkpoint(**kwargs)
        return ckpt.restore(path)
