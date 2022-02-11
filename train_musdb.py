import argparse
import librosa
import crepe
import json
import math
import tqdm
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import soundfile as sf

from scipy.signal import get_window, resample
from scipy.ndimage import filters

from config import Config
from dataset import MUSDB_3SEC
from model import DiffWave

import warnings
warnings.filterwarnings('ignore')

#os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

DATA_DIR =  '/media/genis/musdb18hq/musdb-accomp-4sec/'

class Trainer:
    """WaveGrad trainer.
    """
    def __init__(self, model, lj, config):
        """Initializer.
        Args:
            model: DiffWave, diffwave model.
            lj: LJSpeech, LJ-speec dataset
                which provides already batched and normalized speech dataset.
            config: Config, unified configurations.
        """
        self.model = model
        self.lj = lj
        self.config = config

        self.split = config.train.split // config.data.batch
        self.trainset = self.lj.dataset().take(self.split) \
            .shuffle(config.train.bufsiz) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        self.testset = self.lj.dataset().skip(self.split) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        self.optim = tf.keras.optimizers.Adam(
            config.train.lr(),
            config.train.beta1,
            config.train.beta2,
            config.train.eps)

        self.eval_intval = config.train.eval_intval // config.data.batch
        self.ckpt_intval = config.train.ckpt_intval // config.data.batch

        self.train_log = tf.summary.create_file_writer(
            os.path.join(config.train.log, config.train.name, 'train'))
        self.test_log = tf.summary.create_file_writer(
            os.path.join(config.train.log, config.train.name, 'test'))

        self.ckpt_path = os.path.join(
            config.train.ckpt, config.train.name, config.train.name)

        self.alpha = 1 - config.model.beta
        self.alpha_bar = np.cumprod(1 - config.model.beta)
        self.cmap = tf.constant(plt.get_cmap('viridis').colors, dtype=tf.float32)

        self.loss_path = os.path.join(config.train.ckpt, config.train.name, 'loss.txt')

    def compute_loss(self, vocals, accomp, f0_cond=None):
        """Compute loss for noise estimation.
        Args:
            signal: tf.Tensor, [B, T], raw audio signal segment.
            logmel: tf.Tensor, [B, T // hop, mel], mel-spectrogram.
            T // hop --> num frames
        Returns:
            loss: tf.Tensor, [], L1-loss between noise and estimation.
        """
        # [B]
        bsize = tf.shape(vocals)[0]
        # [B]
        timesteps = tf.random.uniform(
            [bsize], 1, self.config.model.iter + 1, dtype=tf.int32)
        # [B]
        noise_index = timesteps - 1
        # [B]
        noise_alpha = tf.gather(self.alpha, noise_index)
        noise_alpha_bar = tf.gather(self.alpha_bar, noise_index)
        # [B]
        #noise_steps = np.array([self.config.model.alpha_list[x] for x in noise_level])
        # [B, T], [B, T]
        noised, noise = self.model.diffusion(
            vocals, accomp, noise_index, noise_alpha, noise_alpha_bar)
        # [B, T]
        eps = self.model.pred_noise(noised, timesteps, f0_cond)
        # []
        loss = tf.reduce_mean(tf.abs(eps - noise))
        return loss

    def get_f0_conditions(self, signal):
        """Compute conditions
        """
        conditioning = lambda x : self.get_condition(x)
        cond_out = list(map(conditioning, signal))
        return np.array(cond_out, dtype='float32')

    def get_condition(self, audio):
        audio = self.filter_audio(audio, coef=0.0001)
        _, frequency, _, _ = crepe.predict(audio, 22050, viterbi=True, verbose=0)
        frequency[frequency>600] = 0.0
        frequency[frequency<80] = 0.0
        f0_resampled = resample(frequency, int(len(audio)/self.config.data.hop))
        #f0_resampled = resample(frequency, int(len(audio)))
        f0_resampled = f0_resampled.clip(0).astype('float32')
        # One-hot encode F0 track
        freq_grid = librosa.cqt_frequencies(360, 32.7, 60)
        f_bins = self.grid_to_bins(freq_grid, 0.0, freq_grid[-1])
        n_freqs = len(freq_grid)
        freqz = np.zeros((f0_resampled.shape[0], f_bins.shape[0]))
        haha = np.digitize(f0_resampled, f_bins) - 1
        idx2 = haha < n_freqs
        haha = haha[idx2]
        freqz[range(len(haha)), haha] = 1
        atb = filters.gaussian_filter1d(freqz.T, 1, axis=0, mode='constant').T
        min_target = np.min(atb[range(len(haha)), haha])
        atb = atb / min_target
        atb[atb > 1] = 1
        return atb

    @staticmethod
    def filter_audio(audio, coef):
        """
        Code taken from Baris Bozkurt's MIR teaching notebooks
        """
        audio_modif = audio.copy()
        hop_size=256
        frame_size=1024
        start_indexes = np.arange(0, audio.size - frame_size, hop_size, dtype=int)
        num_windows = start_indexes.size
        w = get_window('blackman', frame_size)
        energy = []
        for k in range(num_windows):
            x_win = audio[start_indexes[k]:start_indexes[k] + frame_size] * w
            energy.append(np.sum(np.power(x_win, 2)))
        for k in range(num_windows):
            x_win = audio[start_indexes[k]:start_indexes[k] + frame_size] * w
            energy_frame = np.sum(np.power(x_win, 2))
            if energy_frame < np.max(energy) * coef:
                audio_modif[start_indexes[k]:start_indexes[k] + frame_size] = np.zeros(frame_size)
        return audio_modif

    @staticmethod
    def grid_to_bins(grid, start_bin_val, end_bin_val):
        """Compute the bin numbers from a given grid
        """
        bin_centers = (grid[1:] + grid[:-1])/2.0
        bins = np.concatenate([[start_bin_val], bin_centers, [end_bin_val]])
        return bins

    def train(self, step=0, ir_unit=5):
        """Train wavegrad.
        Args:
            step: int, starting step.
            ir_unit: int, log ir units.
        """
        count = self.get_count()
        loss_file = open(self.loss_path, mode='a')
        # Start training
        for _ in tqdm.trange(step // self.split, self.config.train.epoch):
            train_loss = []
            with tqdm.tqdm(total=self.split, leave=False) as pbar:
                for _, vocal, accomp in self.trainset:
                    #f0_cond = self.get_f0_conditions(vocal.numpy())
                    with tf.GradientTape() as tape:
                        tape.watch(self.model.trainable_variables)
                        #loss = self.compute_loss(vocal, accomp, f0_cond)
                        loss = self.compute_loss(vocal, accomp)
                        train_loss.append(loss)

                    grad = tape.gradient(loss, self.model.trainable_variables)
                    self.optim.apply_gradients(
                        zip(grad, self.model.trainable_variables))

                    norm = tf.reduce_mean([tf.norm(g) for g in grad])
                    del grad

                    step += 1
                    pbar.update()
                    pbar.set_postfix(
                        {'loss': loss.numpy().item(),
                         'step': step,
                         'grad': norm.numpy().item()})

                    #with self.train_log.as_default():
                    #    tf.summary.scalar('loss', loss, step)
                    #    tf.summary.scalar('grad norm', norm, step)
                    #    if step % self.eval_intval == 0:
                    #        pred, _ = self.model(logmel)
                    #        tf.summary.audio(
                    #            'train', pred[..., None], self.config.data.sr, step)
                    #        del pred

                    if step % self.ckpt_intval == 0:
                        self.model.write(
                            '{}_{}.ckpt'.format(self.ckpt_path, step),
                            self.optim)

            train_loss = sum(train_loss) / len(train_loss)
            print('\nTrain loss:', str(round(train_loss.numpy(),5)))
            loss = []
            for _, vocal, accomp in self.testset:
                #f0_cond = self.get_f0_conditions(vocal.numpy())
                #actual_loss = self.compute_loss(vocal, accomp, f0_cond).numpy().item()
                actual_loss = self.compute_loss(vocal, accomp).numpy().item()
                loss.append(actual_loss)
                
            loss = sum(loss) / len(loss)
            print('Eval loss:', str(round(loss, 5)))

            # Writing to file
            loss_file.write('Iter: ' + str(count))
            loss_file.write('Train: ' + str(train_loss.numpy()) + ', Eval: ' + str(loss) + '\n')

            with self.test_log.as_default():
                tf.summary.scalar('loss', loss, step)

                mix_gt, voc_gt, pred = self.eval()
                tf.summary.audio(
                    'gt', voc_gt[None, :, None], self.config.data.sr, step)
                tf.summary.audio(
                    'eval', pred[None, :, None], self.config.data.sr, step)

                filename = os.path.join(self.config.train.sounds, 'iter_' + str(count) + '.wav')
                sf.write(filename, pred, 22050)
                #if pred_avg is not None:
                #    sf.write(filename.replace('iter', 'avg_iter'), pred_avg, 22050)
                if count == 1:
                    sf.write(filename.replace('iter', 'gt_iter'), voc_gt, 22050)
                
                #for i in range(0, len(ir), ir_unit):
                #    tf.summary.audio(
                #        'ir_{}'.format(i),
                #        np.clip(ir[i][None, :, None], -1., 1.),
                #        self.config.data.sr, step)
                
                del mix_gt, voc_gt, pred
            
            count += 1 

    def eval(self):
        """Generate evaluation purpose audio.
        Returns:
            speech: np.ndarray, [T], ground truth.
            pred: np.ndarray, [T], predicted.
            ir: List[np.ndarray], config.model.iter x [T],
                intermediate representations.
        """
        # [T]
        mixture, vocals, accomp = next(iter(lj.validation()))
        # Convert mixture and speech to numpy to write
        accomp = tf.squeeze(accomp, axis=0).numpy()

        hop = self.config.data.hop
        nearest_hop = hop * math.floor(mixture.shape[1]/hop)
        mixture = mixture[:, :nearest_hop]

        # Compute condition
        #f0_cond = self.get_f0_conditions(vocals.numpy())

        # If more than 1 repetition: compute several and average
        # Else: taking single prediction
        #pred, _ = self.model(mixture, f0_cond)
        pred, _ = self.model(mixture)
        pred = tf.squeeze(pred, axis=0).numpy()
        #return mixture, speech, pred, ir
        vocals = tf.squeeze(vocals, axis=0).numpy()
        mixture = tf.squeeze(mixture, axis=0).numpy()
        est_loss = tf.abs(np.sum(vocals - pred) / pred.shape[0])
        print('Estimation loss:', est_loss.numpy())
        print('Max/min/mean vocals: ', tf.reduce_max(vocals).numpy(), tf.reduce_min(vocals).numpy(), tf.reduce_mean(vocals).numpy())
        print('Mix/min/mean prediction: ', tf.reduce_max(pred).numpy(), tf.reduce_min(pred).numpy(), tf.reduce_mean(pred).numpy())
        return mixture, vocals, pred

    def get_count(self):
        iters = glob.glob(os.path.join(self.config.train.sounds, 'iter_*'))
        if len(iters) > 2:
            iter_nums = [int(x.split('/')[-1].replace('iter_', '').replace('.wav', '')) for x in iters]
            return max(iter_nums) + 1
        else:
            return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    parser.add_argument('--load-step', default=0, type=int)
    parser.add_argument('--ir-unit', default=10, type=int)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--download', default=False, action='store_true')
    parser.add_argument('--from-raw', default=False, action='store_true')
    parser.add_argument('--pre-unet', default=None)
    args = parser.parse_args()

    config = Config()
    if args.config is not None:
        print('[*] load config: ' + args.config)
        with open(args.config) as f:
            config = Config.load(json.load(f))

    log_path = os.path.join(config.train.log, config.train.name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    print('hola') 
    ckpt_path = os.path.join(config.train.ckpt, config.train.name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    sounds_path = os.path.join(config.train.sounds, config.train.name)
    if not os.path.exists(sounds_path):
        os.makedirs(sounds_path)

    lj = MUSDB_3SEC(config.data, data_dir=DATA_DIR)
    diffwave = DiffWave(config.model)
    trainer = Trainer(diffwave, lj, config)

    if args.load_step > 0:
        super_path = os.path.join(config.train.ckpt, config.train.name)
        ckpt_path = '{}_{}.ckpt'.format(config.train.name, args.load_step)
        print(os.listdir(super_path))
        ckpt_path = next(name for name in os.listdir(super_path) if name.startswith(ckpt_path) and name.endswith('.index'))
        ckpt_path = os.path.join(super_path, ckpt_path[:-6])
        print('[*] load checkpoint: ' + ckpt_path)
        trainer.model.restore(ckpt_path, trainer.optim)

    #print(config)
    #with open(os.path.join(config.train.ckpt, config.train.name + '.json'), 'w') as f:
    #    json.dump(config.dump(), f)

    trainer.train(args.load_step, args.ir_unit)
