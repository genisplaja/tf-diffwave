import argparse
import random
import os
import glob
import json
import mir_eval

import soundfile as sf
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from scipy.signal import get_window

from config import Config
from dataset.musdb_3sec import MUSDB_3SEC
from model import DiffWave

LJ_DATA_SIZE = 4074
DATA_DIR =  '/media/genis/genis/musdb18hq/musdb-accomp-4sec/'
MUSDB_DIR = '/media/genis/genis/musdb18hq/test/raw_audio/'
musdb_songs = glob.glob(os.path.join(MUSDB_DIR, '*/'))

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


def main(args):
    # prepare directory for samples
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    
    # load checkpoint
    checkpoint_dir = '/home/genis/diffwave_experiments/source_sep_20_soft/ckpt/l1'
    latest_checkpoint = tf.train.latest_checkpoint(
        checkpoint_dir, latest_filename=None)
    print('CKPT:', latest_checkpoint)

    with open(args.config) as f:
        config = Config.load(json.load(f))

    diffwave = DiffWave(config.model)
    diffwave.restore(latest_checkpoint).expect_partial()

    songs_to_evaluate = random.sample(musdb_songs, 25)
    positive_avg = []
    positive_count = 0
    entire_avg = []
    for i in songs_to_evaluate:
        # sample
        print('[*] offset: ', i.split('/')[-2])
        mixture = tf.io.read_file(os.path.join(i, 'mixture.wav'))
        vocals = tf.io.read_file(os.path.join(i, 'vocals.wav'))
        mixture, _ =  tf.audio.decode_wav(mixture, desired_channels=1)
        vocals, _ = tf.audio.decode_wav(vocals, desired_channels=1)

        mixture = tfio.audio.resample(mixture, 44100, 22050)
        vocals = tfio.audio.resample(vocals, 44100, 22050)

        mixture = tf.squeeze(mixture, axis=-1)
        vocals = tf.squeeze(vocals, axis=-1)

        sec = int(mixture.shape[0] / 22050) - 30
        mixture = mixture[sec*22050:(sec+20)*22050]
        vocals = vocals[sec*22050:(sec+20)*22050]

        # Computing actual len
        len_audio = int(len(mixture))
        mixture = mixture[:mixture.shape[0] // config.data.hop * config.data.hop]
        vocals = vocals[:vocals.shape[0] // config.data.hop * config.data.hop]

        # [1, T], iter x [1, T]
        output_signal, _ = diffwave(mixture[None])
        pred_audio = output_signal[:len_audio].numpy()
        pred_audio = tf.squeeze(pred_audio, axis=0).numpy()

        #pred_audio = filter_audio(pred_audio.numpy(), 0.001)

        if tf.reduce_max(vocals).numpy() < 0.005:
            print('silent!')
        else:
            print(tf.reduce_max(vocals), tf.reduce_min(vocals))
            print(np.max(pred_audio), np.max(pred_audio))

            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(vocals.numpy(), pred_audio)
            print(sdr[0], sir[0], sar[0])
            if sdr >= 0:
                positive_count += 1
                positive_avg.append(sdr[0])
                entire_avg.append(sdr[0])
            else:
                entire_avg.append(sdr[0])

            sf.write(
                os.path.join(args.sample_dir, i.split('/')[-2] + '_' + str(np.round(sdr, 2)) + '_filt_pred.wav'),
                pred_audio,
                config.data.sr)

            sf.write(
                os.path.join(args.sample_dir, i.split('/')[-2] + '_' + str(np.round(sdr, 2)) + '_gt.wav'),
                vocals,
                config.data.sr)
    print('Only positives SDR:', sum(positive_avg)/positive_count)
    print('Only positives SDR median:', np.median(positive_avg))
    print('All output SDR:', sum(entire_avg)/25)
    print('All output SDR median:', np.median(entire_avg))

    # open dataset
    lj = MUSDB_3SEC(config.data, data_dir=DATA_DIR)
    '''
    for i in np.arange(20):
        offset = config.train.split + \
                np.random.randint(LJ_DATA_SIZE - config.train.split)

        # sample
        print('[*] offset: ', offset)
        mixture, vocals, _ = next(iter(lj.rawset.skip(offset)))
        # Computing actual len
        len_audio = int(len(mixture))
        mixture = mixture[:mixture.shape[0] // config.data.hop * config.data.hop]
        vocals = vocals[:vocals.shape[0] // config.data.hop * config.data.hop]

        #spec, _ = lj.mel_fn(mixture[None], vocals[None])
        #spec_input = tf.squeeze(spec, axis=0)
        #spec_input = tf.squeeze(spec_input, axis=1)
        #spec_input = tf.transpose(spec_input, [1, 0])
        #segments = prepare_a_song(spec_input, 128, 512)

        # [1, T], iter x [1, T]
        output_signal, _ = diffwave(mixture[None])
        #num_elements = output_signal.shape[0]*output_signal.shape[1]
        # [T too long]
        #pred_concat = tf.reshape(output_signal, num_elements)
        # [Good size]
        pred_audio = output_signal[:len_audio].numpy()
        pred_audio = tf.squeeze(pred_audio, axis=0)

        sf.write(
            os.path.join(args.sample_dir, str(offset) + '_ss_pred.wav'),
            pred_audio,
            config.data.sr)

        sf.write(
            os.path.join(args.sample_dir, str(offset) + '_ss_vocals_gt.wav'),
            vocals.numpy(),
            config.data.sr)

        sf.write(
            os.path.join(args.sample_dir, str(offset) + '_ss_mix_gt.wav'),
            mixture.numpy(),
            config.data.sr)

    print('[*] done')
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-dir', default='/home/genis/diffwave_experiments/source_sep_20_soft/sounds/l1/eval/')
    parser.add_argument('--config', default='/home/genis/diffwave_experiments/source_sep_20_soft/ckpt/l1.json')
    parser.add_argument('--offset', default=None, type=int)
    args = parser.parse_args()
    main(args)
