import argparse
import os
import json

import soundfile as sf
import numpy as np
import tensorflow as tf

from config import Config
from dataset.musdb_3sec import MUSDB_3SEC
from model import DiffWave

LJ_DATA_SIZE = 4074
DATA_DIR =  '/mnt/md1/genis/musdb18hq/musdb-accomp-4sec/'


def main(args):
    # prepare directory for samples
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    
    # load checkpoint
    checkpoint_dir = '/mnt/md1/genis/diffwave_experiments/source_sep_50/ckpt/l1/'
    latest_checkpoint = tf.train.latest_checkpoint(
        checkpoint_dir, latest_filename=None)
    print('CKPT:', latest_checkpoint)

    with open(args.config) as f:
        config = Config.load(json.load(f))

    diffwave = DiffWave(config.model)
    diffwave.restore(latest_checkpoint).expect_partial()

    # open dataset
    lj = MUSDB_3SEC(config.data, data_dir=DATA_DIR)

    for i in np.arange(10):
        offset = np.random.randint(low=1, high=config.train.split)

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
            os.path.join(args.sample_dir, str(offset) + '_ss_pred_train.wav'),
            pred_audio,
            config.data.sr)

        sf.write(
            os.path.join(args.sample_dir, str(offset) + '_ss_vocals_gt_train.wav'),
            vocals.numpy(),
            config.data.sr)

        sf.write(
            os.path.join(args.sample_dir, str(offset) + '_ss_mix_gt_train.wav'),
            mixture.numpy(),
            config.data.sr)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-dir', default='/mnt/md1/genis/diffwave_experiments/source_sep_50/sounds/eval/')
    parser.add_argument('--config', default='/mnt/md1/genis/diffwave_experiments/source_sep_50/ckpt/l1.json')
    parser.add_argument('--offset', default=None, type=int)
    args = parser.parse_args()
    main(args)
