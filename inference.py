import argparse
import os
import json

import librosa
import numpy as np
import tensorflow as tf

from config import Config
from dataset.musdb_3sec import MUSDB_3SEC
from model import DiffWave

LJ_DATA_SIZE = 5633
DATA_DIR = '/mnt/md1/genis/musdb18hq/3-sec-subset/'


def main(args):
    # prepare directory for samples
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    
    # load checkpoint
    checkpoint_dir = '/home/genis/tf-diffwave/ckpt/l1/'
    latest_checkpoint = tf.train.latest_checkpoint(
        checkpoint_dir, latest_filename=None)
    print(latest_checkpoint)

    with open(args.config) as f:
        config = Config.load(json.load(f))

    diffwave = DiffWave(config.model)
    diffwave.restore(latest_checkpoint).expect_partial()

    # open dataset
    lj = MUSDB_3SEC(config.data, data_dir=DATA_DIR)
    if args.offset is None:
        args.offset = config.train.split + \
            np.random.randint(LJ_DATA_SIZE - config.train.split)

    # sample
    print('[*] offset: ', args.offset)
    mixture, vocals = next(iter(lj.rawset.skip(args.offset)))
    # Computing actual len
    len_audio = int(len(mixture))
    mixture = mixture[:mixture.shape[0] // config.data.hop * config.data.hop]
    vocals = vocals[:vocals.shape[0] // config.data.hop * config.data.hop]

    librosa.output.write_wav(
        os.path.join(args.sample_dir, str(args.offset) + '_vocals_gt.wav'),
        vocals.numpy(),
        config.data.sr)

    librosa.output.write_wav(
        os.path.join(args.sample_dir, str(args.offset) + '_mixture_gt.wav'),
        mixture.numpy(),
        config.data.sr)

    spec, _ = lj.mel_fn(mixture[None], vocals[None])
    spec_input = tf.squeeze(spec, axis=0)
    spec_input = tf.squeeze(spec_input, axis=1)
    spec_input = tf.transpose(spec_input, [1, 0])
    segments = prepare_a_song(spec_input, 128, 512)

    # inference    
    noise = tf.random.normal(tf.shape([1, len_audio]))
    librosa.output.write_wav(
        os.path.join(args.sample_dir, str(args.offset) + '_noise.wav'),
        noise.numpy(),
        config.data.sr)

    # [1, T], iter x [1, T]
    output_signal, _ = diffwave(segments)
    num_elements = output_signal.shape[0]*output_signal.shape[1]
    # [T too long]
    pred_concat = tf.reshape(output_signal, num_elements)
    # [Good size]
    pred_audio = pred_concat[:len_audio].numpy()

    librosa.output.write_wav(
        os.path.join(args.sample_dir, '{}.wav'.format(args.offset)),
        pred_audio,
        config.data.sr)

    print('[*] done')

def prepare_a_song(spec, num_frames, num_bands):
    size = spec.shape[1]

    segments = np.zeros(
        (size//(num_frames)+1, num_bands, num_frames, 1),
        dtype=np.float32)

    for index, i in enumerate(np.arange(0, size, num_frames)):
        segment = spec[:num_bands, i:i+num_frames]
        tmp = segment.shape[1]

        if tmp != num_frames:
            segment = np.zeros((num_bands, num_frames), dtype=np.float32)
            segment[:, :tmp] = spec[:num_bands, i:i+num_frames]

        segments[index] = np.expand_dims(segment, axis=2)

    return segments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-dir', default='/mnt/md1/genis/diffwave_experiments/sounds/')
    parser.add_argument('--config', default='./ckpt/l1.json')
    parser.add_argument('--offset', default=None, type=int)
    args = parser.parse_args()
    main(args)
