import argparse
import os
import glob
import tqdm
import random
import json

import librosa
import numpy as np
import tensorflow as tf

from config import Config
from dataset.musdb_3sec import MUSDB_3SEC
from model import DiffWave

DATA_DIR = '/mnt/md1/genis/Saraga-10sec-subset/'


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
    #lj = MUSDB_3SEC(config.data, data_dir=DATA_DIR)
    #if args.offset is None:
    #    args.offset = config.train.split + \
    #        np.random.randint(LJ_DATA_SIZE - config.train.split)

    # sample
    #print('[*] offset: ', args.offset)
    #mixture, vocals = next(iter(lj.rawset.skip(args.offset)))
    dataset = glob.glob(os.path.join(DATA_DIR, '*_mix.wav'))
    mixture_tracks = random.sample(dataset, 20)
    vocal_tracks = [x.replace('_mix.wav', '_vocal.wav') for x in mixture_tracks]
    for mix, voc in tqdm.tqdm(zip(mixture_tracks, vocal_tracks)):

        mixture, vocals = load_audio([mix, voc])
        track_id = mix.split('/')[-1].replace('_mix.wav', '')

        # Computing actual len
        len_audio = int(len(mixture))
        mixture = mixture[:mixture.shape[0] // config.data.hop * config.data.hop]
        vocals = vocals[:vocals.shape[0] // config.data.hop * config.data.hop]

        librosa.output.write_wav(
            os.path.join(args.sample_dir, str(track_id) + '_vocals_gt.wav'),
            vocals.numpy(),
            config.data.sr)

        librosa.output.write_wav(
            os.path.join(args.sample_dir, str(track_id) + '_mixture_gt.wav'),
            mixture.numpy(),
            config.data.sr)

        spec, _ = mel_fn(mixture[None], vocals[None], config.data)
        spec_input = tf.squeeze(spec, axis=0)
        spec_input = tf.squeeze(spec_input, axis=1)
        spec_input = tf.transpose(spec_input, [1, 0])
        segments = prepare_a_song(spec_input, 128, 512)

        # [1, T], iter x [1, T]
        output_signal, _ = diffwave(segments)
        print(output_signal[0, 2028:])
        print(output_signal[1, :20])
        print(output_signal[1, 2028:])
        print(output_signal[2, :20])
        print(output_signal[2, 2028:])
        num_elements = output_signal.shape[0]*output_signal.shape[1]
        # [T too long]
        pred_concat = tf.reshape(output_signal, num_elements)
        # [Good size]
        pred_audio = pred_concat[:len_audio].numpy()

        librosa.output.write_wav(
            os.path.join(args.sample_dir, str(track_id) + '_synthesis.wav'),
            pred_audio,
            config.data.sr)

        print(track_id + ' synthesized!')


def load_audio(paths):
    """Load audio with tf apis.
    Args:
        path: str, wavfile path to read.
    Returns:
        tf.Tensor, [T], mono audio in range (-1, 1).
    """
    mixture = tf.io.read_file(paths[0])
    vocals = tf.io.read_file(paths[1])
    mixture_audio, _ = tf.audio.decode_wav(mixture, desired_channels=1)
    vocal_audio, _ = tf.audio.decode_wav(vocals, desired_channels=1)
    return tf.squeeze(mixture_audio, axis=-1), tf.squeeze(vocal_audio, axis=-1)


def mel_fn(mixture_signal, vocal_signal, config):
    """Generate log mel-spectrogram from input audio segment.
    Args:
        signal: tf.Tensor, [B, T, 2], audio segment.
    Returns:
        tuple,
            signal: tf.Tensor, [B, T], identity to inputs.
            logmel: tf.Tensor, [B, T // hop, mel], log mel-spectrogram.
    """
    padlen = config.win // 2
    # [B, T + win - 1]
    center_pad = tf.pad(mixture_signal, [[0, 0], [padlen, padlen - 1]], mode='reflect')
    # [B, T // hop, fft // 2 + 1]
    stft = tf.signal.stft(
        center_pad,
        frame_length=config.win,
        frame_step=config.hop,
        fft_length=config.fft,
        window_fn=config.window_fn())

    # [B, T // hop, mel]
    #mel = tf.abs(stft) @ self.melfilter
    # [B, T // hop, mel]
    #logmel = tf.math.log(tf.maximum(mel, self.config.eps))
    # tf.abs(stft) --> magnitude spec of mixture, signal[1] --> vocal signal
    return tf.abs(check_shape(stft)), vocal_signal


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


def check_shape(data):
    n = data.shape[-1]
    if n % 2 != 0:
        n = data.shape[-1] - 1
    return tf.expand_dims(data[:, :, :n], axis=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-dir', default='/mnt/md1/genis/diffwave_experiments/sounds/')
    parser.add_argument('--config', default='./ckpt/l1.json')
    parser.add_argument('--offset', default=None, type=int)
    args = parser.parse_args()
    main(args)
