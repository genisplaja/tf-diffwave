import os
import glob
from traceback import TracebackException
import tqdm
import math
import random
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import soundfile as sf

TRACK_PATH = '/mnt/md1/genis/musdb18hq/musdb-accomp-4sec/'
'''
for idx, track in tqdm.tqdm(enumerate(glob.glob(os.path.join(TRACK_PATH, '*_mixture.wav')))):

    mix = tf.io.read_file(track)
    vocals = tf.io.read_file(track.replace('mixture.wav', 'vocals.wav'))
    accompaniment = tf.io.read_file(track.replace('mixture.wav', 'accompaniment.wav'))
    mix, _ = tf.audio.decode_wav(mix, desired_channels=1)
    vocals, _ = tf.audio.decode_wav(vocals, desired_channels=1)
    accompaniment, _ = tf.audio.decode_wav(accompaniment, desired_channels=1)
    vocals = tf.squeeze(vocals, axis=-1)
    accompaniment = tf.squeeze(accompaniment, axis=-1)
    mix = tf.squeeze(mix, axis=-1)

    len_track = math.floor(mix.shape[0]/(22050*4))
    for idx_2, i in enumerate(np.arange(len_track)):
        num_nonzeros = tf.math.count_nonzero(vocals[i*(22050*4):(i+1)*(22050*4)])
        if num_nonzeros.numpy() > int((22050*4) / 2):
            if tf.math.reduce_max(vocals[i*(22050*4):(i+1)*(22050*4)]).numpy() > 0.0025:
                accomp_track = accompaniment[i*(22050*4):(i+1)*(22050*4)].numpy() * 0.9
                vocal_track = vocals[i*(22050*4):(i+1)*(22050*4)].numpy() * 0.9
                mixture_track = tf.add(accomp_track, vocal_track)
                print(tf.reduce_max(mixture_track).numpy())
                sf.write(
                    os.path.join('/mnt/md1/genis/musdb18hq/musdb-accomp-4sec/', str(idx) + '_' + str(idx_2) + '_accompaniment.wav'),
                    accomp_track,
                    22050)
                sf.write(
                    os.path.join('/mnt/md1/genis/musdb18hq/musdb-accomp-4sec/', str(idx) + '_' + str(idx_2) + '_vocals.wav'),
                    vocal_track,
                    22050)
                sf.write(
                    os.path.join('/mnt/md1/genis/musdb18hq/musdb-accomp-4sec/', str(idx) + '_' + str(idx_2) + '_mixture.wav'),
                    mixture_track.numpy(),
                    22050)
            else:
                print('No!')
        else:
            print('No!')

mixture_audio = tf.squeeze(mixture_audio, axis=-1)[:sr*20]
vocals_audio = tf.squeeze(vocals_audio, axis=-1)[:sr*20]
bass_audio = tf.squeeze(bass_audio, axis=-1)[:sr*20]
drums_audio = tf.squeeze(drums_audio, axis=-1)[:sr*20]
other_audio = tf.squeeze(other_audio, axis=-1)[:sr*20]

mixture_stft = tf.signal.stft(mixture_audio,frame_length=1024,frame_step=128,fft_length=1024, window_fn=tf.signal.hann_window)
vocals_stft = tf.signal.stft(vocals_audio,frame_length=1024,frame_step=128,fft_length=1024, window_fn=tf.signal.hann_window)
bass_stft = tf.signal.stft(bass_audio,frame_length=1024,frame_step=128,fft_length=1024, window_fn=tf.signal.hann_window)
drums_stft = tf.signal.stft(drums_audio,frame_length=1024,frame_step=128,fft_length=1024, window_fn=tf.signal.hann_window)
other_stft = tf.signal.stft(other_audio,frame_length=1024,frame_step=128,fft_length=1024, window_fn=tf.signal.hann_window)

stft_shape = mixture_stft.shape
reshaping = stft_shape[0]*stft_shape[1]

accomp = tf.add(bass_stft, drums_stft)
accomp = tf.add(accomp, other_stft)

sample_num = math.floor((stft_shape[0]*stft_shape[1])/STEPS)

voc = tf.reshape(vocals_stft, reshaping)
acc = tf.reshape(accomp, reshaping)

remaining_idx = list(np.arange(reshaping))
for t in tqdm.tqdm(range(20, 0, -1)):
    mask = np.zeros(reshaping)
    samples = list(random.sample(remaining_idx, sample_num))
    print(np.shape(remaining_idx), np.shape(samples))
    remaining_idx = list(np.setdiff1d(remaining_idx, samples))
    for idx in samples:
        mask[idx] = 1.0
    noise_sample = tf.multiply(acc, mask)
    voc = tf.add(voc, noise_sample)
    voc_aux = tf.reshape(voc, stft_shape)
    voc_aux = tf.signal.inverse_stft(
        voc_aux, 1024, 128, window_fn=tf.signal.inverse_stft_window_fn(128, forward_window_fn=tf.signal.hann_window))
    sf.write('/mnt/md1/genis/diffwave_experiments/tf-diffwave/prova_' + str(t) + '.wav', voc_aux.numpy(), sr)
    del voc_aux
mixture_stft = tf.signal.inverse_stft(
        mixture_stft, 1024, 128, window_fn=tf.signal.inverse_stft_window_fn(128, forward_window_fn=tf.signal.hann_window))
sf.write('/mnt/md1/genis/diffwave_experiments/tf-diffwave/prova_mix.wav', mixture_stft.numpy(), sr)
'''
for track in random.sample(glob.glob(os.path.join(TRACK_PATH, '*_mixture.wav')), 30):
    vocals = tf.io.read_file(track.replace('mixture.wav', 'vocals.wav'))
    accompaniment = tf.io.read_file(track.replace('mixture.wav', 'accompaniment.wav'))
    vocals, _ = tf.audio.decode_wav(vocals, desired_channels=1)
    accompaniment, _ = tf.audio.decode_wav(accompaniment, desired_channels=1)
    vocals = tf.squeeze(vocals, axis=-1)
    accompaniment = tf.squeeze(accompaniment, axis=-1)
    for i in np.arange(1, 52, 10):
        print('T:', i)
        frame_size = 1024
        hop_size = 256
        padlen = frame_size // 2
        # [B, T + win - 1]
        accompaniment = accompaniment[None, ...]
        center_pad = tf.pad(accompaniment, [[0, 0], [padlen, padlen - 1]], mode='reflect')
        accomp = tf.signal.stft(
            center_pad,
            frame_length=frame_size,
            frame_step=hop_size,
            fft_length=frame_size,
            window_fn=tf.signal.hann_window)

        accomp = tf.squeeze(accomp, axis=0)

        orig_shape = accomp.shape
        accomp = tf.reshape(accomp, orig_shape[0]*orig_shape[1])

        noise_idx = list(np.arange(orig_shape[0]*orig_shape[1]))
        mask = np.zeros(orig_shape[0]*orig_shape[1], dtype='complex')

        samples = list(random.sample(
            noise_idx,
            int(math.floor((orig_shape[0]*orig_shape[1])/100) * i)))
        #remaining_idx = list(np.setdiff1d(remaining_idx, samples))
        for idx in samples:
            mask[idx] = 1.0 + 1.0j
        accomp = tf.math.multiply(accomp, mask)
        accomp = tf.reshape(accomp, orig_shape)

        accomp = tf.signal.inverse_stft(
            accomp,
            frame_length=frame_size,
            frame_step=hop_size,
            window_fn=tf.signal.inverse_stft_window_fn(hop_size, forward_window_fn=tf.signal.hann_window))

        print(vocals.shape)
        print(accomp.shape)
        noised_voc = tf.add(vocals, accomp)

        sf.write(
            os.path.join('/mnt/md1/genis/musdb18hq/workspace/', track.replace('_mixture.wav', '') + '_' + str(i) + '_accompaniment.wav'),
            accomp.numpy(),
            22050)

        sf.write(
            os.path.join('/mnt/md1/genis/musdb18hq/workspace/', track.replace('_mixture.wav', '') + '_' + str(i) + '_noised_vocals.wav'),
            noised_voc.numpy(),
            22050)


