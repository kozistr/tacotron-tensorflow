# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

from config import get_config

from scipy import signal

import numpy as np
import librosa
import copy
import os

__AUTHOR__ = "kozistr"
__REFERENCE__ = "https://github.com/Kyubyong/tacotron/blob/master/utils.py"
__VERSION__ = "0.1"

cfg, _ = get_config()  # configuration

# set random seed
np.random.seed(cfg.seed)


epsilon = 1e-8


def invert_spectrogram(spectrogram, hop_length, win_length):
    """ Inverting spectrogram
    :param spectrogram: [f, t]
    :param hop_length: An int.
    :param win_length: An int.
    :return:
    """
    return librosa.istft(spectrogram, hop_length, win_length=win_length, window="hann")


def griffin_lim(spectrogram):
    """ Applies Griffin-Lim's raw
    :param spectrogram: [f, t]
    :return:
    """
    hop_length = int(cfg.sample_rate * cfg.frame_shift)
    win_length = int(cfg.sample_rate * cfg.frame_length)

    x_best = copy.deepcopy(spectrogram)
    for i in range(cfg.n_iter):
        x_t = invert_spectrogram(x_best, hop_length, win_length)
        est = librosa.stft(x_t, cfg.n_fft, hop_length, win_length=win_length)
        phase = est / np.maximum(epsilon, np.abs(est))
        x_best = spectrogram * phase
    x_t = invert_spectrogram(x_best, hop_length, win_length)
    y = np.real(x_t)
    return y


def spectrogram2wav(mag):
    """ Spectrogram2Wav
    :param mag: magnitude
    :return:
    """
    # Transpose
    mag = mag.T

    # De-Normalize
    mag = (np.clip(mag, 0., 1.) * cfg.max_db) - cfg.max_db + cfg.ref_db

    # To amplitude
    mag = np.power(10., mag * 0.05)

    # Wav reconstruction
    wav = griffin_lim(mag)

    # De-Preemphasis
    wav = signal.lfilter([1], [1, -cfg.preemphasis], wav)

    # Trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def get_spectrogram(path):
    """ Getting normalized log spectrogram from the audio file
    :param path: A str. Full path of an audio file.
    :return: 2D Arrays of shape (T, n_mels) / (T, 1 + n_fft // 2)
    """
    # Loading sound file
    y, sr = librosa.load(path, sr=cfg.sample_rate)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - cfg.preemphasis * y[:-1])

    # STFT
    hop_length = int(cfg.sample_rate * cfg.frame_shift)
    win_length = int(cfg.sample_rate * cfg.frame_length)

    linear = librosa.stft(y=y,
                          n_fft=cfg.n_fft,
                          hop_length=hop_length,
                          win_length=win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1 + n_fft // 2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(cfg.sample_rate, cfg.n_fft, cfg.n_mels)  # (n_mels, 1 + n_fft // 2)
    mel = np.dot(mel_basis, mag)  # (n_mels, T)

    # To decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # Normalize
    mel = np.clip((mel - cfg.ref_db + cfg.max_db) / cfg.max_db, epsilon, 1)
    mag = np.clip((mag - cfg.ref_db + cfg.max_db) / cfg.max_db, epsilon, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1 + n_fft // 2)
    return mel, mag


def load_spectrogram(path):
    mel, mag = get_spectrogram(path)

    ts = mel.shape[0]
    n_pads = cfg.reduction_factor - (ts % cfg.reduction_factor) if ts % cfg.reduction_factor != 0 else 0

    mel = np.pad(mel, [[0, n_pads], [0, 0]], mode="constant").reshape((-1, cfg.n_mels * cfg.reduction_factor))
    mag = np.pad(mag, [[0, n_pads], [0, 0]], mode="constant")
    return mel, mag
