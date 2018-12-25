# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

from config import get_config

from scipy import signal

import numpy as np
import librosa
import copy

__AUTHOR__ = "kozistr"
__VERSION__ = "0.1"

cfg, _ = get_config()  # configuration

# set random seed
np.random.seed(cfg.seed)


epsilon = 1e-8


def invert_spectrogram(spectrogram, hop_length, win_length):
    """ referenced https://github.com/Kyubyong/tacotron/blob/master/utils.py
    :param spectrogram: [f, t]
    :param hop_length: An int.
    :param win_length: An int.
    :return:
    """
    return librosa.istft(spectrogram, hop_length, win_length=win_length, window="hann")


def griffin_lim(spectrogram):
    """ Applies Griffin-Lim's raw. referenced https://github.com/Kyubyong/tacotron/blob/master/utils.py
    :param spectrogram: [f, t]
    :return:
    """
    hop_length = cfg.sample_rate * cfg.frame_shift
    win_length = cfg.sample_rate * cfg.frame_length

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
    """ Spectrogram2Wav, referenced https://github.com/Kyubyong/tacotron/blob/master/utils.py
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
