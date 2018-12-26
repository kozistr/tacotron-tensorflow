# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

from config import get_config

import numpy as np
import unicodedata
import codecs
import os
import re

__AUTHOR__ = "kozistr"
__VERSION__ = "0.1"

cfg, _ = get_config()  # configuration

# set random seed
np.random.seed(cfg.seed)


class LJSpeech:

    def __init__(self, path, processed_data=None, save_to="npy", load_from=None):
        """ LJSpeech-1.1 DataSet Loader, reference : https://github.com/Kyubyong/tacotron/blob/master/data_load.py
        :param path: A str. DataSet's path.
        :param processed_data: A str. Pre-Processed data.
        :param save_to: A str. File type to save dataset.
        :param load_from: A str. File type for loading dataset.
        """
        self.path = path
        self.processed_data = processed_data
        self.save_to = save_to
        self.load_from = load_from

        # Several Sanity Check
        assert os.path.isdir(self.path)
        if self.processed_data is not None:
            assert os.path.isfile(self.processed_data)
        assert (self.save_to is None or self.save_to == "npy")
        assert (self.load_from is None or self.load_from == "npy")

        self.metadata_path = os.path.join(self.path, "metadata.csv")
        self.audio_data_path = os.path.join(self.path, "wavs")
        self.vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # P : Padding, E : EoS

        self.text_data = list()
        self.text_len_data = list()
        self.audio_data = list()

        self.c2i = self.char2idx()
        self.i2c = self.idx2char()

        self.load_data()

        if self.save_to is not None:
            self.save()

    def char2idx(self):
        return {char: idx for idx, char in enumerate(self.vocab)}

    def idx2char(self):
        return {idx: char for idx, char in enumerate(self.vocab)}

    def normalize(self, text):
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn')  # Strip accents

        text = text.lower()
        text = re.sub("[^{}]".format(self.vocab), " ", text)
        text = re.sub("[ ]+", " ", text)
        return text

    def load_data(self):
        # read .csv file
        data = codecs.open(self.metadata_path, 'r', encoding="UTF-8").readlines()

        for d in data:
            file_name, _, text = d.strip().split("|")  # split by '|'

            file_path = os.path.join(self.audio_data_path, file_name + ".wav")  # audio file path
            self.audio_data.append(file_path)

            text = self.normalize(text) + "E"
            text = [self.c2i[char] for char in text]
            self.text_len_data.append(len(text))
            self.text_data.append(np.array(text, dtype=np.int32).tostring())

    def save(self):
        pass
