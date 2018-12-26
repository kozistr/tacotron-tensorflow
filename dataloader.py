# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

from config import get_config

import numpy as np


__AUTHOR__ = "kozistr"
__VERSION__ = "0.1"

cfg, _ = get_config()  # configuration

# set random seed
np.random.seed(cfg.seed)


class Char2VecEmbeddings:
    """
    Copyright 2018 NAVER Corp.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
    associated documentation files (the "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
    the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial
    portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
    PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
    OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    """

    def __init__(self):
        self.cho = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"  # len = 19
        self.jung = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"  # len = 21
        self.jong = "ㄱ/ㄲ/ㄱㅅ/ㄴ/ㄴㅈ/ㄴㅎ/ㄷ/ㄹ/ㄹㄱ/ㄹㅁ/ㄹㅂ/ㄹㅅ/ㄹㅌ/ㄹㅍ/ㄹㅎ/ㅁ/ㅂ/ㅂㅅ/ㅅ/ㅆ/ㅇ/ㅈ/ㅊ/ㅋ/ㅌ/ㅍ/ㅎ".\
            split('/')  # len = 27
        self.kor_chars = self.cho + self.jung + ''.join(self.jong)

        self.len_jung = len(self.jung)
        self.len_jong = len(self.jong) + 1
        self.hangul_length = len(self.kor_chars)

    def is_valid_char(self, x):
        return x in self.kor_chars

    def decompose(self, x, warning=True):
        in_char = x
        if x < ord('가') or x > ord('힣'):  # not korean char
            return chr(x)

        x -= ord('가')
        y = x // self.len_jong
        z = x % self.len_jong
        x = y // self.len_jung
        y = y % self.len_jung

        zz = self.jong[z - 1] if z > 0 else ''
        if x >= len(self.cho):
            if warning:
                print("[-] Unknown Exception : ", in_char, chr(in_char), x, y, z, zz)
        return self.cho[x] + self.jung[y] + zz

    def decompose_str(self, string, warning=True):
        return ''.join([self.decompose(ord(x), warning=warning) for x in string])

    def decompose_as_one_hot(self, in_char, warning=True):
        # print(ord('ㅣ'), chr(0xac00))
        # [0, 66]: hangul / [67, 194]: ASCII / [195, 245]: hangul danja, danmo / [246, 249]: special characters
        # Total 250 dimensions.

        one_hot = []

        if ord('가') <= in_char <= ord('힣'):  # 가:44032 , 힣: 55203
            x = in_char - ord('가')
            y = x // self.len_jong
            z = x % self.len_jong
            x = y // self.len_jung
            y = y % self.len_jung

            zz = self.jong[z - 1] if z > 0 else ''
            if x >= len(self.cho):
                if warning:
                    print("[-] Unknown Exception : ", in_char, chr(in_char), x, y, z, zz)

            one_hot.append(x)
            one_hot.append(len(self.cho) + y)
            if z > 0:
                one_hot.append(len(self.cho) + len(self.jung) + (z - 1))
            return one_hot
        else:
            if in_char < 128:
                return [self.hangul_length + in_char]  # 67 ~
            elif ord('ㄱ') <= in_char <= ord('ㅣ'):
                return [self.hangul_length + 128 + (in_char - 12593)]  # 194 ~ # [ㄱ:12593] ~ [ㅣ:12643] (len = 51)
            elif in_char == ord('♡'):
                return [self.hangul_length + 128 + 51]  # 245 ~ # ♡
            elif in_char == ord('♥'):
                return [self.hangul_length + 128 + 51 + 1]  # ♥
            elif in_char == ord('★'):
                return [self.hangul_length + 128 + 51 + 2]  # ★
            elif in_char == ord('☆'):
                return [self.hangul_length + 128 + 51 + 3]  # ☆
            else:
                if warning:
                    print("[-] Unhandled character : ", chr(in_char), in_char)
                return []

    def decompose_str_as_one_hot(self, string, warning=True):
        tmp_list = []
        for x in string:
            tmp_list.extend(self.decompose_as_one_hot(ord(x), warning=warning))
        return tmp_list

    def __str__(self):
        return "Char2Vec"


class DataIterator:

    def __init__(self, text, mel, mag, batch_size):
        """ DataLoader
        :param text: An Numpy Object. Text data.
        :param mel: An Numpy Object. mel-spectrogram data.
        :param mag: An Numpy Object. magnitude data.
        :param y: An Numpy Object. Label audio.
        :param batch_size: An int. Batch size.
        """
        self.text = text
        self.mel = mel
        self.mag = mag
        self.batch_size = batch_size

        self.num_examples = num_examples = text.shape[0]
        self.num_batches = num_examples // batch_size
        self.pointer = 0

        assert (self.batch_size <= self.num_examples)

    def next_batch(self):
        start = self.pointer
        self.pointer += self.batch_size

        if self.pointer > self.num_examples:
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)

            self.text = self.text[perm]
            self.mel = self.mel[perm]
            self.mag = self.mag[perm]

            start = 0
            self.pointer = self.batch_size

        end = self.pointer

        return self.text[start:end], self.mel[start:end], self.mag[start:end]

    def iterate(self):
        for step in range(self.num_batches):
            yield self.next_batch()
