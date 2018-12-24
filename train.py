# -*- coding: utf-8 -*-
from __future__ import print_function

from config import get_config

from model import DeepVoiceV3
from model import Tacotron2

import tensorflow as tf
import numpy as np

__AUTHOR__ = "kozistr"
__VERSION__ = "0.1"

cfg, _ = get_config()  # configuration

# set random seed
np.random.seed(cfg.seed)
tf.set_random_seed(cfg.seed)


def main():
    pass


if __name__ == "__main__":
    main()
