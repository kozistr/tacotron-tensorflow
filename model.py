# -*- coding: utf-8 -*-
from __future__ import print_function

from config import get_config

from tfutils import _init, _reg
from tfutils import batch_norm
from tfutils import conv1d
from tfutils import biGRU

from modules import highway_network
from modules import conv1d_banks
from modules import prenet

import tensorflow as tf
import numpy as np

__AUTHOR__ = "kozistr"
__VERSION__ = "0.1"

cfg, _ = get_config()  # configuration

# set random seed
np.random.seed(cfg.seed)
tf.set_random_seed(cfg.seed)


def encoder(inputs, is_training=True, scope="Encoder", reuse=None):
    """ Encoder
    :param inputs: A 2D Tensor with shape of [Seq, E], with dtype of intxx.
    :param is_training: A boolean.
    :param scope: A str, Optional scope for 'variable_scope'.
    :param reuse: A boolean. Whether to reuse the weights of a previous layer
        by the same name.
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Encoder PreNet
        prenet_enc = prenet(inputs, is_training=is_training)

        # Encoder CBHG
        enc = conv1d_banks(prenet_enc, n_kernels=cfg.n_encoder_banks, is_training=is_training)
        enc = tf.layers.max_pooling1d(enc, pool_size=2, strides=1, padding='SAME')

        enc = conv1d(enc, n_filters=cfg.embed_size // 2, kernel=3, scope="conv1D-proj-1")
        enc = batch_norm(enc, is_training=is_training, activation_fn=tf.nn.relu, scope="bn-proj-1")

        enc = conv1d(enc, n_filters=cfg.embed_size // 2, kernel=3, scope="conv1D-proj-2")
        enc = batch_norm(enc, is_training=is_training, activation_fn=tf.nn.relu, scope="bn-proj-2")

        enc += prenet_enc  # long skip connection (LSC)

        # highway networks
        for i in range(cfg.n_highway_blocks):
            enc = highway_network(enc, num_units=cfg.embed_size // 2,
                                  scope="highway_network-%d" % i)

        memory = biGRU(enc, num_units=cfg.embed_size // 2, bidirection=True)

    return memory
