# -*- coding: utf-8 -*-
from __future__ import print_function

from config import get_config

from tfutils import _init, _reg
from tfutils import conv1d

import tensorflow as tf
import numpy as np


__AUTHOR__ = "kozistr"
__VERSION__ = "0.1"


# set random seed
np.random.seed(cfg.seed)
tf.set_random_seed(cfg.seed)


cfg, _ = get_config()  # configuration


def prenet(inputs, num_units=None, is_training=True, scope='prenet', reuse=None):
    """ PreNet for Encoder and Decoder
    :param inputs: A 2D or 3D Tensor.
    :param num_units: A list of two ints or None. FC units.
    :param is_training: A boolean.
    :param scope: A str, Optional scope for 'variable_scope'.
    :param reuse: A boolean. Whether to reuse the weights of a previous layer
        by the same name.
    :return:
    """
    if num_units is None:
        num_units = [cfg.embed_size, cfg.embed_size // 2]

    assert type(num_units) is list

    with tf.variable_scope(scope, reuse=reuse):
        x = tf.layers.dense(inputs=inputs, units=num_units[0],
                            activation=tf.nn.relu,
                            kernel_initializer=_init,
                            kernel_regularizer=_reg,
                            name="fc1")
        x = tf.layers.dropout(x, rate=cfg.dropout,
                              training=is_training,
                              name="do1")

        x = tf.layers.dense(inputs=x, units=num_units[1],
                            activation=tf.nn.relu,
                            kernel_initializer=_init,
                            kernel_regularizer=_reg,
                            name="fc2")
        x = tf.layers.dropout(x, rate=cfg.dropout,
                              training=is_training,
                              name="do2")

    return x
