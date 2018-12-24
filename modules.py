# -*- coding: utf-8 -*-
from __future__ import print_function

from config import get_config

from tfutils import _init, _reg
from tfutils import conv1d

import tensorflow as tf
import numpy as np

__AUTHOR__ = "kozistr"
__VERSION__ = "0.1"

cfg, _ = get_config()  # configuration

# set random seed
np.random.seed(cfg.seed)
tf.set_random_seed(cfg.seed)


def prenet(inputs, num_units=None, is_training=True, scope='prenet', reuse=None):
    """ PreNet for Encoder and Decoder
    :param inputs: A 2D or 3D Tensor.
    :param num_units: A list of two ints or None. FC units.
    :param is_training: A boolean.
    :param scope: A str, Optional scope for 'variable_scope'.
    :param reuse: A boolean. Whether to reuse the weights of a previous layer
        by the same name.
    :return: A 3D Tensor of shape [batch, T, num_units / 2]
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


def highway_network(inputs, num_units=None, scope="highway", reuse=None):
    """ Highway Network, https://arxiv.org/abs/1505.00387
    :param inputs: A 3D Tensor of shape [batch, T, C]
    :param num_units: An int or None. The number of units in the highway network
        uses the input size if 'None'
    :param scope: A str, Optional scope for 'variable_scope'.
    :param reuse: A boolean. Whether to reuse the weights of a previous layer
        by the same name.
    :return: A 3D Tensor of shape [batch, T, C]
    """
    if num_units is None:
        num_units = inputs.get_shape().as_list()[-1]

    with tf.variable_scope(scope, reuse=reuse):
        _h = tf.layers.dense(inputs=inputs, units=num_units,
                             activation=tf.nn.relu,
                             kernel_initializer=_init,
                             kernel_regularizer=_reg,
                             name="H")
        _t = tf.layers.dense(inputs=inputs, units=num_units,
                             activation=tf.nn.sigmoid,
                             kernel_initializer=_init,
                             kernel_regularizer=_reg,
                             bias_initializer=tf.constant_initializer(-1.),
                             name="T")
        outputs = _h * _t + (1. - _t) * inputs

    return outputs
