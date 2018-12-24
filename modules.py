# -*- coding: utf-8 -*-
from __future__ import print_function

from config import get_config

from tfutils import _init, _reg
from tfutils import batch_norm
from tfutils import conv1d

import tensorflow as tf
import numpy as np

__AUTHOR__ = "kozistr"
__VERSION__ = "0.1"

cfg, _ = get_config()  # configuration

# set random seed
np.random.seed(cfg.seed)
tf.set_random_seed(cfg.seed)


def conv1d_banks(inputs, n_kernels=16, is_training=True, scope="conv1d_banks", reuse=None):
    """ Series of conv1d separately
    :param inputs: A 3D Tensor with shape of [batch, T, C]
    :param n_kernels: An int, The size of conv1d banks.
    :param is_training: A boolean.
    :param scope: A str, Optional scope for 'variable_scope'.
    :param reuse: A boolean. Whether to reuse the weights of a previous layer
        by the same name.
    :return: A 3D Tensor with shape of [batch, T, n_kernels * embed_size / 2]
    """
    with tf.variable_scope(scope, reuse=reuse):
        outputs = conv1d(inputs, n_filters=cfg.embed_size // 2, kernel=1)
        for k in range(2, n_kernels + 1):
            with tf.variable_scope("ks-%d" % k) as scope:
                x = conv1d(inputs, n_filters=cfg.embed_size // 2, kernel=k,
                           scope="conv1d-%s" % scope)
                outputs = tf.concat([outputs, x], axis=-1)
        outputs = batch_norm(inputs, is_training=is_training, activation_fn=tf.nn.relu)
    return outputs


def attention_decoder(inputs, memory, num_units=None, attention_method='bahdanau',
                      scope="attention_decoder", reuse=None):
    """
    :param inputs: A 3D tensor with shape of [batch, T', C']. Decoder inputs.
    :param memory: A 3D tensor with shape of [batch, T, C]. Outputs of encoder network.
    :param attention_method: A str. The name of attention mechanism.
    :param num_units: An int, Attention size.
    :param scope: A str, Optional scope for 'variable_scope'.
    :param reuse: A boolean. Whether to reuse the weights of a previous layer
        by the same name.
    :return: A 3D Tensor with shape of [batch, T, num_units]
    """
    if num_units is None:
        num_units = inputs.get_shape().as_list[-1]

    att_mechanism = tf.contrib.seq2seq.BahdanauAttention \
        if attention_method == "bahdanau" else tf.contrib.seq2seq.LuongAttention

    with tf.variable_scope(scope, reuse=reuse):
        decoder_cell = tf.contrib.rnn.GRUCell(num_units)

        attention_mechanism = att_mechanism(num_units=num_units, memory=memory)

        attention = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,
                                                        attention_mechanism=attention_mechanism,
                                                        attention_layer_size=num_units,
                                                        alignment_history=True)

        outputs, state = tf.nn.dynamic_rnn(attention, inputs,
                                           dtype=tf.float32)

    return outputs, state


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
