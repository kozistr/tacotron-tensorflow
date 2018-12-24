# -*- coding: utf-8 -*-
from __future__ import print_function

from config import get_config

import tensorflow as tf
import numpy as np


__AUTHOR__ = "kozistr"
__VERSION__ = "0.1"


cfg, _ = get_config()  # configuration


# set random seed
np.random.seed(cfg.seed)
tf.set_random_seed(cfg.seed)


_init = tf.contrib.layers.variance_scaling_initializer(factor=3., mode='FAN_AVG', uniform=True)
_reg = tf.contrib.layers.l2_regularizer(cfg.l2_reg)


def embedding_table(inputs, vocab_size, embed_size, zero_pad=False,
                    trainable=True, scope="embedding", reuse=None):
    """
    Generating Embedding Table with given parameters
    :param inputs: A 'Tensor' with type 'int8' or 'int16' or 'int32' or 'int64'
        containing the ids to be looked up in 'lookup table'.
    :param vocab_size: An int. Vocabulary size.
    :param embed_size: An int. Number of size of embedding vector.
    :param zero_pad: A boolean. If True, all the values of the first low (id 0)
        should be constant zeros.
    :param trainable: A boolean. Whether freeze the embedding matrix or not.
    :param scope: A str, Optional scope for 'variable_scope'.
    :param reuse: A boolean. Whether to reuse the weights of a previous layer
        by the same name.
    :return: A 'Tensor' with ...
    """
    with tf.variable_scope(scope, reuse=reuse):
        embed_table = tf.get_variable('embedding_table',
                                      shape=[vocab_size, embed_size],
                                      initializer=_init,
                                      trainable=trainable,
                                      dtype=tf.float32)
        if zero_pad:
            embed_table = tf.concat((tf.zeros(shape=[1, embed_size]), embed_table[1:, :]),
                                    axis=0)

    return tf.nn.embedding_lookup(embed_table, inputs)


def conv1d(inputs,
           n_filters=None, kernel=1, stride=1, dilated_rate=1,
           padding="SAME", use_bias=False, activation_fn=None,
           scope="conv1d", reuse=None):
    """
    Convolution 1D Operation
    :param inputs: A '3D Tensor' with shape of [batch, time, depth]
    :param n_filters: An int, Conv1D filter size.
    :param kernel: An int. Conv1D kernel size.
    :param stride: An int. Conv1D stride size.
    :param dilated_rate: An int. Conv1D dilated size.
    :param padding: Either 'SAME' or 'VALID' or 'CAUSAL'.
    :param use_bias: A boolean.
    :param activation_fn: An Object. TF activation function.
    :param scope: A str, Optional scope for 'variable_scope'.
    :param reuse: A boolean. Whether to reuse the weights of a previous layer
        by the same name.
    :return: conv1d result
    """
    with tf.variable_scope(scope, reuse=reuse):
        if n_filters is None:
            n_filters = inputs.get_shape().as_list()[-1]

        if padding.upper() == "CASUAL":
            pad_len = (kernel - 1) * dilated_rate  # padding size

            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "VALID"

        outputs = tf.layers.conv1d(inputs=inputs,
                                   filters=n_filters,
                                   kernel_size=kernel,
                                   strides=stride,
                                   dilation_rate=dilated_rate,
                                   padding=padding,
                                   activation=activation_fn,
                                   use_bias=use_bias,
                                   kernel_initializer=_init,
                                   kernel_regularizer=_reg,
                                   )

    return outputs
