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
    """ Generating Embedding Table with given parameters
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
    """ Convolution 1D Operation
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


def biGRU(inputs, num_units=None, bidirection=False, scope='biGRU', reuse=None):
    """ bi-GRU
    :param inputs: A 3D Tensor with shape of [batch, T, C]
    :param num_units: An int. The number of hidden units.
    :param bidirection: A boolean. If True, bidirectional results
        are concatenated.
    :param scope: A str, Optional scope for 'variable_scope'.
    :param reuse: A boolean. Whether to reuse the weights of a previous layer
        by the same name.
    :return: If bidirection is True, a 3D Tensor with shape of [batch, T, 2 * num_units],
        otherwise [batch, T, num_units]
    """
    if num_units is None:
        num_units = inputs.get_shape().as_list[-1]

    with tf.variable_scope(scope, reuse=reuse):
        cell_fw = tf.contrib.rnn.GRUCell(num_units)

        if bidirection:
            cell_bw = tf.contrib.rnn.GRUCell(num_units)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                         dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell_fw, inputs,
                                           dtype=tf.float32)
    return outputs


def batch_norm(inputs, is_training=True, activation_fn=None, scope="batch_norm", reuse=None):
    """ Batch Normalization, referenced https://github.com/Kyubyong/tacotron/blob/master/modules.py#L43
    :param inputs: A Tensor with 2 or more dimensions, where the first dim has 'batch_size'.

    :param is_training: A boolean.
    :param activation_fn: Activation function.
    :param scope: A str, Optional scope for 'variable_scope'.
    :param reuse: A boolean. Whether to reuse the weights of a previous layer
        by the same name.
    :return:
    """
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims

    if inputs_rank in [2, 3, 4]:
        if not inputs_rank == 4:
            inputs = tf.expand_dims(inputs, axis=1)
        if inputs_rank == 2:
            inputs = tf.expand_dims(inputs, axis=2)

        outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                               center=True,
                                               scale=True,
                                               updates_collections=None,
                                               is_training=is_training,
                                               scope=scope,
                                               fused=True,
                                               reuse=reuse)

        if inputs_rank == 2:
            outputs = tf.squeeze(outputs, axis=[1, 2])
        elif inputs_rank == 3:
            outputs = tf.squeeze(outputs, axis=1)
    else:
        outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                               center=True,
                                               scale=True,
                                               updates_collections=None,
                                               is_training=is_training,
                                               scope=scope,
                                               reuse=reuse,
                                               fused=False)

    if activation_fn is not None:
        outputs = activation_fn(outputs)

    return outputs
