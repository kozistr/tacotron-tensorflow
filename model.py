# -*- coding: utf-8 -*-
from __future__ import print_function

from config import get_config

from tfutils import _init, _reg
from tfutils import batch_norm
from tfutils import conv1d
from tfutils import biGRU

from modules import attention_decoder
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


def encoder(inputs, use_highway_network=True, is_training=True, scope="encoder", reuse=None):
    """ Encoder
    :param inputs: A 2D Tensor with shape of [Seq, E], with dtype of intxx.
    :param use_highway_network: A boolean. Whether using highway network or not
    :param is_training: A boolean.
    :param scope: A str, Optional scope for 'variable_scope'.
    :param reuse: A boolean. Whether to reuse the weights of a previous layer
        by the same name.
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Encoder PreNet
        prenet_enc = prenet(inputs, is_training=is_training)

        # Encoder Convolutional Block
        enc = conv1d_banks(prenet_enc, n_kernels=cfg.n_encoder_banks, is_training=is_training)
        enc = tf.layers.max_pooling1d(enc, pool_size=2, strides=1, padding='SAME')

        # Encoder PostNet
        enc = conv1d(enc, n_filters=cfg.embed_size // 2, kernel=3, scope="conv1d-proj-1")
        enc = batch_norm(enc, is_training=is_training, activation_fn=tf.nn.relu, scope="bn-proj-1")

        enc = conv1d(enc, n_filters=cfg.embed_size // 2, kernel=3, scope="conv1d-proj-2")
        enc = batch_norm(enc, is_training=is_training, activation_fn=tf.nn.relu, scope="bn-proj-2")

        enc += prenet_enc  # long skip connection (LSC)

        # highway networks
        if use_highway_network:
            for i in range(cfg.n_highway_blocks):
                enc = highway_network(enc,
                                      num_units=cfg.embed_size // 2,
                                      scope="highway_network-%d" % i)

        memory = biGRU(enc, num_units=cfg.embed_size // 2, bidirection=True)

    return memory


def pre_decoder(inputs, memory, is_training=False, scope="pre-decoder", reuse=None):
    """ Pre Decoder
    :param inputs: A 3D Tensor with shape of [N, T_y / r, n_mels(*r)], with dtype of intxx.
    :param memory: A 3D Tensor with shape of [N, T_x, E].
    :param is_training: A boolean.
    :param scope: A str, Optional scope for 'variable_scope'.
    :param reuse: A boolean. Whether to reuse the weights of a previous layer
        by the same name.
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder PreNet
        prenet_dec = prenet(inputs, is_training=is_training)

        # Decoder Attention
        dec, state = attention_decoder(prenet_dec, memory,
                                       num_units=cfg.embed_size)

        alignments = tf.transpose(state.alignment_history.stack(), [1, 2, 0])

        # Decoder stacked GRU
        dec += biGRU(dec, num_units=cfg.embed_size, bidirection=False, scope="GRU-1")
        dec += biGRU(dec, num_units=cfg.embed_size, bidirection=False, scope="GRU-2")

        mel_hats = tf.layers.dense(dec, units=cfg.n_mels * cfg.reduction_factor,
                                   kernel_initializer=_init,
                                   kernel_regularizer=_reg)
    return mel_hats, alignments


def post_decoder(inputs, use_highway_network=True, is_training=True, scope="post-decoder", reuse=None):
    """ Post-processing Decoder
    :param inputs: A 3D Tensor with shape of [N, T_y / r, n_mels * r].
    :param use_highway_network: A boolean. Whether using highway network or not
    :param is_training: A boolean.
    :param scope: A str, Optional scope for 'variable_scope'.
    :param reuse: A boolean. Whether to reuse the weights of a previous layer
    :return:
    """
    inputs_shape = inputs.get_shape().as_list()

    with tf.variable_scope(scope, reuse=reuse):
        x = tf.reshape(inputs, (-1, inputs_shape[1] * cfg.reduction_factor, cfg.n_mels))

        # Decoder Convolutional Block
        dec = conv1d_banks(x, n_kernels=cfg.n_decoder_banks, is_training=is_training)
        dec = tf.layers.max_pooling1d(dec, pool_size=2, strides=1, padding='SAME')

        # Encoder PostNet
        dec = conv1d(dec, n_filters=cfg.embed_size // 2, kernel=3, scope="conv1d-proj-1")
        dec = batch_norm(dec, is_training=is_training, activation_fn=tf.nn.relu, scope="bn-proj-1")

        dec = conv1d(dec, n_filters=cfg.n_mels, kernel=3, scope="conv1d-proj-2")
        dec = batch_norm(dec, is_training=is_training, activation_fn=tf.nn.relu, scope="bn-proj-2")

        dec = tf.layers.dense(dec, units=cfg.embed_size // 2,
                              kernel_initializer=_init,
                              kernel_regularizer=_reg)

        # highway networks
        if use_highway_network:
            for i in range(cfg.n_highway_blocks):
                dec = highway_network(dec,
                                      num_units=cfg.embed_size // 2,
                                      scope="highway_network-%d" % i)

        dec = biGRU(dec, num_units=cfg.embed_size // 2, bidirection=True)

        outputs = tf.layers.dense(dec, units=1 + cfg.n_fft // 2,
                                  kernel_initializer=_init,
                                  kernel_regularizer=_reg)
    return outputs
