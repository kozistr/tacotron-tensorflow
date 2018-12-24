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


def embedding_table(inputs, vocab_size, embed_size, zero_pad=False, trainable=True, scope="embedding", reuse=None):
    """
    Generating Embedding Table with given parameters
    :param inputs: A 'Tensor' with type 'int8' or 'int16' or 'int32' or 'int64'
        containing the ids to be tooked up in 'lookup table'.
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
