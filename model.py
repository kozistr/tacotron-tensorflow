# -*- coding: utf-8 -*-
from __future__ import print_function

from config import get_config

from tfutils import embedding_table
from tfutils import _init, _reg
from tfutils import batch_norm
from tfutils import conv1d
from tfutils import biGRU

from modules import attention_decoder
from modules import highway_network
from modules import conv1d_banks
from modules import prenet

from utils import spectrogram2wav

import tensorflow as tf
import numpy as np

__AUTHOR__ = "kozistr"
__VERSION__ = "0.1"

cfg, _ = get_config()  # configuration

# set random seed
np.random.seed(cfg.seed)
tf.set_random_seed(cfg.seed)


class Tacotron:

    def __init__(self, sess, mode="train", sample_rate=22050,
                 vocab_size=251, embed_size=256, n_mels=80, n_fft=2048, reduction_factor=5,
                 n_encoder_banks=16, n_decoder_banks=8, n_highway_blocks=4,
                 lr=1e-3, lr_decay=.95, optimizer="Adam", grad_clip=5.,
                 model_path="./model"):
        """ Tacotron Architecture
        :param sess: A TF Session.
        :param mode: A str. Mode for train/test.
        :param sample_rate: An int, Number of sampling rate.
        :param vocab_size: An int. Number of vocabulary.
        :param embed_size: An int. Embedding vector size.
        :param n_mels: An int.
        :param n_fft: An int.
        :param reduction_factor: An int.
        :param n_encoder_banks: An int. Number of layers of conv banks in Encoder
        :param n_decoder_banks: An int. Number of layers of conv banks in Decoder
        :param n_highway_blocks: An int. Number of layers of highway network
        :param lr: A float, Learning rate.
        :param lr_decay: A float, Learning rate decay factor.
        :param optimizer: A str. Name of Optimizer.
        :param grad_clip: A float. Norm of gradients to clip.
        :param model_path: A str. Path where the model is saved
        """
        self.sess = sess
        self.sample_rate = sample_rate
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.reduction_factor = reduction_factor
        self.n_encoder_banks = n_encoder_banks
        self.n_decoder_banks = n_decoder_banks
        self.n_highway_blocks = n_highway_blocks
        self.lr = lr
        self.lr_decay = lr_decay
        self.optimizer = optimizer.lower()
        self.grad_clip = grad_clip
        self.model_path = model_path

        self.is_training = True if mode.lower() == "train" else False

        # outputs
        self.memory = None
        self.y_hat = None
        self.z_hat = None
        self.alignments = None
        self.audio = None

        self.y_loss = None
        self.z_loss = None
        self.loss = None
        self.opt = None
        self.train_op = None

        self.merged = None
        self.writer = None
        self.saver = None
        self.best_saver = None

        # placeholders
        self.x = tf.placeholder(tf.int32, shape=(None, None),
                                name="x-text")  # (N, T_x)
        self.x_len = tf.placeholder(tf.int32, shape=(None,),
                                    name="x-text-length")  # (N, )
        self.y = tf.placeholder(tf.float32, shape=(None, None, self.n_mels * self.reduction_factor),
                                name="y-mel_spectrogram")  # (N, T_y // r, n_mels * r)
        self.z = tf.placeholder(tf.float32, shape=(None, None, 1 + self.n_fft // 2),
                                name="z-magnitude")  # (N, T_y, 1 + n_fft // 2)

        # global step
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # inputs
        self.encoder_inputs = embedding_table(inputs=self.x,
                                              vocab_size=self.vocab_size,
                                              embed_size=self.embed_size)
        self.decoder_inputs = tf.concat((tf.zeros_like(self.y[:, :1, :]), self.y[:, :-1, :]), axis=1)
        self.decoder_inputs = self.decoder_inputs[:, :, -self.n_mels:]  # feed only last frames

        self.build_model()

    def encoder(self, inputs, use_highway_network=True, is_training=True, scope="encoder", reuse=None):
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
            enc = conv1d_banks(prenet_enc, n_kernels=self.n_encoder_banks, is_training=is_training)
            enc = tf.layers.max_pooling1d(enc, pool_size=2, strides=1, padding='SAME')

            # Encoder PostNet
            enc = conv1d(enc, n_filters=self.embed_size // 2, kernel=3, scope="conv1d-proj-1")
            enc = batch_norm(enc, is_training=is_training, activation_fn=tf.nn.relu, scope="bn-proj-1")

            enc = conv1d(enc, n_filters=self.embed_size // 2, kernel=3, scope="conv1d-proj-2")
            enc = batch_norm(enc, is_training=is_training, activation_fn=tf.nn.relu, scope="bn-proj-2")

            enc += prenet_enc  # long skip connection (LSC)

            # highway networks
            if use_highway_network:
                for i in range(self.n_highway_blocks):
                    enc = highway_network(enc,
                                          num_units=self.embed_size // 2,
                                          scope="highway_network-%d" % i)

            memory = biGRU(enc, num_units=self.embed_size // 2, bidirection=True)

        return memory

    def pre_decoder(self, inputs, memory, is_training=False, scope="pre-decoder", reuse=None):
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
                                           num_units=self.embed_size)

            alignments = tf.transpose(state.alignment_history.stack(), [1, 2, 0])

            # Decoder stacked GRU
            dec += biGRU(dec, num_units=self.embed_size, bidirection=False, scope="GRU-1")
            dec += biGRU(dec, num_units=self.embed_size, bidirection=False, scope="GRU-2")

            mel_hats = tf.layers.dense(dec, units=self.n_mels * self.reduction_factor,
                                       kernel_initializer=_init,
                                       kernel_regularizer=_reg)
        return mel_hats, alignments

    def post_decoder(self, inputs, use_highway_network=True, is_training=True, scope="post-decoder", reuse=None):
        """ Post-processing Decoder
        :param inputs: A 3D Tensor with shape of [N, T_y / r, n_mels * r].
        :param use_highway_network: A boolean. Whether using highway network or not
        :param is_training: A boolean.
        :param scope: A str, Optional scope for 'variable_scope'.
        :param reuse: A boolean. Whether to reuse the weights of a previous layer
        :return:
        """
        with tf.variable_scope(scope, reuse=reuse):
            x = tf.split(inputs, self.reduction_factor, axis=-1)
            x = tf.concat(x, axis=1)

            # Decoder Convolutional Block
            dec = conv1d_banks(x, n_kernels=self.n_decoder_banks, is_training=is_training)
            dec = tf.layers.max_pooling1d(dec, pool_size=2, strides=1, padding='SAME')

            # Encoder PostNet
            dec = conv1d(dec, n_filters=self.embed_size // 2, kernel=3, scope="conv1d-proj-1")
            dec = batch_norm(dec, is_training=is_training, activation_fn=tf.nn.relu, scope="bn-proj-1")

            dec = conv1d(dec, n_filters=self.n_mels, kernel=3, scope="conv1d-proj-2")
            dec = batch_norm(dec, is_training=is_training, activation_fn=tf.nn.relu, scope="bn-proj-2")

            dec = tf.layers.dense(dec, units=self.embed_size // 2,
                                  kernel_initializer=_init,
                                  kernel_regularizer=_reg)

            # highway networks
            if use_highway_network:
                for i in range(self.n_highway_blocks):
                    dec = highway_network(dec,
                                          num_units=self.embed_size // 2,
                                          scope="highway_network-%d" % i)

            dec = biGRU(dec, num_units=self.embed_size // 2, bidirection=True)

            outputs = tf.layers.dense(dec, units=1 + self.n_fft // 2,
                                      kernel_initializer=_init,
                                      kernel_regularizer=_reg)
        return outputs

    def build_model(self):
        with tf.variable_scope("Network"):
            # Encoder
            self.memory = self.encoder(inputs=self.encoder_inputs,
                                       is_training=self.is_training)  # (N, T_x, E)

            # Pre-DecoderA
            self.y_hat, self.alignments = self.pre_decoder(inputs=self.decoder_inputs,
                                                           memory=self.memory,
                                                           is_training=self.is_training)  # (N, T_y // r, n_mels * r)

            # Post-Decoder
            self.z_hat = self.post_decoder(inputs=self.y_hat,
                                           is_training=self.is_training)

        self.audio = tf.py_func(spectrogram2wav, [self.z_hat[0]], tf.float32)

        # Loss
        self.y_loss = tf.reduce_mean(tf.abs(self.y_hat - self.y))
        self.z_loss = tf.reduce_mean(tf.abs(self.z_hat - self.z))
        self.loss = self.y_loss + self.z_loss

        # Optimizer
        learning_rate = tf.train.exponential_decay(self.lr,
                                                   self.global_step,
                                                   327,  # 1 epoch, 327.5 GS
                                                   self.lr_decay,
                                                   staircase=True)

        self.lr = tf.clip_by_value(learning_rate,
                                   clip_value_min=1e-5,
                                   clip_value_max=self.lr,
                                   name='lr-clipped')

        if self.optimizer == "sgd":
            self.opt = tf.train.MomentumOptimizer(self.lr, momentum=.9, use_nesterov=True)
        elif self.optimizer == "adam":
            self.opt = tf.train.AdamOptimizer(self.lr, epsilon=1e-6)
        else:
            raise KeyError

        # Gradient Clipping
        gradients, variables = zip(*self.opt.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.train_op = self.opt.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        # Summary
        tf.summary.scalar("loss/y_loss", self.y_loss)
        tf.summary.scalar("loss/z_loss", self.z_loss)
        tf.summary.scalar("loss/loss", self.loss)
        tf.summary.scalar("misc/lr", self.lr)

        tf.summary.image("y/mel_gt", tf.expand_dims(self.y, axis=-1), max_outputs=1)
        tf.summary.image("y/mel_hat", tf.expand_dims(self.y_hat, axis=-1), max_outputs=1)
        tf.summary.image("z/mel_gt", tf.expand_dims(self.z, axis=-1), max_outputs=1)
        tf.summary.image("z/mel_hat", tf.expand_dims(self.y_hat, axis=-1), max_outputs=1)

        tf.summary.audio("sample", tf.expand_dims(self.audio, axis=0), sample_rate=self.sample_rate)

        self.merged = tf.summary.merge_all()  # merge summaries

        # Model Saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.best_saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter(self.model_path, self.sess.graph)


class Tacotron2:

    def __init__(self):
        pass


class DeepVoiceV3:

    def __init__(self):
        pass
