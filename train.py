# -*- coding: utf-8 -*-
from __future__ import print_function

from config import get_config

from model import DeepVoiceV3
from model import Tacotron2
from model import Tacotron

import tensorflow as tf
import numpy as np
import argparse


__AUTHOR__ = "kozistr"
__VERSION__ = "0.1"

cfg, _ = get_config()  # configuration

# set random seed
np.random.seed(cfg.seed)
tf.set_random_seed(cfg.seed)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
args = parser.parse_args()


def main():
    # Model Loading
    gpu_config = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_config)

    with tf.Session(config=config) as sess:
        model = Tacotron(sess=sess,
                         mode=args.mode,
                         sample_rate=cfg.sample_rate,
                         vocab_size=cfg.vocab_size,
                         embed_size=cfg.embed_size,
                         n_mels=cfg.n_mels,
                         n_fft=cfg.n_fft,
                         reduction_factor=cfg.reduction_factor,
                         n_encoder_banks=cfg.n_encoder_banks,
                         n_decoder_banks=cfg.n_decoder_banks,
                         n_highway_blocks=cfg.n_highway_banks,
                         lr=cfg.lr,
                         lr_decay=cfg.lr_decay,
                         optimizer=cfg.optimizer,
                         grad_clip=cfg.grad_clip,
                         model_path=cfg.model_path)

        # Initializing
        sess.run(tf.global_variables_initializer())

        # Load model & Graph & Weights
        global_step = 0
        ckpt = tf.train.get_checkpoint_state(cfg.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            model.saver.restore(sess, ckpt.model_checkpoint_path)

            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print("[+] global step : %d" % global_step, " successfully loaded")
        else:
            print('[-] No checkpoint file found')


if __name__ == "__main__":
    main()
