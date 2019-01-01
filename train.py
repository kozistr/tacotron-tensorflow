# -*- coding: utf-8 -*-
from __future__ import print_function

from dataloader import DataIterator
from config import get_config
from model import DeepVoiceV3
from model import Tacotron2
from model import Tacotron

import tensorflow as tf
import numpy as np
import argparse
import time
import os


__AUTHOR__ = "kozistr"
__VERSION__ = "0.1"

cfg, _ = get_config()  # configuration

# set random seed
np.random.seed(cfg.seed)
tf.set_random_seed(cfg.seed)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
parser.add_argument('--dataset', type=str, default="ljspeech", choices=["ljspeech"])
args = parser.parse_args()


def main():
    # DataSet Loader
    if args.dataset == "ljspeech":
        from datasets.ljspeech import LJSpeech

        # LJSpeech-1.1 dataset loader
        ljs = LJSpeech(path=cfg.dataset_path,
                       save_to='npy',
                       load_from=None if not os.path.exists(cfg.dataset_path + "/npy") else "npy",
                       verbose=cfg.verbose)
    else:
        raise NotImplementedError("[-] Not Implemented Yet...")

    # Train/Test split
    tr_size = int(len(ljs) * (1. - cfg.test_size))
    ljs.text_data = np.array(ljs.text_data)
    ljs.mels = np.array(ljs.mels)  # .reshape((-1, cfg.n_mels * cfg.sample_rate))
    ljs.mags = np.array(ljs.mags)  # .reshape((-1, 1 + cfg.n_fft // 2))

    tr_text_data, va_text_data = ljs.text_data[:tr_size], ljs.text_data[tr_size:]
    tr_mels, va_mels = ljs.mels[:tr_size], ljs.mels[tr_size:]
    tr_mags, va_mags = ljs.mags[:tr_size], ljs.mags[tr_size:]

    del ljs  # memory release

    # Data Iterator
    di = DataIterator(text=tr_text_data, mel=tr_mels, mag=tr_mags,
                      batch_size=cfg.batch_size)

    if cfg.verbose:
        print("[*] Train/Test split : %d/%d (%.2f/%.2f)" % (tr_text_data.shape[0], va_text_data.shape[0],
                                                            1. - cfg.test_size, cfg.test_size))
        print("  Train")
        print("\ttext : ", tr_text_data.shape)
        print("\tmels : ", tr_mels.shape)
        print("\tmags : ", tr_mags.shape)
        print("  Test")
        print("\ttext : ", va_text_data.shape)
        print("\tmels : ", va_mels.shape)
        print("\tmags : ", va_mags.shape)

    # Model Loading
    gpu_config = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_config)

    with tf.Session(config=config) as sess:
        if cfg.model == "Tacotron":
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
                             n_highway_blocks=cfg.n_highway_blocks,
                             lr=cfg.lr,
                             lr_decay=cfg.lr_decay,
                             optimizer=cfg.optimizer,
                             grad_clip=cfg.grad_clip,
                             model_path=cfg.model_path)
        else:
            raise NotImplementedError("[-] Not Implemented Yet...")

        if cfg.verbose:
            print("[*] %s model is loaded!" % cfg.model)

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

        start_time = time.time()

        batch_size = cfg.batch_size
        model.global_step.assign(tf.constant(global_step))
        restored_epochs = global_step // (di.text.shape[0] // batch_size)
        for epoch in range(restored_epochs, cfg.epochs):
            for text, mel, mag in di.iterate():
                batch_start = time.time()
                _, y_loss, z_loss = sess.run([model.train_op, model.y_loss, model.z_loss],
                                             feed_dict={
                                                 model.x: text,
                                                 model.y: mel,
                                                 model.z: mag,
                                             })
                batch_end = time.time()

                if global_step and global_step % cfg.logging_step == 0:
                    va_y_loss, va_z_loss = 0., 0.

                    va_batch = 20
                    va_iter = len(va_text_data)
                    for idx in range(0, va_iter, va_batch):
                        va_y, va_z = sess.run([model.y_loss, model.z_loss],
                                              feed_dict={
                                                  model.x: va_text_data[va_batch * idx:va_batch * (idx + 1)],
                                                  model.y: va_mels[va_batch * idx:va_batch * (idx + 1)],
                                                  model.z: va_mags[va_batch * idx:va_batch * (idx + 1)],
                                              })

                        va_y_loss += va_y
                        va_z_loss += va_z

                    va_y_loss /= (va_iter // va_batch)
                    va_z_loss /= (va_iter // va_batch)

                    print("[*] epoch %03d global step %07d [%.03f sec/step]" % (epoch,
                                                                                global_step,
                                                                                (batch_end - batch_start)
                                                                                ),
                          " Train \n"
                          " y_loss : {:.6f} z_loss : {:.6f}".format(y_loss, z_loss),
                          " Valid \n"
                          " y_loss : {:.6f} z_loss : {:.6f}".format(va_y_loss, va_z_loss)
                          )

                    # summary
                    summary = sess.run(model.merged,
                                       feed_dict={
                                           model.x: va_text_data[:batch_size * 4],
                                           model.y: va_mels[:batch_size * 4],
                                           model.z: va_mags[:batch_size * 4],
                                       })

                    # Summary saver
                    model.writer.add_summary(summary, global_step)

                    # Model save
                    model.saver.save(sess, cfg.model_path + '%s.ckpt' % cfg.model,
                                     global_step=global_step)

                model.global_step.assign_add(tf.constant(1))
                global_step += 1

        end_time = time.time()

        print("[+] Training Done! Elapsed {:.8f}s".format(end_time - start_time))


if __name__ == "__main__":
    main()
