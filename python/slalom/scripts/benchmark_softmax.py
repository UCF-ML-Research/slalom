from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('/root/slalom')
import os
import json

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from keras import backend

from python import imagenet
from python.slalom.models import get_model
from python.slalom.quant_layers import transform
from python.slalom.utils import Results, timer
from python.slalom.sgxdnn import SGXDNNUtils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_USE_DEEP_CONV2D"] = '0'

BITS = 8


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.set_random_seed(1234)

    # --------------- initialize the SGX device ---------------
    sgxutils = SGXDNNUtils(args.use_sgx)

    # --------------- initialize the input x---------------
    x_raw = np.random.uniform(0, 1, (128))
    x = np.round(2 ** BITS * x_raw).astype(np.float32)

    # --------------- initialize the precompute r ---------------
    r_raw = np.random.uniform(0, 1, (128))
    r = np.round(2 ** BITS * r_raw).astype(np.float32)

    # --------------- initialize the precompute a ---------------
    a_raw = np.random.uniform(0, 1, (128))
    a = np.round(2 ** BITS * a_raw).astype(np.float32)

    # --------------- r_a = r - a ---------------
    r_a_raw = r_raw - a_raw
    r_a = np.round(2 ** BITS * r_a_raw).astype(np.float32)
    
    # --------------- x_r = x - r ---------------
    x_r_raw = x_raw - r_raw
    x_r = np.round(2 ** BITS * x_r_raw).astype(np.float32)

    # --------------- TEE: precompute exp(r) ---------------
    r_exp_raw = np.exp(r_raw.astype(np.float))

    # --------------- TEE: prepare the input ---------------
    total_length = len(x_r) + len(r_a) + len(a)
    total_idx = np.arange(total_length)
    np.random.shuffle(total_idx)

    gpu_x = np.zeros(total_length)
    gpu_x[total_idx[:len(a)]] = a
    gpu_x[total_idx[len(a):len(a) + len(r_a)]] = r_a
    gpu_x[total_idx[len(a) + len(r_a):]] = x_r

    a_idx = total_idx[:len(a)]
    r_a_idx = total_idx[len(a):len(a) + len(r_a)]
    x_r_idx = total_idx[len(a) + len(r_a):]

    # --------------- GPU: compute exp(gpu_x) ---------------
    with tf.Session() as sess:
        gpu_x_raw = gpu_x / (2 ** BITS)
        gpu_x_raw_tensor = tf.convert_to_tensor(gpu_x_raw, dtype=tf.float32)
        with tf.device('/gpu:0'):
            gpu_x_raw_exp_tensor = tf.exp(gpu_x_raw_tensor)

        gpu_x_exp_raw = sess.run(gpu_x_raw_exp_tensor)

    # --------------- TEE: recover exp(gpu_x) to exp (x) ---------------
    x_exp_raw, integrity = sgxutils.benchmark_exp(gpu_x_exp_raw, r_exp_raw, a_idx, r_a_idx, x_r_idx)

    # --------------- TEE: compute softmax ---------------
    x_softmax_raw = sgxutils.benchmark_softmax(x_exp_raw)

    if sgxutils is not None:
        sgxutils.destroy()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_sgx', action='store_true')
    args = parser.parse_args()

    tf.app.run()
