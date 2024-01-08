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

    # --------------- initialize the input X ---------------
    x = np.random.uniform(0, 1, (128, 768))

    # --------------- TEE: compute GELU(X) ---------------
    res = sgxutils.benchmark_gelu(x)

    if sgxutils is not None:
        sgxutils.destroy()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_sgx', action='store_true')
    parser.add_argument('--ratio', type=float, default=0.2)
    args = parser.parse_args()

    tf.app.run()
