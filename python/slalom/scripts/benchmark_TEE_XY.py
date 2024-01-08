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
    
    # --------------- initialize the input X and Y ---------------
    x = np.random.uniform(0, 1, (args.dim_1, args.dim_2))
    y = np.random.uniform(0, 1, (args.dim_2, args.dim_3))

    # --------------- TEE: compute z = xy ---------------
    res = sgxutils.benchmark_TEE_XY(x, y)

    if sgxutils is not None:
        sgxutils.destroy()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_sgx', action='store_true')
    parser.add_argument('--dim_1', type=int, required=True)
    parser.add_argument('--dim_2', type=int, required=True)
    parser.add_argument('--dim_3', type=int, required=True)
    args = parser.parse_args()

    tf.app.run()
