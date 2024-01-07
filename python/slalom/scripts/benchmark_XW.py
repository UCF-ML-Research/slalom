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

    # --------------- initialize the input Q and K ---------------
    X_raw = np.random.uniform(0, 1, (128, 768))
    X = np.round(2 ** BITS * X_raw).astype(np.int32)
    W_raw = np.random.uniform(0, 1, (768, 768))
    W = np.round(2 ** BITS * W_raw).astype(np.int32)

    # --------------- initialize the random matrixs R and S ---------------
    R_raw = np.random.uniform(0, 1, (int(128 * args.ratio), 768))
    R = np.round(2 ** BITS * R_raw).astype(np.int32)
    S_raw = np.random.uniform(0, 1, (768, int(128 * args.ratio)))
    S = np.round(2 ** BITS * S_raw).astype(np.int32)

    # --------------- X -> X_modified, W -> W_modified ---------------
    X_modified = np.empty_like(X)
    X_selected_indices = np.empty(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        random_index = np.random.randint(R.shape[0])
        X_modified[i] = X[i] - R[random_index]
        X_selected_indices[i] = random_index

    W_modified = np.empty_like(W)
    W_selected_indices = np.empty(W.shape[1], dtype=int)

    for j in range(W.shape[1]):
        random_index = np.random.randint(S.shape[1]) 
        W_modified[:, j] = W[:, j] - S[:, random_index]
        W_selected_indices[j] = random_index

    # --------------- TEE: concat X_modified with R, W_modified with S ---------------
    XR = np.concatenate((X_modified, R), axis=0)
    WS = np.concatenate((W_modified, S), axis=1)

    # --------------- TEE: permute the XR and WS, and get the location index to recover ---------------
    permuted_XR_indices = np.random.permutation(XR.shape[0])
    permuted_XR = XR[permuted_XR_indices]

    permuted_WS_indices = np.random.permutation(WS.shape[1])
    permuted_WS = WS[:, permuted_WS_indices]

    # --------------- GPU: compute permuted_XR dot permuted_WS ---------------
    gpu_res = np.matmul(permuted_XR, permuted_WS)

    # --------------- TEE: recover gpu_res to res ---------------
    res = sgxutils.benchmark_XW(gpu_res, X_selected_indices, W_selected_indices, permuted_XR_indices, permuted_WS_indices)

    if sgxutils is not None:
        sgxutils.destroy()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_sgx', action='store_true')
    parser.add_argument('--ratio', type=float, default=0.2)
    args = parser.parse_args()

    tf.app.run()
