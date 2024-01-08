from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('/root/slalom')
import os
import json
import time

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
    Q_raw = np.random.uniform(0, 1, (128, 768))
    Q = np.round(2 ** BITS * Q_raw).astype(np.int32)
    K_raw = np.random.uniform(0, 1, (768, 128))
    K = np.round(2 ** BITS * K_raw).astype(np.int32)

    # --------------- initialize the random matrixs R and S ---------------
    R_raw = np.random.uniform(0, 1, (int(128 * args.ratio), 768))
    R = np.round(2 ** BITS * R_raw).astype(np.int32)
    S_raw = np.random.uniform(0, 1, (768, int(128 * args.ratio)))
    S = np.round(2 ** BITS * S_raw).astype(np.int32)

    # --------------- Q -> Q_modified, K -> K_modified ---------------
    Q_modified = np.empty_like(Q)
    Q_selected_indices = np.empty(Q.shape[0], dtype=int)

    for i in range(Q.shape[0]):
        random_index = np.random.randint(R.shape[0])
        Q_modified[i] = Q[i] - R[random_index]
        Q_selected_indices[i] = random_index

    K_modified = np.empty_like(K)
    K_selected_indices = np.empty(K.shape[1], dtype=int)

    for j in range(K.shape[1]):
        random_index = np.random.randint(S.shape[1]) 
        K_modified[:, j] = K[:, j] - S[:, random_index]
        K_selected_indices[j] = random_index

    # --------------- TEE: concat Q_modified with R, K_modified with S ---------------
    QR = np.concatenate((Q_modified, R), axis=0)
    KS = np.concatenate((K_modified, S), axis=1)

    # --------------- TEE: permute the QR and KS, and get the location index to recover ---------------
    permuted_QR_indices = np.random.permutation(QR.shape[0])
    permuted_QR = QR[permuted_QR_indices]

    permuted_KS_indices = np.random.permutation(KS.shape[1])
    permuted_KS = KS[:, permuted_KS_indices]

    # --------------- GPU: compute permuted_QR dot permuted_KS ---------------
    gpu_res = np.matmul(permuted_QR, permuted_KS)

    # --------------- TEE: recover gpu_res to res ---------------
    res = sgxutils.benchmark_QK(gpu_res, Q_selected_indices, K_selected_indices, permuted_QR_indices, permuted_KS_indices)

    if sgxutils is not None:
        sgxutils.destroy()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_sgx', action='store_true')
    parser.add_argument('--ratio', type=float, default=0.2)
    args = parser.parse_args()

    tf.app.run()
