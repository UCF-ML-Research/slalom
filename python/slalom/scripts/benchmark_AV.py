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

    # --------------- initialize the input A and V ---------------
    A_raw = np.random.uniform(0, 1, (128, 128))
    A = np.round(2 ** BITS * A_raw).astype(np.int32)
    V_raw = np.random.uniform(0, 1, (128, 768))
    V = np.round(2 ** BITS * V_raw).astype(np.int32)

    # --------------- initialize the random matrixs R and S ---------------
    R_raw = np.random.uniform(0, 1, (int(128 * args.ratio), 128))
    R = np.round(2 ** BITS * R_raw).astype(np.int32)
    S_raw = np.random.uniform(0, 1, (128, int(128 * args.ratio)))
    S = np.round(2 ** BITS * S_raw).astype(np.int32)

    # --------------- A -> A_modified, V -> V_modified ---------------
    A_modified = np.empty_like(A)
    A_selected_indices = np.empty(A.shape[0], dtype=int)

    for i in range(A.shape[0]):
        random_index = np.random.randint(R.shape[0])
        A_modified[i] = A[i] - R[random_index]
        A_selected_indices[i] = random_index

    V_modified = np.empty_like(V)
    V_selected_indices = np.empty(V.shape[1], dtype=int)

    for j in range(V.shape[1]):
        random_index = np.random.randint(S.shape[1]) 
        V_modified[:, j] = V[:, j] - S[:, random_index]
        V_selected_indices[j] = random_index

    # --------------- TEE: concat A_modified with R, V_modified with S ---------------
    AR = np.concatenate((A_modified, R), axis=0)
    VS = np.concatenate((V_modified, S), axis=1)

    # --------------- TEE: permute the AR and VS, and get the location index to recover ---------------
    permuted_AR_indices = np.random.permutation(AR.shape[0])
    permuted_AR = AR[permuted_AR_indices]

    permuted_VS_indices = np.random.permutation(VS.shape[1])
    permuted_VS = VS[:, permuted_VS_indices]

    # --------------- GPU: compute permuted_AR dot permuted_VS ---------------
    gpu_res = np.matmul(permuted_AR, permuted_VS)

    # --------------- TEE: recover gpu_res to res ---------------
    res = sgxutils.benchmark_AV(gpu_res, A_selected_indices, V_selected_indices, permuted_AR_indices, permuted_VS_indices)

    if sgxutils is not None:
        sgxutils.destroy()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_sgx', action='store_true')
    parser.add_argument('--ratio', type=float, default=0.2)
    args = parser.parse_args()

    tf.app.run()
