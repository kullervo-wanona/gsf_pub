import os, sys, inspect
sys.path.insert(1, os.path.realpath(os.path.pardir))

import time
import ipdb
trace = ipdb.set_trace

import pprint
pp = pprint.PrettyPrinter(width=81, compact=True).pprint

import numpy as np
np.set_printoptions(precision=2, edgeitems=127, linewidth=150)
np.set_printoptions(suppress=True) # supress scientific notation
import torch

import spectral_schur_det_lib
from multi_channel_invertible_conv_lib import dft_lib
from multi_channel_invertible_conv_lib import complex_lib
from multi_channel_invertible_conv_lib import spatial_conv2D_lib
from multi_channel_invertible_conv_lib import frequency_conv2D_lib
from multi_channel_invertible_conv_lib import matrix_conv2D_lib

batch_size = 10
c = 4
n = 7
k = 3

for backend in ['numpy', 'torch']:
	for channel_mode in ['full', 'lower_block_triangular', 'block_diagonal']:
		X = np.random.randn(batch_size, c, n, n)
		if backend == 'torch': X = torch.tensor(X, dtype=torch.float32)
		K_spatial, K = spatial_conv2D_lib.generate_random_kernel(c, k, channel_mode, backend=backend)
		Lambda = dft_lib.SubmatrixDFT(K, n, backend=backend)

		H, H_blocks = matrix_conv2D_lib.H_and_H_blocks(K, n, backend=backend)
		det_sign, H_log_det = np.linalg.slogdet(H)

		Y_spatial = spatial_conv2D_lib.spatial_circular_conv2D(X, K_spatial, channel_mode=channel_mode, backend=backend)
		Y_matrix = matrix_conv2D_lib.matrix_conv2D(X, H, backend=backend)

		if backend == 'torch':
		    Y_spatial = Y_spatial.numpy()
		    Y_matrix = Y_matrix.numpy()
		assert (np.abs(Y_matrix-Y_spatial).max() < 1e-5)

		H_log_det_MD = spectral_schur_det_lib.schur_complement_log_determinant(Lambda, complement_mode='H/D', backend=backend)
		assert (np.abs(H_log_det-H_log_det_MD.item()) < 1e-4)
		H_log_det_MA = spectral_schur_det_lib.schur_complement_log_determinant(Lambda, complement_mode='H/D', backend=backend)
		assert (np.abs(H_log_det-H_log_det_MA.item()) < 1e-4)

print('All tests passed.')
























