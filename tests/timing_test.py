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
c = 3
n = 100
k = 4
channel_mode = 'full'

for backend in ['numpy', 'torch']:
	print('\nBackend: ' + backend + '\n')
	X = np.random.randn(batch_size, c, n, n)
	if backend == 'torch': X = torch.tensor(X, dtype=torch.float32)
	K_spatial, K = spatial_conv2D_lib.generate_random_kernel(c, k, channel_mode, backend=backend)

	t_start = time.time()
	for i in range(100):
		Y_spatial = spatial_conv2D_lib.spatial_circular_conv2D(X, K_spatial, channel_mode=channel_mode, backend=backend)
	print('Convolution took '+str(time.time()-t_start)+' seconds.')

	t_start = time.time()
	for i in range(100):
		X_rec = frequency_conv2D_lib.frequency_inverse_circular_conv2D(Y_spatial, K, channel_mode, mode='complex', backend=backend)
	print('Inverse took '+str(time.time()-t_start)+' seconds.')

	t_start = time.time()
	for i in range(100):
		Lambda = dft_lib.SubmatrixDFT(K, n, backend=backend)
		H_log_det_MD = spectral_schur_det_lib.spectral_schur_log_determinant(Lambda, complement_mode='H/D', backend=backend)
	print('Determinant 1 took '+str(time.time()-t_start)+' seconds.')

	kernel_to_schur_log_determinant = spectral_schur_det_lib.generate_kernel_to_schur_log_determinant(k, n, backend=backend)
	t_start = time.time()
	for i in range(100):
		H_log_det_MD_2 = kernel_to_schur_log_determinant(K)
	print('Determinant 2 took '+str(time.time()-t_start)+' seconds.')





















