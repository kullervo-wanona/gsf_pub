import os, sys, inspect
sys.path.insert(1, os.path.realpath(os.path.pardir))

from sys import platform
if 'linux' in platform: 
    from IPython.core.debugger import Tracer
    trace = Tracer() #this one triggers the debugger
else:
    import ipdb
    trace = ipdb.set_trace

import numpy as np
import torch

import helper
from multi_channel_invertible_conv_lib import complex_lib
from multi_channel_invertible_conv_lib import dft_lib

def spectral_schur_log_determinant(Lambda, complement_mode='H/D', backend='torch'):
	# overall # O((c)(c+1)(2c+1)/6 n^2) = O((c)^3 n^2) is the overall complexity
	# O(c^3 n^2) when starting with Lambda, O(c^2n(k^2+(k+c)n) starting with K

	if Lambda.shape[1] == 1: 
		if backend == 'torch': 
			return torch.sum(torch.log(complex_lib.abs(Lambda, backend=backend)))  # O(n^2)
		elif backend == 'numpy': 
			return np.sum(np.log(complex_lib.abs(Lambda, backend=backend))) # O(n^2)

	if complement_mode == 'H/D':
		A = Lambda[:, :-1, :-1]
		B = Lambda[:, :-1, -1:]
		C = Lambda[:, -1:, :-1]
		D = Lambda[:, -1:, -1:]

		if backend == 'torch': 
			D_log_det = torch.sum(torch.log(complex_lib.abs(D, backend=backend))) # O(n^2)
		elif backend == 'numpy': 
			D_log_det = np.sum(np.log(complex_lib.abs(D, backend=backend))) # O(n^2)

		BD_rec = complex_lib.mult(B, complex_lib.reciprocal(D, backend=backend), mult_mode='hadamard', mode='complex-complex', backend=backend) # O((c-1) n^2)
		BD_recC = complex_lib.mult(BD_rec, C, mult_mode='hadamard', mode='complex-complex', backend=backend)  # O((c-1)^2 n^2)
		MD = A-BD_recC  # O((c-1)^2 n^2)
		log_det_MD = spectral_schur_log_determinant(MD, complement_mode, backend=backend)

		return log_det_MD + D_log_det

	elif complement_mode == 'H/A':
		A = Lambda[:, :1, :1]
		B = Lambda[:, :1, 1:]
		C = Lambda[:, 1:, :1]
		D = Lambda[:, 1:, 1:]

		if backend == 'torch': 
			A_log_det = torch.sum(torch.log(complex_lib.abs(A, backend=backend))) # O(n^2)
		elif backend == 'numpy': 
			A_log_det = np.sum(np.log(complex_lib.abs(A, backend=backend))) # O(n^2)
		
		CA_rec = complex_lib.mult(C, complex_lib.reciprocal(A, backend=backend), mult_mode='hadamard', mode='complex-complex', backend=backend)  # O((c-1) n^2)
		CA_recB = complex_lib.mult(CA_rec, B, mult_mode='hadamard', mode='complex-complex', backend=backend)  # O((c-1)^2 n^2)
		MA = D-BA_recC  # O((c-1)^2 n^2)
		log_det_MA = spectral_schur_log_determinant(MA, complement_mode, backend=backend)

		return log_det_MA + A_log_det

def generate_SubmatrixDFT(k, n, backend='torch'):
	# FFT of K_hat from K (c x c x k x k)
	F_relevant = helper.cuda_maybe_th(dft_lib.DFT_matrix_F(n, n_rows=k, shift=[+(k-1), +(k-1)], backend=backend), cuda=True)
	F_expand = F_relevant[:, np.newaxis, np.newaxis]
	F_T_expand = complex_lib.transpose(F_relevant, backend=backend)[:, np.newaxis, np.newaxis]
	def func_SubmatrixDFT(K):
		if backend == 'torch':
			K_time_reversed = torch.flip(K, [-1, -2])
		elif backend == 'numpy':
			K_time_reversed = np.flip(K, [-1, -2]).copy()
		Z = complex_lib.mult(K_time_reversed, F_expand, mult_mode='matmul', mode='real-complex', backend=backend)
		Lambda_rolled = complex_lib.mult(F_T_expand, Z, mult_mode='matmul', mode='complex-complex', backend=backend)
		if backend == 'torch':
			Lambda = torch.roll(Lambda_rolled, shifts=[-(k-1), -(k-1)], dims=[-2, -1])
		elif backend == 'numpy':
			Lambda = np.roll(Lambda_rolled, shift=[-(k-1), -(k-1)], axis=[-2, -1])
		return Lambda
	return func_SubmatrixDFT

def generate_kernel_to_schur_log_determinant(k, n, backend='torch'):
	SubmatrixDFT = generate_SubmatrixDFT(k, n, backend=backend)
	def func_kernel_to_schur_log_determinant(K):
		Lambda = SubmatrixDFT(K)
		return spectral_schur_log_determinant(Lambda, complement_mode='H/D', backend=backend)

	return func_kernel_to_schur_log_determinant













