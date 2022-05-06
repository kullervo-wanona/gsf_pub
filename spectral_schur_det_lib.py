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

def generate_SubmatrixDFT(k, n):
	# FFT of K_hat from K (c x c x k x k)
	F_relevant = helper.cuda(dft_lib.DFT_matrix_F(n, n_rows=k, shift=[+(k-1), +(k-1)], backend='torch'))
	F_expand = F_relevant[:, np.newaxis, np.newaxis]
	F_T_expand = complex_lib.transpose(F_relevant, backend='torch')[:, np.newaxis, np.newaxis]
	
	def func_SubmatrixDFT(K):
		K_time_reversed = torch.flip(K, [-1, -2])
		Z = complex_lib.mult(K_time_reversed, F_expand, mult_mode='matmul', mode='real-complex', backend='torch')
		Lambda_rolled = complex_lib.mult(F_T_expand, Z, mult_mode='matmul', mode='complex-complex', backend='torch')
		Lambda = torch.roll(Lambda_rolled, shifts=[-(k-1), -(k-1)], dims=[-2, -1])
		return Lambda
	return func_SubmatrixDFT

def generate_kernel_to_schur_log_determinant(k, n):
	SubmatrixDFT = generate_SubmatrixDFT(k, n)
	def func_kernel_to_schur_log_determinant(K):
		Lambda = SubmatrixDFT(K)
		return spectral_schur_log_determinant(Lambda, complement_mode='H/D', backend='torch')

	return func_kernel_to_schur_log_determinant

# ##############################################################################################################

def frequency_circular_conv2D(X, Lambda):
	# X (b x c x n x n), Lambda (2 x c x c x n x n)
	X_freq = torch.fft.fft2(X) # O(k n^2 log n)
	Lambda_complex = complex_lib.re_im_to_complex(Lambda)

	Y_freq = (Lambda_complex[np.newaxis]*X_freq[:, np.newaxis]).sum(2) # O(k^2 n^2)
	Y = torch.fft.ifft2(Y_freq).real # O(k n^2 log n)
	return Y

def schur_complement_inverse_matmul_complex(Lambda_square_sub_block_T):
	if Lambda_square_sub_block_T.shape[2] == 1: return 1/Lambda_square_sub_block_T
	matmul = torch.matmul
	concat = torch.concat

	A_T = Lambda_square_sub_block_T[:, :, :1, :1]
	B_T = Lambda_square_sub_block_T[:, :, :1, 1:]
	C_T = Lambda_square_sub_block_T[:, :, 1:, :1]
	D_T = Lambda_square_sub_block_T[:, :, 1:, 1:]
	D_inv_T = schur_complement_inverse_matmul_complex(D_T)

	D_inv_C_T = matmul(D_inv_T, C_T)
	B_D_inv_T = matmul(B_T, D_inv_T)
	MD_T = A_T-matmul(B_T, D_inv_C_T)
	MD_inv_T = 1/MD_T

	top_left_T = MD_inv_T
	bottom_left_T = -D_inv_C_T*MD_inv_T
	top_right_T = -MD_inv_T*B_D_inv_T
	bottom_right_T = D_inv_T-bottom_left_T*B_D_inv_T

	Lambda_square_sub_block_inv_T = concat([concat([top_left_T, top_right_T], axis=3), 
				      						concat([bottom_left_T, bottom_right_T], axis=3)], axis=2)
	return Lambda_square_sub_block_inv_T

def frequency_inverse_full_circular_conv2D(Y, K, Lambda):
	# Schur complement spectral inverse
	Lambda = torch.transpose(torch.transpose(Lambda, 3, 1), 4, 2)
	Psi = schur_complement_inverse_matmul_complex(complex_lib.re_im_to_complex(Lambda))
	Psi = torch.transpose(torch.transpose(Psi, 2, 0), 3, 1)
	Psi = complex_lib.complex_to_re_im(Psi, backend='torch')
	X = frequency_circular_conv2D(Y, Psi)
	return X

def generate_frequency_inverse_circular_conv2D(k, n):
	SubmatrixDFT = generate_SubmatrixDFT(k, n)
	def func_frequency_inverse_circular_conv2D(Y, K):
		Lambda = SubmatrixDFT(K)
		return frequency_inverse_full_circular_conv2D(Y, K, Lambda)
	return func_frequency_inverse_circular_conv2D

# ##############################################################################################################
def generate_batch_SubmatrixDFT(k, n):
	# FFT of K_hat from K (c x c x k x k)
	F_relevant = helper.cuda(dft_lib.DFT_matrix_F(n, n_rows=k, shift=[+(k-1), +(k-1)], backend='torch'))
	F_expand = F_relevant[:, np.newaxis, np.newaxis, np.newaxis]
	F_T_expand = complex_lib.transpose(F_relevant, backend='torch')[:, np.newaxis, np.newaxis, np.newaxis]
	
	def func_batch_SubmatrixDFT(K):
		K_time_reversed = torch.flip(K, [-1, -2])
		Z = complex_lib.mult(K_time_reversed, F_expand, mult_mode='matmul', mode='real-complex', backend='torch')
		Lambda_rolled = complex_lib.mult(F_T_expand, Z, mult_mode='matmul', mode='complex-complex', backend='torch')
		Lambda = torch.roll(Lambda_rolled, shifts=[-(k-1), -(k-1)], dims=[-2, -1])
		return Lambda
	return func_batch_SubmatrixDFT

def batch_frequency_circular_conv2D(X, Lambda_complex):
	# X (b x c x n x n), Lambda (2 x c x c x n x n)
	X_freq = torch.fft.fft2(X) # O(k n^2 log n)
	Y_freq = (Lambda_complex*X_freq[:, np.newaxis, :]).sum(2) # O(k^2 n^2)
	Y = torch.fft.ifft2(Y_freq).real # O(k n^2 log n)
	return Y

def batch_schur_complement_inverse_matmul_complex(Lambda_square_sub_block_T):
	if Lambda_square_sub_block_T.shape[3] == 1: return 1/Lambda_square_sub_block_T
	matmul = torch.matmul
	concat = torch.concat

	A_T = Lambda_square_sub_block_T[:, :, :, :1, :1]
	B_T = Lambda_square_sub_block_T[:, :, :, :1, 1:]
	C_T = Lambda_square_sub_block_T[:, :, :, 1:, :1]
	D_T = Lambda_square_sub_block_T[:, :, :, 1:, 1:]
	D_inv_T = batch_schur_complement_inverse_matmul_complex(D_T)

	D_inv_C_T = matmul(D_inv_T, C_T)
	B_D_inv_T = matmul(B_T, D_inv_T)
	MD_T = A_T-matmul(B_T, D_inv_C_T)
	MD_inv_T = 1/MD_T

	top_left_T = MD_inv_T
	bottom_left_T = -D_inv_C_T*MD_inv_T
	top_right_T = -MD_inv_T*B_D_inv_T
	bottom_right_T = D_inv_T-bottom_left_T*B_D_inv_T

	Lambda_square_sub_block_inv_T = concat([concat([top_left_T, top_right_T], axis=4), 
				      						concat([bottom_left_T, bottom_right_T], axis=4)], axis=3)
	return Lambda_square_sub_block_inv_T

def batch_frequency_inverse_full_circular_conv2D(Y, Lambda):
	# Schur complement spectral inverse
	Lambda_T = torch.transpose(torch.transpose(Lambda, 4, 2), 5, 3)
	Psi_complex_T = batch_schur_complement_inverse_matmul_complex(complex_lib.re_im_to_complex(Lambda_T))
	Psi_complex = torch.transpose(torch.transpose(Psi_complex_T, 3, 1), 4, 2)
	X = batch_frequency_circular_conv2D(Y, Psi_complex)
	return X

def generate_batch_frequency_inverse_circular_conv2D(k, n):
	batch_SubmatrixDFT = generate_batch_SubmatrixDFT(k, n)
	def func_batch_frequency_inverse_circular_conv2D(Y, K):
		Lambda = batch_SubmatrixDFT(K)
		return batch_frequency_inverse_full_circular_conv2D(Y, Lambda)
	return func_batch_frequency_inverse_circular_conv2D

##############################################################################################################

def batch_spectral_schur_log_determinant(Lambda):
	# overall # O((c)(c+1)(2c+1)/6 n^2) = O((c)^3 n^2) is the overall complexity
	# O(c^3 n^2) when starting with Lambda, O(c^2n(k^2+(k+c)n) starting with K

	if Lambda.shape[2] == 1: 
		return torch.sum(torch.log(complex_lib.abs(Lambda, backend='torch')), axis=[1, 2, 3, 4])  # O(n^2)

	A = Lambda[:, :, :-1, :-1]
	B = Lambda[:, :, :-1, -1:]
	C = Lambda[:, :, -1:, :-1]
	D = Lambda[:, :, -1:, -1:]
	D_log_det = torch.sum(torch.log(complex_lib.abs(D, backend='torch')), axis=[1, 2, 3, 4]) # O(n^2)

	BD_rec = complex_lib.mult(B, complex_lib.reciprocal(D, backend='torch'), mult_mode='hadamard', mode='complex-complex', backend='torch') # O((c-1) n^2)
	BD_recC = complex_lib.mult(BD_rec, C, mult_mode='hadamard', mode='complex-complex', backend='torch')  # O((c-1)^2 n^2)
	MD = A-BD_recC  # O((c-1)^2 n^2)
	log_det_MD = batch_spectral_schur_log_determinant(MD)
	return log_det_MD + D_log_det

def generate_batch_kernel_to_schur_log_determinant(k, n):
	batch_SubmatrixDFT = generate_batch_SubmatrixDFT(k, n)
	def func_batch_kernel_to_schur_log_determinant(K):
		Lambda = batch_SubmatrixDFT(K)
		return batch_spectral_schur_log_determinant(Lambda)
	return func_batch_kernel_to_schur_log_determinant








# # Setup
# b, c, n, k = 10, 3, 7, 4

# X = torch.randn(b, c, n, n)
# K = torch.randn(b, c, c, k, k)

# import sys, os
# sys.path.insert(1, os.path.realpath(os.path.pardir))
# from multi_channel_invertible_conv_lib import matrix_conv2D_lib
# import spectral_schur_det_lib

# H_log_dets = []
# for i in range(b):
#     H, H_blocks = matrix_conv2D_lib.H_and_H_blocks(K[i], n, backend='torch')
#     det_sign, H_log_det = np.linalg.slogdet(H)
#     H_log_dets.append(H_log_det)

# logdet_func = spectral_schur_det_lib.generate_batch_kernel_to_schur_log_determinant(k, n)
# logdet_all = logdet_func(K)
# print(np.abs(np.asarray(H_log_dets)-logdet_all.numpy()).max())
# trace()

# Y = batch_spatial_circular_conv2D_th(X, K)

# import sys, os
# sys.path.insert(1, os.path.realpath(os.path.pardir))
# import spectral_schur_det_lib
# inverse_func = spectral_schur_det_lib.generate_batch_frequency_inverse_circular_conv2D(k, n)
# X_rec = inverse_func(Y, K)
# print((X_rec - X).abs().max())
# trace()

# # Apply manually
# outputs = []
# for idx in range(N):
#     input = x[idx, np.newaxis]
#     weight = weights[idx]
#     output = torch.nn.functional.conv2d(input, weight, stride=1, padding='valid')
#     outputs.append(output)
# outputs = torch.stack(outputs)
# outputs = outputs.squeeze(1) # remove fake batch dimension

# x_vv = x.view(1, -1, H, W)
# weights_vv = weights.view(-1, weights.shape[-3], weights.shape[-2], weights.shape[-1])

# outputs_grouped_vv = torch.nn.functional.conv2d(x_vv, weights_vv, stride=1, padding='valid', groups=x.shape[0])
# outputs_grouped = outputs_grouped_vv.view(x.shape[0], C_out, outputs_grouped_vv.shape[-2], outputs_grouped_vv.shape[-1])

# print((outputs - outputs_grouped).abs().max())
# aa = batch_spatial_circular_conv2D_th(x, weights)
# trace() #[10, 5, 18, 18]
# print((aa - outputs_grouped).abs().max())
# trace()



