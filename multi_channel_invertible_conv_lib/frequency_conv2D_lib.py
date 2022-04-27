from sys import platform
if 'linux' in platform: 
    from IPython.core.debugger import Tracer
    trace = Tracer() #this one triggers the debugger
else:
    import ipdb
    trace = ipdb.set_trace

import numpy as np
import torch

from multi_channel_invertible_conv_lib import dft_lib
from multi_channel_invertible_conv_lib import complex_lib
from multi_channel_invertible_conv_lib import spatial_conv2D_lib


def frequency_circular_conv2D(X, Lambda, mode='complex', backend='torch'):
	# X (b x c x n x n), Lambda (2 x c x c x n x n)
	if mode == 're_im':
		X_freq = dft_lib.MatmulDFT(X, backend=backend) # O(k n^3)
		Y_freq = complex_lib.mult(Lambda[:, np.newaxis], X_freq[:, :, np.newaxis], 
			mult_mode='hadamard', mode='complex-complex', backend=backend).sum(3) # O(k^2 n^2)
		Y = dft_lib.MatmulDFT(Y_freq, inverse=True, backend=backend) # O(k n^3)
	elif mode == 'complex':
		if backend == 'torch': X_freq = torch.fft.fft2(X)
		elif backend == 'numpy': X_freq = np.fft.fft2(X) # O(k n^2 log n)
		Lambda_complex = complex_lib.re_im_to_complex(Lambda)
		Y_freq = (Lambda_complex[np.newaxis]*X_freq[:, np.newaxis]).sum(2) # O(k^2 n^2)
		if backend == 'torch': Y = torch.fft.ifft2(Y_freq).real # O(k n^2 log n)
		elif backend == 'numpy': Y = np.fft.ifft2(Y_freq).real
	return Y

def frequency_single_channel_inverse_circular_conv2D(Y_i, Lambda_ii, mode='complex', backend='torch'):
	# Y_i (b x n x n), Lambda_ii (2 x n x n)
	# just a simplified version of the below for c=1.
	if mode == 're_im':
		Y_i_freq = dft_lib.MatmulDFT(Y_i, backend=backend) # O(n^3)
		X_i_freq = complex_lib.mult(complex_lib.reciprocal(Lambda_ii, backend=backend)[:, np.newaxis], Y_i_freq, 
			mult_mode='hadamard', mode='complex-complex', backend=backend) # O(n^2)
		X_i = dft_lib.MatmulDFT(X_i_freq, inverse=True, backend=backend) # O(n^3)
	elif mode == 'complex':
		if backend == 'torch': Y_i_freq = torch.fft.fft2(Y_i)
		elif backend == 'numpy': Y_i_freq = np.fft.fft2(Y_i) # O(n^2 log n)
		Lambda_ii_complex = complex_lib.re_im_to_complex(Lambda_ii)
		X_i_freq = ((1/Lambda_ii_complex)[np.newaxis]*Y_i_freq) # O(n^2)
		if backend == 'torch': X_i = torch.fft.ifft2(X_i_freq).real # O(n^2 log n)
		elif backend == 'numpy': X_i = np.fft.ifft2(X_i_freq).real
	return X_i

def frequency_inverse_diagonal_circular_conv2D(Y, Lambda_diag, mode='complex', backend='torch'):
	# Y (b x c x n x n), Lambda_diag (2 x c x n x n)
	if mode == 're_im':
		Y_freq = dft_lib.MatmulDFT(Y, backend=backend) # O(c n^3)
		X_freq = complex_lib.mult(complex_lib.reciprocal(Lambda_diag, backend=backend)[:, np.newaxis], Y_freq, 
			mult_mode='hadamard', mode='complex-complex', backend=backend) # O(c n^2)
		X = dft_lib.MatmulDFT(X_freq, inverse=True, backend=backend) # O(c n^3)
	elif mode == 'complex':
		if backend == 'torch': Y_freq = torch.fft.fft2(Y)
		elif backend == 'numpy': Y_freq = np.fft.fft2(Y) # O(c n^2 log n)
		Lambda_complex = complex_lib.re_im_to_complex(Lambda_diag)
		X_freq = ((1/Lambda_complex)[np.newaxis]*Y_freq)# O(c n^2)
		if backend == 'torch': X = torch.fft.ifft2(X_freq).real # O(c n^2 log n)
		elif backend == 'numpy': X = np.fft.ifft2(X_freq).real
	return X

def frequency_inverse_triangular_circular_conv2D(Y, K, Lambda_diag, mode='complex', backend='torch'):
	# Y (b x c x n x n), K (c x c x k x k), Lambda_diag (2 x c x n x n)
	X_0 = frequency_single_channel_inverse_circular_conv2D(Y[:, 0], Lambda_diag[:, 0], mode=mode, backend=backend)[:, np.newaxis]
	if Y.shape[1] == 1: return X_0
	Y_residual = Y[:, 1:]-spatial_conv2D_lib.spatial_circular_conv2D(X_0, K[1:, 0, np.newaxis], channel_mode='full', backend=backend)
	X_1k = frequency_inverse_triangular_circular_conv2D(Y_residual, K[1:, 1:], Lambda_diag[:, 1:], mode=mode, backend=backend)
	if backend == 'torch':
		return  torch.concat([X_0, X_1k], axis=1)
	elif backend == 'numpy': 
		return  np.concatenate([X_0, X_1k], axis=1)

def schur_complement_inverse_dot(Lambda_square_sub_block, complement_mode='H/D', backend='torch'):
	if Lambda_square_sub_block.shape[1] == 1: return complex_lib.reciprocal(Lambda_square_sub_block, backend=backend)
	if backend == 'torch': concat = torch.concat
	elif backend == 'numpy': concat = np.concatenate

	if complement_mode == 'H/D':
		A = Lambda_square_sub_block[:, :1, :1]
		B = Lambda_square_sub_block[:, :1, 1:]
		C = Lambda_square_sub_block[:, 1:, :1]
		D = Lambda_square_sub_block[:, 1:, 1:]
		D_inv = schur_complement_inverse_dot(D, complement_mode, backend=backend)

		D_inv_C = complex_lib.mult(D_inv, C[:, np.newaxis, :, 0], mult_mode='hadamard', mode='complex-complex', backend=backend).sum(axis=2, keepdims=True)
		B_D_inv = complex_lib.mult(B[:, 0, :, np.newaxis], D_inv, mult_mode='hadamard', mode='complex-complex', backend=backend).sum(axis=1, keepdims=True)
		MD = A-complex_lib.mult(B[:, 0], D_inv_C[:, :, 0], mult_mode='hadamard', mode='complex-complex', backend=backend).sum(axis=1, keepdims=True)[:, :, np.newaxis, ...]
		MD_inv = complex_lib.reciprocal(MD, backend=backend)

		top_left = MD_inv
		bottom_left = -complex_lib.mult(D_inv_C, MD_inv, mult_mode='hadamard', mode='complex-complex', backend=backend)
		top_right = -complex_lib.mult(MD_inv, B_D_inv, mult_mode='hadamard', mode='complex-complex', backend=backend)
		bottom_right = D_inv-complex_lib.mult(bottom_left, B_D_inv, mult_mode='hadamard', mode='complex-complex', backend=backend)
	elif complement_mode == 'H/A':
		A = Lambda_square_sub_block[:, :-1, :-1]
		B = Lambda_square_sub_block[:, :-1, -1:]
		C = Lambda_square_sub_block[:, -1:, :-1]
		D = Lambda_square_sub_block[:, -1:, -1:]
		A_inv = schur_complement_inverse_dot(A, complement_mode, backend=backend)

		A_inv_B = complex_lib.mult(A_inv, B[:, np.newaxis, :, 0], mult_mode='hadamard', mode='complex-complex', backend=backend).sum(axis=2, keepdims=True)
		C_A_inv = complex_lib.mult(C[:, 0, :, np.newaxis], A_inv, mult_mode='hadamard', mode='complex-complex', backend=backend).sum(axis=1, keepdims=True)
		MA = D-complex_lib.mult(C[:,0], A_inv_B[:, :, 0], mult_mode='hadamard', mode='complex-complex', backend=backend).sum(axis=1, keepdims=True)[:, :, np.newaxis, ...]
		MA_inv = complex_lib.reciprocal(MA, backend=backend)

		bottom_right = MA_inv
		top_right = -complex_lib.mult(A_inv_B, MA_inv, mult_mode='hadamard', mode='complex-complex', backend=backend)
		bottom_left = -complex_lib.mult(MA_inv, C_A_inv, mult_mode='hadamard', mode='complex-complex', backend=backend)
		top_left = A_inv-complex_lib.mult(top_right, C_A_inv, mult_mode='hadamard', mode='complex-complex', backend=backend)
	
	Lambda_square_sub_block_inv = concat([concat([top_left, top_right], axis=2), 
				    					  concat([bottom_left, bottom_right], axis=2)], axis=1)
	return Lambda_square_sub_block_inv

def schur_complement_inverse_matmul(Lambda_square_sub_block_T, complement_mode='H/D', backend='torch'):
	if Lambda_square_sub_block_T.shape[3] == 1: return complex_lib.reciprocal(Lambda_square_sub_block_T, backend=backend)
	if backend == 'torch': concat = torch.concat
	elif backend == 'numpy': concat = np.concatenate

	if complement_mode == 'H/D':
		A_T = Lambda_square_sub_block_T[:, :, :, :1, :1]
		B_T = Lambda_square_sub_block_T[:, :, :, :1, 1:]
		C_T = Lambda_square_sub_block_T[:, :, :, 1:, :1]
		D_T = Lambda_square_sub_block_T[:, :, :, 1:, 1:]
		D_inv_T = schur_complement_inverse_matmul(D_T, complement_mode, backend=backend)

		D_inv_C_T = complex_lib.mult(D_inv_T, C_T, mult_mode='matmul', mode='complex-complex', backend=backend)
		B_D_inv_T = complex_lib.mult(B_T, D_inv_T, mult_mode='matmul', mode='complex-complex', backend=backend)
		MD_T = A_T-complex_lib.mult(B_T, D_inv_C_T, mult_mode='matmul', mode='complex-complex', backend=backend)
		MD_inv_T = complex_lib.reciprocal(MD_T, backend=backend)

		top_left_T = MD_inv_T
		bottom_left_T = -complex_lib.mult(D_inv_C_T, MD_inv_T, mult_mode='hadamard', mode='complex-complex', backend=backend)
		top_right_T = -complex_lib.mult(MD_inv_T, B_D_inv_T, mult_mode='hadamard', mode='complex-complex', backend=backend)
		bottom_right_T = D_inv_T-complex_lib.mult(bottom_left_T, B_D_inv_T, mult_mode='hadamard', mode='complex-complex', backend=backend)
	elif complement_mode == 'H/A':
		A_T = Lambda_square_sub_block_T[:, :, :, :-1, :-1]
		B_T = Lambda_square_sub_block_T[:, :, :, :-1, -1:]
		C_T = Lambda_square_sub_block_T[:, :, :, -1:, :-1]
		D_T = Lambda_square_sub_block_T[:, :, :, -1:, -1:]
		A_inv_T = schur_complement_inverse_matmul(A_T, complement_mode, backend=backend)

		A_inv_B_T = complex_lib.mult(A_inv_T, B_T, mult_mode='matmul', mode='complex-complex', backend=backend)
		C_A_inv_T = complex_lib.mult(C_T, A_inv_T, mult_mode='matmul', mode='complex-complex', backend=backend)
		MA_T = D_T-complex_lib.mult(C_T, A_inv_B_T, mult_mode='matmul', mode='complex-complex', backend=backend)
		MA_inv_T = complex_lib.reciprocal(MA_T, backend=backend)

		bottom_right_T = MA_inv_T
		top_right_T = -complex_lib.mult(A_inv_B_T, MA_inv_T, mult_mode='hadamard', mode='complex-complex', backend=backend)
		bottom_left_T = -complex_lib.mult(MA_inv_T, C_A_inv_T, mult_mode='hadamard', mode='complex-complex', backend=backend)
		top_left_T = A_inv_T-complex_lib.mult(top_right_T, C_A_inv_T, mult_mode='hadamard', mode='complex-complex', backend=backend)

	Lambda_square_sub_block_inv_T = concat([concat([top_left_T, top_right_T], axis=4), 
				      						concat([bottom_left_T, bottom_right_T], axis=4)], axis=3)
	return Lambda_square_sub_block_inv_T
	
def schur_complement_inverse_dot_complex(Lambda_square_sub_block, complement_mode='H/D', backend='torch'):
	if Lambda_square_sub_block.shape[0] == 1: return 1/Lambda_square_sub_block
	if backend == 'torch': concat = torch.concat
	elif backend == 'numpy': concat = np.concatenate

	if complement_mode == 'H/D':
		A = Lambda_square_sub_block[:1, :1]
		B = Lambda_square_sub_block[:1, 1:]
		C = Lambda_square_sub_block[1:, :1]
		D = Lambda_square_sub_block[1:, 1:]
		D_inv = schur_complement_inverse_dot_complex(D, complement_mode, backend=backend)

		D_inv_C = (D_inv*C[np.newaxis, :, 0]).sum(axis=1, keepdims=True)
		B_D_inv = (B[0, :, np.newaxis]*D_inv).sum(axis=0, keepdims=True)
		MD = A-(B[0]*D_inv_C[:, 0]).sum(axis=0, keepdims=True)[:, np.newaxis, ...]
		MD_inv = 1/MD
		
		top_left = MD_inv
		bottom_left = -D_inv_C*MD_inv
		top_right = -MD_inv*B_D_inv
		bottom_right = D_inv-bottom_left*B_D_inv
	elif complement_mode == 'H/A':
		A = Lambda_square_sub_block[:-1, :-1]
		B = Lambda_square_sub_block[:-1, -1:]
		C = Lambda_square_sub_block[-1:, :-1]
		D = Lambda_square_sub_block[-1:, -1:]
		A_inv = schur_complement_inverse_dot_complex(A, complement_mode, backend=backend)

		A_inv_B = (A_inv*B[np.newaxis, :, 0]).sum(axis=1, keepdims=True)
		C_A_inv = (C[0, :, np.newaxis]*A_inv).sum(axis=0, keepdims=True)
		MA = D-(C[0]*A_inv_B[:, 0]).sum(axis=0, keepdims=True)[:, np.newaxis, ...]
		MA_inv = 1/MA

		bottom_right = MA_inv
		top_right= -A_inv_B*MA_inv
		bottom_left = -MA_inv*C_A_inv
		top_left = A_inv-top_right*C_A_inv

	Lambda_square_sub_block_inv = concat([concat([top_left, top_right], axis=1), 
				    					  concat([bottom_left, bottom_right], axis=1)], axis=0)
	return Lambda_square_sub_block_inv

def schur_complement_inverse_matmul_complex(Lambda_square_sub_block_T, complement_mode='H/D', backend='torch'):
	if Lambda_square_sub_block_T.shape[2] == 1: return 1/Lambda_square_sub_block_T
	if backend == 'torch': 
		matmul = torch.matmul
		concat = torch.concat
	elif backend == 'numpy': 
		matmul = np.matmul
		concat = np.concatenate

	if complement_mode == 'H/D':
		A_T = Lambda_square_sub_block_T[:, :, :1, :1]
		B_T = Lambda_square_sub_block_T[:, :, :1, 1:]
		C_T = Lambda_square_sub_block_T[:, :, 1:, :1]
		D_T = Lambda_square_sub_block_T[:, :, 1:, 1:]
		D_inv_T = schur_complement_inverse_matmul_complex(D_T, complement_mode, backend=backend)

		D_inv_C_T = matmul(D_inv_T, C_T)
		B_D_inv_T = matmul(B_T, D_inv_T)
		MD_T = A_T-matmul(B_T, D_inv_C_T)
		MD_inv_T = 1/MD_T

		top_left_T = MD_inv_T
		bottom_left_T = -D_inv_C_T*MD_inv_T
		top_right_T = -MD_inv_T*B_D_inv_T
		bottom_right_T = D_inv_T-bottom_left_T*B_D_inv_T
	elif complement_mode == 'H/A':
		A_T = Lambda_square_sub_block_T[:, :, :-1, :-1]
		B_T = Lambda_square_sub_block_T[:, :, :-1, -1:]
		C_T = Lambda_square_sub_block_T[:, :, -1:, :-1]
		D_T = Lambda_square_sub_block_T[:, :, -1:, -1:]
		A_inv_T = schur_complement_inverse_matmul_complex(A_T, complement_mode, backend=backend)

		A_inv_B_T = matmul(A_inv_T, B_T)
		C_A_inv_T = matmul(C_T, A_inv_T)
		MA_T = D_T-matmul(C_T, A_inv_B_T)
		MA_inv_T = 1/MA_T

		bottom_right_T = MA_inv_T
		top_right_T = -A_inv_B_T*MA_inv_T
		bottom_left_T = -MA_inv_T*C_A_inv_T
		top_left_T = A_inv_T-top_right_T*C_A_inv_T

	Lambda_square_sub_block_inv_T = concat([concat([top_left_T, top_right_T], axis=3), 
				      						concat([bottom_left_T, bottom_right_T], axis=3)], axis=2)
	return Lambda_square_sub_block_inv_T

def frequency_inverse_full_circular_conv2D(Y, K, Lambda, mode='complex', backend='torch', shur_mode='matmul'):
	# Schur complement spectral inverse
	if shur_mode == 'dot':
		if mode == 'complex': 
			Psi = schur_complement_inverse_dot_complex(complex_lib.re_im_to_complex(Lambda), complement_mode='H/D', backend=backend)
			Psi = complex_lib.complex_to_re_im(Psi, backend=backend)
		elif mode == 're_im': 
			Psi = schur_complement_inverse_dot(Lambda, complement_mode='H/D', backend=backend)
	else:
		if backend == 'torch': Lambda = torch.transpose(torch.transpose(Lambda, 3, 1), 4, 2)
		elif backend == 'numpy': Lambda = np.transpose(Lambda, [0, 3, 4, 1, 2])
		if mode == 'complex': 
			Psi = schur_complement_inverse_matmul_complex(complex_lib.re_im_to_complex(Lambda), complement_mode='H/D', backend=backend)
			if backend == 'torch': Psi = torch.transpose(torch.transpose(Psi, 2, 0), 3, 1)
			elif backend == 'numpy': Psi = np.transpose(Psi, [2, 3, 0, 1])
			Psi = complex_lib.complex_to_re_im(Psi, backend=backend)
		elif mode == 're_im': 
			Psi = schur_complement_inverse_matmul(Lambda, complement_mode='H/D', backend=backend)
			if backend == 'torch': Psi = torch.transpose(torch.transpose(Psi, 3, 1), 4, 2)
			elif backend == 'numpy': Psi = np.transpose(Psi, [0, 3, 4, 1, 2])
	X = frequency_circular_conv2D(Y, Psi, mode=mode, backend=backend)
	return X

def frequency_inverse_circular_conv2D(Y, K, channel_mode, mode='complex', backend='torch'):
	if channel_mode in ['block_diagonal', 'lower_block_triangular']:
		if backend == 'torch':
			K_diag = torch.concat([K[i, i, np.newaxis, np.newaxis] for i in range(K.shape[0])], axis=0)
		elif backend == 'numpy':
			K_diag = np.concatenate([K[i, i, np.newaxis, np.newaxis] for i in range(K.shape[0])], axis=0)	
		Lambda_diag = dft_lib.SubmatrixDFT(K_diag, Y.shape[-1], backend=backend)[:, :, 0]
	else:
		Lambda = dft_lib.SubmatrixDFT(K, Y.shape[-1], backend=backend)

	if channel_mode == 'block_diagonal':
		return frequency_inverse_diagonal_circular_conv2D(Y, Lambda_diag, mode=mode, backend=backend)
	elif channel_mode == 'lower_block_triangular':
		return frequency_inverse_triangular_circular_conv2D(Y, K, Lambda_diag, mode=mode, backend=backend)
	elif channel_mode == 'full':
		return frequency_inverse_full_circular_conv2D(Y, K, Lambda, mode=mode, backend=backend)














