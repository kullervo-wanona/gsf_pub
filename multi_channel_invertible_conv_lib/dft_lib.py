from sys import platform
if 'linux' in platform: 
    from IPython.core.debugger import Tracer
    trace = Tracer() #this one triggers the debugger
else:
    import ipdb
    trace = ipdb.set_trace

import numpy as np
import torch

from multi_channel_invertible_conv_lib import complex_lib

def n_embedded_fft_kernel(K, n, backend='torch'):
	# Eq. 4 create K_hat (c x c x n x n) from K (c x c x m x m)
	if backend == 'torch':
		K_hat = torch.zeros(list(K.shape[:-2])+[n, n], dtype=K.dtype)
		K_hat[..., :K.shape[-2], :K.shape[-1]] = K
		K_hat = torch.roll(K_hat, shifts=[-1, -1], dims=[-2, -1])
		K_hat = torch.flip(K_hat, [-1, -2])
	elif backend == 'numpy':
		K_hat = np.zeros(list(K.shape[:-2])+[n, n])
		K_hat[..., :K.shape[-2], :K.shape[-1]] = K
		K_hat = np.roll(K_hat, shift=[-1, -1], axis=[-2, -1])
		K_hat = np.flip(K_hat, [-1, -2]).copy()
	return K_hat

def DFT_matrix_F(n, n_rows=None, shift=None, backend='torch'):
	# n x n, complex, symmetrical, sqrt(n)*orthogonal matrix.
	# 1/F = F.conj() since np.abs(F) = [1] for all elements
	# np.matmul(F, 1/F) = np.matmul(F, F.conj()) = np.matmul(F, F.conj().T) = n*np.eye(n)
	i, j = np.meshgrid(np.arange(n), np.arange(n))
	omega_complex = np.exp(-2*np.pi*1J/n)
	W_complex = np.power(omega_complex, i*j)#/np.sqrt(n) 
	if shift is not None: W_complex = np.roll(W_complex, shift=shift, axis=[-2, -1])
	if n_rows is not None: W_complex = W_complex[:n_rows, :]
	F = np.concatenate([W_complex.real[np.newaxis, ...], W_complex.imag[np.newaxis, ...]], axis=0)
	if backend == 'torch': F = torch.tensor(F, dtype=torch.float32)
	return F

def fourier_2D_circular_conv_eigenvectors(n, backend='torch'):
	# n^2 x n^2, complex, symmetrical, orthogonal matrix.
	# np.matmul(Q, Q.conj()) = np.matmul(Q, Q.conj().T) = np.eye(n)
	F = DFT_matrix_F(n, n_rows=None, shift=None, backend=backend)	
	F_sq_kronocker_ext = complex_lib.mult(
		F[:, :, np.newaxis, :, np.newaxis], F[:, np.newaxis, :, np.newaxis, :], 
		mult_mode='hadamard', mode='complex-complex', backend=backend)
	Q = (1/n)*F_sq_kronocker_ext.reshape(2, n**2, n**2)
	return Q

def MatmulDFT(tensor, inverse=False, backend='torch'): 
	# Assumes the last two axes has the same size where DFT/inverse DFT is applied. 
	# Tensor is real if applying DFT and result is real if inverse DFT is applied.
	n = tensor.shape[-1]
	F = DFT_matrix_F(n, n_rows=None, shift=None, backend=backend)
	if inverse:
		F_ext = F[tuple([slice(None)]+[np.newaxis]*(len(tensor.shape)-3))]
		F_conj_ext = (1/n)*complex_lib.conj(F_ext, backend=backend)
		inter = complex_lib.mult(tensor, F_conj_ext, mult_mode='matmul', mode='complex-complex', backend=backend)
		return complex_lib.mult(F_conj_ext, inter, mult_mode='matmul', mode='complex-complex-real-part', backend=backend)
	else:
		F_ext = F[tuple([slice(None)]+[np.newaxis]*(len(tensor.shape)-2))]
		inter = complex_lib.mult(tensor, F_ext, mult_mode='matmul', mode='real-complex', backend=backend)
		return complex_lib.mult(F_ext, inter, mult_mode='matmul', mode='complex-complex', backend=backend)

def SubmatrixDFT(K, n, backend='torch'):
	# FFT of K_hat from K (c x c x k x k)
	k = K.shape[-1]
	F_relevant = DFT_matrix_F(n, n_rows=k, shift=[+(k-1), +(k-1)], backend=backend)
	if backend == 'torch':
		K_time_reversed = torch.flip(K, [-1, -2])
	elif backend == 'numpy':
		K_time_reversed = np.flip(K, [-1, -2]).copy()
	Z = complex_lib.mult(K_time_reversed, F_relevant[:, np.newaxis, np.newaxis], mult_mode='matmul', mode='real-complex', backend=backend)
	Lambda_rolled = complex_lib.mult(complex_lib.transpose(F_relevant, backend=backend)[:, np.newaxis, np.newaxis], Z, mult_mode='matmul', mode='complex-complex', backend=backend)
	if backend == 'torch':
		Lambda = torch.roll(Lambda_rolled, shifts=[-(k-1), -(k-1)], dims=[-2, -1])
	elif backend == 'numpy':
		Lambda = np.roll(Lambda_rolled, shift=[-(k-1), -(k-1)], axis=[-2, -1])
	return Lambda











