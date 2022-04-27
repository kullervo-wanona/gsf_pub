from sys import platform
if 'linux' in platform: 
    from IPython.core.debugger import Tracer
    trace = Tracer() #this one triggers the debugger
else:
    import ipdb
    trace = ipdb.set_trace

import numpy as np
import torch

def complex_to_re_im(tensor, backend='torch'):
	if backend == 'torch':
		return torch.concat([tensor.real[np.newaxis], tensor.imag[np.newaxis]], axis=0)
	elif backend == 'numpy':
		return np.concatenate([tensor.real[np.newaxis], tensor.imag[np.newaxis]], axis=0)

def re_im_to_complex(tensor):
	return tensor[0]+1J*tensor[1]

def abs(tensor, backend):
	abs_sq = (tensor[0]**2+tensor[1]**2)
	if backend == 'torch':
		return torch.sqrt(abs_sq)
	elif backend == 'numpy':
		return np.sqrt(abs_sq)

def transpose(tensor, backend='torch'):
	if backend == 'torch':
		return torch.transpose(tensor, -1, -2)
	elif backend == 'numpy':
		return np.transpose(tensor, list(range(len(tensor.shape[:-2])))+[-1, -2])

def conj(tensor, backend='torch'):
	if backend == 'torch':
		tensor_copy = torch.clone(tensor)
	elif backend == 'numpy':
		tensor_copy = tensor.copy()
	tensor_copy[1] = -tensor_copy[1]
	return tensor_copy

def hermitian_transpose(tensor, backend='torch'):
	return transpose(conj(tensor, backend), backend)

def reciprocal(tensor, backend='torch'):
	if backend == 'torch':
		denom = torch.sum(tensor**2, axis=0, keepdims=True)
		result_re_im = torch.clone(tensor)
	elif backend == 'numpy':
		denom = np.sum(tensor**2, axis=0, keepdims=True)
		result_re_im = tensor.copy()
	result_re_im[1, ...] = -result_re_im[1, ...]
	result_re_im = result_re_im/denom
	return result_re_im


def mult(tensor_1, tensor_2, mult_mode='matmul', mode='complex-complex', backend='torch'):
	if mode == 'real-complex':
		if mult_mode == 'matmul':
			if backend == 'torch':
				result_re_im = torch.matmul(tensor_1[np.newaxis, ...], tensor_2)
			elif backend == 'numpy':
				result_re_im = np.matmul(tensor_1[np.newaxis, ...], tensor_2)			
		elif mult_mode == 'hadamard':
			result_re_im = tensor_1[np.newaxis, ...]*tensor_2

	elif mode == 'complex-real':
		if mult_mode == 'matmul':
			if backend == 'torch':
				result_re_im = torch.matmul(tensor_1, tensor_2[np.newaxis, ...])
			elif backend == 'numpy':
				result_re_im = np.matmul(tensor_1, tensor_2[np.newaxis, ...])			
		elif mult_mode == 'hadamard':
			result_re_im = tensor_1*tensor_2[np.newaxis, ...]

	elif mode == 'complex-complex':
		if mult_mode == 'matmul':
			if backend == 'torch':
				result_matrix = torch.matmul(tensor_1[np.newaxis, ...], tensor_2[:, np.newaxis, ...])
			elif backend == 'numpy':
				result_matrix = np.matmul(tensor_1[np.newaxis, ...], tensor_2[:, np.newaxis, ...])			
		elif mult_mode == 'hadamard':
			result_matrix = tensor_1[np.newaxis, ...]*tensor_2[:, np.newaxis, ...]
		
		if backend == 'torch':
			result_re_im = torch.concat([(result_matrix[0, 0]-result_matrix[1, 1])[np.newaxis, ...], 
										 (result_matrix[0, 1]+result_matrix[1, 0])[np.newaxis, ...]], axis=0)
		elif backend == 'numpy':
			result_re_im = np.concatenate([(result_matrix[0, 0]-result_matrix[1, 1])[np.newaxis, ...], 
										   (result_matrix[0, 1]+result_matrix[1, 0])[np.newaxis, ...]], axis=0)
	
	elif mode == 'complex-complex-real-part':
		if mult_mode == 'matmul':
			if backend == 'torch':
				result_re_im = torch.matmul(tensor_1[0], tensor_2[0])-torch.matmul(tensor_1[1], tensor_2[1])
			elif backend == 'numpy':
				result_re_im = np.matmul(tensor_1[0], tensor_2[0])-np.matmul(tensor_1[1], tensor_2[1])		
		elif mult_mode == 'hadamard':
			result_re_im = tensor_1[0]*tensor_2[0]-tensor_1[1]*tensor_2[1]
	return result_re_im





