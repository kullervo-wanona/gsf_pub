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

def conv2D_spatial_connectivity_mask(n, k, spatial_mode='circular', backend='torch'):
    assert (spatial_mode in ['circular', 'linear'])
    vals = np.arange(n*n).reshape(n, n)
    mask = np.zeros((n*n, n*n))
    for i in range(n):
        for j in range(n):
            row = vals[i, j]
            for l in range(k):
                for o in range(k):
                    add_row = (i+l)
                    add_col = (j+o)
                    if spatial_mode == 'circular':
                        col = vals[add_row % n, add_col % n]
                        mask[row, col] = 1
                    else:
                        if add_row < n and add_col < n: 
                            col = vals[add_row, add_col]
                            mask[row, col] = 1
    if backend == 'torch': mask = torch.tensor(mask, dtype=torch.float32)
    return mask

def linear_conv2D_masking(H, Hs, n, k, backend='torch'):
	linear_connectivity_mask = conv2D_spatial_connectivity_mask(n, k, spatial_mode='linear', backend=backend)
	Hs_masked = Hs*linear_connectivity_mask[np.newaxis, np.newaxis, :, :]

	block_n_row_col = [Hs.shape[0], Hs.shape[1]]
	block_size = [Hs.shape[2], Hs.shape[3]]

	inject_dict_for_H_mask = {}
	for i in range(Hs.shape[0]):
		for j in range(Hs.shape[1]):
			inject_dict_for_H_mask[(i, j)] = linear_connectivity_mask

	H_mask = generate_block_matrix(block_size, block_n_row_col, inject_dict_for_H_mask, backend=backend)
	H_masked = H*H_mask
	return H_masked, Hs_masked, H_mask, linear_connectivity_mask

def create_circulant_mat(row, n_rows, backend='torch'):
	rows = [row[np.newaxis, :]]
	for r in range(1, n_rows):
		if backend == 'torch': rows.append(torch.concat([row[-r:], row[:-r]], axis=0)[np.newaxis, :])
		elif backend == 'numpy': rows.append(np.concatenate([row[-r:], row[:-r]], axis=0)[np.newaxis, :])

	if backend == 'torch': circulant_mat = torch.concat(rows, axis=0)
	elif backend == 'numpy': circulant_mat = np.concatenate(rows, axis=0)
	return circulant_mat

def circulant_K_embed_per_channel(K, n, backend='torch'):
	# assert (len(K.shape) == 4)
	# assert (K.shape[-1] == K.shape[-2])

	K_embed = np.zeros(list(K.shape[:-2])+[n, n])
	if backend == 'torch': K_embed = torch.tensor(K_embed, dtype=torch.float32)
	K_embed[..., :K.shape[-2], :K.shape[-1]] = K

	n_rows = K_embed.shape[-2]
	n_cols = K_embed.shape[-1] 
	circ_K_embed = np.zeros([K_embed.shape[0], K_embed.shape[1], n_rows, n_cols, n_cols])
	if backend == 'torch': circ_K_embed = torch.tensor(circ_K_embed, dtype=torch.float32)

	for i in range(K_embed.shape[0]):
		for j in range(K_embed.shape[1]):
			n_circulant_mats = [create_circulant_mat(K_embed[i, j, row_ind, :], n_rows=n_cols, backend=backend)[np.newaxis, :, :] 
				for row_ind in range(n_rows)]
			if backend == 'torch': circ_K_embed[i, j, :, :, :] = torch.concat(n_circulant_mats, axis=0)
			elif backend == 'numpy': circ_K_embed[i, j, :, :, :] = np.concatenate(n_circulant_mats, axis=0)
	return circ_K_embed

def generate_block_matrix(block_size, block_n_row_col, inject_dict={}, backend='torch'):
	block_mat = np.zeros((block_size[0]*block_n_row_col[0], block_size[1]*block_n_row_col[1]))
	if backend == 'torch': block_mat = torch.tensor(block_mat, dtype=torch.float32)
	for index in inject_dict:
		block_row, block_col = index
		block_mat[block_row*block_size[0]:(block_row+1)*block_size[0], \
				  block_col*block_size[1]:(block_col+1)*block_size[1]] = inject_dict[index] 
	return block_mat

def H_and_H_blocks(K_full, n, backend='torch'):
	circ_K_embed = circulant_K_embed_per_channel(K_full, n, backend=backend)

	block_size = [circ_K_embed.shape[3], circ_K_embed.shape[4]]
	block_n_row_col = [circ_K_embed.shape[2], circ_K_embed.shape[2]]

	H_blocks = np.zeros(list(circ_K_embed.shape[:2])+[block_size[0]*block_n_row_col[0], block_size[1]*block_n_row_col[1]])
	if backend == 'torch': H_blocks = torch.tensor(H_blocks, dtype=torch.float32)
	
	inject_dict_for_H = {}
	for i in range(circ_K_embed.shape[0]):
		for j in range(circ_K_embed.shape[1]):
			inject_dict = {}
			for block_row in range(circ_K_embed.shape[2]):
				for block_col in range(circ_K_embed.shape[2]):
					inject_dict[(block_row, block_col)] = circ_K_embed[i, j, block_col-block_row]
			H_blocks[i, j] = generate_block_matrix(block_size, block_n_row_col, inject_dict, backend=backend)
			inject_dict_for_H[(i, j)] = H_blocks[i, j]

	H = generate_block_matrix([H_blocks.shape[2], H_blocks.shape[3]], [H_blocks.shape[0], H_blocks.shape[1]], inject_dict_for_H, backend=backend)
	return H, H_blocks

def matrix_conv2D(X, H, backend='torch'):
	if backend == 'torch':
		return helper.unvectorize(torch.matmul(H, helper.vectorize(X).T).T, [X.shape[1], X.shape[2], X.shape[3]])
	if backend == 'numpy':
		return helper.unvectorize(np.matmul(H, helper.vectorize(X).T).T, [X.shape[1], X.shape[2], X.shape[3]])












