from sys import platform
if 'linux' in platform: 
    from IPython.core.debugger import Tracer
    trace = Tracer() #this one triggers the debugger
else:
    import ipdb
    trace = ipdb.set_trace

import numpy as np
import torch

def generate_identity_kernel(c, k, channel_mode, backend='torch'):
    assert (channel_mode in ['full', 'lower_block_triangular', 'block_diagonal'])

    if channel_mode == 'full':
        K_spatial = np.zeros([c, c, k, k])
        for i in range(c): K_spatial[i, i, 0, 0] = 1
    elif channel_mode == 'lower_block_triangular':
        K_spatial = np.zeros([(c*(c+1))//2, 1, k, k])
        for i in range(c): K_spatial[(i*(i+1))//2+i, 0, 0, 0] = 1
    elif channel_mode == 'block_diagonal':
        K_spatial = np.zeros([c, 1, k, k])
        for i in range(c): K_spatial[i, 0, 0, 0] = 1
    
    if channel_mode == 'full':
        K_full = K_spatial.copy()
    elif channel_mode == 'lower_block_triangular':
        # c, k = int(np.floor(np.sqrt(K_spatial.shape[0]*2))), K_spatial.shape[2]
        K_full = np.zeros((c, c, k, k))
        for row in range(c):
            for col in range(row+1):
                K_full[row, col, :, :] = K_spatial[(row*(row+1))//2+col, 0, :, :]
    elif channel_mode == 'block_diagonal':
        # c, k = K_spatial.shape[0], K_spatial.shape[2]
        K_full = np.zeros((c, c, k, k))
        for i in range(c): K_full[i, i, :, :] = K_spatial[i, 0, :, :]

    if backend == 'torch': 
        K_spatial = torch.tensor(K_spatial, dtype=torch.float32)
        K_full = torch.tensor(K_full, dtype=torch.float32)
    return K_spatial, K_full

def generate_random_kernel(c, k, channel_mode, backend='torch'):
    assert (channel_mode in ['full', 'lower_block_triangular', 'block_diagonal'])
    
    if channel_mode == 'full':
        K_spatial = np.random.randn(c, c, k, k)
    elif channel_mode == 'lower_block_triangular':
        K_spatial = np.random.randn((c*(c+1))//2, 1, k, k)
    elif channel_mode == 'block_diagonal':
        K_spatial = np.random.randn(c, 1, k, k)
    
    if channel_mode == 'full':
        K_full = K_spatial.copy()
    elif channel_mode == 'lower_block_triangular':
        # c, k = int(np.floor(np.sqrt(K_spatial.shape[0]*2))), K_spatial.shape[2]
        K_full = np.zeros((c, c, k, k))
        for row in range(c):
            for col in range(row+1):
                K_full[row, col, :, :] = K_spatial[(row*(row+1))//2+col, 0, :, :]
    elif channel_mode == 'block_diagonal':
        # c, k = K_spatial.shape[0], K_spatial.shape[2]
        K_full = np.zeros((c, c, k, k))
        for i in range(c): K_full[i, i, :, :] = K_spatial[i, 0, :, :]

    if backend == 'torch': 
        K_spatial = torch.tensor(K_spatial, dtype=torch.float32)
        K_full = torch.tensor(K_full, dtype=torch.float32)
    return K_spatial, K_full

def spatial_circular_conv2D_th(X_th, K_spatial_th, channel_mode='full'):
    # assert (len(X_th.shape) == 4)
    # assert (len(K_spatial_th.shape) == 4)
    # assert (channel_mode in ['full', 'lower_block_triangular', 'block_diagonal'])
    # if channel_mode == 'full': assert (K_spatial_th.shape[1] == X_th.shape[1])
    # else: assert (K_spatial_th.shape[1] == 1)
    # assert (K_spatial_th.shape[2] < X_th.shape[2])
    # assert (K_spatial_th.shape[3] < X_th.shape[3])
    # assert (K_spatial_th.shape[1] == X_th.shape[1])
    # assert (K_spatial_th.shape[0] == K_spatial_th.shape[1]) #channel-preserving

    c = X_th.shape[1]
    ###### PREPROCESS IMAGE ######
    # In order to simulate circular convolutions using linear convolutions, we circularly pad the input tensor in its spatial dimensions.
    padded_X_th = X_th
    if K_spatial_th.shape[2] > 1: padded_X_th = torch.concat([padded_X_th, padded_X_th[:, :, :K_spatial_th.shape[2]-1, :]], axis=2)
    if K_spatial_th.shape[3] > 1: padded_X_th = torch.concat([padded_X_th, padded_X_th[:, :, :, :K_spatial_th.shape[3]-1]], axis=3)            

    if channel_mode == 'lower_block_triangular': 
        padded_X_th = torch.concat([padded_X_th[:, j, np.newaxis, :, :] 
            for i in range(c) for j in range(i+1)], axis=1)

    ###### PERFORM CONVOLUTION ######
    if channel_mode == 'full':
        Y_th = torch.nn.functional.conv2d(padded_X_th, K_spatial_th, stride=(1, 1), padding='valid', dilation=(1, 1))
    else:
        Y_th = torch.nn.functional.conv2d(padded_X_th, K_spatial_th, stride=(1, 1), padding='valid', dilation=(1, 1), groups=padded_X_th.shape[1])

    ###### POST-PROCESS CONVOLUTION ######
    if channel_mode == 'lower_block_triangular': 
        Y_th = torch.concat([Y_th[:, (i*(i+1))//2:((i+2)*(i+1))//2, :, :].sum(1, keepdim=True) for i in range(c)], axis=1)
    return Y_th

def spatial_linear_conv2D_th(X_th, K_spatial_th, channel_mode='full'):
    # assert (len(X_th.shape) == 4)
    # assert (len(K_spatial_th.shape) == 4)
    # assert (channel_mode in ['full', 'lower_block_triangular', 'block_diagonal'])
    # if channel_mode == 'full': assert (K_spatial_th.shape[1] == X_th.shape[1])
    # else: assert (K_spatial_th.shape[1] == 1)
    # assert (K_spatial_th.shape[2] < X_th.shape[2])
    # assert (K_spatial_th.shape[3] < X_th.shape[3])
    # assert (K_spatial_th.shape[1] == X_th.shape[1])
    # assert (K_spatial_th.shape[0] == K_spatial_th.shape[1]) #channel-preserving

    c = X_th.shape[1]
    ###### PREPROCESS IMAGE ######
    # linear convolutions, we pad the input tensor with zeros in its spatial dimensions.
    padded_X_th = X_th
    if K_spatial_th.shape[2] > 1: 
        zero_pad_shape = list(padded_X_th.shape[:2])+[K_spatial_th.shape[2]-1, padded_X_th.shape[-1]]
        padded_X_th = torch.concat([padded_X_th, torch.zeros(zero_pad_shape)], axis=2)
    if K_spatial_th.shape[3] > 1: 
        zero_pad_shape = list(padded_X_th.shape[:2])+[padded_X_th.shape[-2], K_spatial_th.shape[2]-1]
        padded_X_th = torch.concat([padded_X_th, torch.zeros(zero_pad_shape)], axis=3)   

    if channel_mode == 'lower_block_triangular': 
        padded_X_th = torch.concat([padded_X_th[:, j, np.newaxis, :, :] 
            for i in range(c) for j in range(i+1)], axis=1)

    ###### PERFORM CONVOLUTION ######
    if channel_mode == 'full':
        Y_th = torch.nn.functional.conv2d(padded_X_th, K_spatial_th, stride=(1, 1), padding='valid', dilation=(1, 1))
    else:
        Y_th = torch.nn.functional.conv2d(padded_X_th, K_spatial_th, stride=(1, 1), padding='valid', dilation=(1, 1), groups=padded_X_th.shape[1])

    ###### POST-PROCESS CONVOLUTION ######
    if channel_mode == 'lower_block_triangular': 
        Y_th = torch.concat([Y_th[:, (i*(i+1))//2:((i+2)*(i+1))//2, :, :].sum(1, keepdim=True) for i in range(c)], axis=1)
    return Y_th

def spatial_circular_conv2D(X, K_spatial, channel_mode='full', backend='torch'):
    if backend == 'torch':
        return spatial_circular_conv2D_th(X, K_spatial, channel_mode)
    if backend == 'numpy':
        return spatial_circular_conv2D_th(torch.tensor(X, dtype=torch.float32), 
            torch.tensor(K_spatial, dtype=torch.float32), channel_mode).numpy()

def spatial_linear_conv2D(X, K_spatial, channel_mode='full', backend='torch'):
    if backend == 'torch':
        return spatial_linear_conv2D_th(X, K_spatial, channel_mode)
    if backend == 'numpy':
        return spatial_linear_conv2D_th(torch.tensor(X, dtype=torch.float32), 
            torch.tensor(K_spatial, dtype=torch.float32), channel_mode).numpy()







    