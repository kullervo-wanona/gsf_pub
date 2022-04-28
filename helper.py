from sys import platform
if 'linux' in platform: 
    from IPython.core.debugger import Tracer
    trace = Tracer() #this one triggers the debugger
else:
    import ipdb
    trace = ipdb.set_trace

from PIL import Image
from matplotlib import pyplot as plt

import numpy as np
import torch

# CUDA_FLAG = False
CUDA_FLAG = True

def cuda(x):
    global CUDA_FLAG
    if torch.cuda.is_available() and CUDA_FLAG and not x.is_cuda: return x.to(device='cuda')
    else: return x

def display(mat, rescale=False):
    if rescale: mat = (mat-mat.min())/(mat.max()-mat.min())
    plt.imshow(np.clip(mat, 0, 1), interpolation='nearest')
    plt.draw()
    plt.pause(0.001)

def save_image(mat, path, rescale=False):
    if rescale: mat = (mat-mat.min())/(mat.max()-mat.min())
    im = Image.fromarray((np.clip(mat, 0, 1)*255.).astype(np.uint8))
    im.save(path)

def load_image(path, size=None):
    im_Image_format = Image.open(path)
    if size is not None: im_Image_format = im_Image_format.resize(size, Image.BICUBIC)
    im = np.asarray(im_Image_format, dtype="uint8").astype(float)[:, :, :3]/255.0
    return im 
    
def vectorize(tensor):
    # vectorize last d-1 dims, column indeces together (entire row), then row indices (entire channel), then channel indices (entire image)
    # return tensor.reshape(tensor.shape[0], tensor.shape[1]*tensor.shape[2]*tensor.shape[3])
    return tensor.reshape(tensor.shape[0], np.prod(tensor.shape[1:]))

def unvectorize(tensor, shape_list):
    # vectorize last d-1 dims, column indeces together (entire row), then row indices (entire channel), then channel indices (entire image)
    # return tensor.reshape(tensor.shape[0], n_in_chan, n_rows, n_cols)
    return tensor.reshape(tensor.shape[0], *shape_list)

def get_conv_initial_weight_kernel_np(kernel_shape, n_in_channels, n_out_channels, initializer_mode='he_uniform'):
    # Written with convolution in mind, using terminology related to convolutions.
    fan_in, fan_out = n_in_channels*np.prod(kernel_shape), n_out_channels*np.prod(kernel_shape)
    if 'uniform' in initializer_mode:
        if initializer_mode == 'uniform':
            uniform_scale = 0.05
        elif initializer_mode == 'lecun_uniform': # http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
            uniform_scale = np.sqrt(3./fan_in)
        elif initializer_mode == 'xavier_uniform': # http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
            uniform_scale = np.sqrt(6./(fan_in + fan_out))
        elif initializer_mode == 'he_uniform': # http://arxiv.org/abs/1502.01852
            uniform_scale = np.sqrt(6./fan_in)
        elif initializer_mode == 'he_uniform_2': # http://arxiv.org/abs/1502.01852
            uniform_scale = np.sqrt(1./fan_in)
        initial_weight_kernel = np.random.uniform(low=-uniform_scale, high=uniform_scale, 
            size=[n_out_channels, n_in_channels]+kernel_shape).astype(np.float32)

    elif 'gaussian' in initializer_mode:
        if initializer_mode == 'gaussian' or initializer_mode == 'truncated_gaussian':
            # gaussian_std = 0.02
            gaussian_std = 0.4
        elif initializer_mode == 'xavier_gaussian': # http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
            gaussian_std = np.sqrt(2./(fan_in+fan_out))
        elif initializer_mode == 'he_gaussian': # http://arxiv.org/abs/1502.01852
            gaussian_std = np.sqrt(2./fan_in)
        initial_weight_kernel = np.random.randn(
            *([n_out_channels, n_in_channels]+kernel_shape)).astype(np.float32)*gaussian_std

    return initial_weight_kernel











    