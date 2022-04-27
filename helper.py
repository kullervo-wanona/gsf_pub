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

def cuda_maybe_th(x, cuda=True):
    if torch.cuda.is_available() and cuda and not x.is_cuda: return x.to(device='cuda')
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













    