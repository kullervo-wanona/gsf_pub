import os, sys, inspect
sys.path.insert(1, os.path.realpath(os.path.pardir))

from sys import platform
if 'linux' in platform: 
    from IPython.core.debugger import set_trace
    trace = set_trace
else:
    import ipdb
    trace = ipdb.set_trace

import time
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt

import numpy as np
import torch

import helper
import Transforms

#######################  TEST SQUEEZE  #######################

b, c, n = 10, 3, 4
tensor = torch.from_numpy(np.arange(b*c*n*n).reshape(b, c, n, n))
for chan_mode in ['input_channels_adjacent', 'input_channels_apart']:
    for spatial_mode in ['tl-tr-bl-br', 'tl-br-tr-bl']:
        squeezer = Transforms.Squeeze(chan_mode, spatial_mode)
        tensor_squeezed = squeezer.forward(tensor)
        tensor_rec = squeezer.inverse(tensor_squeezed)
        assert (torch.abs(tensor_rec-tensor).max() < 1e-6)

#######################  TEST LOGIT  #######################

b, c, n = 10, 3, 4
tensor = torch.from_numpy(np.arange(b*c*n*n).reshape(b, c, n, n))
tensor = tensor/(b*c*n*n)
logit = Transforms.Logit(c, n)
logit_out, logdet = logit.forward_with_logdet(tensor)
tensor_rec = logit.inverse(logit_out)
assert (torch.abs(tensor_rec-tensor).max() < 1e-6)


print('All tests passed.')






















