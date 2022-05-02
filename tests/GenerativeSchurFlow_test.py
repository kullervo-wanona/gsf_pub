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
import spectral_schur_det_lib
import GenerativeSchurFlow
from multi_channel_invertible_conv_lib import spatial_conv2D_lib
from multi_channel_invertible_conv_lib import frequency_conv2D_lib

from DataLoaders.CelebA.CelebA32Loader import DataLoader
data_loader = DataLoader(batch_size=100)
data_loader.setup('Training', randomized=True, verbose=True)
_, _, example_batch = next(data_loader) 

c_in = 3
n_in = 7
flow_net = GenerativeSchurFlow.GenerativeSchurFlow(c_in=c_in, n_in=n_in, k_list=[3, 4, 5])
flow_net.set_actnorm_parameters(data_loader, setup_mode='Training', n_batches=500, test_normalization=True, sub_image=[c_in, n_in, n_in])

example_input = helper.cuda(torch.from_numpy(example_batch['Image']))[:, :c_in, :n_in, :n_in]
example_input_np = helper.to_numpy(example_input)

example_out, logdet_computed = flow_net.transform(example_input)
logdet_computed_np = helper.to_numpy(logdet_computed)
J, J_flat = flow_net.jacobian(example_input)
logdet_desired_np = np.log(np.abs(np.linalg.det(J_flat)))

logdet_desired_error = np.abs(logdet_desired_np-logdet_computed_np).max()
print("Desired Logdet: \n", logdet_desired_np)
print("Computed Logdet: \n", logdet_computed_np)
print('Logdet error:' + str(logdet_desired_error))
assert (logdet_desired_error < 1e-4)

example_input_reconst = flow_net.inverse_transform(example_out) 
example_input_reconst_np = helper.to_numpy(example_input_reconst)

inversion_error = np.abs(example_input_reconst_np-example_input_np).max()
print('Inversion error:' + str(inversion_error))
assert (inversion_error < 1e-4)

z, x, log_pdf_z, log_pdf_x = flow_net(example_input)
x_sample = flow_net.sample_x(n_samples=10)

print('All tests passed.')







