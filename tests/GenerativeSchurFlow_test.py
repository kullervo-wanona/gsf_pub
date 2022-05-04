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
from GenerativeSchurFlow import GenerativeSchurFlow

from DataLoaders.CelebA.CelebA32Loader import DataLoader
data_loader = DataLoader(batch_size=10)
data_loader.setup('Training', randomized=True, verbose=True)
_, _, example_batch = next(data_loader) 

c_in = 3
n_in = 16
flow_net = GenerativeSchurFlow(c_in=c_in, n_in=n_in, k_list=[3, 4, 4], squeeze_list=[0, 1, 0])
flow_net.set_actnorm_parameters(data_loader, setup_mode='Training', n_batches=500, 
    test_normalization=True, sub_image=[c_in, n_in, n_in])

n_param = 0
for name, e in flow_net.named_parameters():
    print(name, e.requires_grad, e.shape)
    n_param += np.prod(e.shape)
print('Total number of parameters: ' + str(n_param))

example_input = helper.cuda(torch.from_numpy(example_batch['Image']))[:, :c_in, :n_in, :n_in]
example_input_np = helper.to_numpy(example_input)

example_out, logdet_computed = flow_net.transform(example_input)
logdet_computed_np = helper.to_numpy(logdet_computed)
J, J_flat = flow_net.jacobian(example_input)
det_sign, logdet_desired_np = np.linalg.slogdet(J_flat)

logdet_desired_error = np.abs(logdet_desired_np-logdet_computed_np).max()
print("Desired Logdet: \n", logdet_desired_np)
print("Computed Logdet: \n", logdet_computed_np)
print('Logdet error:' + str(logdet_desired_error))

example_input_reconst = flow_net.inverse_transform(example_out) 
example_input_reconst_np = helper.to_numpy(example_input_reconst)

inversion_error = np.abs(example_input_reconst_np-example_input_np).max()
print('Inversion error:' + str(inversion_error))

z, x, log_pdf_z, log_pdf_x = flow_net(example_input)
x_sample = flow_net.sample_x(n_samples=10)

assert (logdet_desired_error < 1e-3)
assert (inversion_error < 1e-3)
print('All tests passed.')

































# squeezed = squeeze(test_image, init_chan_together=False)
# test_image_rec = undo_squeeze(squeezed, init_chan_together=False)

# helper.vis_samples_np(helper.cpu(squeezed[:, 0:12:4, :, :]).detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/train_real/', prefix='real1', resize=[256, 256])
# helper.vis_samples_np(helper.cpu(squeezed[:, 1:12:4, :, :]).detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/train_real/', prefix='real2', resize=[256, 256])
# helper.vis_samples_np(helper.cpu(squeezed[:, 2:12:4, :, :]).detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/train_real/', prefix='real3', resize=[256, 256])
# helper.vis_samples_np(helper.cpu(squeezed[:, 3:12:4, :, :]).detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/train_real/', prefix='real4', resize=[256, 256])

# squeezed = squeeze(test_image, init_chan_together=True)

# helper.vis_samples_np(helper.cpu(squeezed[:, 0:3, :, :]).detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/train_real/', prefix='real', resize=[256, 256])
# helper.vis_samples_np(helper.cpu(squeezed[:, 3:6, :, :]).detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/train_real/', prefix='real2', resize=[256, 256])
# helper.vis_samples_np(helper.cpu(squeezed[:, 6:9, :, :]).detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/train_real/', prefix='real3', resize=[256, 256])
# helper.vis_samples_np(helper.cpu(squeezed[:, 9:12, :, :]).detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/train_real/', prefix='real4', resize=[256, 256])
