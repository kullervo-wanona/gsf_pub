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
from GenerativeConditionalSchurFlow import GenerativeConditionalSchurFlow

from DataLoaders.CelebA.CelebA32Loader import DataLoader
data_loader = DataLoader(batch_size=10)
data_loader.setup('Training', randomized=True, verbose=True)
_, _, example_batch = next(data_loader) 

c_in = 2
n_in = 10
flow_net = GenerativeConditionalSchurFlow(c_in=c_in, n_in=n_in, n_blocks=3)
# flow_net.set_actnorm_parameters(data_loader, setup_mode='Training', n_batches=50, 
#     test_normalization=True, sub_image=[c_in, n_in, n_in])

n_param = 0
for name, e in flow_net.named_parameters():
    print(name, e.requires_grad, e.shape)
    n_param += np.prod(e.shape)
print('Total number of parameters: ' + str(n_param))

example_input = helper.cuda(torch.from_numpy(example_batch['Image']))[:, :c_in, :n_in, :n_in]
example_input_np = helper.to_numpy(example_input)

example_out, logdet_computed = flow_net.transform_with_logdet(example_input)
example_input_reconst = flow_net.inverse_transform(example_out) 
example_input_reconst_np = helper.to_numpy(example_input_reconst)

inversion_error_max = np.abs(example_input_reconst_np-example_input_np).max()
inversion_error_mean = np.abs(example_input_reconst_np-example_input_np).mean()
inversion_error_median = np.median(np.abs(example_input_reconst_np-example_input_np))
print('Inversion error max:' + str(inversion_error_max))
print('Inversion error mean:' + str(inversion_error_mean))
print('Inversion error median:' + str(inversion_error_median))
trace()

z, x, logdet, log_pdf_z, log_pdf_x = flow_net(example_input)
x_sample = flow_net.sample_x(n_samples=10)

J, J_flat = flow_net.jacobian(example_input)
det_sign, logdet_desired_np = np.linalg.slogdet(J_flat)

logdet_computed_np = helper.to_numpy(logdet_computed)
logdet_desired_error = np.abs(logdet_desired_np-logdet_computed_np).max()
print("Desired Logdet: \n", logdet_desired_np)
print("Computed Logdet: \n", logdet_computed_np)
print('Logdet error:' + str(logdet_desired_error))

print('Inversion error max:' + str(inversion_error_max))
print('Inversion error mean:' + str(inversion_error_mean))
print('Inversion error median:' + str(inversion_error_median))

assert (logdet_desired_error < 1e-3)
assert (inversion_error_max < 1e-3)
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
