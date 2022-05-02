import os, sys, inspect

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
from multi_channel_invertible_conv_lib import spatial_conv2D_lib
from multi_channel_invertible_conv_lib import frequency_conv2D_lib

from DataLoaders.CelebA.CelebA32Loader import DataLoader
data_loader = DataLoader(batch_size=10)
data_loader.setup('Training', randomized=True, verbose=True)
_, _, example_batch = next(data_loader) 

def jacobian(func_to_J, point):
    optimizer = torch.optim.Adam(func_to_J.parameters(), lr=0.0001, betas=(0.9, 0.95), eps=1e-08)
    point.requires_grad = True

    out, _ = func_to_J(point)
    assert (out.shape == point.shape)

    J = np.zeros(out.shape+point.shape[1:])
    for i in range(out.shape[1]):
        for a in range(out.shape[2]):
            for b in range(out.shape[3]):
                print(i, a, b)
                optimizer.zero_grad() # zero the parameter gradients
                if point.grad is not None: point.grad.zero_()

                out, _ = func_to_J(point)
                loss = torch.sum(out[:, i, a, b])
                loss.backward()
                J[:, i, a, b, ...] = point.grad.numpy()

    J_flat = J.reshape(out.shape[0], np.prod(out.shape[1:]), np.prod(point.shape[1:]))
    return J, J_flat

def compute_actnorm_stats(data_loader, net, layer_id, n_batches=500, sub_image=None):
    data_loader.setup('Training', randomized=False, verbose=False)
    print('Layer: ' + str(layer_id) + ', mean computation.' )

    n_examples = 0
    accum_mean = None
    for i, curr_batch_size, batch_np in data_loader:     
        if n_batches is not None and i > n_batches: break
        image_np = batch_np['Image']
        if sub_image is not None: image_np = image_np[:, :sub_image[0], :sub_image[1], :sub_image[2]]
        image = helper.cuda(torch.from_numpy(image_np))

        input_to_layer, _ = net.forward(image, until_layer=layer_id)
        input_to_layer = helper.cpu(input_to_layer).detach().numpy()
        curr_mean = input_to_layer.mean(axis=(2, 3)).sum(0)
        if accum_mean is None: accum_mean = curr_mean
        else: accum_mean += curr_mean
        n_examples += input_to_layer.shape[0]

    mean = accum_mean/n_examples

    data_loader.setup('Training', randomized=False, verbose=False)
    print('Layer: ' + str(layer_id) + ', std computation.' )
    
    n_examples = 0
    accum_var = None
    for i, curr_batch_size, batch_np in data_loader:  
        if n_batches is not None and i > n_batches: break
        image_np = batch_np['Image']
        if sub_image is not None: image_np = image_np[:, :sub_image[0], :sub_image[1], :sub_image[2]]
        image = helper.cuda(torch.from_numpy(image_np))

        input_to_layer, _ = net.forward(image, until_layer=layer_id)
        input_to_layer = helper.cpu(input_to_layer).detach().numpy()
        curr_var = ((input_to_layer-mean[np.newaxis, :, np.newaxis, np.newaxis])**2).mean(axis=(2, 3)).sum(0)
        if accum_var is None: accum_var = curr_var
        else: accum_var += curr_var
        n_examples += input_to_layer.shape[0]

    var = accum_var/n_examples
    log_std = 0.5*np.log(var)
    bias = -mean/(np.exp(log_std)+1e-5)
    log_scale = -log_std

    bias = bias[np.newaxis, :, np.newaxis, np.newaxis].astype(np.float32)
    log_scale = log_scale[np.newaxis, :, np.newaxis, np.newaxis].astype(np.float32)

    return bias, log_scale

def set_actnorm_parameters_for_net(data_loader, net, n_batches=500, test_normalization=True, sub_image=None):
    for layer_id in range(net.n_layers):
        actnorm_bias_np, actnorm_log_scale_np = compute_actnorm_stats(data_loader, net, layer_id, n_batches=n_batches, sub_image=sub_image)
        net.set_actnorm_parameters(layer_id, actnorm_bias_np, actnorm_log_scale_np)
        if test_normalization:
            actnorm_bias_np, actnorm_log_scale_np = compute_actnorm_stats(data_loader, net, layer_id, n_batches=n_batches, sub_image=sub_image)
            assert (np.abs(actnorm_bias_np).max() < 1e-4)
            assert (np.abs(actnorm_log_scale_np).max() < 1e-4)

class Net(torch.nn.Module):
    def __init__(self, c_in, n_in, k_list):
        super().__init__()
        self.n_in = n_in
        self.c_in = c_in
        self.k_list = k_list
        self.K_to_log_determinants = []
        self.n_layers = len(self.k_list)

        for layer_id, curr_k in enumerate(self.k_list):
            curr_c = self.c_in
            curr_n = self.n_in

            curr_temp_actnorm_bias = helper.cuda(torch.tensor(np.zeros([1, curr_c, 1, 1]), dtype=torch.float32))
            curr_temp_actnorm_log_scale = helper.cuda(torch.tensor(np.zeros([1, curr_c, 1, 1]), dtype=torch.float32))
            setattr(self, 'actnorm_bias_'+str(layer_id+1), curr_temp_actnorm_bias)
            setattr(self, 'actnorm_log_scale_'+str(layer_id+1), curr_temp_actnorm_log_scale)

            _, iden_K = spatial_conv2D_lib.generate_identity_kernel(curr_c, curr_k, 'full', backend='numpy')
            rand_kernel_np = helper.get_conv_initial_weight_kernel_np([curr_k, curr_k], curr_c, curr_c, 'he_uniform')
            curr_kernel_np = iden_K + 0.1*rand_kernel_np 
            curr_conv_kernel_param = torch.nn.parameter.Parameter(data=helper.cuda(torch.tensor(curr_kernel_np, dtype=torch.float32)), requires_grad=True)
            setattr(self, 'conv_kernel_'+str(layer_id+1), curr_conv_kernel_param)
            curr_conv_bias_param = torch.nn.parameter.Parameter(data=helper.cuda(torch.zeros((1, curr_c, curr_n, curr_n), dtype=torch.float32)), requires_grad=True)
            setattr(self, 'conv_bias_'+str(layer_id+1), curr_conv_bias_param)
            curr_conv_log_scale_param = torch.nn.parameter.Parameter(data=helper.cuda(torch.zeros((1, curr_c, curr_n, curr_n), dtype=torch.float32)), requires_grad=True)
            setattr(self, 'conv_log_scale_'+str(layer_id+1), curr_conv_log_scale_param)

            curr_slog_log_alpha_param = torch.nn.parameter.Parameter(data=helper.cuda(torch.zeros((1, curr_c, 1, 1), dtype=torch.float32)), requires_grad=True)
            setattr(self, 'slog_log_alpha_'+str(layer_id+1), curr_slog_log_alpha_param)

            self.K_to_log_determinants.append(spectral_schur_det_lib.generate_kernel_to_schur_log_determinant(curr_k, curr_n, backend='torch'))

    def set_actnorm_parameters(self, layer_id, actnorm_bias_np, actnorm_log_scale_np):
        actnorm_bias = torch.nn.parameter.Parameter(data=helper.cuda(torch.tensor(actnorm_bias_np, dtype=torch.float32)), requires_grad=True)
        setattr(self, 'actnorm_bias_'+str(layer_id+1), actnorm_bias)

        actnorm_log_scale = torch.nn.parameter.Parameter(data=helper.cuda(torch.tensor(actnorm_log_scale_np, dtype=torch.float32)), requires_grad=True)
        setattr(self, 'actnorm_log_scale_'+str(layer_id+1), actnorm_log_scale)

    ################################################################################################

    def actnorm_with_logdet(self, actnorm_in, layer_id):
        bias = getattr(self, 'actnorm_bias_'+str(layer_id+1))
        log_scale = getattr(self, 'actnorm_log_scale_'+str(layer_id+1))
        scale = torch.exp(log_scale)
        actnorm_out = actnorm_in*scale+bias

        actnorm_logdet = (actnorm_in.shape[-1]**2)*log_scale.sum()
        return actnorm_out, actnorm_logdet

    def actnorm_inverse(self, actnorm_out, layer_id):
        bias = getattr(self, 'actnorm_bias_'+str(layer_id+1)).detach()
        log_scale = getattr(self, 'actnorm_log_scale_'+str(layer_id+1)).detach()
        scale = torch.exp(log_scale)
        actnorm_in = (actnorm_out-bias)/(scale+1e-5)
        return actnorm_in

    ################################################################################################

    def conv_with_logdet(self, conv_in, layer_id):
        K = getattr(self, 'conv_kernel_'+str(layer_id+1))
        bias = getattr(self, 'conv_bias_'+str(layer_id+1))
        log_scale = getattr(self, 'conv_log_scale_'+str(layer_id+1))
        scale = torch.exp(log_scale)
        conv_out = scale*spatial_conv2D_lib.spatial_circular_conv2D_th(conv_in, K)+bias
        conv_logdet = log_scale.sum()+self.K_to_log_determinants[layer_id](K)
        return conv_out, conv_logdet

    def conv_inverse(self, conv_out, layer_id):
        K = getattr(self, 'conv_kernel_'+str(layer_id+1)).detach()
        bias = getattr(self, 'conv_bias_'+str(layer_id+1)).detach()
        log_scale = getattr(self, 'conv_log_scale_'+str(layer_id+1)).detach()
        scale = torch.exp(log_scale)
        conv_out = (conv_out-bias)/(scale+1e-5)
        conv_in = frequency_conv2D_lib.frequency_inverse_circular_conv2D(conv_out, K, 'full', mode='complex', backend='torch')
        return conv_in

    ################################################################################################

    def slog_gate_with_logit(self, nonlin_in, layer_id):
        log_alpha = getattr(self, 'slog_log_alpha_'+str(layer_id+1))
        alpha = torch.exp(log_alpha)
        nonlin_out = (torch.sign(nonlin_in)/alpha)*torch.log(1+alpha*torch.abs(nonlin_in))
        slog_gate_logdet = (-alpha*torch.abs(nonlin_out)).sum(axis=[1, 2, 3])
        return nonlin_out, slog_gate_logdet

    def slog_gate_inverse(self, nonlin_out, layer_id):
        log_alpha = getattr(self, 'slog_log_alpha_'+str(layer_id+1)).detach()
        alpha = torch.exp(log_alpha)
        nonlin_in = (torch.sign(nonlin_out)/alpha)*(torch.exp(alpha*torch.abs(nonlin_out))-1)
        return nonlin_in

    ################################################################################################

    def forward(self, x, until_layer=None):
        if until_layer is not None: assert (until_layer <= self.n_layers)
        actnorm_logdets, conv_logdets, nonlin_logdets = [], [], []

        layer_in = x
        for layer_id, k in enumerate(self.k_list): 

            actnorm_out, actnorm_logdet = self.actnorm_with_logdet(layer_in, layer_id)
            actnorm_logdets.append(actnorm_logdet)

            if until_layer is not None and layer_id >= until_layer:
                layer_out = actnorm_out
                break

            conv_out, conv_logdet = self.conv_with_logdet(actnorm_out, layer_id)
            conv_logdets.append(conv_logdet)

            nonlin_out, nonlin_logdet = self.slog_gate_with_logit(conv_out, layer_id)
            nonlin_logdets.append(nonlin_logdet)

            layer_out = nonlin_out
            layer_in = layer_out

        y = layer_out
        total_log_det = sum(actnorm_logdets)+sum(conv_logdets)+sum(nonlin_logdets) 
        return y, total_log_det

    def inverse(self, y):
        y = y.detach()

        layer_out = y
        for layer_id in list(range(len(self.k_list)))[::-1]:
            conv_out = self.slog_gate_inverse(layer_out, layer_id)
            actnorm_out = self.conv_inverse(conv_out, layer_id)
            layer_in = self.actnorm_inverse(actnorm_out, layer_id)
            layer_out = layer_in

        x = layer_in
        return x

c_in = 3
n_in = 7
net = Net(c_in=c_in, n_in=n_in, k_list=[3, 4, 5])
set_actnorm_parameters_for_net(data_loader, net, n_batches=500, test_normalization=True, sub_image=[c_in, n_in, n_in])

example_input = helper.cuda(torch.from_numpy(example_batch['Image']))[:, :c_in, :n_in, :n_in]

example_out, log_abs_det_computed = net(example_input)
log_abs_det_computed_np = log_abs_det_computed.detach().numpy()

J, J_flat = jacobian(net, example_input)
log_abs_det_desired_np = np.log(np.abs(np.linalg.det(J_flat)))

print("Desired: ", log_abs_det_desired_np)
print("Computed: ", log_abs_det_computed_np)
assert(np.abs(log_abs_det_desired_np-log_abs_det_computed_np).max() < 1e-4)

example_input_reconst = net.inverse(example_out) 
try: assert (np.abs(example_input_reconst.numpy()-example_input.detach().numpy()).max() < 1e-4)
except: print('Inversion error high:' + str(np.abs(example_input_reconst.numpy()-example_input.detach().numpy()).max()))










