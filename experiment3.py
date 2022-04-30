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
# torch.set_flush_denormal(True)

import helper
import spectral_schur_det_lib
from multi_channel_invertible_conv_lib import spatial_conv2D_lib
from multi_channel_invertible_conv_lib import frequency_conv2D_lib

# from DataLoaders.CelebA.CelebA32Loader import DataLoader
# # from DataLoaders.CelebA.CelebA128Loader import DataLoader
# # from DataLoaders.CelebA.CelebA64Loader import DataLoader
# data_loader = DataLoader(batch_size=10)
# data_loader.setup('Training', randomized=True, verbose=False)
# data_loader.setup('Test', randomized=False, verbose=False)
# _, _, batch = next(data_loader)

# from DataLoaders.MNIST.MNISTLoader import DataLoader
from DataLoaders.CelebA.CelebA32Loader import DataLoader
data_loader = DataLoader(batch_size=10)
data_loader.setup('Training', randomized=True, verbose=True)
# data_loader.setup('Test', randomized=True, verbose=True)
# data_loader.setup('Validation', randomized=True, verbose=True)
_, _, example_batch = next(data_loader) 


class Net4(torch.nn.Module):
    def __init__(self, c_in, n_in, k_list, squeeze_list, logit_layer=True, actnorm_layers=True, squeeze_layers=False):
        super().__init__()
        self.n_in = n_in
        self.c_in = c_in
        self.k_list = k_list
        self.squeeze_list = squeeze_list
        assert (len(self.squeeze_list) == len(self.k_list))
        self.n_conv_blocks = len(self.k_list)
        self.logit_layer = logit_layer
        self.actnorm_layers = actnorm_layers
        self.squeeze_layers = squeeze_layers

        self.K_to_schur_log_determinant_funcs = []
        # accum_squeeze = 0
        for layer_id, curr_k in enumerate(self.k_list):
            curr_c = self.c_in
            curr_n = self.n_in
            # if self.squeeze_layers:
            #     accum_squeeze += self.squeeze_list[layer_id]
            #     curr_c = self.c_in*(4**accum_squeeze)
            #     curr_n = self.n_in//(2**accum_squeeze)
            print(curr_c, (curr_n, curr_n), curr_c*curr_n*curr_n)

            _, iden_K = spatial_conv2D_lib.generate_identity_kernel(curr_c, curr_k, 'full', backend='numpy')
            rand_kernel_np = helper.get_conv_initial_weight_kernel_np([curr_k, curr_k], curr_c, curr_c, 'he_uniform')
            curr_kernel_np = iden_K + 0.001*rand_kernel_np 
            curr_conv_kernel_param = torch.nn.parameter.Parameter(data=helper.cuda(torch.tensor(curr_kernel_np, dtype=torch.float32)), requires_grad=True)
            setattr(self, 'conv_kernel_'+str(layer_id+1), curr_conv_kernel_param)
            curr_conv_bias_param = torch.nn.parameter.Parameter(data=helper.cuda(torch.zeros((curr_c), dtype=torch.float32)), requires_grad=True)
            setattr(self, 'conv_bias_'+str(layer_id+1), curr_conv_bias_param)

            if self.actnorm_layers:
                curr_actnorm_bias_np = np.zeros([1, curr_c, 1, 1])
                curr_actnorm_bias = torch.nn.parameter.Parameter(data=helper.cuda(torch.tensor(curr_actnorm_bias_np, dtype=torch.float32)), requires_grad=True)
                setattr(self, 'actnorm_bias_'+str(layer_id+1), curr_actnorm_bias)
                curr_actnorm_log_scale_np = np.zeros([1, curr_c, 1, 1])
                curr_actnorm_log_scale = torch.nn.parameter.Parameter(data=helper.cuda(torch.tensor(curr_actnorm_log_scale_np, dtype=torch.float32)), requires_grad=True)
                setattr(self, 'actnorm_log_scale_'+str(layer_id+1), curr_actnorm_log_scale)

            self.K_to_schur_log_determinant_funcs.append(spectral_schur_det_lib.generate_kernel_to_schur_log_determinant(curr_k, curr_n, backend='torch'))
        
        self.c_out = curr_c
        self.n_out = curr_n
        self.normal_dist = torch.distributions.Normal(helper.cuda(torch.tensor([0.0])), helper.cuda(torch.tensor([1.0])))
        # self.normal_dist_delta = torch.distributions.Normal(helper.cuda(torch.tensor([0.0])), helper.cuda(torch.tensor([0.2])))

    # def squeeze(self, x):
    #     """Squeezes a C x H x W tensor into a 4C x H/2 x W/2 tensor.
    #     (See Fig 3 in the real NVP paper.)
    #     Args:
    #         x: input tensor (B x C x H x W).
    #     Returns:
    #         the squeezed tensor (B x 4C x H/2 x W/2).
    #     """
    #     [B, C, H, W] = list(x.size())
    #     x = x.reshape(B, C, H//2, 2, W//2, 2)
    #     x = x.permute(0, 1, 3, 5, 2, 4)
    #     x = x.reshape(B, C*4, H//2, W//2)
    #     return x

    # def undo_squeeze(self, x):
    #     """unsqueezes a C x H x W tensor into a C/4 x 2H x 2W tensor.
    #     (See Fig 3 in the real NVP paper.)
    #     Args:
    #         x: input tensor (B x C x H x W).
    #     Returns:
    #         the squeezed tensor (B x C/4 x 2H x 2W).
    #     """
    #     [B, C, H, W] = list(x.size())
    #     x = x.reshape(B, C//4, 2, 2, H, W)
    #     x = x.permute(0, 1, 4, 2, 5, 3)
    #     x = x.reshape(B, C//4, H*2, W*2)
    #     return x

    def logit_with_logdet(self, x):
        x_safe = 0.0005+x*0.999
        y = torch.log(x_safe)-torch.log(1-x_safe)
        y_logdet = (-torch.log(x_safe)-torch.log(1-x_safe)).sum(axis=[1, 2, 3])
        return y, y_logdet

    def inverse_logit(self, y):
        x_safe = torch.sigmoid(y)
        x = (x_safe-0.0005)/0.999
        return x


    def leaky_relu_with_logdet(self, x, pos_slope=1.2, neg_slope=0.8):
        x_pos = torch.relu(x)
        x_neg = x-x_pos
        y = pos_slope*x_pos+neg_slope*x_neg
        x_ge_zero = x_pos/(x+0.001)
        y_deriv = pos_slope*x_ge_zero
        y_deriv += neg_slope*(1-y_deriv)
        y_logdet = torch.log(y_deriv).sum(axis=[1, 2, 3])
        return y, y_logdet
    
    def inverse_leaky_relu(self, y, pos_slope=1.2, neg_slope=0.8):
        y_pos = torch.relu(y)
        y_neg = y-y_pos
        x = (1/pos_slope)*y_pos+(1/neg_slope)*y_neg
        return x

    # def tanh_with_logdet(self, x):
    #     y = torch.tanh(x)
    #     y_deriv = 1-y*y
    #     y_logdet = torch.log(y_deriv).sum(axis=[1, 2, 3])
    #     return y, y_logdet

    # def inverse_tanh(self, y):
    #     return 0.5*(torch.log(1+y+1e-4)-torch.log(1-y+1e-4))

    def compute_conv_logdet_from_K(self, layer_id):
        K = getattr(self, 'conv_kernel_'+str(layer_id+1))
        return self.K_to_schur_log_determinant_funcs[layer_id](K)

    def compute_normal_log_pdf(self, y):
        return self.normal_dist.log_prob(y).sum(axis=[1, 2, 3])

    def sample_y(self, n_samples=10):
        return self.normal_dist.sample([n_samples, self.c_out, self.n_out, self.n_out])[..., 0]
        # return self.normal_dist_delta.sample([n_samples, self.c_out, self.n_out, self.n_out])[..., 0]

    def sample_x(self, n_samples=10):
        return self.inverse(self.sample_y(n_samples))

    def actnorm_with_logdet(self, x, layer_id):
        bias = getattr(self, 'actnorm_bias_'+str(layer_id+1))
        log_scale = getattr(self, 'actnorm_log_scale_'+str(layer_id+1))
        scale = torch.exp(log_scale)
        normalized_x = x*scale+bias
        logdet = (x.shape[-1]**2)*log_scale.sum()
        return normalized_x, logdet

    def inverse_actnorm(self, x, layer_id):
        bias = getattr(self, 'actnorm_bias_'+str(layer_id+1))
        log_scale = getattr(self, 'actnorm_log_scale_'+str(layer_id+1))
        scale = torch.exp(log_scale)
        unnormalized_x = (x-bias)/scale
        return unnormalized_x

    def forward(self, x, until_layer_id=None):
        if until_layer_id is not None: assert (until_layer_id <= self.n_conv_blocks)
        conv_log_dets, nonlin_logdets, actnorm_logdets = [], [], []
        
        curr_inp, logit_logdet = x, 0
        if self.logit_layer: curr_inp, logit_logdet = self.logit_with_logdet(curr_inp)

        for layer_id, k in enumerate(self.k_list):
            # for squeeze_i in range(self.squeeze_list[layer_id]):
            #     curr_inp = self.squeeze(curr_inp)

            # if until_layer_id is not None and layer_id >= until_layer_id:
            #     break

            if self.actnorm_layers: 
                curr_inp, actnorm_logdet = self.actnorm_with_logdet(curr_inp, layer_id)
                actnorm_logdets.append(actnorm_logdet)
            
            conv_out = spatial_conv2D_lib.spatial_circular_conv2D_th(
                curr_inp, getattr(self, 'conv_kernel_'+str(layer_id+1)), 
                bias=getattr(self, 'conv_bias_'+str(layer_id+1)))
            # print(conv_out.max(), conv_out.mean(), conv_out.min())

            conv_log_det = self.compute_conv_logdet_from_K(layer_id)
            conv_log_dets.append(conv_log_det)
            if layer_id < len(self.k_list)-1:
                # nonlin_out, nonlin_logdet = self.tanh_with_logdet(conv_out)
                nonlin_out, nonlin_logdet = self.leaky_relu_with_logdet(conv_out)
                nonlin_logdets.append(nonlin_logdet)
                curr_inp = nonlin_out
            else:
                curr_inp = conv_out

        y = curr_inp
        actnorm_logdets_sum = sum(actnorm_logdets)
        nonlin_logdets_sum = sum(nonlin_logdets)
        conv_log_dets_sum = sum(conv_log_dets)

        log_det = logit_logdet+conv_log_dets_sum+nonlin_logdets_sum+actnorm_logdets_sum
        log_pdf_y = self.compute_normal_log_pdf(y)
        log_pdf_x = log_pdf_y + log_det
        # print('conv_log_dets_sum:', conv_log_dets_sum)
        # print('log_pdf_y:', log_pdf_y)
        # print('log_pdf_x:', log_pdf_x)
        # trace()
        return y, log_pdf_x

    def inverse(self, y):
        y = y.detach()
        nonlin_out = y
        for layer_id in list(range(len(self.k_list)))[::-1]:
            if layer_id < len(self.k_list)-1:
                # conv_out = self.inverse_tanh(nonlin_out)
                conv_out = self.inverse_leaky_relu(nonlin_out)
            else: conv_out = nonlin_out 
            # print(conv_out.min(), conv_out.max())

            curr_inp = frequency_conv2D_lib.frequency_inverse_circular_conv2D(conv_out-getattr(self, 'conv_bias_'+str(layer_id+1))[np.newaxis, :, np.newaxis, np.newaxis], getattr(self, 'conv_kernel_'+str(layer_id+1)), 'full', mode='complex', backend='torch')
            # print(curr_inp.min(), curr_inp.max())
            if self.actnorm_layers: 
                curr_inp = self.inverse_actnorm(curr_inp, layer_id)

            # print(curr_inp.shape)
            # for squeeze_i in range(self.squeeze_list[layer_id]):
            #     curr_inp = self.undo_squeeze(curr_inp)
            nonlin_out = curr_inp
        # print(curr_inp.shape)

        x = nonlin_out
        if self.logit_layer: x = self.inverse_logit(x)
        return x

# net = Net4(c_in=data_loader.image_size[1], n_in=data_loader.image_size[3], k_list=[5, 5, 5, 4, 4, 4, 2, 2, 2], squeeze_list=[0, 1, 0, 0, 0, 0, 0, 0, 0])
# net = Net4(c_in=data_loader.image_size[1], n_in=data_loader.image_size[3], k_list=[4, 4, 4], squeeze_list=[0, 0, 0])
net = Net4(c_in=data_loader.image_size[1], n_in=data_loader.image_size[3], k_list=[3, 3, 4, 6, 8], squeeze_list=[0, 0, 0, 0, 0])
criterion = torch.nn.CrossEntropyLoss()

n_param = 0 
for e in net.parameters():
    print(e.shape)
    n_param += np.prod(e.shape)
print('Total number of parameters: ' + str(n_param))
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.5, 0.9), eps=1e-08)
# optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.95, 0.999), eps=1e-08)

helper.vis_samples_np(example_batch['Image'], sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/real/', prefix='real')

for layer_id in range(net.n_conv_blocks):
    data_loader.setup('Training', randomized=False, verbose=True)
    print('Layer: ' + str(layer_id) + ', mean computation.' )

    n_examples = 0
    accum_mean = None
    for i, curr_batch_size, batch_np in data_loader:     
        if i > 400: break

        image = helper.cuda(torch.from_numpy(batch_np['Image']))

        input_to_layer_id, _ = net.forward(image, until_layer_id=layer_id)
        input_to_layer_id = helper.cpu(input_to_layer_id).detach().numpy()
        curr_mean = input_to_layer_id.mean(axis=(2, 3)).sum(0)
        if accum_mean is None: accum_mean = curr_mean
        else: accum_mean += curr_mean
        n_examples += input_to_layer_id.shape[0]
    mean = accum_mean/n_examples

    data_loader.setup('Training', randomized=False, verbose=True)
    print('Layer: ' + str(layer_id) + ', std computation.' )
    
    n_examples = 0
    accum_var = None
    for i, curr_batch_size, batch_np in data_loader:  
        if i > 400: break

        image = helper.cuda(torch.from_numpy(batch_np['Image']))

        input_to_layer_id, _ = net.forward(image, until_layer_id=layer_id)
        input_to_layer_id = helper.cpu(input_to_layer_id).detach().numpy()
        curr_var = ((input_to_layer_id-mean[np.newaxis, :, np.newaxis, np.newaxis])**2).mean(axis=(2, 3)).sum(0)
        if accum_var is None: accum_var = curr_var
        else: accum_var += curr_var
        n_examples += input_to_layer_id.shape[0]
    var = accum_var/n_examples
    log_std = np.log(np.sqrt(var))
    scale = 1/(np.exp(log_std)+0.001)
    log_scale = np.log(scale)
    bias = -mean*scale

    curr_actnorm_bias_var = getattr(net, 'actnorm_bias_'+str(layer_id+1))
    curr_actnorm_log_scale_var = getattr(net, 'actnorm_log_scale_'+str(layer_id+1))
    curr_actnorm_bias_var.data = torch.from_numpy(bias[np.newaxis, :, np.newaxis, np.newaxis].astype(np.float32))
    curr_actnorm_log_scale_var.data = torch.from_numpy(log_scale[np.newaxis, :, np.newaxis, np.newaxis].astype(np.float32))

    # normalized = input_to_layer_id*np.exp(log_scale[np.newaxis, :, np.newaxis, np.newaxis])+bias[np.newaxis, :, np.newaxis, np.newaxis]
    # print('Mean after normalization: ', normalized.mean(axis=(0, 2, 3)))
    # print('Std after normalization: ', normalized.std(axis=(0, 2, 3)))


exp_t_start = time.time()
running_loss = 0.0
for epoch in range(10):

    data_loader.setup('Training', randomized=True, verbose=True)
    for i, curr_batch_size, batch_np in data_loader:     
        image = helper.cuda(torch.from_numpy(batch_np['Image']))

        optimizer.zero_grad() # zero the parameter gradients

        latent, log_pdf_image = net(image)
        # assert (torch.abs(latent-image).max() > 0.1)
        # print(torch.abs(image_reconst-image).max())
        # assert (torch.abs(image_reconst-image).max() < 1e-3)
        loss = -torch.mean(log_pdf_image)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 0:
            image_reconst = net.inverse(latent)
            image_sample = net.sample_x(n_samples=10)            
            helper.vis_samples_np(helper.cpu(image).detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/real/', prefix='real')
            helper.vis_samples_np(helper.cpu(image_reconst).detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/reconst/', prefix='reconst')
            helper.vis_samples_np(helper.cpu(image_sample).detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/network/', prefix='network')

            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item()}')
            running_loss = 0.0

print('Experiment took '+str(time.time()-exp_t_start)+' seconds.')
print('Finished Training')

























# def leaky_relu_with_logdet(x, neg_slope=0.2):
#     x_pos = torch.nn.functional.relu(x)
#     x_neg = x-x_pos
#     y = x_pos+neg_slope*x_neg
#     y_deriv = helper.cuda(torch.ge(x, 0).type(torch.float32))
#     y_deriv += neg_slope*(1-y_deriv)
#     y_logdet = torch.log(y_deriv).sum(axis=[1, 2, 3])
#     return y, y_logdet
    
# def sigmoid_with_logdet(x):
#     y = torch.sigmoid(x)
#     y_deriv = (1-y)*y
#     y_logdet = torch.log(y_deriv).sum(axis=[1, 2, 3])
#     return y, y_logdet

    