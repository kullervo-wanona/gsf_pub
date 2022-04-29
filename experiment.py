import os, sys, inspect

from sys import platform
if 'linux' in platform: 
    from IPython.core.debugger import set_trace
    trace = set_trace
else:
    import ipdb
    trace = ipdb.set_trace

import time
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
    def __init__(self, c, n, k_list):
        super().__init__()
        self.n = n
        self.c = c
        self.k_list = k_list

        for layer_id, k in enumerate(self.k_list):
            _, iden_K = spatial_conv2D_lib.generate_identity_kernel(self.c, k, 'full', backend='numpy')
            rand_kernel_np = helper.get_conv_initial_weight_kernel_np([k, k], self.c, self.c, 'he_uniform')
            curr_kernel_np = iden_K + 0.001*rand_kernel_np 
            curr_conv_kernel = helper.cuda(torch.nn.parameter.Parameter(data=torch.tensor(curr_kernel_np, dtype=torch.float32), requires_grad=True))
            trace()
            setattr(self, 'conv_kernel_'+str(layer_id+1), curr_conv_kernel)
            curr_conv_bias = helper.cuda(torch.nn.parameter.Parameter(data=torch.zeros((self.c), dtype=torch.float32), requires_grad=True))
            setattr(self, 'conv_bias_'+str(layer_id+1), curr_conv_bias)

        self.K_to_schur_log_determinant_funcs = {(k, self.n): 
            spectral_schur_det_lib.generate_kernel_to_schur_log_determinant(k, self.n, backend='torch') for k in self.k_list}
        self.normal_dist = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.normal_dist_delta = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.2]))

    def leaky_relu_with_logdet(self, x, pos_slope=1.1, neg_slope=0.9):
        x_pos = torch.relu(x)
        x_neg = x-x_pos
        y = pos_slope*x_pos+neg_slope*x_neg
        x_ge_zero = x_pos/(x+0.001)
        y_deriv = pos_slope*x_ge_zero
        y_deriv += neg_slope*(1-y_deriv)
        y_logdet = torch.log(y_deriv).sum(axis=[1, 2, 3])
        return y, y_logdet
    
    def inverse_leaky_relu(self, y, pos_slope=1., neg_slope=0.7):
        y_pos = torch.relu(y)
        y_neg = y-y_pos
        x = (1/pos_slope)*y_pos+(1/neg_slope)*y_neg
        return x

    def tanh_with_logdet(self, x):
        y = torch.tanh(x)
        y_deriv = 1-y*y
        y_logdet = torch.log(y_deriv).sum(axis=[1, 2, 3])
        return y, y_logdet

    def inverse_tanh(self, y):
        return 0.5*(torch.log(1+y+1e-4)-torch.log(1-y+1e-4))

    def compute_conv_logdet_from_K(self, K):
        return self.K_to_schur_log_determinant_funcs[K.shape[-1], self.n](K)

    def compute_normal_log_pdf(self, y):
        return self.normal_dist.log_prob(y).sum(axis=[1, 2, 3])

    def sample_y(self, n_samples=10):
        return self.normal_dist.sample([n_samples, self.c, self.n, self.n])[..., 0]
        # return self.normal_dist_delta.sample([n_samples, self.c, self.n, self.n])[..., 0]

    def sample_x(self, n_samples=10):
        return self.inverse(self.sample_y(n_samples))

    def forward(self, x):
        conv_log_dets, nonlin_logdets = [], []
        
        curr_inp = x
        for layer_id, k in enumerate(self.k_list):
            conv_out = spatial_conv2D_lib.spatial_circular_conv2D_th(
                curr_inp, getattr(self, 'conv_kernel_'+str(layer_id+1)), 
                bias=getattr(self, 'conv_bias_'+str(layer_id+1)))
            # print(conv_out.max(), conv_out.mean(), conv_out.min())
            conv_log_det = self.compute_conv_logdet_from_K(getattr(self, 'conv_kernel_'+str(layer_id+1)))
            conv_log_dets.append(conv_log_det)
            if layer_id < len(self.k_list)-1:
                # nonlin_out, nonlin_logdet = self.tanh_with_logdet(conv_out)
                nonlin_out, nonlin_logdet = self.leaky_relu_with_logdet(conv_out)
                nonlin_logdets.append(nonlin_logdet)
                curr_inp = nonlin_out
            else:
                curr_inp = conv_out

        y = curr_inp
        nonlin_logdets_sum = sum(nonlin_logdets)
        conv_log_dets_sum = sum(conv_log_dets)

        log_det = conv_log_dets_sum + nonlin_logdets_sum
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
            nonlin_out = curr_inp

        x = nonlin_out
        return x

net = Net4(c=data_loader.image_size[1], n=data_loader.image_size[3], k_list=[3, 3, 4, 6, 8])
criterion = torch.nn.CrossEntropyLoss()

n_param = 0 
for e in net.parameters(): 
    print(e.shape)
    n_param += np.prod(e.shape)
print('Total number of parameters: ' + str(n_param))
trace()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.5, 0.9), eps=1e-08)


helper.vis_samples_np(example_batch['Image'], sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/real/', prefix='real')

exp_t_start = time.time()
running_loss = 0.0
for epoch in range(10):

    data_loader.setup('Training', randomized=True, verbose=True)
    for i, curr_batch_size, batch_np in data_loader:     
        image = helper.cuda(torch.from_numpy(batch_np['Image'])) #-0.5

        optimizer.zero_grad() # zero the parameter gradients

        latent, log_pdf_image = net(image)

        # assert (torch.abs(latent-image).max() > 0.1)
        # print(torch.abs(image_reconst-image).max())
        # assert (torch.abs(image_reconst-image).max() < 1e-3)
        loss = -torch.mean(log_pdf_image)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 300 == 0:
            image_reconst = net.inverse(latent)
            image_sample = net.sample_x(n_samples=10)            
            helper.vis_samples_np(image_reconst.detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/reconst/', prefix='reconst')
            helper.vis_samples_np(image_sample.detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/network/', prefix='network')

            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
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

    