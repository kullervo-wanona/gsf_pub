from sys import platform
if 'linux' in platform: 
    from IPython.core.debugger import set_trace
    trace = set_trace
else:
    import ipdb
    trace = ipdb.set_trace

import numpy as np
import torch

import helper
import spectral_schur_det_lib
from multi_channel_invertible_conv_lib import spatial_conv2D_lib
from multi_channel_invertible_conv_lib import frequency_conv2D_lib

########################################################################################################

class CondMultiChannel2DCircularConv(torch.nn.Module):
    def __init__(self, c, n, k, kernel_init='I + net', bias_mode='non-spatial', name=''):
        super().__init__()
        assert (kernel_init in ['I + net', 'net'])
        assert (bias_mode in ['no-bias', 'non-spatial'])

        self.name = 'CondMultiChannel2DCircularConv_' + name
        self.n = n
        self.c = c
        self.k = k
        self.bias_mode = bias_mode
        self.kernel_init = kernel_init

        if self.kernel_init == 'I + net':
            _, iden_kernel_np = spatial_conv2D_lib.generate_identity_kernel(self.c, self.k, 'full', backend='numpy')
            self.iden_kernel = helper.cuda(torch.tensor(iden_kernel_np, dtype=torch.float32))

        self.conv_batch_inverse_func = spectral_schur_det_lib.generate_batch_frequency_inverse_circular_conv2D(self.k, self.n)
        self.conv_batch_kernel_to_logdet = spectral_schur_det_lib.generate_batch_kernel_to_schur_log_determinant(self.k, self.n)

        self.parameter_sizes = {}
        self.parameter_sizes['kernel'] = [-1, self.c, self.c, self.k, self.k]
        if self.bias_mode == 'non-spatial': self.parameter_sizes['bias'] = [-1, self.c, 1, 1]
        else: self.parameter_sizes['bias'] = None

    def transform_with_logdet(self, conv_in, K, bias, check_sizes=False):
        if check_sizes: 
            assert (K.shape[1:] == self.parameter_sizes['kernel'])
            if self.bias_mode  == 'non-spatial': 
                assert (bias.shape[1:] == self.parameter_sizes['bias'])
        
        if self.kernel_init == 'I + net': K = K + self.iden_kernel[np.newaxis]

        conv_out = spatial_conv2D_lib.batch_spatial_circular_conv2D_th(conv_in, K)
        if self.bias_mode  == 'non-spatial': conv_out = conv_out+bias

        logdet = self.conv_batch_kernel_to_logdet(K)
        return conv_out, logdet

    def inverse_transform(self, conv_out, K, bias, check_sizes=False):
        with torch.no_grad():
            if check_sizes: 
                assert (K.shape == self.parameter_sizes['kernel'])
                if self.bias_mode  == 'non-spatial': 
                    assert (bias.shape == self.parameter_sizes['bias'])
            
            if self.bias_mode == 'non-spatial': conv_out = conv_out-bias
    
            if self.kernel_init == 'I + net': K = K + self.iden_kernel[np.newaxis]
            conv_in = self.conv_batch_inverse_func(conv_out, K)
            return conv_in

########################################################################################################

class CondAffine(torch.nn.Module):
    def __init__(self, c, n, bias_mode='spatial', scale_mode='spatial', name=''):
        super().__init__()
        assert (bias_mode in ['no-bias', 'non-spatial', 'spatial'])
        assert (scale_mode in ['no-scale', 'non-spatial', 'spatial'])
        assert (bias_mode != 'no-bias' or scale_mode != 'no-bias')

        self.name = 'CondAffine_' + name
        self.n = n
        self.c = c
        self.bias_mode = bias_mode
        self.scale_mode = scale_mode
        self.parameter_sizes = {}

        self.parameter_sizes['bias'] = None
        if self.bias_mode == 'spatial':
            self.parameter_sizes['bias'] = [-1, self.c, self.n, self.n]
        elif self.bias_mode == 'non-spatial': 
            self.parameter_sizes['bias'] = [-1, self.c, 1, 1]

        self.parameter_sizes['log_scale'] = None
        if self.scale_mode == 'spatial':
            self.parameter_sizes['log_scale'] = [-1, self.c, self.n, self.n]
        elif self.scale_mode == 'non-spatial': 
            self.parameter_sizes['log_scale'] = [-1, self.c, 1, 1]

    def transform_with_logdet(self, affine_in, bias, log_scale, check_sizes=False):
        if check_sizes: 
            if self.bias_mode != 'no-bias': 
                assert (bias.shape == self.parameter_sizes['bias'])
            if self.scale_mode != 'no-scale': 
                assert (log_scale.shape == self.parameter_sizes['log_scale'])
        
        if self.scale_mode != 'no-scale': 
            scale = torch.exp(log_scale)

        affine_out = affine_in
        if self.scale_mode != 'no-scale': 
            affine_out = affine_out*scale
        if self.bias_mode != 'no-bias': 
            affine_out = affine_out+bias

        logdet = 0
        if self.scale_mode == 'spatial': 
            logdet = log_scale.sum(axis=[1, 2, 3])
        elif self.scale_mode == 'non-spatial':
            logdet = (self.n*self.n)*log_scale.sum(axis=[1, 2, 3])
            
        return affine_out, logdet

    def inverse_transform(self, affine_out, bias, log_scale, check_sizes=False):
        with torch.no_grad():
            if check_sizes: 
                if self.bias_mode != 'no-bias': 
                    assert (bias.shape == self.parameter_sizes['bias'])
                if self.scale_mode != 'no-scale': 
                    assert (log_scale.shape == self.parameter_sizes['log_scale'])
            
            if self.scale_mode != 'no-scale': 
                scale = torch.exp(log_scale)

            if self.bias_mode != 'no-bias': 
                affine_out = affine_out-bias
            if self.scale_mode != 'no-scale': 
                affine_out = affine_out/(scale+1e-6)
            affine_in = affine_out 
            
            return affine_in

########################################################################################################

class CondPReLU(torch.nn.Module):
    def __init__(self, c, n, mode='non-spatial', name=''):
        super().__init__()
        assert (mode in ['non-spatial', 'spatial'])
        self.name = 'CondPReLU_' + name
        self.n = n
        self.c = c
        self.mode = mode

        if self.mode == 'spatial': 
            self.pos_log_scale_shape = [1, self.c, self.n, self.n]
            self.neg_log_scale_shape = [1, self.c, self.n, self.n]
        elif self.mode == 'non-spatial': 
            self.pos_log_scale_shape = [1, self.c, 1, 1]
            self.neg_log_scale_shape = [1, self.c, 1, 1]

    def transform_with_logdet(self, nonlin_in, pos_log_scale, neg_log_scale, check_sizes=False):
        if check_sizes: 
            assert (pos_log_scale.shape == self.pos_log_scale_shape)
            assert (neg_log_scale.shape == self.neg_log_scale_shape)
        pos_scale = torch.exp(pos_log_scale)
        neg_scale = torch.exp(neg_log_scale)

        x = nonlin_in
        x_pos = torch.relu(x)
        x_neg = x-x_pos
        x_ge_zero = x_pos/(x+1e-7)

        nonlin_out = pos_scale*x_pos+neg_scale*x_neg
        log_deriv = pos_log_scale*x_ge_zero+neg_log_scale*(1-x_ge_zero)
        logdet = log_deriv.sum(axis=[1, 2, 3])
        return nonlin_out, logdet

    def inverse_transform(self, nonlin_out, pos_log_scale, neg_log_scale, check_sizes=False):
        with torch.no_grad():
            if check_sizes: 
                assert (pos_log_scale.shape == self.pos_log_scale_shape)
                assert (neg_log_scale.shape == self.neg_log_scale_shape)
            pos_scale = torch.exp(pos_log_scale)
            neg_scale = torch.exp(neg_log_scale)

            y = nonlin_out
            y_pos = torch.relu(y)
            y_neg = y-y_pos
            nonlin_in = y_pos/(pos_scale+1e-6)+y_neg/(neg_scale+1e-6)
            return nonlin_in

########################################################################################################

class CondSLogGate(torch.nn.Module):
    def __init__(self, c, n, mode='non-spatial', name=''):
        super().__init__()
        assert (mode in ['non-spatial', 'spatial'])
        self.name = 'CondSLogGate_' + name
        self.n = n
        self.c = c
        self.mode = mode

        if self.mode == 'spatial': 
            self.log_alpha_shape = [1, self.c, self.n, self.n]
        elif self.mode == 'non-spatial': 
            self.log_alpha_shape = [1, self.c, 1, 1]

    def transform_with_logdet(self, nonlin_in, log_alpha, check_sizes=False):
        if check_sizes: assert (log_alpha.shape == self.log_alpha_shape)
        alpha = torch.exp(log_alpha)
        nonlin_out = (torch.sign(nonlin_in)/alpha)*torch.log(1+alpha*torch.abs(nonlin_in))
        logdet = (-alpha*torch.abs(nonlin_out)).sum(axis=[1, 2, 3])
        return nonlin_out, logdet

    def inverse_transform(self, nonlin_out, log_alpha, check_sizes=False):
        with torch.no_grad():
            if check_sizes: assert (log_alpha.shape == self.log_alpha_shape)
            alpha = torch.exp(log_alpha)
            nonlin_in = (torch.sign(nonlin_out)/alpha)*(torch.exp(alpha*torch.abs(nonlin_out))-1)
            return nonlin_in



