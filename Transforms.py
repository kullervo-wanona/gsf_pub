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

########################################################################################################

class MultiChannel2DCircularConv(torch.nn.Module):
    def __init__(self, c, n, k, kernel_init='I + he_uniform', bias_mode='spatial', scale_mode='no-scale', name=''):
        super().__init__()
        assert (kernel_init in ['I + he_uniform', 'he_uniform'])
        assert (bias_mode in ['no-bias', 'non-spatial', 'spatial'])
        assert (scale_mode in ['no-scale', 'non-spatial', 'spatial'])

        self.name = 'MultiChannel2DCircularConv_' + name
        self.n = n
        self.c = c
        self.k = k
        self.kernel_init = kernel_init
        self.bias_mode = bias_mode
        self.scale_mode = scale_mode

        rand_kernel_np = helper.get_conv_initial_weight_kernel_np([self.k, self.k], self.c, self.c, 'he_uniform')
        if self.kernel_init == 'I + he_uniform': 
            _, iden_kernel_np = spatial_conv2D_lib.generate_identity_kernel(self.c, self.k, 'full', backend='numpy')
            kernel_np = iden_kernel_np + 0.1*rand_kernel_np 
        elif self.kernel_init == 'he_uniform': 
            kernel_np = rand_kernel_np

        kernel_th = helper.cuda(torch.tensor(kernel_np, dtype=torch.float32))
        kernel_param = torch.nn.parameter.Parameter(data=kernel_th, requires_grad=True)
        setattr(self, 'kernel', kernel_param)
        self.conv_inverse_func = spectral_schur_det_lib.generate_frequency_inverse_circular_conv2D(self.k, self.n)
        self.conv_kernel_to_logdet = spectral_schur_det_lib.generate_kernel_to_schur_log_determinant(self.k, self.n)

        if self.bias_mode == 'spatial': 
            bias_th = helper.cuda(torch.zeros((1, self.c, self.n, self.n), dtype=torch.float32))
        elif self.bias_mode == 'non-spatial': 
            bias_th = helper.cuda(torch.zeros((1, self.c, 1, 1), dtype=torch.float32))
        if self.bias_mode in ['non-spatial', 'spatial']: 
            bias_param = torch.nn.parameter.Parameter(data=bias_th, requires_grad=True)
            setattr(self, 'bias', bias_param)
        
        if self.scale_mode == 'spatial': 
            log_scale_th = helper.cuda(torch.zeros((1, self.c, self.n, self.n), dtype=torch.float32))
        elif self.scale_mode == 'non-spatial': 
            log_scale_th = helper.cuda(torch.zeros((1, self.c, 1, 1), dtype=torch.float32))
        if self.scale_mode in ['non-spatial', 'spatial']: 
            log_scale_param = torch.nn.parameter.Parameter(data=log_scale_th, requires_grad=True)
            setattr(self, 'log_scale', log_scale_param)

    def forward_with_logdet(self, conv_in):
        if self.bias_mode in ['non-spatial', 'spatial']: 
            bias = getattr(self, 'bias')
            conv_in = conv_in+bias

        K = getattr(self, 'kernel')
        conv_out = spatial_conv2D_lib.spatial_circular_conv2D_th(conv_in, K)
        logdet = self.conv_kernel_to_logdet(K)

        if self.scale_mode in ['non-spatial', 'spatial']: 
            log_scale = getattr(self, 'log_scale')
            scale = torch.exp(log_scale)
            conv_out = scale*conv_out
            if self.scale_mode == 'non-spatial':
                logdet += (self.n*self.n)*log_scale.sum()
            elif self.scale_mode == 'spatial':
                logdet += log_scale.sum()

        return conv_out, logdet

    def inverse(self, conv_out):
        with torch.no_grad():
            if self.scale_mode in ['non-spatial', 'spatial']: 
                log_scale = getattr(self, 'log_scale')
                scale = torch.exp(log_scale)
                conv_out = conv_out/(scale+1e-6)

            K = getattr(self, 'kernel')
            conv_in = self.conv_inverse_func(conv_out, K)

            if self.bias_mode in ['non-spatial', 'spatial']: 
                bias = getattr(self, 'bias')
                conv_in = conv_in-bias

            return conv_in

########################################################################################################

class Logit(torch.nn.Module):
    def __init__(self, c, n, scale=0.1, safe_mult=0.99, name=''):
        super().__init__()
        self.name = 'Logit_' + name
        self.n = n
        self.c = c
        self.scale = scale
        self.safe_mult = safe_mult

    def forward_with_logdet(self, nonlin_in):
        nonlin_in_safe = (1-self.safe_mult)/2+nonlin_in*self.safe_mult
        log_nonlin_in_safe = torch.log(nonlin_in_safe)
        log_one_min_nonlin_in_safe = torch.log(1-nonlin_in_safe)
        nonlin_out = self.scale*(log_nonlin_in_safe-log_one_min_nonlin_in_safe)

        logdet = np.prod(nonlin_in.shape[1:])*(np.log(self  .scale)+np.log(self.safe_mult)) + \
            (-log_nonlin_in_safe-log_one_min_nonlin_in_safe).sum(axis=[1, 2, 3])
        return nonlin_out, logdet

    def inverse(self, nonlin_out):
        with torch.no_grad():
            nonlin_in_safe = torch.sigmoid(nonlin_out/self.scale)
            nonlin_in = (nonlin_in_safe-(1-self.safe_mult)/2)/self.safe_mult
            return nonlin_in

########################################################################################################

class Tanh(torch.nn.Module):
    def __init__(self, c, n, name=''):
        super().__init__()
        self.name = 'Tanh_' + name
        self.n = n
        self.c = c

    def forward_with_logdet(self, nonlin_in):
        nonlin_out = torch.tanh(nonlin_in)
        deriv = 1-nonlin_out*nonlin_out
        logdet = torch.log(deriv).sum(axis=[1, 2, 3])
        return nonlin_out, logdet

    def inverse(self, nonlin_out):
        with torch.no_grad():
            nonlin_in = 0.5*(torch.log(1+nonlin_out)-torch.log(1-nonlin_out))
            return nonlin_in

########################################################################################################

class PReLU(torch.nn.Module):
    def __init__(self, c, n, mode='non-spatial', name=''):
        super().__init__()
        assert (mode in ['non-spatial', 'spatial'])
        self.name = 'PReLU_' + name
        self.n = n
        self.c = c
        self.mode = mode

        if self.mode == 'spatial': 
            pos_log_scale_th = helper.cuda(torch.zeros((1, self.c, self.n, self.n), dtype=torch.float32))
            neg_log_scale_th = helper.cuda(-1.609*torch.ones((1, self.c, self.n, self.n), dtype=torch.float32))
        elif self.mode == 'non-spatial': 
            pos_log_scale_th = helper.cuda(torch.zeros((1, self.c, 1, 1), dtype=torch.float32))
            neg_log_scale_th = helper.cuda(-1.609*torch.ones((1, self.c, 1, 1), dtype=torch.float32))
        pos_log_scale_param = torch.nn.parameter.Parameter(data=pos_log_scale_th, requires_grad=True)
        neg_log_scale_param = torch.nn.parameter.Parameter(data=neg_log_scale_th, requires_grad=True)
        setattr(self, 'pos_log_scale', pos_log_scale_param)
        setattr(self, 'neg_log_scale', neg_log_scale_param)

    def forward_with_logdet(self, nonlin_in):
        pos_log_scale = getattr(self, 'pos_log_scale')
        neg_log_scale = getattr(self, 'neg_log_scale')
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

    def inverse(self, nonlin_out):
        with torch.no_grad():
            pos_log_scale = getattr(self, 'pos_log_scale')
            neg_log_scale = getattr(self, 'neg_log_scale')
            pos_scale = torch.exp(pos_log_scale)
            neg_scale = torch.exp(neg_log_scale)

            y = nonlin_out
            y_pos = torch.relu(y)
            y_neg = y-y_pos
            nonlin_in = y_pos/(pos_scale+1e-6)+y_neg/(neg_scale+1e-6)
            return nonlin_in

########################################################################################################

class FixedSLogGate(torch.nn.Module):
    def __init__(self, c, n, name=''):
        super().__init__()
        self.name = 'FixedSLogGate_' + name
        self.n = n
        self.c = c
        # self.alpha = 2.14
        self.alpha = 0.5

    def forward_with_logdet(self, nonlin_in):
        nonlin_out = (torch.sign(nonlin_in)/self.alpha)*torch.log(1+self.alpha*torch.abs(nonlin_in))
        logdet = (-self.alpha*torch.abs(nonlin_out)).sum(axis=[1, 2, 3])
        return nonlin_out, logdet

    def inverse(self, nonlin_out):
        with torch.no_grad():
            nonlin_in = (torch.sign(nonlin_out)/self.alpha)*(torch.exp(self.alpha*torch.abs(nonlin_out))-1)
            return nonlin_in

class SLogGate(torch.nn.Module):
    def __init__(self, c, n, mode='non-spatial', name=''):
        super().__init__()
        assert (mode in ['non-spatial', 'spatial'])
        self.name = 'SLogGate_' + name
        self.n = n
        self.c = c
        self.mode = mode

        if self.mode == 'spatial': 
            log_alpha_th = helper.cuda(torch.zeros((1, self.c, self.n, self.n), dtype=torch.float32))
        elif self.mode == 'non-spatial': 
            log_alpha_th = helper.cuda(torch.zeros((1, self.c, 1, 1), dtype=torch.float32))
        log_alpha_param = torch.nn.parameter.Parameter(data=log_alpha_th, requires_grad=True)
        setattr(self, 'log_alpha', log_alpha_param)

    def forward_with_logdet(self, nonlin_in):
        log_alpha = getattr(self, 'log_alpha')
        alpha = torch.exp(log_alpha)
        nonlin_out = (torch.sign(nonlin_in)/alpha)*torch.log(1+alpha*torch.abs(nonlin_in))
        logdet = (-alpha*torch.abs(nonlin_out)).sum(axis=[1, 2, 3])
        return nonlin_out, logdet

    def inverse(self, nonlin_out):
        with torch.no_grad():
            log_alpha = getattr(self, 'log_alpha')
            alpha = torch.exp(log_alpha)
            nonlin_in = (torch.sign(nonlin_out)/alpha)*(torch.exp(alpha*torch.abs(nonlin_out))-1)
            return nonlin_in

########################################################################################################

class Actnorm(torch.nn.Module):
    def __init__(self, c, n, mode='non-spatial', name=''):
        super().__init__()
        assert (mode in ['non-spatial', 'spatial'])
        self.name = 'Actnorm_' + name
        self.n = n
        self.c = c
        self.mode = mode
        self.initialized = False

        if self.mode == 'spatial': 
            temp_bias_th = helper.cuda(torch.tensor(np.zeros([1, self.c, self.n, self.n]), dtype=torch.float32))
        elif self.mode == 'non-spatial': 
            temp_bias_th = helper.cuda(torch.tensor(np.zeros([1, self.c, 1, 1]), dtype=torch.float32))
        setattr(self, 'bias', temp_bias_th)

        if self.mode == 'spatial': 
            temp_log_scale_th = helper.cuda(torch.tensor(np.zeros([1, self.c, self.n, self.n]), dtype=torch.float32))
        elif self.mode == 'non-spatial': 
            temp_log_scale_th = helper.cuda(torch.tensor(np.zeros([1, self.c, 1, 1]), dtype=torch.float32))
        setattr(self, 'log_scale', temp_log_scale_th)

    def set_parameters(self, bias_np, log_scale_np):
        if self.mode == 'spatial': 
            assert (bias_np.shape == (1, self.c, self.n, self.n) and log_scale_np.shape == (1, self.c, self.n, self.n))
        elif self.mode == 'non-spatial':
            assert (bias_np.shape == (1, self.c, 1, 1) and log_scale_np.shape == (1, self.c, 1, 1))

        bias_th = helper.cuda(torch.tensor(bias_np, dtype=torch.float32))
        bias_param = torch.nn.parameter.Parameter(data=bias_th, requires_grad=True)
        setattr(self, 'bias', bias_param)

        log_scale_th = helper.cuda(torch.tensor(log_scale_np, dtype=torch.float32))
        log_scale_param = torch.nn.parameter.Parameter(data=log_scale_th, requires_grad=True)
        setattr(self, 'log_scale', log_scale_param)

    def set_initialized(self):
        self.initialized = True

    def forward_with_logdet(self, actnorm_in):
        bias = getattr(self, 'bias')
        log_scale = getattr(self, 'log_scale')

        scale = torch.exp(log_scale)
        actnorm_out = actnorm_in*scale+bias

        if self.mode == 'spatial': 
            logdet = log_scale.sum()
        elif self.mode == 'non-spatial':
            print('SOMETHING SEEMS WRONG WITH THIS')
            trace() 
            logdet = (self.n*self.n)*log_scale.sum()
        return actnorm_out, logdet

    def inverse(self, actnorm_out):
        with torch.no_grad():
            bias = getattr(self, 'bias')
            log_scale = getattr(self, 'log_scale')

            scale = torch.exp(log_scale)
            actnorm_in = (actnorm_out-bias)/(scale+1e-6)
            return actnorm_in

########################################################################################################

class Squeeze(torch.nn.Module):
    def __init__(self, chan_mode='input_channels_adjacent', spatial_mode='tl-br-tr-bl', name=''):
        super().__init__()
        assert (chan_mode in ['input_channels_adjacent', 'input_channels_apart'])
        assert (spatial_mode in ['tl-tr-bl-br', 'tl-br-tr-bl'])
        self.name = 'Squeeze_' + name
        self.chan_mode = chan_mode
        self.spatial_mode = spatial_mode

    def forward(self, x):
        """Squeezes a C x H x W tensor into a 4C x H/2 x W/2 tensor.
        (See Fig 3 in the real NVP paper.)
        Args:
            x: input tensor (B x C x H x W).
        Returns:
            the squeezed tensor (B x 4C x H/2 x W/2).
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H//2, 2, W//2, 2)
        if self.chan_mode == 'input_channels_adjacent':
            x = x.permute(0, 3, 5, 1, 2, 4)
            if self.spatial_mode == 'tl-tr-bl-br':
                x = x.reshape(B, C*4, H//2, W//2)
            elif self.spatial_mode == 'tl-br-tr-bl':
                x = torch.concat([x[:, 0, 0, np.newaxis], x[:, 1, 1, np.newaxis], 
                                  x[:, 0, 1, np.newaxis], x[:, 1, 0, np.newaxis]], axis=1)
                x = x.reshape(B, C*4, H//2, W//2)            
        elif self.chan_mode == 'input_channels_apart': 
            x = x.permute(0, 1, 3, 5, 2, 4)
            if self.spatial_mode == 'tl-tr-bl-br':
                x = x.reshape(B, C*4, H//2, W//2)
            elif self.spatial_mode == 'tl-br-tr-bl':
                x = torch.concat([x[:, :, 0, 0, np.newaxis], x[:, :, 1, 1, np.newaxis], 
                                  x[:, :, 0, 1, np.newaxis], x[:, :, 1, 0, np.newaxis]], axis=2)
                x = x.reshape(B, C*4, H//2, W//2)            
        return x

    def inverse(self, x):
        """unsqueezes a C x H x W tensor into a C/4 x 2H x 2W tensor.
        (See Fig 3 in the real NVP paper.)
        Args:
            x: input tensor (B x C x H x W).
        Returns:
            the squeezed tensor (B x C/4 x 2H x 2W).
        """
        with torch.no_grad():
            B, C, H, W = x.shape
            if self.chan_mode == 'input_channels_adjacent':
                if self.spatial_mode == 'tl-tr-bl-br':
                    x = x.reshape(B, 2, 2, C//4, H, W)
                elif self.spatial_mode == 'tl-br-tr-bl':
                    x = x.reshape(B, 4, C//4, H, W)
                    x = torch.concat([torch.concat([x[:, 0, np.newaxis, np.newaxis], x[:, 2, np.newaxis, np.newaxis]], axis=2),
                                      torch.concat([x[:, 3, np.newaxis, np.newaxis], x[:, 1, np.newaxis, np.newaxis]], axis=2)], axis=1)
                x = x.permute(0, 3, 4, 1, 5, 2)
            elif self.chan_mode == 'input_channels_apart':
                if self.spatial_mode == 'tl-tr-bl-br':
                    x = x.reshape(B, C//4, 2, 2, H, W)
                elif self.spatial_mode == 'tl-br-tr-bl':
                    x = x.reshape(B, C//4, 4, H, W)
                    x = torch.concat([torch.concat([x[:, :, 0, np.newaxis, np.newaxis], x[:, :, 2, np.newaxis, np.newaxis]], axis=3),
                                      torch.concat([x[:, :, 3, np.newaxis, np.newaxis], x[:, :, 1, np.newaxis, np.newaxis]], axis=3)], axis=2)
                x = x.permute(0, 1, 4, 2, 5, 3)
            x = x.reshape(B, C//4, H*2, W*2)
            return x

# class BatchNorm(torch.nn.Module):
#     def __init__(self, c, n, name=''):
#         super().__init__()
#         self.name = 'BatchNormTransform_' + name
#         self.epsilon = 1e-05
#         self.n = n
#         self.c = c

#         beta_th = helper.cuda(torch.zeros((1, self.c, 1, 1), dtype=torch.float32))
#         log_gamma_th = helper.cuda(torch.zeros((1, self.c, 1, 1), dtype=torch.float32))
#         beta_param = torch.nn.parameter.Parameter(data=beta_th, requires_grad=True)
#         log_gamma_param = torch.nn.parameter.Parameter(data=log_gamma_th, requires_grad=True)
#         setattr(self, 'beta', beta_param)
#         setattr(self, 'log_gamma', log_gamma_param)

#     def forward_with_logdet(self, batch_norm_in):
#         beta = getattr(self, 'beta')
#         log_gamma = getattr(self, 'log_gamma')
#         gamma = torch.exp(log_scale)

#         mu = torch.mean(batch_norm_in, axis=[0, 2, 3], keepdims=True)
#         mu.requires_grad = False
#         centered = batch_norm_in-mu
#         var = torch.mean(centered**2, axis=[0, 2, 3], keepdims=True)
#         var.requires_grad = False
#         scale = 1/torch.sqrt(var+self.epsilon)
#         log_scale = torch.log(scale)
#         normalized = centered*scale

#         batch_norm_out = normalized*gamma+beta

#         logdet = (self.n*self.n)*(log_gamma+log_scale).sum()
#         return actnorm_out, logdet

#     def inverse(self, actnorm_out):
#         with torch.no_grad():
#             beta = getattr(self, 'beta')
#             log_gamma = getattr(self, 'log_gamma')
#             gamma = torch.exp(log_scale)

#             scale = torch.exp(log_scale)
#             actnorm_in = (actnorm_out-bias)/(scale+1e-6)
#             return actnorm_in

    



