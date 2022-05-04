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
            kernel_np = iden_kernel_np + 0.01*rand_kernel_np 
        elif self.kernel_init == 'he_uniform': 
            kernel_np = rand_kernel_np

        kernel_th = helper.cuda(torch.tensor(kernel_np, dtype=torch.float32))
        kernel_param = torch.nn.parameter.Parameter(data=kernel_th, requires_grad=True)
        setattr(self, 'kernel', kernel_param)
        self.kernel_to_logdet = spectral_schur_det_lib.generate_kernel_to_schur_log_determinant(self.k, self.n, backend='torch')

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
        K = getattr(self, 'kernel')
        conv_out = spatial_conv2D_lib.spatial_circular_conv2D_th(conv_in, K)
        logdet = self.kernel_to_logdet(K)

        if self.bias_mode in ['non-spatial', 'spatial']: 
            bias = getattr(self, 'bias')
            conv_out = conv_out+bias

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
        if self.scale_mode in ['non-spatial', 'spatial']: 
            log_scale = getattr(self, 'log_scale').detach()
            scale = torch.exp(log_scale)
            conv_out = conv_out/(scale+1e-6)

        if self.bias_mode in ['non-spatial', 'spatial']: 
            bias = getattr(self, 'bias').detach()
            conv_out = conv_out-bias

        K = getattr(self, 'kernel').detach()
        conv_in = frequency_conv2D_lib.frequency_inverse_circular_conv2D(conv_out, K, 'full', mode='complex', backend='torch')
        return conv_in

########################################################################################################
class Tanh(torch.nn.Module):
    def __init__(self, c, n, name=''):
        super().__init__()
        self.name = 'Tanh_' + name
        self.n = n
        self.c = c

    def forward_with_logdet(self, nonlin_in):
        nonlin_out = torch.tanh(x)
        deriv = 1-nonlin_out*nonlin_out
        logdet = torch.log(deriv).sum(axis=[1, 2, 3])
        return nonlin_out, logdet

    def inverse(self, nonlin_out):
        nonlin_in = 0.5*(torch.log(1+y)-torch.log(1-y))
        return nonlin_in

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
        pos_log_scale = getattr(self, 'pos_log_scale').detach()
        neg_log_scale = getattr(self, 'neg_log_scale').detach()
        pos_scale = torch.exp(pos_log_scale)
        neg_scale = torch.exp(neg_log_scale)

        y = nonlin_out
        y_pos = torch.relu(y)
        y_neg = y-y_pos
        nonlin_in = y_pos/(pos_scale+1e-6)+y_neg/(neg_scale+1e-6)
        return nonlin_in

########################################################################################################

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
        log_alpha = getattr(self, 'log_alpha').detach()
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
            logdet = (self.n*self.n)*log_scale.sum()
        return actnorm_out, logdet

    def inverse(self, actnorm_out):
        bias = getattr(self, 'bias').detach()
        log_scale = getattr(self, 'log_scale').detach()

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

########################################################################################################

class GenerativeSchurFlow(torch.nn.Module):
    def __init__(self, c_in, n_in, k_list, squeeze_list, final_actnorm=True):
        super().__init__()
        assert (len(k_list) == len(squeeze_list))
        self.name = 'GenerativeSchurFlow'
        self.n_in = n_in
        self.c_in = c_in
        self.k_list = k_list
        self.squeeze_list = squeeze_list
        self.final_actnorm = final_actnorm
        self.n_layers = len(self.k_list)

        self.uniform_dist = torch.distributions.Uniform(helper.cuda(torch.tensor([0.0])), helper.cuda(torch.tensor([1.0])))
        # self.normal_dist = torch.distributions.Normal(helper.cuda(torch.tensor([0.0])), helper.cuda(torch.tensor([0.1])))
        # self.normal_sharper_dist = torch.distributions.Normal(helper.cuda(torch.tensor([0.0])), helper.cuda(torch.tensor([0.07])))
        self.normal_dist = torch.distributions.Normal(helper.cuda(torch.tensor([0.0])), helper.cuda(torch.tensor([1.0])))
        self.normal_sharper_dist = torch.distributions.Normal(helper.cuda(torch.tensor([0.0])), helper.cuda(torch.tensor([0.7])))

        print('\n**********************************************************')
        print('Creating GenerativeSchurFlow: ')
        print('**********************************************************\n')
        conv_layers, nonlin_layers, actnorm_layers = [], [], []

        accum_squeeze = 0
        for layer_id in range(self.n_layers):
            accum_squeeze += self.squeeze_list[layer_id]
            curr_c = self.c_in*(4**accum_squeeze)
            curr_n = self.n_in//(2**accum_squeeze)
            curr_k = self.k_list[layer_id]
            print('Layer '+str(layer_id)+': c='+str(curr_c)+', n='+str(curr_n)+', k='+str(curr_k))
            assert (curr_n >= curr_k)

            actnorm_layers.append(Actnorm(curr_c, curr_n, name=str(layer_id)))
            conv_layers.append(MultiChannel2DCircularConv(
                curr_c, curr_n, curr_k, kernel_init='I + he_uniform', 
                bias_mode='spatial', scale_mode='no-scale', name=str(layer_id)))
            # conv_layers.append(MultiChannel2DCircularConv(
            #     curr_c, curr_n, curr_k, kernel_init='he_uniform', 
            #     bias_mode='spatial', scale_mode='no-scale', name=str(layer_id)))

            if layer_id != self.n_layers-1:
                # nonlin_layers.append(SLogGate(curr_c, curr_n, mode='spatial', name=str(layer_id)))
                # nonlin_layers.append(PReLU(curr_c, curr_n, mode='spatial', name=str(layer_id)))
                nonlin_layers.append(Tanh(curr_c, curr_n, name=str(layer_id)))

        if self.final_actnorm: actnorm_layers.append(Actnorm(curr_c, curr_n, name='final'))

        self.conv_layers = torch.nn.ModuleList(conv_layers)
        self.nonlin_layers = torch.nn.ModuleList(nonlin_layers)
        self.actnorm_layers = torch.nn.ModuleList(actnorm_layers)
        self.squeeze_layer = Squeeze()

        self.c_out = curr_c
        self.n_out = curr_n
        print('\n**********************************************************\n')

    ################################################################################################

    def dequantize(self, x, quantization_levels=255.):
        # https://arxiv.org/pdf/1511.01844.pdf
        scale = 1/quantization_levels
        uniform_sample = self.uniform_dist.sample(x.shape)[..., 0]
        return x+scale*uniform_sample

    def jacobian(self, x):
        dummy_optimizer = torch.optim.Adam(self.parameters())
        x.requires_grad = True

        func_to_J = self.transform
        z, _ = func_to_J(x)
        assert (len(z.shape) == 4 and len(x.shape) == 4)
        assert (z.shape[0] == x.shape[0])
        assert (np.prod(z.shape[1:]) == np.prod(x.shape[1:]))

        J = np.zeros(z.shape+x.shape[1:])
        for i in range(z.shape[1]):
            for a in range(z.shape[2]):
                for b in range(z.shape[3]):
                    print(i, a, b)
                    dummy_optimizer.zero_grad() # zero the parameter gradients
                    if x.grad is not None: x.grad.zero_()

                    z, _ = func_to_J(x)
                    loss = torch.sum(z[:, i, a, b])
                    loss.backward()
                    J[:, i, a, b, ...] = helper.to_numpy(x.grad)

        J_flat = J.reshape(z.shape[0], np.prod(z.shape[1:]), np.prod(x.shape[1:]))
        return J, J_flat

    ################################################################################################

    def compute_uninitialized_actnorm_stats(self, data_loader, setup_mode='Training', n_batches=500, sub_image=None):
        data_loader.setup(setup_mode, randomized=False, verbose=False)
        print('Mean computation.' )

        n_examples = 0
        accum_mean = None
        for i, curr_batch_size, batch_np in data_loader:     
            if n_batches is not None and i > n_batches: break
            image_np = batch_np['Image']
            if sub_image is not None: image_np = image_np[:, :sub_image[0], :sub_image[1], :sub_image[2]]
            image = helper.cuda(torch.from_numpy(image_np))

            actnorm_out, actnorm_object_mean = self.transform(image, initialization=True)
            if type(actnorm_object_mean) is not Actnorm: return None, None, None, None, None

            actnorm_out = helper.to_numpy(actnorm_out)
            if actnorm_object_mean.mode == 'spatial': curr_mean = actnorm_out.sum(0)
            elif actnorm_object_mean.mode == 'non-spatial': curr_mean = actnorm_out.mean(axis=(2, 3)).sum(0)

            if accum_mean is None: accum_mean = curr_mean
            else: accum_mean += curr_mean
            n_examples += actnorm_out.shape[0]

        mean = accum_mean/n_examples

        data_loader.setup(setup_mode, randomized=False, verbose=False)
        print('Std computation.' )
        
        n_examples = 0
        accum_var = None
        for i, curr_batch_size, batch_np in data_loader:  
            if n_batches is not None and i > n_batches: break
            image_np = batch_np['Image']
            if sub_image is not None: image_np = image_np[:, :sub_image[0], :sub_image[1], :sub_image[2]]
            image = helper.cuda(torch.from_numpy(image_np))

            actnorm_out, actnorm_object_var = self.transform(image, initialization=True)
            if type(actnorm_object_var) is not Actnorm: return None, None, None, None, None

            actnorm_out = helper.to_numpy(actnorm_out)
            if actnorm_object_var.mode == 'spatial': curr_var = ((actnorm_out-mean[np.newaxis, :, :, :])**2).sum(0)
            elif actnorm_object_var.mode == 'non-spatial':
                curr_var = ((actnorm_out-mean[np.newaxis, :, np.newaxis, np.newaxis])**2).mean(axis=(2, 3)).sum(0)

            if accum_var is None: accum_var = curr_var
            else: accum_var += curr_var
            n_examples += actnorm_out.shape[0]
        
        var = accum_var/n_examples
        std = np.sqrt(var)
        log_std = 0.5*np.log(var)
        bias = -mean/(np.exp(log_std)+1e-5)
        log_scale = -log_std

        assert (actnorm_object_mean == actnorm_object_var)
        if actnorm_object_var.mode == 'spatial': 
            bias = bias[np.newaxis, :, :, :].astype(np.float32)
            log_scale = log_scale[np.newaxis, :, :, :].astype(np.float32)
        elif actnorm_object_var.mode == 'non-spatial':
            bias = bias[np.newaxis, :, np.newaxis, np.newaxis].astype(np.float32)
            log_scale = log_scale[np.newaxis, :, np.newaxis, np.newaxis].astype(np.float32)

        return actnorm_object_var, bias, log_scale, mean, std

    def set_actnorm_parameters(self, data_loader, setup_mode='Training', n_batches=500, test_normalization=True, sub_image=None):
        while True:
            print('\n')
            actnorm_object, actnorm_bias_np, actnorm_log_scale_np, _, _ = \
                self.compute_uninitialized_actnorm_stats(data_loader, setup_mode, n_batches, sub_image)
            if actnorm_object is None: break

            actnorm_object.set_parameters(actnorm_bias_np, actnorm_log_scale_np)
            print(actnorm_object.name + ' is initialized.\n')

            if test_normalization:
                print('Testing normalization: ')
                actnorm_object_test, actnorm_bias_np, actnorm_log_scale_np, _, _ = \
                    self.compute_uninitialized_actnorm_stats(data_loader, setup_mode, n_batches, sub_image)
                assert (np.abs(actnorm_bias_np).max() < 1e-4 and np.abs(actnorm_log_scale_np).max() < 1e-4)
                assert (actnorm_object_test == actnorm_object)
                print('PASSED: ' + actnorm_object_test.name + ' is normalizing.\n')

            actnorm_object.set_initialized()

    ################################################################################################

    def compute_normal_log_pdf(self, z):
        return self.normal_dist.log_prob(z).sum(axis=[1, 2, 3])

    def sample_z(self, n_samples=10):
        return self.normal_dist.sample([n_samples, self.c_out, self.n_out, self.n_out])[..., 0].detach()

    def sample_sharper_z(self, n_samples=10):
        return self.normal_sharper_dist.sample([n_samples, self.c_out, self.n_out, self.n_out])[..., 0].detach()

    def sample_x(self, n_samples=10):
        return self.inverse_transform(self.sample_z(n_samples))

    def sample_sharper_x(self, n_samples=10):
        return self.inverse_transform(self.sample_sharper_z(n_samples))

    ################################################################################################

    def transform(self, x, initialization=False):
        actnorm_logdets, conv_logdets, nonlin_logdets = [], [], []

        layer_in = x
        for layer_id, k in enumerate(self.k_list): 
            for squeeze_i in range(self.squeeze_list[layer_id]):
                layer_in = self.squeeze_layer(layer_in)

            actnorm_out, actnorm_logdet = self.actnorm_layers[layer_id].forward_with_logdet(layer_in)
            if initialization and not self.actnorm_layers[layer_id].initialized:
                return actnorm_out, self.actnorm_layers[layer_id]
            actnorm_logdets.append(actnorm_logdet)

            conv_out, conv_logdet = self.conv_layers[layer_id].forward_with_logdet(actnorm_out)
            conv_logdets.append(conv_logdet)

            if layer_id != self.n_layers-1:
                nonlin_out, nonlin_logdet = self.nonlin_layers[layer_id].forward_with_logdet(conv_out)
                nonlin_logdets.append(nonlin_logdet)
            else:
                nonlin_out = conv_out

            layer_out = nonlin_out
            layer_in = layer_out

        if self.final_actnorm: 
            layer_out, actnorm_logdet = self.actnorm_layers[self.n_layers].forward_with_logdet(layer_out)
            if initialization and not self.actnorm_layers[self.n_layers].initialized:
                return layer_out, self.actnorm_layers[self.n_layers]
            actnorm_logdets.append(actnorm_logdet)

        y = layer_out
        total_log_det = sum(actnorm_logdets)+sum(conv_logdets)+sum(nonlin_logdets) 
        return y, total_log_det

    def inverse_transform(self, y):
        y = y.detach()

        layer_out = y
        if self.final_actnorm: layer_out = self.actnorm_layers[self.n_layers].inverse(layer_out)

        for layer_id in list(range(len(self.k_list)))[::-1]:
            if layer_id != self.n_layers-1:
                conv_out = self.nonlin_layers[layer_id].inverse(layer_out)
            else:
                conv_out = layer_out
            
            actnorm_out = self.conv_layers[layer_id].inverse(conv_out)
            layer_in = self.actnorm_layers[layer_id].inverse(actnorm_out)

            for squeeze_i in range(self.squeeze_list[layer_id]):
                layer_in = self.squeeze_layer.inverse(layer_in)
            layer_out = layer_in

        x = layer_in
        return x

    def forward(self, x, dequantize=True):
        if dequantize: x = self.dequantize(x)
        z, logdet = self.transform(x)
        log_pdf_z = self.compute_normal_log_pdf(z)
        log_pdf_x = log_pdf_z + logdet
        return z, x, log_pdf_z, log_pdf_x























    # def logit_with_logdet(self, x, scale=0.1):
    #     x_safe = 0.005+x*0.99
    #     y = scale*(torch.log(x_safe)-torch.log(1-x_safe))
    #     y_logdet = np.prod(x_safe.shape[1:])*(np.log(scale)+np.log(0.99))+(-torch.log(x_safe)-torch.log(1-x_safe)).sum(axis=[1, 2, 3])
    #     return y, y_logdet

    # def inverse_logit(self, y, scale=0.1):
    #     x_safe = torch.sigmoid(y/scale)
    #     x = (x_safe-0.005)/0.99
    #     return x

    # def leaky_relu_with_logdet(self, x, pos_slope=1.2, neg_slope=0.8):
    #     x_pos = torch.relu(x)
    #     x_neg = x-x_pos
    #     y = pos_slope*x_pos+neg_slope*x_neg
    #     x_ge_zero = x_pos/(x+0.001)
    #     y_deriv = pos_slope*x_ge_zero
    #     y_deriv += neg_slope*(1-y_deriv)
    #     y_logdet = torch.log(y_deriv).sum(axis=[1, 2, 3])
    #     return y, y_logdet
    
    # def inverse_leaky_relu(self, y, pos_slope=1.2, neg_slope=0.8):
    #     y_pos = torch.relu(y)
    #     y_neg = y-y_pos
    #     x = (1/pos_slope)*y_pos+(1/neg_slope)*y_neg
    #     return x


    # def tanh_with_logdet(self, x):
    #     y = torch.tanh(x)
    #     y_deriv = 1-y*y
    #     y_logdet = torch.log(y_deriv).sum(axis=[1, 2, 3])
    #     return y, y_logdet

    # def inverse_tanh(self, y):
    #     y = torch.clamp(y, min=-0.98, max=0.98)
    #     return 0.5*(torch.log(1+y+1e-4)-torch.log(1-y+1e-4))




# class Net(torch.nn.Module):
#     def __init__(self, c_in, n_in, k_list, squeeze_list, logit_layer=True, actnorm_layers=True, squeeze_layers=True):
#         super().__init__()
#         self.n_in = n_in
#         self.c_in = c_in
#         self.k_list = k_list
#         self.squeeze_list = squeeze_list
#         assert (len(self.squeeze_list) == len(self.k_list))
#         self.n_conv_blocks = len(self.k_list)
#         self.logit_layer = logit_layer
#         self.actnorm_layers = actnorm_layers
#         self.squeeze_layers = squeeze_layers

#         self.K_to_schur_log_determinant_funcs = []
#         accum_squeeze = 0
#         for layer_id, curr_k in enumerate(self.k_list):
#             curr_c = self.c_in
#             curr_n = self.n_in
#             if self.squeeze_layers:
#                 accum_squeeze += self.squeeze_list[layer_id]
#                 curr_c = self.c_in*(4**accum_squeeze)
#                 curr_n = self.n_in//(2**accum_squeeze)
#             print(curr_c, (curr_n, curr_n), curr_c*curr_n*curr_n)



            # for squeeze_i in range(self.squeeze_list[layer_id]):
            #     curr_inp = self.squeeze(curr_inp)

            # for squeeze_i in range(self.squeeze_list[layer_id]):
            #     curr_inp = self.undo_squeeze(curr_inp)



