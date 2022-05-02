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


class GenerativeSchurFlow(torch.nn.Module):
    def __init__(self, c_in, n_in, k_list):
        super().__init__()
        self.n_in = n_in
        self.c_in = c_in
        self.k_list = k_list
        self.K_to_log_determinants = []
        self.n_layers = len(self.k_list)

        self.uniform_dist = torch.distributions.Uniform(helper.cuda(torch.tensor([0.0])), helper.cuda(torch.tensor([1.0])))
        self.normal_dist = torch.distributions.Normal(helper.cuda(torch.tensor([0.0])), helper.cuda(torch.tensor([1.0])))

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

            if layer_id < (self.n_layers-1):
                curr_slog_log_alpha_param = torch.nn.parameter.Parameter(data=helper.cuda(torch.zeros((1, curr_c, 1, 1), dtype=torch.float32)), requires_grad=True)
                setattr(self, 'slog_log_alpha_'+str(layer_id+1), curr_slog_log_alpha_param)

            self.K_to_log_determinants.append(spectral_schur_det_lib.generate_kernel_to_schur_log_determinant(curr_k, curr_n, backend='torch'))

        self.c_out = curr_c
        self.n_out = curr_n

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
        assert (z.shape == x.shape)

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

    def squeeze(self, x):
        """Squeezes a C x H x W tensor into a 4C x H/2 x W/2 tensor.
        (See Fig 3 in the real NVP paper.)
        Args:
            x: input tensor (B x C x H x W).
        Returns:
            the squeezed tensor (B x 4C x H/2 x W/2).
        """
        trace()
        [B, C, H, W] = list(x.size())
        x = x.reshape(B, C, H//2, 2, W//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C*4, H//2, W//2)
        return x

    def undo_squeeze(self, x):
        """unsqueezes a C x H x W tensor into a C/4 x 2H x 2W tensor.
        (See Fig 3 in the real NVP paper.)
        Args:
            x: input tensor (B x C x H x W).
        Returns:
            the squeezed tensor (B x C/4 x 2H x 2W).
        """
        trace()
        [B, C, H, W] = list(x.size())
        x = x.reshape(B, C//4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C//4, H*2, W*2)
        return x

    ################################################################################################

    def compute_actnorm_stats_for_layer(self, data_loader, layer_id, setup_mode='Training', n_batches=500, sub_image=None):
        data_loader.setup(setup_mode, randomized=False, verbose=False)
        print('Layer: ' + str(layer_id) + ', mean computation.' )

        n_examples = 0
        accum_mean = None
        for i, curr_batch_size, batch_np in data_loader:     
            if n_batches is not None and i > n_batches: break
            image_np = batch_np['Image']
            if sub_image is not None: image_np = image_np[:, :sub_image[0], :sub_image[1], :sub_image[2]]
            image = helper.cuda(torch.from_numpy(image_np))

            input_to_layer, _ = self.transform(image, until_layer=layer_id)
            input_to_layer = helper.to_numpy(input_to_layer)
            curr_mean = input_to_layer.mean(axis=(2, 3)).sum(0)
            if accum_mean is None: accum_mean = curr_mean
            else: accum_mean += curr_mean
            n_examples += input_to_layer.shape[0]

        mean = accum_mean/n_examples

        data_loader.setup(setup_mode, randomized=False, verbose=False)
        print('Layer: ' + str(layer_id) + ', std computation.' )
        
        n_examples = 0
        accum_var = None
        for i, curr_batch_size, batch_np in data_loader:  
            if n_batches is not None and i > n_batches: break
            image_np = batch_np['Image']
            if sub_image is not None: image_np = image_np[:, :sub_image[0], :sub_image[1], :sub_image[2]]
            image = helper.cuda(torch.from_numpy(image_np))

            input_to_layer, _ = self.transform(image, until_layer=layer_id)
            input_to_layer = helper.to_numpy(input_to_layer)
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

    def set_actnorm_parameters_of_layer(self, layer_id, actnorm_bias_np, actnorm_log_scale_np):
        actnorm_bias = torch.nn.parameter.Parameter(data=helper.cuda(torch.tensor(actnorm_bias_np, dtype=torch.float32)), requires_grad=True)
        setattr(self, 'actnorm_bias_'+str(layer_id+1), actnorm_bias)
        actnorm_log_scale = torch.nn.parameter.Parameter(data=helper.cuda(torch.tensor(actnorm_log_scale_np, dtype=torch.float32)), requires_grad=True)
        setattr(self, 'actnorm_log_scale_'+str(layer_id+1), actnorm_log_scale)

    def set_actnorm_parameters(self, data_loader, setup_mode='Training', n_batches=500, test_normalization=True, sub_image=None):
        for layer_id in range(self.n_layers):
            actnorm_bias_np, actnorm_log_scale_np = self.compute_actnorm_stats_for_layer(data_loader, layer_id, setup_mode,  n_batches, sub_image)
            self.set_actnorm_parameters_of_layer(layer_id, actnorm_bias_np, actnorm_log_scale_np)
            if test_normalization:
                print('Testing normalization: ')
                actnorm_bias_np, actnorm_log_scale_np = self.compute_actnorm_stats_for_layer(data_loader, layer_id, setup_mode,  n_batches, sub_image)
                assert (np.abs(actnorm_bias_np).max() < 1e-4)
                assert (np.abs(actnorm_log_scale_np).max() < 1e-4)
                print('Passed.')

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

    def compute_normal_log_pdf(self, z):
        return self.normal_dist.log_prob(z).sum(axis=[1, 2, 3])

    def sample_z(self, n_samples=10):
        return self.normal_dist.sample([n_samples, self.c_out, self.n_out, self.n_out])[..., 0].detach()

    def sample_x(self, n_samples=10):
        return self.inverse_transform(self.sample_z(n_samples))

    ################################################################################################

    def transform(self, x, until_layer=None):
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

            if layer_id < (self.n_layers-1):
                nonlin_out, nonlin_logdet = self.slog_gate_with_logit(conv_out, layer_id)
                nonlin_logdets.append(nonlin_logdet)
            else:
                nonlin_out = conv_out

            layer_out = nonlin_out
            layer_in = layer_out

        y = layer_out
        total_log_det = sum(actnorm_logdets)+sum(conv_logdets)+sum(nonlin_logdets) 
        return y, total_log_det

    def inverse_transform(self, y):
        y = y.detach()

        layer_out = y
        for layer_id in list(range(len(self.k_list)))[::-1]:
            if layer_id < (self.n_layers-1):
                conv_out = self.slog_gate_inverse(layer_out, layer_id)
            else:
                conv_out = layer_out
            actnorm_out = self.conv_inverse(conv_out, layer_id)
            layer_in = self.actnorm_inverse(actnorm_out, layer_id)
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



