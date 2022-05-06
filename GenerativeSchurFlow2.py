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
from Transforms import Actnorm, Squeeze, SLogGate
from ConditionalTransforms import CondMultiChannel2DCircularConv, CondAffine #, CondPReLU, CondSLogGate

class ConditionalSchurTransform(torch.nn.Module):
    def __init__(self, c_in, n_in, k_list, squeeze_list, final_actnorm=True):
        super().__init__()
        assert (len(k_list) == len(squeeze_list))
        self.name = 'GenerativeSchurFlow2'
        self.n_in = n_in
        self.c_in = c_in
        self.k_list = k_list
        self.squeeze_list = squeeze_list
        self.final_actnorm = final_actnorm
        self.n_layers = len(self.k_list)

        print('\n**********************************************************')
        print('Creating GenerativeSchurFlow: ')
        print('**********************************************************\n')
        conv_layers, affine_layers, conv_nonlin_layers, affine_nonlin_layers, actnorm_layers = [], [], [], [], []

        self.non_spatial_conditional_transforms = {}
        self.spatial_conditional_transforms = {}

        accum_squeeze = 0
        for layer_id in range(self.n_layers):
            accum_squeeze += self.squeeze_list[layer_id]
            curr_c = self.c_in*(4**accum_squeeze)
            curr_n = self.n_in//(2**accum_squeeze)
            curr_k = self.k_list[layer_id]
            print('Layer '+str(layer_id)+': c='+str(curr_c)+', n='+str(curr_n)+', k='+str(curr_k))
            assert (curr_n >= curr_k)

            actnorm_layers.append(Actnorm(curr_c, curr_n, name=str(layer_id)))

            conv_layer = CondMultiChannel2DCircularConv(curr_c, curr_n, curr_k, 
                bias_mode='non-spatial', name=str(layer_id))
            self.non_spatial_conditional_transforms[conv_layer.name] = conv_layer
            conv_layers.append(conv_layer)

            if layer_id != self.n_layers-1:
                conv_nonlin_layers.append(SLogGate(curr_c, curr_n, mode='non-spatial', name='conv_nonlin_'+str(layer_id)))

            affine_layer = CondAffine(curr_c, curr_n, mode='spatial', name=str(layer_id))
            self.spatial_conditional_transforms[affine_layer.name] = affine_layer
            affine_layers.append(affine_layer)
            
            if layer_id != self.n_layers-1:
                affine_nonlin_layers.append(SLogGate(curr_c, curr_n, mode='non-spatial', name='affine_nonlin_'+str(layer_id)))

        if self.final_actnorm: actnorm_layers.append(Actnorm(curr_c, curr_n, name='final'))

        self.conv_layers = torch.nn.ModuleList(conv_layers)
        self.conv_nonlin_layers = torch.nn.ModuleList(conv_nonlin_layers)
        self.affine_layers = torch.nn.ModuleList(affine_layers)
        self.affine_nonlin_layers = torch.nn.ModuleList(affine_nonlin_layers)
        self.actnorm_layers = torch.nn.ModuleList(actnorm_layers)
        self.squeeze_layer = Squeeze()        

        self.c_out = curr_c
        self.n_out = curr_n
        self.non_spatial_n_cond_params, self.non_spatial_cond_param_sizes_list = self.non_spatial_conditional_param_sizes()
        self.spatial_cond_param_shape, self.spatial_cond_param_sizes_list = self.spatial_conditional_param_sizes()
        print('\n**********************************************************\n')
        
    ################################################################################################

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

    ###############################################################################################

    def non_spatial_conditional_param_sizes(self):
        param_sizes_list = []
        total_n_params = 0
        for transform_name in sorted(self.non_spatial_conditional_transforms):
            for param_name in sorted(self.non_spatial_conditional_transforms[transform_name].parameter_sizes):
                if self.non_spatial_conditional_transforms[transform_name].parameter_sizes[param_name] is None:
                    param_sizes_list.append((transform_name + '__' + param_name, None))
                else:
                    curr_n_param = np.prod(self.non_spatial_conditional_transforms[transform_name].parameter_sizes[param_name][1:])
                    param_sizes_list.append((transform_name + '__' + param_name, 
                        self.non_spatial_conditional_transforms[transform_name].parameter_sizes[param_name]))
                    total_n_params += curr_n_param
        return total_n_params, param_sizes_list

    def non_spatial_conditional_param_assignments(self, tensor):
        param_assignments = {}
        total_n_params = 0
        for transform_name in sorted(self.non_spatial_conditional_transforms):
            param_assignments[transform_name] = {}
            for param_name in sorted(self.non_spatial_conditional_transforms[transform_name].parameter_sizes):
                if self.non_spatial_conditional_transforms[transform_name].parameter_sizes[param_name] is None:
                    param_assignments[transform_name][param_name] = None
                else:
                    curr_n_param = np.prod(self.non_spatial_conditional_transforms[transform_name].parameter_sizes[param_name][1:])
                    curr_param_flat = tensor[:, total_n_params:total_n_params+curr_n_param]
                    curr_param = curr_param_flat.reshape(self.non_spatial_conditional_transforms[transform_name].parameter_sizes[param_name])
                    param_assignments[transform_name][param_name] = curr_param
                    total_n_params += curr_n_param
        return total_n_params, param_assignments

    def spatial_conditional_param_sizes(self):
        param_sizes_list = []
        total_param_shape = None
        for transform_name in sorted(self.spatial_conditional_transforms):
            for param_name in sorted(self.spatial_conditional_transforms[transform_name].parameter_sizes):
                if self.spatial_conditional_transforms[transform_name].parameter_sizes[param_name] is None:
                    param_sizes_list.append((transform_name + '__' + param_name, None))
                else:
                    curr_param_shape = self.spatial_conditional_transforms[transform_name].parameter_sizes[param_name][1:]
                    param_sizes_list.append((transform_name + '__' + param_name, 
                        self.spatial_conditional_transforms[transform_name].parameter_sizes[param_name]))
                    if total_param_shape is None: total_param_shape = curr_param_shape
                    else:
                        assert (total_param_shape[1:] == curr_param_shape[1:])
                        total_param_shape[0] += curr_param_shape[0]
        return total_param_shape, param_sizes_list

    def spatial_conditional_param_assignments(self, tensor):
        param_assignments = {}
        total_param_shape = None
        for transform_name in sorted(self.spatial_conditional_transforms):
            param_assignments[transform_name] = {}
            for param_name in sorted(self.spatial_conditional_transforms[transform_name].parameter_sizes):
                if self.spatial_conditional_transforms[transform_name].parameter_sizes[param_name] is None:
                    param_assignments[transform_name][param_name] = None
                else:
                    curr_param_shape = self.spatial_conditional_transforms[transform_name].parameter_sizes[param_name][1:]
                    if total_param_shape is None:
                        curr_param = tensor[:, :curr_param_shape[0], :, :]
                    else:
                        curr_param = tensor[:, total_param_shape[0]:total_param_shape[0]+curr_param_shape[0], :, :]
                    param_assignments[transform_name][param_name] = curr_param

                    if total_param_shape is None: total_param_shape = curr_param_shape
                    else:
                        assert (total_param_shape[1:] == curr_param_shape[1:])
                        total_param_shape[0] += curr_param_shape[0]        
        return total_param_shape, param_assignments

    def transform(self, x, non_spatial_param, spatial_param, initialization=False):
        non_spatial_n_params, non_spatial_param_assignments = self.non_spatial_conditional_param_assignments(non_spatial_param)
        spatial_cond_param_shape, spatial_param_assignments = self.spatial_conditional_param_assignments(spatial_param)
        actnorm_logdets, conv_logdets, affine_logdets, nonlin_logdets = [], [], [], []

        layer_in = x
        for layer_id, k in enumerate(self.k_list): 
            for squeeze_i in range(self.squeeze_list[layer_id]):
                layer_in = self.squeeze_layer(layer_in)

            actnorm_out, actnorm_logdet = self.actnorm_layers[layer_id].forward_with_logdet(layer_in)
            if initialization and not self.actnorm_layers[layer_id].initialized:
                return actnorm_out, self.actnorm_layers[layer_id]
            actnorm_logdets.append(actnorm_logdet)

            curr_params = non_spatial_param_assignments[self.conv_layers[layer_id].name]
            curr_kernel, curr_bias = curr_params["kernel"], curr_params["bias"]
            conv_out, conv_logdet = self.conv_layers[layer_id].forward_with_logdet(actnorm_out, curr_kernel, curr_bias)
            conv_logdets.append(conv_logdet)
            
            if layer_id != self.n_layers-1:
                conv_nonlin_out, nonlin_logdet = self.conv_nonlin_layers[layer_id].forward_with_logdet(conv_out)
                nonlin_logdets.append(nonlin_logdet)
            else:
                conv_nonlin_out = conv_out

            curr_params = spatial_param_assignments[self.affine_layers[layer_id].name]
            curr_affine_bias, curr_affine_log_scale =  curr_params["bias"], curr_params["log_scale"]
            affine_out, affine_logdet = self.affine_layers[layer_id].forward_with_logdet(conv_nonlin_out, curr_affine_bias, curr_affine_log_scale)
            affine_logdets.append(affine_logdet)

            if layer_id != self.n_layers-1:
                affine_nonlin_out, nonlin_logdet = self.affine_nonlin_layers[layer_id].forward_with_logdet(affine_out)
                nonlin_logdets.append(nonlin_logdet)
            else:
                affine_nonlin_out = affine_out

            layer_out = affine_nonlin_out
            layer_in = layer_out

        if self.final_actnorm: 
            layer_out, actnorm_logdet = self.actnorm_layers[self.n_layers].forward_with_logdet(layer_out)
            if initialization and not self.actnorm_layers[self.n_layers].initialized:
                return layer_out, self.actnorm_layers[self.n_layers]
            actnorm_logdets.append(actnorm_logdet)

        y = layer_out
        total_log_det = sum(actnorm_logdets)+sum(conv_logdets)+sum(affine_logdets)+sum(nonlin_logdets) 
        return y, total_log_det

    def inverse_transform(self, y, non_spatial_param, spatial_param):
        with torch.no_grad():
            non_spatial_n_params, non_spatial_param_assignments = self.non_spatial_conditional_param_assignments(non_spatial_param)
            spatial_cond_param_shape, spatial_param_assignments = self.spatial_conditional_param_assignments(spatial_param)

            layer_out = y
            if self.final_actnorm: layer_out = self.actnorm_layers[self.n_layers].inverse(layer_out)

            for layer_id in list(range(len(self.k_list)))[::-1]:
                if layer_id != self.n_layers-1:
                    affine_out = self.affine_nonlin_layers[layer_id].inverse(layer_out)
                else:
                    affine_out = layer_out
                
                curr_params = spatial_param_assignments[self.affine_layers[layer_id].name]
                curr_affine_bias, curr_affine_log_scale =  curr_params["bias"], curr_params["log_scale"]
                conv_nonlin_out = self.affine_layers[layer_id].inverse(affine_out, curr_affine_bias, curr_affine_log_scale)

                if layer_id != self.n_layers-1:
                    conv_out = self.conv_nonlin_layers[layer_id].inverse(conv_nonlin_out)
                else:
                    conv_out = conv_nonlin_out
                
                curr_params = non_spatial_param_assignments[self.conv_layers[layer_id].name]
                curr_kernel, curr_bias = curr_params["kernel"], curr_params["bias"]
                
                actnorm_out = self.conv_layers[layer_id].inverse(conv_out, curr_kernel, curr_bias)
                
                layer_in = self.actnorm_layers[layer_id].inverse(actnorm_out)

                for squeeze_i in range(self.squeeze_list[layer_id]):
                    layer_in = self.squeeze_layer.inverse(layer_in)
                layer_out = layer_in

            x = layer_in
            return x
    
    ################################################################################################


class GenerativeSchurFlow(torch.nn.Module):
    def __init__(self, c_in, n_in):
        super().__init__()
        self.name = 'GenerativeSchurFlow'
        self.c_in = c_in
        self.n_in = n_in

        self.uniform_dist = torch.distributions.Uniform(helper.cuda(torch.tensor([0.0])), helper.cuda(torch.tensor([1.0])))
        self.normal_dist = torch.distributions.Normal(helper.cuda(torch.tensor([0.0])), helper.cuda(torch.tensor([1.0])))
        self.normal_sharper_dist = torch.distributions.Normal(helper.cuda(torch.tensor([0.0])), helper.cuda(torch.tensor([0.7])))

        self.squeeze_layer = Squeeze()
        self.cond_schur_transform = ConditionalSchurTransform(c_in=self.c_in*4//2, n_in=n_in//2, 
            k_list=[3, 4, 4], squeeze_list=[0, 0, 0])

        self.base_cond_net_c_out = 128
        self.base_cond_net = self.create_base_cond_net(c_in=(self.c_in*4//2), c_out=self.base_cond_net_c_out)
        self.spatial_cond_net = self.create_spatial_cond_net(c_in=self.base_cond_net_c_out, 
            c_out=self.cond_schur_transform.spatial_cond_param_shape[0])
        self.non_spatial_cond_net = self.create_non_spatial_cond_net(c_in=self.base_cond_net_c_out, n_in=(self.n_in//2), 
            c_out=self.cond_schur_transform.non_spatial_n_cond_params)

    ################################################################################################

    def create_base_cond_net(self, c_in, c_out, channel_multiplier=1):
        net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=c_in, out_channels=c_in*2*channel_multiplier, kernel_size=3, stride=1, padding='same',
                            dilation=1, groups=1, bias=True, padding_mode='zeros'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=c_in*2*channel_multiplier, out_channels=c_in*4*channel_multiplier, kernel_size=3, stride=1, padding='same',
                            dilation=1, groups=1, bias=True, padding_mode='zeros'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=c_in*4*channel_multiplier, out_channels=c_out, kernel_size=3, stride=1, padding='same',
                            dilation=1, groups=1, bias=True, padding_mode='zeros'),
            torch.nn.ReLU(),
            )
        net = helper.cuda(net)
        return net

    def create_spatial_cond_net(self, c_in, c_out, channel_multiplier=1):
        net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=c_in, out_channels=c_in*channel_multiplier, kernel_size=3, stride=1, padding='same',
                            dilation=1, groups=1, bias=True, padding_mode='zeros'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=c_in*channel_multiplier, out_channels=c_out, kernel_size=3, stride=1, padding='same',
                            dilation=1, groups=1, bias=True, padding_mode='zeros'),
            )
        net = helper.cuda(net)
        return net

    def create_non_spatial_cond_net(self, c_in, n_in, c_out, channel_multiplier=1):
        if n_in == 14:
            net = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=c_in, out_channels=c_in//2*channel_multiplier, kernel_size=4, stride=2, padding='valid', 
                                dilation=1, groups=1, bias=True, padding_mode='zeros'),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=c_in//2*channel_multiplier, out_channels=c_in//4*channel_multiplier, kernel_size=4, stride=1, padding='valid', 
                                dilation=1, groups=1, bias=True, padding_mode='zeros'),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=c_in//4*channel_multiplier, out_channels=c_out//4, kernel_size=3, stride=1, padding='valid', 
                                dilation=1, groups=1, bias=True, padding_mode='zeros'),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(c_out//4, c_out)
                )
        if n_in == 16:
            net = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=c_in, out_channels=c_in//2*channel_multiplier, kernel_size=4, stride=2, padding='valid', 
                                dilation=1, groups=1, bias=True, padding_mode='zeros'),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=c_in//2*channel_multiplier, out_channels=c_in//4*channel_multiplier, kernel_size=4, stride=1, padding='valid', 
                                dilation=1, groups=1, bias=True, padding_mode='zeros'),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=c_in//4*channel_multiplier, out_channels=c_out//8, kernel_size=4, stride=1, padding='valid', 
                                dilation=1, groups=1, bias=True, padding_mode='zeros'),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(c_out//8, c_out)
                )
        if n_in == 32:
            net = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=c_in, out_channels=c_in//2*channel_multiplier, kernel_size=4, stride=2, padding='valid', 
                                dilation=1, groups=1, bias=True, padding_mode='zeros'),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=c_in//2*channel_multiplier, out_channels=c_in//2*channel_multiplier, kernel_size=4, stride=2, padding='valid', 
                                dilation=1, groups=1, bias=True, padding_mode='zeros'),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=c_in//2*channel_multiplier, out_channels=c_in//4*channel_multiplier, kernel_size=4, stride=1, padding='valid', 
                                dilation=1, groups=1, bias=True, padding_mode='zeros'),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=c_in//4*channel_multiplier, out_channels=c_out//8, kernel_size=3, stride=1, padding='valid', 
                                dilation=1, groups=1, bias=True, padding_mode='zeros'),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(c_out//8, c_out)
                )
        net = helper.cuda(net)
        return net

    ################################################################################################

    def dequantize(self, x, quantization_levels=255.):
        # https://arxiv.org/pdf/1511.01844.pdf
        scale = 1/quantization_levels
        uniform_sample = self.uniform_dist.sample(x.shape)[..., 0]
        return x+scale*uniform_sample

    def jacobian(self, x):
        dummy_optimizer = torch.optim.Adam(self.parameters())
        x.requires_grad = True

        func_to_J = self.forward
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

    ###############################################################################################
    def compute_normal_log_pdf(self, z):
        return self.normal_dist.log_prob(z).sum(axis=[1, 2, 3])

    def sample_z(self, n_samples=10):
        with torch.no_grad():
            return self.normal_dist.sample([n_samples, self.c_out, self.n_out, self.n_out])[..., 0]

    def sample_sharper_z(self, n_samples=10):
        with torch.no_grad():
            return self.normal_sharper_dist.sample([n_samples, self.c_out, self.n_out, self.n_out])[..., 0]

    def sample_x(self, n_samples=10):
        with torch.no_grad():
            return self.inverse_transform(self.sample_z(n_samples))

    def sample_sharper_x(self, n_samples=10):
        with torch.no_grad():
            return self.inverse_transform(self.sample_sharper_z(n_samples))

################################################################################################

    def transform(self, x, initialization=False):
        x_squeezed = self.squeeze_layer(x)
        x_base, x_update = x_squeezed[:, :x_squeezed.shape[1]//2], x_squeezed[:, x_squeezed.shape[1]//2:]

        base_cond = self.base_cond_net(x_base)
        non_spatial_param = self.non_spatial_cond_net(base_cond)
        spatial_param = self.spatial_cond_net(base_cond)
        
        z_update, update_logdet = self.cond_schur_transform.transform(x_update, non_spatial_param, spatial_param, initialization)
        if type(update_logdet) is Actnorm: return z_update, update_logdet # init run unparameterized actnorm

        base_cond_2 = self.base_cond_net(z_update)
        non_spatial_param_2 = self.non_spatial_cond_net(base_cond_2)
        spatial_param_2 = self.spatial_cond_net(base_cond_2)
        
        z_base, base_logdet = self.cond_schur_transform.transform(x_base, non_spatial_param_2, spatial_param_2, initialization)
        if type(base_logdet) is Actnorm: return z_base, base_logdet # init run unparameterized actnorm

        total_lodget = update_logdet+base_logdet
        z_squeezed = torch.concat([z_base, z_update], axis=1)

        return z_squeezed, total_lodget

    def inverse_transform(self, z_squeezed):
        with torch.no_grad():
            z_base, z_update = z_squeezed[:, :z_squeezed.shape[1]//2], z_squeezed[:, z_squeezed.shape[1]//2:]
            
            base_cond_2 = self.base_cond_net(z_update)
            non_spatial_param_2 = self.non_spatial_cond_net(base_cond_2)
            spatial_param_2 = self.spatial_cond_net(base_cond_2)
            x_base = self.cond_schur_transform.inverse_transform(z_base, non_spatial_param_2, spatial_param_2)

            base_cond = self.base_cond_net(x_base)
            non_spatial_param = self.non_spatial_cond_net(base_cond)
            spatial_param = self.spatial_cond_net(base_cond)
            x_update = self.cond_schur_transform.inverse_transform(z_update, non_spatial_param, spatial_param)
            
            x_squeezed = torch.concat([x_base, x_update], axis=1)
            x = self.squeeze_layer.inverse(x_squeezed)
            return x 

    def forward(self, x, dequantize=True):
        if dequantize: x = self.dequantize(x)
        z, logdet = self.transform(x)
        log_pdf_z = self.compute_normal_log_pdf(z)
        log_pdf_x = log_pdf_z + logdet
        return z, x, log_pdf_z, log_pdf_x

from DataLoaders.MNIST.MNISTLoader import DataLoader
# from DataLoaders.CelebA.CelebA32Loader import DataLoader
train_data_loader = DataLoader(batch_size=10)
train_data_loader.setup('Training', randomized=True, verbose=True)
_, _, example_batch = next(train_data_loader) 
example_input = helper.cuda(torch.from_numpy(example_batch['Image']))

c_in=train_data_loader.image_size[1]
n_in=train_data_loader.image_size[3]
flow_net = GenerativeSchurFlow(c_in, n_in)
flow_net.set_actnorm_parameters(train_data_loader, setup_mode='Training', n_batches=500, test_normalization=True)

n_param = 0
for name, e in flow_net.named_parameters():
    print(name, e.requires_grad, e.shape)
    n_param += np.prod(e.shape)
print('Total number of parameters: ' + str(n_param))

example_out, logdet_computed = flow_net.transform(example_input)
example_input_rec = flow_net.inverse_transform(example_out)
print(torch.abs(example_input-example_input_rec).max())

z, x, log_pdf_z, log_pdf_x = flow_net(example_input)


J, J_flat = flow_net.jacobian(example_input)
det_sign, logdet_desired_np = np.linalg.slogdet(J_flat)

example_out, logdet_computed = flow_net(example_input)
logdet_computed_np = helper.to_numpy(logdet_computed)

logdet_desired_error = np.abs(logdet_desired_np-logdet_computed_np).max()
print("Desired Logdet: \n", logdet_desired_np)
print("Computed Logdet: \n", logdet_computed_np)
print('Logdet error:' + str(logdet_desired_error))

trace()

























# class net1(torch.nn.Module):
#     def __init__(self):
#         super(net1, self).__init__()
#         self.seq = torch.nn.Sequential(
#                         torch.nn.Conv2d(1,20,5),
#                          torch.nn.ReLU(),
#                           torch.nn.Conv2d(20,64,5),
#                        torch.nn.ReLU()
#                        )   

#     def forward(self, x):
#         return self.seq(x)

#     #Note: the same result can be obtained by using the for loop as follows
#     #def forward(self, x):
#     #    for s in self.seq:
#     #        x = s(x)
#     #    return x


# net = net1()
# n_param = 0
# for name, e in net.named_parameters():
#     print(name, e.requires_grad, e.shape)
#     n_param += np.prod(e.shape)
# print('Total number of parameters: ' + str(n_param))

# trace()

