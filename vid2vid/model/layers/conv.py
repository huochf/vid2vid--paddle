# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------
from types import SimpleNamespace
import numpy as np

import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.nn as nn
import paddle.fluid.dygraph as dg

import vid2vid.model.layers.misc as misc_layer

class _BaseConvBlock(dg.Layer):
    """
    An abstract wrapper class the wraps convolution or linear layer
    with normalization and nonlinearity
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation,
                 groups, bias, padding_mode, weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params, nonlinearity,
                 inplace_nonlinearity, apply_noise, order, input_dim):
        super().__init__()
        from .nonlinearity import get_nonlinearity_layer
        from .weight_norm import get_weight_norm_layer
        from .activation_norm import get_activation_norm_layer

        self.weight_norm_type = weight_norm_type

        # Convolutional layer.
        if weight_norm_params is None:
            weight_norm_params = SimpleNamespace()
        weight_norm = get_weight_norm_layer(weight_norm_type, **vars(weight_norm_params))
        conv_layer = weight_norm(self._get_conv_layer(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias, padding_mode, input_dim))
        
        # Noise injection layer.
        noise_layer = misc_layer.ApplyNoise() if apply_noise else None

        # Normalization layer.
        conv_before_norm = order.find('C') < order.find('N')
        norm_channels = out_channels if conv_before_norm else in_channels
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace()
        activation_norm_layer = get_activation_norm_layer(
            norm_channels, activation_norm_type, input_dim, **vars(activation_norm_params))
        
        # Nonlinearity layer.
        nonlinearity_layer = get_nonlinearity_layer(nonlinearity, inplace=inplace_nonlinearity)

        # Mapping from operation names to layers.
        mappings = {'C': ('conv', conv_layer),
                    'N': ('norm', activation_norm_layer),
                    'A': ('nonlinearity', nonlinearity_layer)}
        
        # All layers in order.
        self.layers = []
        for op in order:
            if mappings[op][1] is not None:
                self.add_sublayer(mappings[op][0], mappings[op][1])
                self.layers.append((mappings[op][0], mappings[op][1]))
                if op == 'C' and noise_layer is not None:
                    # Inject noise after convolution.
                    self.add_sublayer('noise', noise_layer)
                    self.layers.append(('noise', noise_layer))
        
        # Whether this block expects conditional inputs.
        self.conditional = getattr(conv_layer, 'conditional', False) or \
                           getattr(activation_norm_layer, 'conditional', False)
    

    def forward(self, x, *cond_inputs, **kw_cond_inputs):
        for _, layer in self.layers:
            if getattr(layer, 'conditional', False):
                # Layers that require conditional inputs.
                x = layer(x, *cond_inputs, **kw_cond_inputs)
            else:
                x = layer(x)
        return x


    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding,
                        dilation, groups, bias, padding_mode, input_dim):
        # Returns the convolutional layer.
        if input_dim == 0:
            layer = dg.Linear(in_channels, out_channels, bias_attr=bias)
        else:
            layer_type = getattr(dg, 'Conv%dD' % input_dim)
            layer = layer_type(num_channels=in_channels, num_filters=out_channels, 
                filter_size=kernel_size, stride=stride, padding=padding, dilation=dilation, 
                groups=groups, bias_attr=bias, )
        return layer
    

    def __repr__(self):
        main_str = self._get_name() + '('
        child_lines = []
        for name, layer in self.layers:
            mod_str = repr(layer)
            if name == 'conv' and self.weight_norm_type != 'none' and self.weight_norm_type != '':
                mod_str = mod_str[:-1] + ', weight_norm={}'.format(self.weight_norm_type) + ')'
            mod_str = self._addindent(mod_str, 2)
            child_lines.append(mod_str)
        if len(child_lines) == 1:
            main_str += child_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(child_lines) + '\n'
        
        main_str += ')'
        return main_str
    

    @staticmethod
    def _addindent(s_, numSpaces):
        s = s_.split('\n')
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * ' ') + line for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s
 
                
class LinearBlock(_BaseConvBlock):
    """
    A Wrapper class that wraps Linear with normalization and nonlinearity.
    """
    def __init__(self, in_features, out_features, bias=None,
        weight_norm_type='none', weight_norm_params=None,
        activation_norm_type='none', activation_norm_params=None,
        nonlinearity='none', inplace_nonlinearity=False,
        apply_noise=False, order='CNA'):
        super().__init__(in_features, out_features, None, None, None, None, None,
            bias, None, weight_norm_type, weight_norm_params, activation_norm_type,
            activation_norm_params, nonlinearity, inplace_nonlinearity, apply_noise, order, 0)


class Conv2dBlock(_BaseConvBlock):
    """
    A Wrapper class that wraps Conv2D with normalization and nonlinearity
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=None, padding_mode='zeros', weight_norm_type='none', 
        weight_norm_params=None, activation_norm_type='none', activation_norm_params=None,
        nonlinearity='none', inplace_nonlinearity=False, apply_noise=False, order='CNA'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
            bias, padding_mode, weight_norm_type, weight_norm_params, activation_norm_type,
            activation_norm_params, nonlinearity, inplace_nonlinearity, apply_noise, order, 2)


class _BaseHyperConvBlock(_BaseConvBlock):
    """
    An abstract wrapper class that wraps a hyper convolutional layer
    with normalization and nonlinearity.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
        padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params,
        nonlinearity, inplace_nonlinearity, apply_noise, is_hyper_conv, is_hyper_norm, order, input_dim):

        self.is_hyper_conv = is_hyper_conv
        if is_hyper_conv:
            weight_norm_type = 'none'
        if is_hyper_norm:
            activation_norm_type= 'hyper_' + activation_norm_type
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                         padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity, apply_noise, order, input_dim)
    

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, dilation,
        groups, bias, padding_mode, input_dim):
        if input_dim == 0:
            raise ValueError("HyperLinearBlock is not suppprted.")
        else:
            name = 'HyperConv' if self.is_hyper_conv else 'dg.Conv'
            layer_type = eval(name + '%dD' % input_dim)
            layer = layer_type(in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, bias_attr=bias)
            return layer


class HyperConv2dBlock(_BaseHyperConvBlock):
    """
    A Wrapper class that wraps HyperConv2D with normalization and nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None,
        padding_mode='zeros', weight_norm_type='none', weight_norm_params=None,
        activation_norm_type='none', activation_norm_params=None, is_hyper_conv=False, is_hyper_norm=False, 
        nonlinearity='none', inplace_nonlinearity=False, apply_noise=False, order='CNA'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
            padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params,
            nonlinearity, inplace_nonlinearity, apply_noise, is_hyper_conv, is_hyper_norm, order, 2)


class HyperConv2D(dg.Layer):
    """
    Hyper Conv2D initialization.
    """
    def __init__(self, in_channels=0, out_channels=0, kernel_size=3, stride=1, padding=1,
        dilation=1, groups=1, bias_attr=None, padding_mode='zeros'):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias_attr
        self.padding_mode = padding_mode
        self.conditional = True
    

    def forward(self, x, *args, conv_weights=(None, None), **kwargs):
        """
        Hyper Conv2D forward, Convolve x using the provided weight and bias.

        Args:
            x (N x C x H x W): input
            conv_weights (N x C2 x C1 x k x k): convolution weights or [weight, bias].
        
        Returns:
            y (N x C2 x H x W): output
        """
        if conv_weights is None:
            conv_weight, conv_bias = None, None
        elif isinstance(conv_weights, F.Variable):
            conv_weight, conv_bias = conv_weights, None
        else:
            conv_weight, conv_bias = conv_weights
        
        if conv_weight is None:
            return x
        if conv_bias is None:
            if self.use_bias:
                raise ValueError("bias not provided but set to true during initialization")
            conv_bias = [None] * x.shape[0]
        
        if self.padding_mode != 'zeros':
            x = L.pad2d(x, [self.padding] * 4, mode=self.padding_mode)
            padding = 0
        else:
            padding = self.padding
        
        y = None
        for i in range(x.shape[0]):
            if self.stride >= 1:
                yi = nn.functional.conv2d(x[i : i+1], weight=conv_weight[i], bias=conv_bias[i],
                    stride=self.stride, padding=padding, dilation=self.dilation, groups=self.groups)
            else:
                yi = nn.functional.conv2d_transpose(x[i : i+1], weight=conv_weight[i], bias=conv_bias[i],
                    stride=int(1 / self.stride), dilation=self.dilation, output_padding=self.padding, groups=self.groups)
            y = L.concat([y, yi]) if y is not None else yi
        
        return y
    

class _BasePartialConvBlock(_BaseConvBlock):
    """
    An abstract wrapper class that wraps a partial convolutional layer
    with normalization and nonlinearity.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
        padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params,
        nonlinearity, inplace_nonlinearity, multi_channel, return_mask, apply_noise, order, input_dim):
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        self.partial_conv = True
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
            padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params,
            nonlinearity, inplace_nonlinearity, apply_noise, order, input_dim)
    

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, dilation,
        groups, bias, padding_mode, input_dim):
        if input_dim == 2:
            layer_type = PartialConv2D
        elif input_dim == 3:
            raise NotImplementedError()
        else:
            raise ValueError("Partial conv only supports 2D and 3D conv now.")
        layer = layer_type(in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode, multi_channel=self.multi_channel, return_max=self.return_mask)
        return layer
    

    def forward(self, x, *cond_inputs, mask_in=None, **kw_cond_inputs):
        """
        Args:
            x: input
            cond_inputs (list): conditional input
            mask_in (tensor): it masks the valid input region.
            kw_cond_inputs (dict): keyword conditional inputs.
        Returns:
            x: output
            mask_out: masks the valid output region.
        """
        mask_out = None
        for _, layer in self.layers:
            if getattr(layer, 'conditional', False):
                # Layers that require conditional inputs.
                x = layer(x, *cond_inputs, **kw_cond_inputs)
            elif getattr(layer, 'partial_conv', False):
                x = layer(x, mask_in=mask_in, **kw_cond_inputs)
                if type(x) == tuple:
                    x, mask_out = x
            else:
                x = layer(x)
        if mask_out is not None:
            return x, mask_out

        return x


class PartialConv2dBlock(_BasePartialConvBlock):
    """
    A Wrapper class that wraps ``PartialConv2D`` with normalization and nonlinearity.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None,
        padding_mode='zeros', weight_norm_type='none', weight_norm_params=None,
        activation_norm_type='none', activation_norm_params=None, nonlinearity='none', inplace_nonlinearity=False,
        multi_channel=False, return_mask=True, apply_noise=False, order='CNA'):
        super.__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
            padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params,
            nonlinearity, inplace_nonlinearity, multi_channel, return_mask, apply_noise, order, 2)


###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################
class PartialConv2D(dg.Layer):
    """
    Partial 2D convolution in
    "Image inpainting for irregular holes using partial convolutions."
    Liu et al., ECCV 2018
    """
    def __init__(self, *args, multi_channel=False, return_mask=True, **kwargs):
        # whether the mask is multi-channel or not
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        super(PartialConv2D, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = L.ones((self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
        else:
            self.weight_maskUpdater = L.ones((1, 1, self.kernel_size[0], self.kernel_size[1]))
        
        shape = self.weight_maskUpdater.shape
        self.slide_winsize = shape[1] * shape[2] * shape[3]
        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None
        self.partial_conv = True
    

    def forward(self, x, mask_in=None):
        assert len(x.shape) == 4

        if mask_in is not None or self.last_size != tuple(x.shape):
            self.last_size = tuple(x.shape)

            with dg.no_grad():
                if self.weight_maskUpdater.dtype != x.dtype:
                    self.weight_maskUpdater = self.weight_maskUpdater.astype(x.dtype)
                
                if mask_in is None:
                    # If mask is not provided, create a mask.
                    if self.multi_channel:
                        mask = L.ones(x.shape, dtype=x.dtype)
                    else:
                        mask = L.ones((1, 1, x.shape[2], x.shape[3]), dtype=x.dtype)
                else:
                    mask = mask_in
                
                self.update_mask = nn.functional.conv2d(mask, self.weight_maskUpdater, bias=None,
                    stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
                # For mixed precision training, eps from 1e-8 ~ 1e-6
                eps = 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + eps)
                self.update_mask = L.clamp(self.update_mask, 0, 1)
                self.mask_ratio = self.mask_ratio * self.update_mask
        
        raw_out = super(PartialConv2D, self).forward(x * mask if mask_in is not None else x)

        if self.bias is not None:
            bias_view = L.reshape(self.bias, (1, self.out_channels, 1, 1))
            output = (raw_out - bias_view) * self.mask_ratio + bias_view
            output = output * self.update_mask
        else:
            output = raw_out * self.mask_ratio
        
        if self.return_mask:
            return output, self.update_mask
        else:
            return output































