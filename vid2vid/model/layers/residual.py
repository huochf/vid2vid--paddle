# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------
import functools

import paddle.fluid.dygraph as dg

from .conv import Conv2dBlock, HyperConv2dBlock, LinearBlock, PartialConv2dBlock


class _BaseResBlock(dg.Layer):

    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, groups, bias,
        padding_mode, weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params,
        skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity,
        apply_noise, hidden_channels_equal_out_channels, order, block, learn_shortcut):
        super().__init__()
        if order == 'pre_act':
            order = 'NACNAC'
        if isinstance(bias, bool):
            # The bias for conv_block_0, conv_block_1, and conv_block_s.
            biases = [bias, bias, bias]
        elif isinstance(bias, list):
            if len(bias) == 3:
                biases = bias
            else:
                raise ValueError("Bias list must be 3.")
        else:
            raise ValueError('Bias must either an integer or a list.')
        
        self.learn_shortcut = (in_channels != out_channels) or learn_shortcut

        if len(order) > 6 or len(order) < 5:
            raise ValueError('order must be either 5 or 5 charancters')
        
        if hidden_channels_equal_out_channels:
            hidden_channels = out_channels
        else:
            hidden_channels = min(in_channels, out_channels)
        
        # Parameters that are specific for convolutions.
        conv_main_params = {}
        conv_skip_params = {}
        if block != LinearBlock:
            conv_base_params = dict(stride=1, dilation=dilation, groups=groups, padding_mode=padding_mode)
            conv_main_params.update(conv_base_params)
            conv_main_params.update(dict(kernel_size=kernel_size, padding=padding,
                activation_norm_type=activation_norm_type, activation_norm_params=activation_norm_params,))
            conv_skip_params.update(conv_base_params)
            conv_skip_params.update(dict(kernel_size=1))
            if skip_activation_norm:
                conv_skip_params.update(dict(activation_norm_type=activation_norm_type,
                    activation_norm_params=activation_norm_params))
        
        # Other parameters.
        other_params = dict(weight_norm_type=weight_norm_type,
            weight_norm_params=weight_norm_params, apply_noise=apply_noise)
        
        # Residual branch
        if order.find('A') < order.find('C') and \
            (activation_norm_type == '' or activation_norm_type == 'none'):
            # Nonlinearity is the first operation in the residual path.
            # In-place nonlinearity will modify the input variable and cause backward error.
            first_inplace=False
        else:
            first_inplace=inplace_nonlinearity
        self.conv_block_0 = block(in_channels, hidden_channels, bias=biases[0],
            nonlinearity=nonlinearity, order=order[0:3], inplace_nonlinearity=first_inplace, **conv_main_params, **other_params)
        self.conv_block_1 = block(hidden_channels, out_channels, bias=biases[1], 
            nonlinearity=nonlinearity, order=order[3:], inplace_nonlinearity=inplace_nonlinearity, **conv_main_params, **other_params)
        
        # Shortcut branch
        if self.learn_shortcut:
            if skip_nonlinearity:
                skip_nonlinearity_type = nonlinearity
            else:
                skip_nonlinearity_type = ''
            self.conv_block_s = block(in_channels, out_channels, bias=biases[2],
                nonlinearity=skip_nonlinearity_type, order=order[0:3], **conv_skip_params, **other_params)
            
        # Whether this block expects conditional inputs.
        self.conditional = getattr(self.conv_block_0, 'conditional', False) or \
            getattr(self.conv_block_1, 'conditional', False)
    

    def conv_blocks(self, x, *cond_inputs, **kw_cond_inputs):
        """
        Returns the output of the residual branch.

        Args:
            x: input tensor,
            cond_inputs (list of tensors): conditional input tensors
            kw_cond_inputs (dict):keyword conditional inputs
        """
        dx = self.conv_block_0(x, *cond_inputs, **kw_cond_inputs)
        dx = self.conv_block_1(dx, *cond_inputs, **kw_cond_inputs)
        return dx
    

    def forward(self, x, *cond_inputs, do_checkpoint=False, **kw_cond_inputs):
        if do_checkpoint:
            raise NotImplementedError()
        else:
            dx = self.conv_blocks(x, *cond_inputs, **kw_cond_inputs)
        
        if self.learn_shortcut:
            x_shortcut = self.conv_block_s(x, *cond_inputs, **kw_cond_inputs)
        else:
            x_shortcut = x
        
        output = x_shortcut + dx
        return output


class ResLinearBlock(_BaseResBlock):
    """
    Residual block with full-connected layers.
    """
    def __init__(self, in_channels, out_channels, bias=True, weight_norm_type='none',
        weight_norm_params=None, activation_norm_type='none', activation_norm_params=None,
        skip_activation_norm=True, skip_nonlinearity=False, nonlinearity='leakyrelu', inplace_nonlinearity=False,
        apply_noise=False, hidden_channels_equal_out_channels=False, order='CNACNA', learn_shortcut=False):
        super().__init__(in_channels, out_channels, None, None, None, None, None, bias, None,
            weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params,
            skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, 
            apply_noise, hidden_channels_equal_out_channels, order, LinearBlock, learn_shortcut)


class Res2dBlock(_BaseResBlock):
    """
    Residual block for 2D input. 
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, groups=1, bias=True,
        padding_mode='zeros', weight_norm_type='none', weight_norm_params=None, 
        activation_norm_type='none', activation_norm_params=None, 
        skip_activation_norm=True, skip_nonlinearity=False, nonlinearity='leakyrelu',
        inplace_nonlinearity=False, apply_noise=False, hidden_channels_equal_out_channels=False, order='CNACNA', learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding, dilation, groups, bias, padding_mode, 
            weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params,
            skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity,
            apply_noise, hidden_channels_equal_out_channels, order, Conv2dBlock, learn_shortcut)


class _BaseHyperResBlock(_BaseResBlock):
    """
    An abstract class for hyper residual blocks.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, groups, bias, padding_mode,
        weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params,
        skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise,
        hidden_channels_equal_out_channels, order, is_hyper_conv, is_hyper_norm, block, learn_shortcut):
        block = functools.partial(block, is_hyper_conv=is_hyper_conv, is_hyper_norm=is_hyper_norm)
        super().__init__(in_channels, out_channels, kernel_size, padding, dilation, groups, bias, padding_mode,
            weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params,
            skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity,
            apply_noise, hidden_channels_equal_out_channels, order, block, learn_shortcut)
    

    def forward(self, x, *cond_inputs, conv_weights=(None,) * 3, norm_weights=(None,) * 3, **kw_cond_inputs):
        dx = self.conv_block_0(x, *cond_inputs, conv_weights=conv_weights[0], norm_weights=norm_weights[0])
        dx = self.conv_block_1(dx, *cond_inputs, conv_weights=conv_weights[1], norm_weights=norm_weights[1])

        if self.learn_shortcut:
            x_shortcut = self.conv_block_s(x, *cond_inputs, conv_weights=conv_weights[2], norm_weights=norm_weights[2])
        else:
            x_shortcut = x
        
        output = x_shortcut + dx
        return output


class HyperRes2dBlock(_BaseHyperResBlock):
    """
    Hyper residual block for 2D input.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros',
        weight_norm_type='', weight_norm_params=None, activation_norm_type='', activation_norm_params=None,
        skip_activation_norm=True, skip_nonlinearity=False, nonlinearity='leakyrelu', inplace_nonlinearity=False,
        apply_noise=False, hidden_channels_equal_out_channels=False, order='CNACNA', is_hyper_conv=False, is_hyper_norm=False, learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding, dilation, groups, bias, padding_mode,
            weight_norm_type, weight_norm_params, activation_norm_type, activation_norm_params,
            skip_activation_norm, skip_nonlinearity, nonlinearity, inplace_nonlinearity, apply_noise,
            hidden_channels_equal_out_channels, order, is_hyper_conv, is_hyper_norm, HyperConv2dBlock, learn_shortcut)
















