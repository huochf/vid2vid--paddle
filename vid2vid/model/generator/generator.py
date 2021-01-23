# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------
from functools import partial

import paddle.fluid.layers as L 
import paddle.fluid.dygraph as dg 

from vid2vid.model.layers.residual import HyperRes2dBlock
from vid2vid.model.layers.conv import Conv2dBlock
import vid2vid.utils.data as data_utils
import vid2vid.utils.misc as misc_utils


class Generator(dg.Layer):
    """
    Few-shot vid2vid generator constructor.
    """
    def __init__(self, gen_cfg, data_cfg):
        super().__init__()
        self.gen_cfg = gen_cfg
        self.data_cfg = data_cfg
        self.num_frames_G = data_cfg.num_frames_G # 2
        self.flow_cfg = flow_cfg = gen_cfg.flow

        num_img_channels = data_utils.get_paired_input_image_channel_number(data_cfg)
        self.num_downsamples = num_downsamples = misc_utils.get_and_setattr(gen_cfg, 'num_downsamples', 5)
        conv_kernel_size = misc_utils.get_and_setattr(gen_cfg, 'kernel_size', 3)
        num_filters = misc_utils.get_and_setattr(gen_cfg, 'num_filters', 32)
        max_num_filters = misc_utils.get_and_setattr(gen_cfg, 'max_num_filters', 1024)
        self.max_num_filters = gen_cfg.max_num_filters = min(max_num_filters, num_filters * (2 ** num_downsamples))

        # Get number of filters at each layer in the main branch
        num_filters_each_layer = [min(self.max_num_filters, num_filters * (2 ** i)) for i in range(num_downsamples + 2)]

        # Hyper normalization / convolution.
        hyper_cfg = gen_cfg.hyper
        # Use adaptive weight generation for SPADE
        self.use_hyper_spade = hyper_cfg.is_hyper_spade # True
        # Use adaptive for convolutional layers in the main branch.
        self.use_hyper_conv = hyper_cfg.is_hyper_conv # True
        # Number of hyper layers.
        self.num_hyper_layers = getattr(hyper_cfg, 'num_hyper_layers', 4)
        if self.num_hyper_layers == -1:
            self.num_hyper_layers = num_downsamples
        gen_cfg.hyper.num_hyper_layers = self.num_hyper_layers

        # Number of layers to perform multi-spade combine. 
        self.num_multi_spade_layers = getattr(flow_cfg.multi_spade_combine, 'num_layers', 3)

        # Whether to generate raw output for additional losses.
        self.generate_raw_output = getattr(flow_cfg, 'generate_raw_output', False)

        # Main branch image generation.
        padding = conv_kernel_size // 2
        activation_norm_type = misc_utils.get_and_setattr(gen_cfg, 'activation_norm_type', 'sync_batch')
        weight_norm_type = misc_utils.get_and_setattr(gen_cfg, 'weight_norm_type', 'spectral')
        activation_norm_params = misc_utils.get_and_setattr(gen_cfg, 'activation_norm_params', None) # spatially_adaptive

        spade_in_channels = [] # Input channel size in SPADE module.
        for i in range(num_downsamples + 1):
            spade_in_channels += [[num_filters_each_layer[i]]] \
                if i >= self.num_multi_spade_layers else [[num_filters_each_layer[i]] * 3]
        
        order = getattr(gen_cfg.hyper, 'hyper_block_order', 'NAC')
        for i in reversed(range(num_downsamples + 1)): # 5 -> 0
            activation_norm_params.cond_dims = spade_in_channels[i]
            is_hyper_conv = self.use_hyper_conv and i < self.num_hyper_layers
            is_hyper_norm = self.use_hyper_spade and i < self.num_hyper_layers

            self.add_sublayer('up_%d' % i, HyperRes2dBlock(
                num_filters_each_layer[i + 1], num_filters_each_layer[i], conv_kernel_size, padding=padding,
                weight_norm_type=weight_norm_type, activation_norm_type=activation_norm_type,
                activation_norm_params=activation_norm_params, order=order*2, 
                is_hyper_conv=is_hyper_conv, is_hyper_norm=is_hyper_norm))
        
        self.conv_img = Conv2dBlock(num_filters, num_img_channels, conv_kernel_size, padding=padding,
            nonlinearity='leakyrelu', order='AC')
        self.upsample = partial(L.image_resize, scale=2)


    def forward(self, encoded_ref, encoded_label, encoded_label_raw, conv_weights, norm_weights):
        x = encoded_ref
        x_raw = None
        for i in range(self.num_downsamples, -1, -1): # 5 -> 0
            conv_weight = norm_weight = [None] * 3
            if self.use_hyper_conv and i < self.num_hyper_layers:
                conv_weight = conv_weights[i]
            if self.use_hyper_spade and i < self.num_hyper_layers:
                norm_weight = norm_weights[i]
            
            # Main branch residual blocks.
            x = self.one_up_conv_layer(x, encoded_label, conv_weight, norm_weight, i)

            # For raw output generation.
            if self.generate_raw_output and i < self.num_multi_spade_layers:
                x_raw = self.one_up_conv_layer(x_raw, encoded_label_raw, conv_weight, norm_weight, i)
            else:
                x_raw = x
        
        # Final conv layer. 
        if self.generate_raw_output:
            img_raw = L.tanh(self.conv_img(x_raw))
        else:
            img_raw = None
        img_final = L.tanh(self.conv_img(x))

        return img_final, img_raw


    def one_up_conv_layer(self, x, encoded_label, conv_weight, norm_weight, i):
        """
        One residual block layer in the main branch.
        """
        layer = getattr(self, 'up_' + str(i))
        x = layer(x, *encoded_label[i], conv_weights=conv_weight, norm_weights=norm_weight)

        if i != 0:
            x = self.upsample(x)
        return x

