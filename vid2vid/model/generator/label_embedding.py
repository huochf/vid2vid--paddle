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

from vid2vid.model.model_lib import Upsample
from vid2vid.model.layers.conv import HyperConv2dBlock
import vid2vid.utils.data as data_utils
import vid2vid.utils.misc as misc_utils


class LabelEmbedding(dg.Layer):
    """
    Embed the input label map to get embedded features.

    Args:
        emb_cfg: Embed network configuration.
        num_input_channels (int): Number of input channels.
        num_hyper_layers (int): Number of hyper layers.
    """
    def __init__(self, gen_cfg, emb_cfg, data_cfg, in_channels=-1, num_hyper_layers=0):
        super().__init__()
        num_input_channels = data_utils.get_paired_input_label_channel_number(data_cfg)
        if num_input_channels == 0:
            num_input_channels = getattr(data_cfg, 'label_channels', 1)
        elif misc_utils.get_nested_attr(data_cfg, 'for_pose_dataset.pose_type', 'both') == 'open':
            num_input_channels -= 3
        
        if in_channels != -1:
            num_input_channels = in_channels
        
        hyper_cfg = gen_cfg.hyper # 4
        if num_hyper_layers != -1:
            self.num_hyper_layers = num_hyper_layers
        else:
            self.num_hyper_layers = num_hyper_layers = hyper_cfg.num_hyper_layers

        num_filters = getattr(emb_cfg, 'num_filters', 32) # 32
        max_num_filters = getattr(emb_cfg, 'max_num_filters', 1024) # 1024
        self.arch = getattr(emb_cfg, 'arch', 'encoderdecoder') # encoderdecoder
        self.num_downsamples = num_downsamples = getattr(emb_cfg, 'num_downsamples', 5) # 5
        kernel_size = getattr(emb_cfg, 'kernel_size', 3) # 3
        weight_norm_type = getattr(emb_cfg, 'weight_norm_type', 'spectral')
        activation_norm_type = getattr(emb_cfg, 'activation_norm_type', 'none')

        self.unet = 'unet' in self.arch
        self.has_decoder = 'decoder' in self.arch or self.unet
        self.num_hyper_layers = num_hyper_layers if num_hyper_layers != -1 else num_downsamples

        base_conv_block = partial(HyperConv2dBlock, kernel_size=kernel_size, padding=(kernel_size // 2),
            weight_norm_type=weight_norm_type, activation_norm_type=activation_norm_type, nonlinearity='leakyrelu')
        
        ch = [min(max_num_filters, num_filters * (2 ** i)) for i in range(num_downsamples + 1)]

        self.conv_first = base_conv_block(num_input_channels, num_filters, activation_norm_type='none')

        # Downsample.
        for i in range(num_downsamples):
            is_hyper_conv = (i < num_hyper_layers) and not self.has_decoder
            self.add_sublayer('down_%d' % i, base_conv_block(ch[i], ch[i + 1], stride=2, is_hyper_conv=is_hyper_conv))
        
        # Upsample
        if self.has_decoder:
            self.upsample = Upsample(scale=2)
            for i in reversed(range(num_downsamples)):
                ch_i = ch[i + 1] * (2 if self.unet and i != num_downsamples - 1 else 1)
                self.add_sublayer('up_%d' % i, base_conv_block(ch_i, ch[i], is_hyper_conv=(i < num_hyper_layers)))

    
    def forward(self, input, weights=None):
        """
        Embedding network forward.
        """
        if input is None:
            return None
        output = [self.conv_first(input)]

        for i in range(self.num_downsamples): # 0 -> 4
            layer = getattr(self, 'down_%d' % i)
            # For hyper networks, the hyper layers are at the last few layers
            # of the decoder (if the network has a decoder). Otherwise, the hyper
            # layers will be at the first few layers of the network.
            if i >= self.num_hyper_layers or self.has_decoder:
                conv = layer(output[-1])
            else:
                conv = layer(output[-1], conv_weights=weights[i])
            # We will use outputs from different layers as input to different
            # SPADE layers in the main branch.
            output.append(conv)
        
        if not self.has_decoder:
            return output # 0 -> 4
        
        # If the nerwork has a decoder, will use outputs from the decoder
        # layers instead of the encoding layers. 
        if not self.unet:
            output = [output[-1]]
        
        for i in reversed(range(self.num_downsamples)): # 4 -> 0
            input_i = output[-1]

            if self.unet and i != self.num_downsamples - 1:
                input_i = L.concat([input_i, output[i + 1]], 1)

            input_i = self.upsample(input_i)
            layer = getattr(self, 'up_%d' % i)
            # The last few layers will be hyper layers if necessary
            if i >= self.num_hyper_layers:
                conv = layer(input_i)
            else:
                conv = layer(input_i, conv_weights=weights[i])
            output.append(conv)

        if self.unet:
            output = output[self.num_downsamples:]
        return output[::-1]
