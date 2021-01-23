# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------
from functools import partial
import copy

import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg

from vid2vid.model.layers.conv import Conv2dBlock
from vid2vid.model.layers.residual import Res2dBlock
from vid2vid.model.model_lib import Upsample
import vid2vid.utils.data as data_utils


class FlowGenerator(dg.Layer):
    """
    Flow generator constructor.
    """
    def __init__(self, flow_cfg, data_cfg, num_frames):
        super().__init__()
        num_input_channels = data_cfg.num_input_channels # 6
        if num_input_channels == 0:
            num_input_channels = 1
        num_prev_img_channels = data_utils.get_paired_input_image_channel_number(data_cfg) # 3
        num_downsamples = getattr(flow_cfg, 'num_downsamples', 3)
        kernel_size = getattr(flow_cfg, 'kernel_size', 3)
        padding = kernel_size // 2
        num_blocks = getattr(flow_cfg, 'num_blocks', 6)
        num_filters = getattr(flow_cfg, 'num_filters', 32)
        max_num_filters = getattr(flow_cfg, 'max_num_filters', 1024)
        num_filters_each_layer = [min(max_num_filters, num_filters * (2 ** i)) for i in range(num_downsamples + 1)]

        self.flow_output_multiplier = getattr(flow_cfg, 'flow_output_multiplier', 20)
        self.sep_up_mask = getattr(flow_cfg, 'sep_up_mask', False)
        activation_norm_type = getattr(flow_cfg, 'activation_norm_type', 'sync_batch')
        weight_norm_type = getattr(flow_cfg, 'weight_norm_type', 'spectral')
        base_conv_block = partial(Conv2dBlock, kernel_size=kernel_size, padding=padding,
            weight_norm_type=weight_norm_type, activation_norm_type=activation_norm_type, nonlinearity='leakyrelu')
        
        num_input_channels = num_input_channels * num_frames + num_prev_img_channels * (num_frames - 1)

        # First layer.
        down_flow = [('0', base_conv_block(num_input_channels, num_filters))]

        # Downsamples.
        for i in range(num_downsamples):
            down_flow += [(str(i + 1), base_conv_block(num_filters_each_layer[i], num_filters_each_layer[i + 1], stride=2))]
        
        # Resnet blocks.
        res_flow = []
        ch = num_filters_each_layer[num_downsamples]
        for i in range(num_blocks):
            res_flow += [(str(i), Res2dBlock(ch, ch, kernel_size, padding=padding, weight_norm_type=weight_norm_type,
                activation_norm_type=activation_norm_type, order='NACNAC'))]
        
        # Upsamples.
        up_flow = []
        for i in reversed(range(num_downsamples)):
            up_flow += [(str((num_downsamples - 1 - i) * 2), Upsample(scale=2)), 
                        (str((num_downsamples - 1 - i) * 2 + 1), base_conv_block(num_filters_each_layer[i + 1], num_filters_each_layer[i]))]

        conv_flow = [('0', Conv2dBlock(num_filters, 2, kernel_size, padding=padding))]
        conv_mask = [('0', Conv2dBlock(num_filters, 1, kernel_size, padding=padding, nonlinearity='sigmoid'))]

        self.down_flow = dg.Sequential(*down_flow)
        self.res_flow = dg.Sequential(*res_flow)
        self.up_flow = dg.Sequential(*up_flow)

        if self.sep_up_mask:
            self.up_mask = dg.Sequential(*copy.deepcopy(up_flow))
        self.conv_flow = dg.Sequential(*conv_flow)
        self.conv_mask = dg.Sequential(*conv_mask)
    

    def forward(self, label, ref_label, ref_image):
        """
        Flow generator forward.

        Args:
            label (4D tensor): Input label tensor.
            ref_label (4D tensor): Reference label tensors.
            ref_image (4D tensor): Reference image tensors.
        Returns:
            - flow (4D tensor): Generated flow map.
            - mask (4D tensor): Generated occlusion mask.
        """
        label_concat = L.concat([label, ref_label, ref_image], 1)
        downsample = self.down_flow(label_concat)
        res = self.res_flow(downsample)
        flow_feat = self.up_flow(res)
        flow = self.conv_flow(flow_feat) * self.flow_output_multiplier

        mask_feat = self.up_mask(res) if self.sep_up_mask else flow_feat
        mask = self.conv_mask(mask_feat)
        return flow, mask
