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

from .attention_module import AttentionModule
from vid2vid.model.layers.conv import Conv2dBlock
import vid2vid.utils.data as data_utils
import vid2vid.utils.misc as misc_utils

class ReferenceEncoder(dg.Layer):

    def __init__(self, gen_cfg, data_cfg):
        super().__init__()

        hyper_cfg = gen_cfg.hyper

        num_filters = gen_cfg.num_filters # 32
        self.max_num_filters = gen_cfg.max_num_filters # 1024
        self.num_downsamples = num_downsamples = gen_cfg.num_downsamples # 5
        self.num_filters_each_layer = num_filters_each_layer = \
            [min(self.max_num_filters, num_filters * (2 ** i)) for i in range(num_downsamples + 2)]

        kernel_size = getattr(gen_cfg.activation_norm_params, 'kernel_size', 1)
        activation_norm_type = getattr(hyper_cfg, 'activation_norm_type', 'instance') # instance
        weight_norm_type = getattr(hyper_cfg, 'weight_norm_type', 'spectral')


        self.concat_ref_label = 'concat' in hyper_cfg.method_to_use_ref_labels
        self.mul_ref_label = 'mul' in hyper_cfg.method_to_use_ref_labels

        num_input_channels = data_utils.get_paired_input_label_channel_number(data_cfg) # 6: densepose + openpose
        if num_input_channels == 0:
            num_input_channels = getattr(data_cfg, 'label_channels', 1)
        elif misc_utils.get_nested_attr(data_cfg, 'for_pose_dataset.pose_type', 'both') == 'open':
            num_input_channels -= 3
        data_cfg.num_input_channels = num_input_channels
        num_img_channels = data_utils.get_paired_input_image_channel_number(data_cfg) # 3
        num_ref_channels = num_img_channels + (num_input_channels if self.concat_ref_label else 0) # 3 for mul_ref

        conv_2d_block = partial(
            Conv2dBlock, kernel_size=kernel_size,
            padding=(kernel_size // 2), weight_norm_type=weight_norm_type,
            activation_norm_type=activation_norm_type,
            nonlinearity='leakyrelu',
        )
        self.ref_img_first = conv_2d_block(num_ref_channels, num_filters)
        if self.mul_ref_label:
            self.ref_label_first = conv_2d_block(num_input_channels, num_filters)

        for i in range(num_downsamples):
            in_ch, out_ch = num_filters_each_layer[i], num_filters_each_layer[i + 1]
            self.add_sublayer('ref_img_down_%d' % i, conv_2d_block(in_ch, out_ch, stride=2))
            self.add_sublayer('ref_img_up_%d' % i, conv_2d_block(out_ch, in_ch))
            if self.mul_ref_label:
                self.add_sublayer('ref_label_down_%d' % i, conv_2d_block(in_ch, out_ch, stride=2))
                self.add_sublayer('ref_label_up_%d' % i, conv_2d_block(out_ch, in_ch))
        
        if hasattr(hyper_cfg, 'attention'):
            self.num_downsample_atn = misc_utils.get_and_setattr(hyper_cfg.attention, 'num_downsamples', 2) # 2
            if data_cfg.initial_few_shot_K > 1:
                self.attention_module = AttentionModule(hyper_cfg, data_cfg, conv_2d_block, num_filters_each_layer)
        else:
            self.num_downsample_atn = 0


    def forward(self, ref_image, ref_label, label, k):
        """
        Encode the reference image to get features for weight generation.

        Args:

            ref_image ((NxK)x3xHxW): Reference images.
            ref_label ((NxK)xCxHxW): Reference labels.
            label (NxCxHxW): Target label.
            k (int): Number of reference images.
        
        Returns: (tuple)
            - x (NxC2xH2xW2): Encoded features from reference images
              for the main branch (as input to the decoder).
            - encoded_ref (list of Variable): Encoded features from reference
              images for the weight generation branch.
            - attention (Nx(KxH1xW1)x(H1xW1)): Attention maps.
            - atn_vis (1x1xH1xW1): Visualization for attention scores.
            - ref_idx (Nx1): Index for which image to use from the
              reference image.
        """
        if self.concat_ref_label:
            # concat reference label map and image together for encoding.
            concat_ref = L.concat([ref_image, ref_label], axis=1)
            x = self.ref_img_first(concat_ref)
        elif self.mul_ref_label:
            x = self.ref_img_first(ref_image)
            x_label = self.ref_label_first(ref_label)
        else:
            x = self.ref_img_first(ref_image)
        
        atn_ref_image = atn_ref_label = None
        atn = atn_vis = ref_idx = None
        for i in range(self.num_downsamples):
            x = getattr(self, 'ref_img_down_' + str(i))(x)
            if self.mul_ref_label:
                x_label = getattr(self, 'ref_label_down_' + str(i))(x_label)
            # Preserve reference for attention module.
            if k > 1 and i == self.num_downsample_atn - 1:
                x, atn, atn_vis = self.attention_module(x, label, ref_label)
                if self.mul_ref_label:
                    x_label, _, _ = self.attention_module(x_label, None, None, atn)
                
                atn_sum = L.reshape(atn, (label.shape[0], k, -1)) # [b, k, h*w*h*w]
                atn_sum = L.reduce_sum(atn_sum, dim=2)
                ref_idx = L.argmax(atn_sum, axis=1)

        # Get all corresponding layers in the encoder output for generating
        # weights in corresponding layers.
        encoded_image_ref = [x]
        if self.mul_ref_label:
            encoded_ref_label = [x_label]
        
        for i in reversed(range(self.num_downsamples)): # 4 -> 0
            conv = getattr(self, 'ref_img_up_' + str(i))(encoded_image_ref[-1])
            encoded_image_ref.append(conv)
            if self.mul_ref_label:
                conv_label = getattr(self, 'ref_label_up_' + str(i))(encoded_ref_label[-1])
                encoded_ref_label.append(conv_label)
        
        if self.mul_ref_label:
            encoded_ref = []
            for i in range(len(encoded_image_ref)):
                conv, conv_label = encoded_image_ref[i], encoded_ref_label[i]
                b, c, h, w = conv.shape
                conv_label = L.softmax(conv_label, axis=1)
                conv_label = L.reshape(conv_label, (b, 1, c, h * w))
                # conv_label = L.expand(conv_label, (1, c, 1, 1))
                conv = L.reshape(conv, (b, c, 1, h * w))
                # conv = L.expand(conv, (1, 1, c, 1))
                conv_prod = conv * conv_label # (b, c, c, h * w)
                conv_prod = L.reduce_sum(conv_prod, dim=3, keep_dim=True) # (b, c, c, 1)
                encoded_ref.append(conv_prod)
        else:
            encoded_ref = encoded_image_ref
        
        encoded_ref = encoded_ref[::-1] # level0 -> level4
        return x, encoded_ref, atn, atn_vis, ref_idx
