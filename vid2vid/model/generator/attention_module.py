# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------

import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg


class AttentionModule(dg.Layer):

    def __init__(self, atn_cfg, data_cfg, conv_2d_block, num_filters_each_layer):
        super().__init__()
        self.initial_few_shot_K = data_cfg.initial_few_shot_K # 1
        num_input_channels = data_cfg.num_input_channels # 6 for densepose and openpose
        num_filters = getattr(atn_cfg, 'num_filters', 32) # 32

        self.num_downsample_atn = getattr(atn_cfg, 'num_downsamples', 2) # 2
        self.atn_query_first = conv_2d_block(num_input_channels, num_filters)
        self.atn_key_first = conv_2d_block(num_input_channels, num_filters)
        for i in range(self.num_downsample_atn): # 0 -> 1
            f_in, f_out = num_filters_each_layer[i], num_filters_each_layer[i + 1]
            self.add_sublayer('atn_key_%d' % i, conv_2d_block(f_in, f_out, stride=2))
            self.add_sublayer('atn_query_%d' % i, conv_2d_block(f_in, f_out, stride=2))
    

    def forward(self, in_features, label, ref_label, attention=None):
        """
        Get the attention map to combine multiple image features in the
        case of multiple reference images.

        Args:   
            in_features ((NxK)xC1xH1xW1): input features.
            label (NxC2xH2xW2): Target label.
            ref_label (NxC2xH2xW2): Reference label.
            attention (Nx(KxH1xW1)x(H1xW1)): attention maps
        Returns:
            out_features (NxC1xH1xW1): attention-combined features
            attention (Nx(KxH1xW1)x(H1xW1)): attention maps
            atn_vis (1x1xH1xW1): Visualization for attention scores
        """

        b, c, h, w = in_features.shape
        k = self.initial_few_shot_K # 1
        b = b // k

        if attention is None:
            atn_key = self.attention_encode(ref_label, 'atn_key')
            atn_query = self.attention_encode(label, 'atn_query')

            atn_key = L.reshape(atn_key, (b, k, c, -1))
            atn_key = L.transpose(atn_key, (0, 1, 3, 2)) # [b, k, h*w, c]
            atn_key = L.reshape(atn_key, (b, -1, c))     # [b, k*h*w, c]
            atn_query = L.reshape(atn_query, (b, c, -1)) # [b, c, h*w]
            energy = L.matmul(atn_key, atn_query)        # [b, k*h*w, h*w]
            attention = L.softmax(energy, axis=1)

        in_features = L.reshape(in_features, (b, k, c, h * w))
        in_features = L.transpose(in_features, (0, 2, 1, 3)) # [b, c, k, h*w]
        in_features = L.reshape(in_features, (b, c, -1))     # [b, c, k*h*w]
        out_features = L.matmul(in_features, attention)   # [b, c, h*w]
        out_features = L.reshape(out_features, (b, c, h, w))

        atn_vis = L.reshape(attention, (b, k, h * w, h * w))
        atn_vis = L.reduce_sum(atn_vis, dim=2) # [b, k, h * w]

        return out_features, attention, atn_vis[-1:, 0:1]


    def attention_encode(self, img, net_name):
        """
        Encode the input image to get the attention map.

        Args:
            img (NxCxHxW): input image.
            net_name (str): name for attention network
        Returns:
            x (NxC2xH2xW2): Encoded feature
        """
        x = getattr(self, net_name + '_first')(img)
        for i in range(self.num_downsample_atn):
            x = getattr(self, net_name + '_' + str(i))(x)
        return x
