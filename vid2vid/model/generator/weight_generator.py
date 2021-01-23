# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------
from functools import partial
import numpy as np

import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg

from vid2vid.model.layers.conv import LinearBlock


class WeightGenerator(dg.Layer):

    def __init__(self, gen_cfg, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.embed_cfg = embed_cfg = gen_cfg.embed
        self.embed_arch = embed_cfg.arch # encoderdecoder

        num_filters = gen_cfg.num_filters # 32
        self.max_num_filters = gen_cfg.max_num_filters # 1024
        self.num_downsamples = num_downsamples = gen_cfg.num_downsamples # 5
        self.num_filters_each_layer = num_filters_each_layer = \
            [min(self.max_num_filters, num_filters * (2 ** i)) for i in range(num_downsamples + 2)]

        # Normalization params.
        # Conv kernel size in main branch.
        self.conv_kernel_size = conv_kernel_size = gen_cfg.kernel_size # 3
        # Conv kernel size in embedding branch.
        self.embed_kernel_size = embed_kernel_size = getattr(gen_cfg.embed, 'kernel_size', 3) # 3
        # Conv kernel size in SPADE.
        self.kernel_size = kernel_size = getattr(gen_cfg.activation_norm_params, 'kernel_size', 1) # 1
        # Input channel size in SPADE module.
        self.spade_in_channels = []
        for i in range(num_downsamples + 1):
            self.spade_in_channels += [num_filters_each_layer[i]]


        hyper_cfg = gen_cfg.hyper

        # For reference image encoding.
        # How to utilize the reference label map: concat | mul
        self.mul_ref_label = 'mul' in hyper_cfg.method_to_use_ref_labels
        # Output spatial size for adaptive pooling layer.
        self.sh_fix = self.sw_fix = 32
        # Number of fc layers in weight generation.
        self.num_fc_layers = getattr(hyper_cfg, 'num_fc_layers', 2) # 2

        # Number of hyper layers.
        self.num_hyper_layers = hyper_cfg.num_hyper_layers # 4
        # Use adaptive for convolutional layers in the main branch.
        self.use_hyper_conv = hyper_cfg.is_hyper_conv # False
        # Use adaptive weight generation for SPADE
        self.use_hyper_spade = hyper_cfg.is_hyper_spade # True
        # Use adaptive for label embedding network.
        self.use_hyper_embed = hyper_cfg.is_hyper_embed # True
        # Order of operations in the conv block.
        order = getattr(gen_cfg.hyper, 'hyper_block_order', 'NAC')
        self.conv_before_norm = order.find('C') < order.find('N')

        # Normalization / main branch conv weight generation.
        if self.use_hyper_spade or self.use_hyper_conv:
            for i in range(self.num_hyper_layers):
                ch_in, ch_out = num_filters_each_layer[i], num_filters_each_layer[i + 1]
                conv_ks2 = conv_kernel_size ** 2
                embed_ks2 = embed_kernel_size ** 2
                spade_ks2 = kernel_size ** 2
                spade_in_ch = self.spade_in_channels[i]

                fc_names, fc_ins, fc_outs = [], [], []
                if self.use_hyper_spade:
                    fc0_out = fcs_out = (spade_in_ch * spade_ks2 + 1) * (1 if self.conv_before_norm else 2)
                    fc1_out = (spade_in_ch * spade_ks2 + 1) * (1 if ch_in != ch_out else 2)
                    fc_names += ['fc_spade_0', 'fc_spade_1', 'fc_spade_s']
                    fc_ins += [ch_out] * 3
                    fc_outs += [fc0_out, fc1_out, fcs_out]

                    if self.use_hyper_embed:
                        fc_names += ['fc_spade_e']
                        fc_ins += [ch_out]
                        fc_outs += [ch_in * embed_ks2 + 1]
                
                if self.use_hyper_conv:
                    fc0_out = ch_out * conv_ks2 + 1
                    fc1_out = ch_in * conv_ks2 + 1
                    fcs_out = ch_out + 1
                    fc_names += ['fc_conv_0', 'fc_conv_1', 'fc_conv_s']
                    fc_ins += [ch_in] * 3
                    fc_outs += [fc0_out, fc1_out, fcs_out]
                
                linear_block = partial(LinearBlock, weight_norm_type='spectral', nonlinearity='leakyrelu')

                for n, l in enumerate(fc_names):
                    fc_in = fc_ins[n] if self.mul_ref_label else self.sh_fix * self.sw_fix
                    fc_layer = [('0', linear_block(fc_in, ch_out))]
                    for k in range(1, self.num_fc_layers):
                        fc_layer += [(str(k), linear_block(ch_out, ch_out))]
                    fc_layer += [(str(len(fc_layer)), LinearBlock(ch_out, fc_outs[n], weight_norm_type='spectral'))]
                    self.add_sublayer('%s_%d' % (l, i), dg.Sequential(*fc_layer))

    

    def forward(self, encoded_ref, k, is_first_frame):

        if self.training or is_first_frame or k > 1:
            embedding_weights, norm_weights, conv_weights = [], [], []
            for i in range(self.num_hyper_layers):
                if self.use_hyper_spade:
                    feat = encoded_ref[min(len(encoded_ref) - 1, i + 1)]
                    embedding_weight, norm_weight = self.get_norm_weight(feat, i)
                    embedding_weights.append(embedding_weight)
                    norm_weights.append(norm_weight)
                
                if self.use_hyper_conv:
                    feat = encoded_ref[min(len(encoded_ref) - 1, i)]
                    conv_weights.append(self.get_conv_weight(feat, i))
            
            if not self.training:
                self.embedding_weights, self.conv_weights, self.norm_weights = \
                    embedding_weights, conv_weights, norm_weights
        else:
            embedding_weights, conv_weights, norm_weights = \
                self.embedding_weights, self.conv_weights, self.norm_weights
        

        return embedding_weights, norm_weights, conv_weights
    

    def get_norm_weight(self, x, i):
        """
        Adaptively generate weights for SPADE in layer i of generator.

        Args:
            x (NxCxHxW): input features.
            i (int): Layer index
        Returns:
            embedding_weights (list of tensors): weights for the label embedding network
            norm_weights (list of tensors): weights for the SPADE layers
        """
        if not self.mul_ref_label:
            x = L.adaptive_pool2d(x, pool_size=(self.sh_fix, self.sw_fix), pool_type='avg')
         
        in_ch = self.num_filters_each_layer[i]
        out_ch = self.num_filters_each_layer[i + 1]
        spade_ch = self.spade_in_channels[i]
        eks, sks = self.embed_kernel_size, self.kernel_size

        b = x.shape[0]
        weight_reshaper = WeightReshaper()
        x = weight_reshaper.reshape_embed_input(x)

        # Weights for the label embedding network.
        embedding_weights = None
        if self.use_hyper_embed:
            fc_e = L.reshape(getattr(self, 'fc_spade_e_' + str(i))(x), (b, -1))
            if 'decoder' in self.embed_arch:
                weight_shape = [in_ch, out_ch, eks, eks]
                fc_e = fc_e[:, :-in_ch]
            else:
                weight_shape = [out_ch, in_ch, eks, eks]
            embedding_weights = weight_reshaper.reshape_weight(fc_e, weight_shape)

        # weights for the 3 layers in SPADE module: conv_0, conv_1, and shortcut.
        fc_0 = L.reshape(getattr(self, 'fc_spade_0_' + str(i))(x), (b, -1))
        fc_1 = L.reshape(getattr(self, 'fc_spade_1_' + str(i))(x), (b, -1))
        fc_s = L.reshape(getattr(self, 'fc_spade_s_' + str(i))(x), (b, -1))
        if self.conv_before_norm:
            out_ch = in_ch
        weight_0 = weight_reshaper.reshape_weight(fc_0, [out_ch * 2, spade_ch, sks, sks])
        weight_1 = weight_reshaper.reshape_weight(fc_1, [in_ch * 2, spade_ch, sks, sks])
        weight_s = weight_reshaper.reshape_weight(fc_s, [out_ch * 2, spade_ch, sks, sks])
        
        norm_weights = [weight_0, weight_1, weight_s]

        return embedding_weights, norm_weights

    
    def get_conv_weight(self, x, i):
        """
        Adaptively generate weights for layer i in main branch convolutions.

        Args:
            x (NxCxHxW): input features
            i (int): layer index
        Returns:
            conv_weights (list of tensors): Weights for the conv layers in the main branch.
        """
        if not self.mul_ref_label:
            x = L.adaptive_pool2d(x, pool_size=(self.sh_fix, self.sw_fix), pool_type='avg')
        in_ch = self.num_filters_each_layer[i]
        out_ch = self.num_filters_each_layer[i + 1]
        cks = self.conv_kernel_size
        b = x.shape[0]
        weight_reshaper = WeightReshaper()
        x = weight_reshaper.reshape_embed_input(x)

        fc_0 = L.reshape(getattr(self, 'fc_conv_0_' + str(i))(x), (b, -1))
        fc_1 = L.reshape(getattr(self, 'fc_conv_1_' + str(i))(x), (b, -1))
        fc_s = L.reshape(getattr(self, 'fc_conv_s_' + str(i))(x), (b, -1))

        weight_0 = weight_reshaper.reshape_weight(fc_0, [in_ch, out_ch, cks, cks])
        weight_1 = weight_reshaper.reshape_weight(fc_1, [in_ch, in_ch, cks, cks])
        weight_s = weight_reshaper.reshape_weight(fc_s, [in_ch, out_ch, 1, 1])
        return [weight_0, weight_1, weight_s]         


    def reset(self, ):
        """
        Reset the network at the beginning of a sequence.
        """
        self.embedding_weights = self.conv_weights = self.norm_weights = None


class WeightReshaper():
    """
    Handles all weight reshape related tasks.
    """
    def reshape_weight(self, x, weight_shape):
        """
        Reshape input x to the desired weight shape.

        Args:
            x: Input features.
            weight_shape (list of int): Desired shape of the weight.
        Returns:
            weight: Network weights
            bias: Network bias.
        """
        # If desired shape is a list, first divide x into the target list of features.
        if type(weight_shape[0]) == list and type(x) != list:
            x = self.split_weights(x, self.sum_mul(weight_shape))
        
        if type(x) == list:
            return [self.reshape_weight(xi, wi) for xi, wi in zip(x, weight_shape)]
        
        # Get output shape, and deivide x into either weight + bias or just weight.
        weight_shape = [x.shape[0]] + weight_shape
        bias_size = weight_shape[1]
        try:
            weight = L.reshape(x[:, :-bias_size], weight_shape)
            bias = x[:, -bias_size:]
        except Exception:
            weight = L.reshape(x, weight_shape)
            bias = None
        return [weight, bias]
    

    def split_weights(self, weight, sizes):
        """
        When the desired shape is a list, first divide the input to each
        corresponding weight shape in the list.

        Args:
            weight: Input weight.
            sizes: Target sizes
        Return:
            weight (list of tensors): Divided weights.
        """
        if isinstance(sizes, list):
            weights = []
            cur_size = 0
            for i in range(len(sizes)):
                # For each target size in sizes, get the number of elements needed.
                next_size = cur_size + self.sum(sizes[i])
                # Recursively divide the weights.
                weights.append(self.split_weights(weight[:, cur_size:next_size], sizes[i]))
                cur_size = next_size
            assert (next_size == weight.shape[1])
            return weights
        return weight
    

    def reshape_embed_input(self, x):
        """
        Reshape input to be (B x C) x H x W
        """
        if isinstance(x, list):
            return [self.reshape_embed_input(xi) for xi in zip(x)]
        b, c, _, _ = x.shape
        x = L.reshape(x, (b * c, -1))
        return x
    

    def sum_mul(self, x):
        """
        Given a weight shape, compute the number elements needed for
        weight + bias. If input is a list of shapes, sum all theelements.
        """
        assert (type(x) == list)
        if type(x[0]) != list:
            return np.prod(x) + x[0] # x[0] accounts for bias.
        return [self.sum_mul(xi) for xi in x]
