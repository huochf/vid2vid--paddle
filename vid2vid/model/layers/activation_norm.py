# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------
from types import SimpleNamespace

import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg

from .conv import LinearBlock, Conv2dBlock, PartialConv2dBlock, HyperConv2D
from .misc import PartialSequential

class AdaptiveNorm(dg.Layer):
    """
    Adaptive normalization layer. The layer first normalizes the input, then
    performs an affine transformation using parameters computed from the
    conditional inputs.
    """
    def __init__(self, num_features, cond_dims, weight_norm_type='', projection=True,
        separate_projection=False, input_dim=2, activation_norm_type='instance', activation_norm_params=None):
        super().__init__()
        self.projection = projection
        self.separate_projection = separate_projection
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace(affine=False)
        self.norm = get_activation_norm_layer(num_features, activation_norm_type, input_dim, **vars(activation_norm_params))

        if self.projection:
            if self.separate_projection:
                self.fc_gamma = LinearBlock(cond_dims, num_features, weight_norm_type=weight_norm_type)
                self.fc_beta = LinearBlock(cond_dims, num_features, weight_norm_type=weight_norm_type)
            else:
                self.fc = LinearBlock(cond_dims, num_features * 2, weight_norm_type=weight_norm_type)
        
        self.conditional = True
    

    def forward(self, x, y, **kargs):
        """
        Adaptive Normalization forward.

        Args:
            x (N x C1 x *): input, 
            y (N x C2): Conditional information.
        Returns:
            out (N x c1 x *): output
        """
        residual_dim = len(x.shape) - len(y.shape)
        if self.projection:
            if self.separate_projection:
                gamma = self.fc_gamma(y)
                beta = self.fc_beta(y)
                for _ in range(residual_dim):
                    gamma = L.unsqueeze(gamma, -1)
                    beta = L.unsqueeze(beta, -1)
            else:
                y = self.fc(x)
                for _ in range(residual_dim):
                    y = L.unsqueeze(y, -1)
                gamma, beta = L.split(y, num_or_sections=2, dim=1)
        else:
            for _ in range(residual_dim):
                y = L.unsqueeze(y, -1)
            gamma, beta = L.split(y, 2, 1)
        
        x = self.norm(x) if self.norm is not None else x
        out = x * (1 + gamma) + beta
        return out


class SpatiallyAdaptiveNorm(dg.Layer):
    """
    Spatially Adaptive Normalization (SPADE) initialization
    """
    def __init__(self, num_features, cond_dims, num_filters=128, kernel_size=3, weight_norm_type='',
        separate_projection=False, activation_norm_type='sync_batch', activation_norm_params=None, partial=False):
        super().__init__()
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace(affine=False)
        padding = kernel_size // 2
        self.separate_projection = separate_projection
        mlps = []
        gammas = []
        betas = []

        # Make cond_dims a list.
        if type(cond_dims) != list:
            cond_dims = [cond_dims]
        
        # Make num_filters a list
        if not isinstance(num_filters, list):
            num_filters = [num_filters] * len(cond_dims)
        else:
            assert len(num_filters) >= len(cond_dims)
        
        # Make partial a list.
        if not isinstance(partial, list):
            partial = [partial] * len(cond_dims)
        else:
            assert len(partial) >= len(cond_dims)
        
        for i, cond_dim in enumerate(cond_dims):
            mlp = []
            conv_block = PartialConv2dBlock if partial[i] else Conv2dBlock
            sequential = PartialSequential if partial[i] else dg.Sequential

            if num_filters[i] > 0:
                mlp += [(str(i), conv_block(cond_dim, num_filters[i], kernel_size, padding=padding,
                                   weight_norm_type=weight_norm_type, nonlinearity='relu'))]
            mlp_ch = cond_dim if num_filters[i] == 0 else num_filters[i]

            if self.separate_projection:
                if partial[i]:
                    raise NotImplementedError("Separate projection not yet implemented for partial conv")
                mlps.append(dg.Sequential(*mlp))
                gammas.append((str(i), conv_block(mlp_ch, num_features, kernel_size, padding=padding, weight_norm_type=weight_norm_type)))
                betas.append((str(i), conv_block(mlp_ch, num_features, kernel_size, padding=padding, weight_norm_type=weight_norm_type)))
            else:
                mlp += [(str(i), conv_block(mlp_ch, num_features * 2, kernel_size, padding=padding, weight_norm_type=weight_norm_type))]
                mlps.append(sequential(*mlp))
        
        self.mlps = dg.LayerList(mlps)
        self.gammas = dg.LayerList(gammas)
        self.betas = dg.LayerList(betas)
        
        self.norm = get_activation_norm_layer(num_features, activation_norm_type, 2, **vars(activation_norm_params))

        self.conditional = True
    

    def forward(self, x, *cond_inputs, **kwargs):
        output = self.norm(x) if self.norm is not None else x
        for i in range(len(cond_inputs)):
            if cond_inputs[i] is None:
                continue
            label_map = L.image_resize(cond_inputs[i], out_shape=x.shape[2:], resample='NEAREST')

            if self.separate_projection:
                hidden = self.mlps[i](label_map)
                gamma = self.gammas[i](hidden)
                beta = self.betas[i](hidden)
            else:
                affine_params = self.mlps[i](label_map)
                gamma, beta = L.split(affine_params, 2, 1)
            output = output * (1 + gamma) + beta
        
        return output


class HyperSpatiallyAdaptiveNorm(dg.Layer):
    """
    Spatially Adaptive Normalization (SPADE) intialization
    """

    def __init__(self, num_features, cond_dims, num_filters=0, kernel_size=3,
        weight_norm_type='', activation_norm_type='sync_batch', is_hyper=True):
        super().__init__()
        padding = kernel_size // 2
        mlps = []
        if type(cond_dims) != list:
            cond_dims = [cond_dims]
        
        for i, cond_dim in enumerate(cond_dims):
            mlp = []
            if not is_hyper or (i != 0):
                if num_filters > 0:
                    mlp += [(str(i), Conv2dBlock(cond_dim, num_filters, kernel_size, padding=padding,
                                                 weight_norm_type=weight_norm_type, nonlinearity='relu'))]
                mlp_ch = cond_dim if num_filters == 0 else num_filters
                mlp += [(str(len(mlp)), Conv2dBlock(mlp_ch, num_features * 2, kernel_size, 
                                                    padding=padding, weight_norm_type=weight_norm_type))]
                mlp = dg.Sequential(*mlp)
            else:
                if num_filters > 0:
                    raise ValueError('Multi hyper layer not supported yet.')
                mlp = HyperConv2D(padding=padding)
            mlps.append(mlp)
        
        self.mlps = dg.LayerList(mlps)
        self.norm = get_activation_norm_layer(num_features, activation_norm_type, 2, affine=False)
        self.conditional = True
    

    def forward(self, x, *cond_inputs, norm_weights=(None, None), **kwargs):
        """
        Spatially Adaptive Normalization (SPADE) forward.
        """
        output = self.norm(x)
        for i in range(len(cond_inputs)):
            if cond_inputs[i] is None:
                continue
            
            if type(cond_inputs[i]) == list:
                cond_input, mask = cond_inputs[i]
                mask = L.image_resize(mask, size=x.shape[2:], resample='BILINEAR', align_corners=False)
            else:
                cond_input = cond_inputs[i]
                mask = None

            label_map = L.image_resize(cond_input, x.shape[2:])
            if norm_weights is None or norm_weights[0] is None or i != 0:
                affine_params = self.mlps[i](label_map)
            else:
                affine_params = self.mlps[i](label_map, conv_weights=norm_weights)
            
            gamma, beta = L.split(affine_params, 2, 1)
            if mask is not None:
                gamma = gamma * (1 - mask)
                beta = beta * (1 - mask)
            output = output * (1 + gamma) + beta
        
        return output


def get_activation_norm_layer(num_features, norm_type, input_dim, **norm_params):
    """
    Return an activation normalization layer.
    """
    input_dim = max(input_dim, 1)
    assert input_dim == 2 or input_dim == 1, 'Only support for 2D currently'

    if norm_type == 'none' or norm_type == '':
        norm_layer = None
    elif norm_type == 'batch':
        norm_layer = dg.BatchNorm(num_features, **norm_params)
    elif norm_type == 'instance':
        affine = norm_params.pop('affine', True)
        if not affine:
            norm_params['param_attr'] = False
            norm_params['bias_attr'] = False
        norm_layer = dg.InstanceNorm(num_features, **norm_params) #affine=affine, **norm_params)
    elif norm_type == 'sync_batch':
        affine = norm_params.pop('affine', True)
        norm_layer = dg.BatchNorm(num_features, **norm_params)
        F.BuildStrategy().sync_batch_norm = True
    elif norm_type == 'layer':
        norm_layer = dg.LayerNorm(num_features, **norm_params)
    elif norm_type == 'layer_2d':
        raise NotImplementedError()
    elif norm_type == 'adaptive':
        norm_layer = AdaptiveNorm(num_features, **norm_params)
    elif norm_type == 'spatially_adaptive':
        if input_dim != 2:
            raise ValueError("Spatially adaptive normalization layers only supports 2D input")
        norm_layer = SpatiallyAdaptiveNorm(num_features, **norm_params)
    elif norm_type == 'hyper_spatially_adaptive':
        if input_dim != 2:
            raise ValueError("Spatially adaptive normalization layers only supports 2D input")
        norm_layer = HyperSpatiallyAdaptiveNorm(num_features, **norm_params)
    else:
        raise ValueError("Activation norm layer %s is not recognized" % norm_type)

    return norm_layer
    































