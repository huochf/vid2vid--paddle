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
import paddle.fluid.dygraph.nn as nn

class SpectralNormalization(dg.Layer):

    def __init__(self,
                 layer,
                 name='',
                 n_power_iterations=1,
                 eps=1e-12,
                 dim=0,
                 dtype='float32'):
        super(SpectralNormalization, self).__init__()
        self.spectral_norm = nn.SpectralNorm(layer.weight.shape, dim, n_power_iterations, eps, dtype)
        self.dim = dim
        self.power_iters = n_power_iterations
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out


def spectral_norm(**kwargs):
    return partial(SpectralNormalization, **kwargs)


def get_weight_norm_layer(norm_type, **norm_params):
    """
    Return weight normalization.

    Args:
        norm_type (str): Type of weight normalization.
    """

    if norm_type == 'none' or norm_type == '':
        return lambda x: x
    elif norm_type == 'spectral': # spectral normalization
        return spectral_norm(**norm_params)
    else:
        raise NotImplementedError()
























