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


class ReLU(dg.Layer):
    def forward(self, x):
        return L.relu(x)


class LeakyReLU(dg.Layer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
    

    def forward(self, x):
        return L.leaky_relu(x, alpha=self.alpha)


class Tanh(dg.Layer):
    def forward(self, x):
        return L.tanh(x)


class Sigmoid(dg.Layer):
    def forward(self, x):
        return L.sigmoid(x)


class Softmax(dg.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    

    def forward(self, x):
        return L.softmax(x, axis=self.dim)


def get_nonlinearity_layer(nonlinearity_type, inplace):
    """
    Return a nonlinearity layer.
    """
    if nonlinearity_type == 'relu':
        nonlinearity = ReLU()
    elif nonlinearity_type == 'leakyrelu':
        nonlinearity = LeakyReLU(0.2)
    elif nonlinearity_type == 'prelu':
        nonlinearity = dg.PRelu()
    elif nonlinearity_type == 'tanh':
        nonlinearity = Tanh()
    elif nonlinearity_type == 'sigmoid':
        nonlinearity = Sigmoid()
    elif nonlinearity_type.startswith('softmax'):
        dim = nonlinearity_type.split(',')[1] if ',' in nonlinearity_type else 1
        nonlinearity = Softmax(int(dim))
    elif nonlinearity_type == 'none' or nonlinearity_type == '':
        nonlinearity = None
    else:
        raise ValueError('Nonlinearity %s is not recognized' % nonlinearity_type)
    
    return nonlinearity











