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


class ApplyNoise(dg.Layer):
    """
    Add Gaussian noise to the input tensor.
    """
    def __init__(self):
        super().__init__()
        # scale of the noise
        # self.weight = 
        raise NotImplementedError()


class PartialSequential(dg.Sequential):
    """
    Sequential block for partial convolutions.
    """
    def __init__(self, *modules):
        super(PartialSequential, self).__init__(*modules)
    

    def forward(self, x):
        act = x[:, :-1]
        mask = L.unsqueeze(x[:, -1], 1)
        for module in self:
            act, mask = module(act, mask_in = mask)
        return act