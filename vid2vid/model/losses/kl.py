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


class GaussianKLLoss(dg.Layer):
    """
    Compute KL loss in VAE for Gaussian distributions
    """
    def __init__(self,):
        super(GaussianKLLoss, self).__init__()
    

    def forward(self, mu, logvar=None):
        """
        Compute loss

        Args:
            mu (tensor): mean
            logvar (tensor): logarithm of variance
        """
        if logvar is None:
            logvar = L.zeros_like(mu)
        return -0.5 * L.reduce_sum(1 + logvar - L.pow(mu, 2) - L.exp(logvar))
