# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------

import numpy as np

import paddle.nn as nn
import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg


def fuse_math_min_mean_pos(x):
    """
    Fuse operation min mean for hinge loss computation of positive samples
    """
    minval = L.clip(x - 1, -1e8, 0)
    loss = - L.reduce_mean(minval)
    return loss


def fuse_math_min_mean_neg(x):
    """
    Fuse operation min mean for hinge loss computation of negative samples
    """
    minval = L.clip(-x - 1, -1e8, 0)
    loss = - L.reduce_mean(minval)
    return loss


class GANLoss(dg.Layer):
    """
    GAN loss constructor.

    Args:
        gan_mode (str): Type of GAN loss, ``hinge``, ``least_square``, ``non_saturated``, ``wasserstein``.
        target_real_label (float): The desired output label for real images.
        target_fake_label (float): The desired output label for fake images.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fakce_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.gan_mode = gan_mode
        print('GAN mode: %s' % gan_mode)
    

    def forward(self, dis_output, t_real, dis_update=True):
        """
        GAN loss computation.

        Args:
            dis_output (tensor or list of tensors): Discriminator outputs.
            t_real (bool): If ``True``, uses the real label as target, otherwise uses the fake label as target.
            dis_update (bool): If ``True``, the loss will be used to update the discriminator, otherwise the generator.
        
        Returns:
            loss (tensor): Loss value.
        """
        if isinstance(dis_output, list):
            # For multi-scale discriminators.
            # In this implementation, the loss is first averaged for each scale
            # (batch size and number of locations) then averaged across scales,
            # so that the gradient is not dominated by the discriminator that
            # has the most output values (highest resolution).
            loss = 0
            for dis_output_i in dis_output:
                loss += self.loss(dis_output_i, t_real, dis_update)
            return loss / len(dis_output)
        else:
            return self.loss(dis_output, t_real, dis_update)
    

    def loss(self, dis_output, t_real, dis_update=True):
        """
        GAN loss computation.

        Args:
            dis_output (tensor): Discriminator outputs.
            t_real (bool): If ``True``, uses the real label as target, otherwise uses the fake label as target.
            dis_update (bool): Updating the discriminator or the generator.
        
        Returns:
            loss (tensor): Loss value.
        """
        if not dis_update:
            assert t_real, "The target should be real when updating the generator."
        
        if self.gan_mode == 'non_saturated':
            target_tensor = self.get_target_tensor(dis_output, t_real)
            loss = nn.functional.binary_cross_entropy_with_logits(dis_output, target_tensor)
        elif self.gan_mode == 'least_square':
            target_tensor = self.get_target_tensor(dis_output, t_real)
            loss = 0.5 * nn.functional.mse_loss(dis_output, target_tensor)
        elif self.gan_mode == 'hinge':
            if dis_update:
                if t_real:
                    loss = fuse_math_min_mean_pos(dis_output)
                else:
                    loss = fuse_math_min_mean_neg(dis_output)
            else:
                loss = -L.reduce_mean(dis_output)
        elif self.gan_mode == 'wasserstein':
            if t_real:
                loss = - L.reduce_mean(dis_output)
            else:
                loss = L.reduce_mean(dis_output)
        else:
            raise ValueError("Unexpected gan_mode {}".format(self.gan_mode))
        return loss
    

    def get_target_tensor(self, dis_output, t_real):
        """
        Return the target vector for the binary cross entropy loss computation.

        Args:
            dis_output (tensor): Discriminator outputs.
            t_real (bool): If ``True``, uses the real label as target, otherwise uses the fake label as target.
        
        Returns:
            target (tensor): Target tensor vector.
        """
        if t_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = dg.to_variable(np.ones(dis_output.shape, dtype="float32") * self.real_label)
            return L.expand_as(self.real_label_tensor, dis_output)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = dg.to_variable(np.ones(dis_output.shape, dtype="float32") * self.fake_label)
            return L.expand_as(self.fake_label_tensor, dis_output)
