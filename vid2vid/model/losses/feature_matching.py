# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------

import paddle.nn as nn
import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg 

class FeatureMatchingLoss(dg.Layer):
    """
    Compute feature matching loss
    """
    def __init__(self, criterion='l1'):
        super(FeatureMatchingLoss, self).__init__()
        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction='mean')
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = dg.MSELoss(reduction='mean')
        else:
            raise ValueError("Criterion %s is not recognized" % criterion)
    

    def forward(self, fake_features, real_features):
        """
        Return the target vector for the binary cross entropy loss computation.

        Args:
            fake_features (list of lists): Discriminator features of fake images.
            real_features (list of lists): Discriminator features of real images.
        """
        num_d = len(fake_features)
        dis_weight = 1.0 / num_d
        loss = 0
        for i in range(num_d):
            for j in range(len(fake_features[i])):
                tmp_loss = self.criterion(fake_features[i][j], real_features[i][j])
                loss += dis_weight * tmp_loss
        return loss
