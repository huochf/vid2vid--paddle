# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------
import numpy as np 

import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg
import paddle.nn as nn

from .vgg import build_model

class PerceptualLoss(dg.Layer):

    def __init__(self, cfg, criterion='l1', num_scales=1):
        super().__init__()
        self.model = build_model()
        self.num_scales = num_scales
        
        self.criterion = nn.L1Loss()
        self.resize = False
        self.weights = [0.03125, 0.0625, 0.125, 0.25, 1.0]
        

    def forward(self, inp, target):
        """
        Perceptual loss forward.

        Args:
            inp (4D tensor): Input tensor. 
            target (4D tensor): Ground truth tensor, same shape as the input. 
        """
        self.model.eval()
        inp, target = apply_imagenet_normalization(inp), apply_imagenet_normalization(target)
        
        # Evaluate perceptual loss at each scale.
        loss = 0
        input_features, target_features = self.model(inp), self.model(target)

        for i, weight in enumerate(self.weights):
            loss += weight * self.criterion(input_features[i], target_features[i])
        return loss


def apply_imagenet_normalization(input):
    # The input images are assumed to be [-1, 1]
    normalized_input = (input + 1) / 2
    mean = dg.to_variable(np.array([0.485, 0.456, 0.406]).astype("float32"))
    std = dg.to_variable(np.array([0.229, 0.224, 0.225]).astype("float32"))
    mean = L.reshape(mean, (1, 3, 1, 1))
    std = L.reshape(std, (1, 3, 1, 1))
    output = (normalized_input - mean) / std
    return output

