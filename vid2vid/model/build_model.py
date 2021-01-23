# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------

import paddle.fluid.optimizer as optimizer
from paddle.optimizer.lr import StepDecay

from .generator.fewshot_gen_model import FewShotGenerator
from .discriminator.discriminator import Discriminator


def get_model_optimizer_and_scheduler(cfg):
    net_G = FewShotGenerator(cfg.gen, cfg.data)
    net_D = Discriminator(cfg.dis, cfg.data)

    scheduler_G = get_scheduler(cfg.gen_opt)
    scheduler_D = get_scheduler(cfg.dis_opt)

    opt_G = get_optimizer(cfg.gen_opt, net_G, scheduler_G)
    opt_D = get_optimizer(cfg.dis_opt, net_D, scheduler_D)

    return net_G, net_D, opt_G, opt_D, scheduler_G, scheduler_D


def get_scheduler(cfg_opt, ):
    if cfg_opt.lr_policy.type == 'step':
        scheduler = StepDecay(learning_rate=cfg_opt.lr, 
                              step_size=cfg_opt.lr_policy.step_size,
                              gamma=cfg_opt.lr_policy.gamma)
    else:
        raise NotImplementedError()
    
    return scheduler


def get_optimizer(cfg_opt, net, scheduler=None):
    params = net.parameters()
    return get_optimizer_for_params(cfg_opt, params, scheduler)


def get_optimizer_for_params(cfg_opt, params, scheduler):
    """
    Return the scheduler object.

    Args:
        cfg_opt (obj): Config for the specific optimization module (gen/dis). 
        params (obj): Parameters to be trained by the parameters.
    
    Returns:
        (obj): Optimizer
    """
    # We will use fuse optimizers by default.
    fused_opt = cfg_opt.fused_opt
    
    if cfg_opt.type == 'adam':
        if fused_opt:
            print("Not implemented fused Adam")
            opt = optimizer.AdamOptimizer(learning_rate=scheduler if scheduler is not None else cfg_opt.lr, 
                epsilon=cfg_opt.eps, beta1=cfg_opt.adam_beta1, beta2=cfg_opt.adam_beta2, parameter_list=params)
        else:
            opt = optimizer.AdamOptimizer(learning_rate=scheduler if scheduler is not None else cfg_opt.lr, 
                epsilon=cfg_opt.eps, beta1=cfg_opt.adam_beta1, beta2=cfg_opt.adam_beta2, parameter_list=params)
    else:
        raise NotImplementedError()
    
    return opt
