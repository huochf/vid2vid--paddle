# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------


def get_and_setattr(cfg, name, default):
    """
    Get attribute with default choice. If attribute does not exist, set it using the default value.
    """
    if not hasattr(cfg, name) or name not in cfg.__dict__:
        setattr(cfg, name, default)
    return getattr(cfg, name)


def get_nested_attr(cfg, attr_name, default):
    """
    Iteratively try to get the attribute from cfg. If not found, return default.
    """
    names = attr_name.split('.')
    atr = cfg
    for name in names:
        if not hasattr(atr, name):
            return default
        atr = getattr(atr, name)
    return atr
    