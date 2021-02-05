# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------


def get_paired_input_image_channel_number(data_cfg):
    """
    Get number of channels for the input image.
    """
    num_channels = 0
    for ix, data_type in enumerate(data_cfg.input_types):
        for k in data_type:
            if k in data_cfg.input_image:
                num_channels += data_type[k].num_channels
                # print('Concatenate %s for input.' % data_type)
    print('\tNum. of channels in the input image: %d' % num_channels)
    return num_channels



def get_paired_input_label_channel_number(data_cfg, video=False):
    """
    Get number of channels for the input label map.

    Args:
        data_cfg (obj): Data configuration structure.
        video (bool): Whether we are dealing with video data.
    Returns:
        num_channels (int): Number of input label map channels.
    """
    num_labels = 0
    if not hasattr(data_cfg, 'input_labels'):
        return num_labels
    for ix, data_type in enumerate(data_cfg.input_types):
        for k in data_type:
            if k in data_cfg.input_labels:
                num_labels += data_type[k].num_channels
                if getattr(data_type[k], 'use_dont_care', False):
                    print(data_type[k].use_dont_care)
                    num_labels += 1
            # print('Concatenate %s for input.' % data_type)
    
    if video:
        num_time_steps = getattr(data_cfg.train, 'initial_sequence_length', None)
        num_labels *= num_time_steps
        num_labels += get_paired_input_label_channel_number(data_cfg) * (num_time_steps - 1)
    
    print('\tNum. of channels in the input label: %d' % num_labels)
    return num_labels































