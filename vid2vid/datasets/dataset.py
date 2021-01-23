# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------

from .pose_dataset import PoseDataset
from .face_dataset import FaceDataset


def get_train_and_val_dataloader(cfg):
    """
    Return dataset objects for the training and validation sets.

    Args:
        cfg (obj): Global configuration file.
    
    Returns:
        (dict):
            - train_data_loader (obj): Train data loader.
            - val_data_loader (obj): Val data loader.
    """
    train_dataset, val_dataset = _get_train_and_val_dataset_objects(cfg)
    # train_reader = train_dataset.batch_reader(cfg.train_data.batch_size)
    # val_reader = val_dataset.batch_reader(cfg.val_data.batch_size)
    return train_dataset, val_dataset


def _get_train_and_val_dataset_objects(cfg):
    if cfg.data.name == 'pose':
        train_dataset = PoseDataset()
        val_dataset = PoseDataset()
        train_dataset.initialize(cfg.train_data)
        val_dataset.initialize(cfg.val_data)
    else:
        raise NotImplementedError()
    
    print("Train dataset length: ", len(train_dataset))
    print("Val dataset length: ", len(val_dataset))

    return train_dataset, val_dataset


def get_val_dataset(cfg):
    val_dataset = PoseDataset()
    val_dataset.initialize(cfg.val_data)
    print("Val dataset length: ", len(val_dataset))
    return val_dataset


def get_test_data_loader(cfg):
    """
    Return dataset objects for the training and validation sets.

    Args:
        cfg (obj): Global configuration file.
    
    Returns:
        (dict):
            - train_data_loader (obj): Train data loader.
            - val_data_loader (obj): Val data loader.
    """
    if cfg.data.name == 'pose':
        test_dataset = PoseDataset()
        test_dataset.initialize(cfg.inference_data)
    elif cfg.data.name == 'faceForensics':
        test_dataset = FaceDataset()
        test_dataset.initialize(cfg.inference_data)

    return test_dataset
