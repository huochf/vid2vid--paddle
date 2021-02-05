# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------
import os
import numpy as np
import cv2
import importlib

from vid2vid.model.model_utils import extract_valid_pose_labels


def visualize_dataset_image(images, save_dir, image_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n, c, h, w = images.shape

    for i in range(n):
        show = np.transpose(images[i], (1, 2, 0))
        show = show * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        show = (show * 255).astype(np.uint8)
        show = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
        path = os.path.join(save_dir, '{}_{}.png'.format(image_name, i))
        cv2.imwrite(path, show)


def visualize_dataset_face(images, save_dir, image_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n, c, h, w = images.shape

    for i in range(n):
        show = np.transpose(images[i], (1, 2, 0))
        show = (show).astype(np.uint8)
        show = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
        path = os.path.join(save_dir, '{}_{}.png'.format(image_name, i))
        cv2.imwrite(path, show)


def tensor2im(image_tensor, imtype=np.uint8, normalize=True, three_channel_output=True):
    if image_tensor is None:
        return None
    if isinstance(image_tensor, list):
        return [tensor2im(x, imtype, normalize) for x in image_tensor]
    if len(image_tensor.shape) == 5 or len(image_tensor.shape) == 4:
        return [tensor2im(image_tensor[idx], imtype, normalize) for idx in range(image_tensor.shape[0])]
    
    if len(image_tensor.shape) == 3:
        image_numpy = image_tensor.numpy()
        if normalize:
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        else:
            image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        image_numpy = np.clip(image_numpy, 0, 255)
        if image_numpy.shape[2] == 1 and three_channel_output:
            image_numpy = np.repeat(image_numpy, 3, axis=2)
        elif image_numpy.shape[2] > 3:
            image_numpy = image_numpy[:, :, :3]
        return image_numpy.astype(imtype)


def tensor2flow(tensor, imtype=np.uint8):
    if tensor is None:
        return None
    if isinstance(tensor, list):
        tensor = [t for t in tensor if t is not None]
        if not tensor:
            return None
        return [tensor2flow(t, imtype) for t in tensor]
    if len(tensor.shape) == 5 or len(tensor.shape) == 4:
        return [tensor2flow(tensor[b]) for b in range(tensor.shape[0])]
    
    tensor = tensor.numpy()
    tensor = np.transpose(tensor, (1, 2, 0))

    hsv = np.zeros((tensor.shape[0], tensor.shape[1], 3), dtype=imtype)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(tensor[..., 0], tensor[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def tensor2pose(cfg, label_tensor):

    if len(label_tensor.shape) == 5 or len(label_tensor.shape) == 4:
        return [tensor2pose(cfg, label_tensor[idx]) for idx in range(label_tensor.shape[0])]

    # If adding additional discriminators, draw the bbox for the regions (e.g. faces) too. 
    add_dic_cfg = getattr(cfg.dis, 'additional_discriminators', None)
    if add_dic_cfg is not None:
        crop_coords = []
        for name in add_dic_cfg:
            v = add_dic_cfg[name].vis
            file, crop_func = v.split('::')
            file = importlib.import_module(file)
            crop_func = getattr(file, crop_func)
            crop_coord = crop_func(cfg.data, label_tensor)
            if len(crop_coord) > 0:
                if type(crop_coord[0]) == list:
                    crop_coords.extend(crop_coord)
                else:
                    crop_coords.append(crop_coord)
    
    pose_cfg = cfg.data.for_pose_dataset
    pose_type = getattr(pose_cfg, 'pose_type', 'both')
    remove_face_labels = getattr(pose_cfg, 'remove_face_labels', False)
    label_tensor = extract_valid_pose_labels(label_tensor, pose_type, remove_face_labels)

    # If using both DensePose and OpenPose, overlay one image onto the other
    # to get the visualization map. 
    dp_key = 'pose_maps-densepose'
    op_key = 'poses-openpose'

    op_ch = 3
    dp_ch = 3
    label_img = tensor2im(label_tensor[:dp_ch])
    openpose = label_tensor[-op_ch:]
    openpose = tensor2im(openpose)
    label_img[openpose != 0] = openpose[openpose != 0]

    # Draw the bbox for the regions for the additional discriminator. 
    if add_dic_cfg is not None:
        for crop_coord in crop_coords:
            ys, ye, xs, xe = crop_coord
            label_img[ys, xs:xe, :] = label_img[ye - 1, xs:xe, :] \
                = label_img[ys:ye, xs, :] = label_img[ys:ye, xe - 1, :] = 255
    
    if len(label_img.shape) == 2:
        raise NotImplementedError()
    
    return label_img



def flow2img(flow_data):
    """
    convert optical flow into color image
    :param" flowa-data
    :return: color image
    """
    u = flow_data[:, :, 0]
    v = flow_data[:, :, 1]

    UNKNOW_FLOW_THRESHOLD = 1e7
    pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
    pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
    idx_unknown = (pr1 | pr2)
    u[idx_unknown] = v[idx_unknown] = 0

    # get max value in each direction
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    maxu = max(maxu, np.max(u))
    maxv = max(maxv, np.max(v))
    minu = min(minu, np.min(u))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u / maxrad + np.finfo(float).eps
    v = v / maxrad + np.finfo(float).eps

    img = compute_color(u, v)

    idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u, v):
    height, width = u.shape
    img = np.zeros((height, width, 3))

    NAN_idx = np.isnan(u) | np.isnan(v)
    u[NAN_idx] = v[NAN_idx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel	















