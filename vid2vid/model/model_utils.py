# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------
import numpy as np

import paddle as P
import paddle.tensor as T
import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg
import paddle.nn as nn


def extract_valid_pose_labels(pose_map, pose_type, remove_face_labels, do_remove=True):
    """
    Remove some labels (e.g. face regions) in the pose map if necessary.

    Args:
        pose_map (3D, 4D or 5D tensor): input pose map.
        pose_type (str): 'both' or 'open'
        remove_face_labels (bool): Whether to remove labels for the face region.
        do_remove (bool): Do remove face labels.
    
    Returns:
        pose_map (3D, 4D or 5D tensor): Output pose map.
    """
    if pose_map is None:
        return pose_map
    
    if type(pose_map) == list:
        return [extract_valid_pose_labels(p, pose_type, remove_face_labels, do_remove) for p in pose_map]
    
    orig_dim = len(pose_map.shape)
    assert (orig_dim >= 3 and orig_dim <= 5)
    if orig_dim == 3:
        pose_map = L.unsqueeze(pose_map, axes=[0, 1])
    elif orig_dim == 4:
        pose_map = L.unsqueeze(pose_map, [0])
    
    if pose_type == 'open':
        # If input is only openpose, remove densepose part.
        pose_map = pose_map[:, :, 3:]
    elif remove_face_labels and do_remove:
        # Remove face part for densepose input.
        densepose, openpose = pose_map[:, :, :3], pose_map[:, :, 3:]
        face_mask = get_face_mask(pose_map[:, :, 2])
        face_mask = L.unsqueeze(face_mask, [2])
        pose_map = L.concat([densepose * (1 - face_mask) - face_mask, openpose], axis=2)
    
    if orig_dim == 3:
        pose_map = pose_map[0, 0]
    elif orig_dim == 4:
        pose_map = pose_map[0]
    return pose_map


def get_face_mask(densepose_map):
    """
    Obtain mask of faces. 

    Args:
        densepose_map (3D or 4D tensor)
    """
    need_reshape = len(densepose_map.shape) == 4
    if need_reshape:
        bo, t, h, w = densepose_map.shape
        densepose_map = L.reshape(densepose_map, (-1, h, w))
    
    b, h, w = densepose_map.shape
    part_map = (densepose_map / 2 + 0.5) * 24
    assert L.reduce_all((part_map >= 0)) and L.reduce_all((part_map < 25))

    mask = dg.to_variable(np.zeros((b, h, w)).astype('bool'))

    for j in [23, 24]:
        mask = L.logical_or(mask, L.logical_and((part_map > j - 0.1), (part_map < j + 0.1)))
    
    if need_reshape:
        mask = L.reshape(mask, (bo, t, h, w))
    
    return P.cast(mask, "float32")


def resample(image, flow):
    """
    Resamples an image using the provided flow.

    Args:
        image (NxCxHxW tensor): Image to resample.
        flow (Nx2xHxW): Optical flow to resample the image.
    Returns:
        Output (NxCxHxW tensor): Resamples image.
    """
    assert flow.shape[1] == 2
    b, c, h, w = image.shape
    grid = get_grid(b, (h, w))
    flow = L.concat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0),
                     flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], 1)
    final_grid = L.transpose((grid + flow), (0, 2, 3, 1))
    image.stop_gradient = False
    try:
        output = nn.functional.grid_sample(image, final_grid, mode='bilinear', padding_mode='border', align_corners=True)
    except Exception:
        output = nn.functional.grid_sample(image, final_grid, mode='bilinear', padding_mode='border')
    
    return output
    # return image
    # return L.zeros_like(image)


def get_grid(batchsize, size, minval=-1.0, maxval=1.0):
    if len(size) == 2:
        rows, cols = size
    elif len(size) == 3:
        deps, rows, cols = size
    else:
        raise ValueError("Dimension can only be 2 or 3.")
    
    x = L.linspace(minval, maxval, cols)
    x = L.reshape(x, (1, 1, 1, cols))
    x = L.expand(x, (batchsize, 1, rows, 1))

    y = L.linspace(minval, maxval, rows)
    y = L.reshape(y, (1, 1, rows, 1))
    y = L.expand(y, (batchsize, 1, 1, cols))

    t_grid = L.concat([x, y], 1)
    
    if len(size) == 3:
        z = L.linspace(minval, maxval, deps)
        z = L.reshape(z, (1, 1, deps, 1, 1))
        z = L.expand(z, (batchsize, 1, 1, rows, cols))

        t_grid = L.expand(L.unsqueeze(t_grid, [2]), (1, 1, deps, 1, 1))
        t_grid = L.concat([t_grid, z], 1)
    
    t_grid.stop_gradient = True
    return t_grid


def pick_image(images, idx):
    """
    Pick the image among images according to idx.

    Args:
        images (B x N x C x H x W), N images, 
        idx (B ) indices to select.
    """
    if type(images) == list:
        return [pick_image(r, idx) for r in images]
    if idx is None:
        return images[:, 0]
    elif type(idx) == int:
        return images[:, idx]
    
    idx = idx.astype('long').numpy()
    images = L.stack([images[i][int(idx[i])] for i in range(images.shape[0])])
    return images


def get_fg_mask(densepose_map, has_fg):
    """
    Obtain the foreground mask for pose sequences, which only includes
    the human. This is done by looking at the body part map from DensePose.

    Args:
        densepose_map (NxCxHxW tensor): DensePose map.
        has_fg (bool): Whether data has foreground or not.
    Returns:
        mask (Nx1xHxW tensor): fg mask.
    """
    if type(densepose_map) == list:
        return [get_fg_mask(label, has_fg) for label in densepose_map]
    if not has_fg or densepose_map is None:
        return 1
    if len(densepose_map.shape) == 5:
        densepose_map = densepose_map[:, 0]
    # Get the body part map from DensePose.
    mask = densepose_map[:, 2:3]

    # Make the mask slightly larger.
    mask = L.pool2d(mask, pool_size=15, pool_type='max', pool_stride=1, pool_padding=7)
    # mask = dg.to_variable(((mask > -1).numpy().astype("float32")))
    mask = P.cast((mask > -1), "float32")
    return mask


def get_part_mask(densepose_map):
    """
    Obtain mask of different body parts of humans. This is done by looking
    at the body part map from DensePose.

    Args:
        densepose_map (NxCxHxW tensor): DensePose map.
    Returns:
        mask (NxKxHxW tensor): Body part mask, where K is the number of parts.
    """
    # Group of body parts. Each group contains IDs of body labels in DensePose.
    # The 9 groups here are: background, torso, hands, feet, upper legs, lower legs,
    # upper arms, lower arms, head.
    part_groups = [[0], [1, 2], [3, 4], [5, 6], [7, 9, 8, 10], [11, 13, 12, 14],
                   [15, 17, 16, 18], [19, 21, 20, 22], [23, 24]]
    n_parts = len(part_groups)

    densepose_map = densepose_map.numpy()
    need_reshape = len(densepose_map.shape) == 4
    if need_reshape:
        bo, t, h, w = densepose_map.shape
        densepose_map = np.reshape(densepose_map, (-1, h, w))
    b, h, w = densepose_map.shape
    part_map = (densepose_map / 2 + 0.5) * 24
    assert np.all(part_map >= 0) and np.all(part_map < 25)

    mask = np.zeros((b, n_parts, h, w)).astype("bool")
    for i in range(n_parts):
        for j in part_groups[i]:
            # Account for numerical errors.
            mask[:, i] = np.logical_or(mask[:, i],
                np.logical_and((part_map > j - 0.1), (part_map < j + 0.1)))
    if need_reshape:
        mask = np.reshape(mask, (bo, t, -1, h, w))
    mask = dg.to_variable(mask.astype("float32"))
    return mask


def concat_frames(prev, now, n_frames):
    now = L.unsqueeze(now, [1]) # [b, T, C, H, W]
    if prev is None or (n_frames == 1):
        return now
    if prev.shape[1] == n_frames:
        prev = prev[:, 1:]
    return L.concat([prev, now], 1)


def detach(output):
    if type(output) == dict:
        new_dict = dict()
        for k, v in output.items():
            new_dict[k] = detach(v)
        return new_dict
    elif type(output) == F.Variable:
        return output.detach()
    else:
        return output


def crop_face_from_data(cfg, is_inference, data):
    """
    Crop the face regions in input data and resize to the target size.
    """
    raise NotImplementedError()


def crop_face_from_output(data_cfg, image, input_label, crop_smaller=0):
    """
    Crop out the face region of the image (and resize if necessary to feed into generator/discriminator).
    """
    if type(image) == list:
        return [crop_face_from_output(data_cfg, im, input_label, crop_smaller)  for im in image]
    
    output = None
    face_size = image.shape[-2] // 32 * 8
    for i in range(input_label.shape[0]):
        ys, ye, xs, xe = get_face_bbox_for_output(data_cfg, input_label[i:i+1], crop_smaller=crop_smaller)
        output_i = L.image_resize(image[i:i+1, -3:, ys:ye, xs:xe], out_shape=(face_size, face_size), )
        output = L.concat([output, output_i]) if i != 0 else output_i
    
    return output


def get_face_bbox_for_output(data_cfg, pose, crop_smaller=0):
    """
    Get pixel coordinates of the face bounding box.
    """
    if len(pose.shape) == 3:
        pose = L.unsqueeze(pose, [0])
    elif len(pose.shape) == 5:
        pose = pose[-1, -1:]
    _, _, h, w = pose.shape

    use_openpose = False # 'pose_maps-densepose' not in data_cfg.input_labels
    if use_openpose: # Use openpose face keypoints to identify face region. 
        raise NotImplementedError()
    else: # Use densepose labels. 
        # face = T.search.nonzero(dg.to_variable((pose[:, 2] > 0.9).numpy().astype("int64")), as_tuple=False)
        face = T.search.nonzero((pose[:, 2] > 0.9).astype("int64"), as_tuple=False)
    
    ylen = xlen = h // 32 * 8
    if face.shape[0]:
        y, x = face[:, 1], face[:, 2]
        ys, ye = L.reduce_min(y), L.reduce_max(y)
        xs, xe = L.reduce_min(x), L.reduce_max(x)
        if use_openpose:
            xc, yc = (xs + xe) // 2, (ys * 3 + ye * 2) // 5
            ylen = int((xe - xs) * 2.5)
        else:
            xc, yc = (xs + xe) // 2, (ys + ye) // 2
            ylen = int((ye - ys) * 1.25)
        ylen = xlen = min(w, max(32, ylen))
        yc = max(ylen // 2, min(h - 1 - ylen // 2, yc))
        xc = max(xlen // 2, min(w - 1 - xlen // 2, xc))
    else:
        yc = h // 4
        xc = w // 2
    
    ys, ye = yc - ylen // 2, yc + ylen // 2
    xs, xe = xc - xlen // 2, xc + xlen // 2
    if crop_smaller != 0: # Crop slightly smaller inside face.
        ys += crop_smaller
        xs += crop_smaller
        ye -= crop_smaller
        xe -= crop_smaller

    if not isinstance(ys, int):
        ys = int(ys.numpy()[0])
    if not isinstance(ye, int):
        ye = int(ye.numpy()[0])
    if not isinstance(xs, int):
        xs = int(xs.numpy()[0])
    if not isinstance(xe, int):
        xe = int(xe.numpy()[0])

    return [ys, ye, xs, xe]
























