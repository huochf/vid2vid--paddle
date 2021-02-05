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
from scipy import linalg
import imageio

import paddle.nn as nn
import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg

from vid2vid.model.backbones.inception_v4 import build as build_inception


def compute_fid(fid_path, data_loader, net_G, key_real='tgt_image', key_fake='fake_images',
    sample_size=None, preprocess=None, is_video=False, few_shot_video=False):
    """
    Compute the fid score.

    Args:
        fid_path (str): Location for the numpy file to store or to load the statistics.
        data_loader (obj): data_loader object. 
        net_G (obj): 
        key_real (str): Dictionary key value for the real data. 
        key_fake (str): Dictionary key value for the fake data. 
        sample_size (int or tuple): How many samples to be used. 
        prerpocess (func): The preprocess function to be applied to the data. 
        is_video (bool): Whether we are handling video sequences. 
        few_shot_video(bool): If True, uses few-shot video synthesis. 
    """
    print("Computing FID.")
    with dg.no_grad():
        # Get the fake mean and covariance.
        fake_mean, fake_cov = load_or_compute_stats(fid_path, data_loader, key_real, key_fake, 
            net_G, sample_size, preprocess, is_video, few_shot_video)
        
        # Get the ground truth mean and covariance.
        mean_cov_path = os.path.join(os.path.dirname(fid_path), 'real_mean_cov.npz')
        real_mean, real_cov = load_or_compute_stats(mean_cov_path, data_loader, key_real, key_fake,
            None, sample_size, preprocess, is_video, few_shot_video)
    
    fid = calculate_frechet_distance(real_mean, real_cov, fake_mean, fake_cov)
    return fid


def load_or_compute_stats(fid_path, data_loader, key_real, key_fake, 
    generator=None, sample_size=None, preprocess=None, is_video=None, few_shot_video=False):
    """
    Load mean and covariance from saved npy file if exists. Otherwise, compute the mean
    and covariance. 
    """
    if os.path.exists(fid_path):
        print("Load FID mean and cov from {}".format(fid_path))
        npz_file = np.load(fid_path)
        mean = npz_file['mean']
        cov = npz_file['cov']
    else:
        print("Get FID mean and cov and save to {}".format(fid_path))
        mean, cov = get_inception_mean_cov(data_loader, key_real, key_fake,
            generator, sample_size, preprocess, is_video, few_shot_video)
        os.makedirs(os.path.dirname(fid_path), exist_ok=True)
        np.savez(fid_path, mean=mean, cov=cov)
    
    return mean, cov


def get_inception_mean_cov(data_loader, key_real, key_fake, generator, 
    sample_size, preprocess, is_video=False, few_shot_video=False):
    """
    Load mean and covariance from saved npy file if exists. Otherwise,
    compute the mean and covariance.
    """
    print("Extract mean and covariance.")
    if is_video:
        with dg.no_grad():
            y = get_video_activations(data_loader, key_real, key_fake, generator,
                sample_size, preprocess, few_shot_video)
    else:
        y = get_activations(data_loader, key_real, key_fake, generator, sample_size, preprocess)
    
    m = np.mean(y, axis=0)
    s = np.cov(y, rowvar=False)

    return m, s


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance. 
    The Frechet distance between two multivariate Gaussins X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is 
    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2 * sqrt(C_1 * C_2)). 
    Stable version by Dougal J. Sutherland.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu1)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; adding %s to diagonal of cov estimates' % eps)
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print('Imaginary component {}'.format(m))
            # raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(
        sigma2) - 2 * tr_covmean)


def get_activations(data_loader, key_real, key_fake, generator=None, sample_size=None, preprocess=None):
    inception = build_inception()
    inception.eval()
    batch_y = []
    for it, data in enumerate(data_loader.batch_reader(2)()):
        if preprocess is not None:
            data = preprocess(data)
        if generator is None:
            images = data[key_real]
        else:
            net_G_output = generator(data)
            images = net_G_output[key_fake]
        # Clamp the image for models that do not set the output to between
        # -1, 1. For models that employ tanh, this has not effect. 
        # images = L.clip(images, -1, 1)
        images = apply_image_net_normalization(images)
        images = nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=True)
        y = inception(images)
        batch_y += [y]
    batch_y = L.concat(batch_y).numpy()
    if sample_size is not None:
        batch_y = batch_y[:sample_size]
    # print(batch_y.shape)
    return batch_y


def get_video_activations(data_loader, key_real, key_fake, trainer=None, sample_size=None, preprocess=None, few_shot=False):
    inception = build_inception()
    inception.eval()
    batch_y = []
    num_sequences = data_loader.num_inference_sequences()

    if sample_size is None:
        num_videos_to_test = 10
        num_frames_per_video = 5
    else:
        num_videos_to_test = sample_size[0]
        num_frames_per_video = sample_size[1]

    for sequence_idx in range(num_sequences):
        print("Sequence index %d" % sequence_idx)
        if sequence_idx > num_videos_to_test:
            break 

        data_loader.reset()
        data_loader.set_inference_sequence_idx(sequence_idx, sequence_idx, 0)
        if trainer is not None:
            trainer.reset()
        
        max_length = data_loader.get_current_sequence_length()
        video = []
        for it in range(max_length):
            data = [data_loader.get_items(it)]
            if it >= num_frames_per_video:
                break
            
            if trainer is not None:
                data = trainer.pre_process(data)
            elif preprocess is not None:
                data = preprocess(data)
            else:
                data = data[0]
            
            if trainer is None:
                images = data[key_real][:, -1]
            else:
                key = data['path'][0]
                filename = key.split('/')[-1]

                # Create output dir for this sequence. 
                if it == 0:
                    seq_name = '%03d' % sequence_idx
                    root_output_dir = trainer.get_val_output_dir()
                    output_dir = os.path.join(root_output_dir, seq_name)
                    os.makedirs(output_dir, exist_ok=True)
                    video_path = output_dir
                
                data['img_name'] = filename

                net_G_output, show_image = trainer.test_single(data, output_dir, return_fake_image=False)
                images = net_G_output[key_fake]
                video.append(show_image)
            # Clamp the image for models that do not set the output to between
            # -1, 1. For models that employ tanh, this has not effect. 
            # images = L.clip(images, -1, 1)
            images = apply_image_net_normalization(images)
            images = nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=True)
            y = inception(images)
            batch_y += [y.numpy()]

        if video != []:
            # Save output as mp4. 
            imageio.mimsave(video_path + '.mp4', video, fps=15)
        
    batch_y = np.concatenate(batch_y) # L.concat(batch_y).numpy()
    return batch_y


def apply_image_net_normalization(input):
    normalized_input = (input + 1) / 2
    mean = dg.to_variable(np.array([0.485, 0.456, 0.406]).astype("float32"))
    mean = L.reshape(mean, (1, 3, 1, 1))
    std = dg.to_variable(np.array([0.229, 0.224, 0.225]).astype("float32"))
    std = L.reshape(std, (1, 3, 1, 1))
    output = (normalized_input - mean) / std 
    return output
