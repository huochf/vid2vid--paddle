# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------
import time
import os
import random
import numpy as np
import imageio
from PIL import Image

import paddle.fluid as F 
import paddle.fluid.dygraph as dg

from vid2vid.model.losses.gan import GANLoss
from vid2vid.model.losses.flow import FlowLoss
from vid2vid.model.losses.perceptual import PerceptualLoss
from vid2vid.model.losses.feature_matching import FeatureMatchingLoss
from vid2vid.model.model_utils import concat_frames, get_fg_mask, detach, get_face_bbox_for_output
from vid2vid.utils.misc import get_nested_attr
from vid2vid.utils.meters import Meter
from vid2vid.utils.visualize import tensor2im, tensor2flow, tensor2pose
from vid2vid.utils.evaluation import compute_fid


def get_trainer(cfg, net_G, net_D, opt_G, opt_D, sch_G, sch_D, train_dataset, val_dataset):
    return Trainer(cfg, net_G, net_D, opt_G, opt_D, sch_G, sch_D, train_dataset, val_dataset)


class Trainer():
    
    def __init__(self, cfg, net_G, net_D, opt_G, opt_D, sch_G, sch_D, train_dataset, val_dataset):
        print("Setup trainer.")

        # Initialize models and data loaders.
        self.cfg = cfg
        self.net_G = net_G
        self.net_D = net_D
        self.opt_G = opt_G
        self.opt_D = opt_D
        self.sch_G = sch_G
        self.sch_D = sch_D
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Initialize logging attributes.
        self.current_iteration = 0
        self.current_epoch = 0
        self.start_iteration_time = None
        self.elapsed_iteration_time = 0
        self.time_iteration = -1
        self.time_epoch = -1

        self.sequence_length = 1
        self.sequence_length_max = 16

        # Initialize loss functions.
        self.criteria = dg.LayerList()
        # Mapping from loss names to loss weights. 
        self.weights = dict()
        self.losses = dict(gen_update=dict(), dis_update=dict())
        self.gen_losses = self.losses['gen_update']
        self.dis_losses = self.losses['dis_update']
        self._init_loss(cfg)

        self.meters = {}

        self.is_inference = cfg.is_inference
        self.has_fg = getattr(cfg.data, 'has_foreground', False)

        self.temporal_network_initialized = False
        self.gt_flow = [None, None]

        self.sample_size = (
            getattr(cfg.trainer, 'num_videos_to_test', 16),
            getattr(cfg.trainer, 'num_frames_per_video', 10)
        )


    def _init_loss(self, cfg):
        self.criteria = dict()
        self.weights = dict()
        trainer_cfg = cfg.trainer
        loss_weight = cfg.trainer.loss_weight

        # GAN loss and feature matching loss. 
        self._assign_criteria('GAN', GANLoss(trainer_cfg.gan_mode), loss_weight.gan) # 1.0
        self._assign_criteria('FeatureMatching', FeatureMatchingLoss(), loss_weight.feature_matching) # 10.0

        # Perceptual loss.
        perceptual_loss = cfg.trainer.perceptual_loss
        self._assign_criteria('Perceptual', PerceptualLoss(cfg=cfg, num_scales=getattr(perceptual_loss, 'num_scales', 1)),
            loss_weight.perceptual) # 10
        
        # Whether to add an additional discriminator for specific regions.
        self.add_dis_cfg = getattr(self.cfg.dis, 'additional_discriminators', None)
        if self.add_dis_cfg is not None:
            for name in self.add_dis_cfg:
                add_dis_cfg = self.add_dis_cfg[name]
                self.weights['GAN_' + name] = add_dis_cfg.loss_weight
                self.weights['FeatureMatching_' + name] = loss_weight.feature_matching
        
        # Temporal GAN loss. 
        self.num_temporal_scales = get_nested_attr(self.cfg.dis, 'temporal.num_scales', 0)
        for s in range(self.num_temporal_scales):
            self.weights['GAN_T%d' % s] = loss_weight.temporal_gan
            self.weights['FeatureMatching_T%d' % s] = loss_weight.feature_matching
        
        # Flow loss. It consists of three parts: L1 loss compared to GT,
        # warping loss when used to warp images, and loss on the occlusion mask. 
        self.use_flow = hasattr(cfg.gen, 'flow') and cfg.gen.use_flow
        if self.use_flow:
            self.criteria['Flow'] = FlowLoss(cfg)
            self.weights['Flow'] = self.weights['Flow_L1'] = \
                self.weights['Flow_Warp'] = self.weights['Flow_Mask'] = loss_weight.flow


    def _assign_criteria(self, name, criterion, weight):
        self.criteria[name] = criterion
        self.weights[name] = weight       


    def save_checkpoint(self, current_epoch, current_iteration):
        latest_checkpoint_path = 'epoch_{:05}_iteration_{:09}_checkpoint'.format(current_epoch, current_iteration)
        save_path = os.path.join(self.cfg.logdir, 'checkpoints', latest_checkpoint_path, latest_checkpoint_path)
        F.save_dygraph(self.net_G.state_dict(), save_path + '_net_G')
        F.save_dygraph(self.net_D.state_dict(), save_path + '_net_D')
        F.save_dygraph(self.opt_G.state_dict(), save_path + '_net_G')
        F.save_dygraph(self.opt_D.state_dict(), save_path + '_net_D')
        print("Save checkpoint to {}".format(save_path))


    def load_checkpoint(self, cfg, checkpoint_path, resume=None):
        if os.path.exists(checkpoint_path + '_net_G.pdparams'):
            if resume is None:
                resume = False
        else:
            current_epoch = 0
            current_iteration = 0
            print("No checkpoint found.")
            return current_epoch, current_iteration
        
        net_G_dict, opt_G_dict = dg.load_dygraph(checkpoint_path + '_net_G')
        if not self.is_inference:
            net_D_dict, opt_D_dict = dg.load_dygraph(checkpoint_path + '_net_D')
        current_epoch, current_iteration = int(checkpoint_path.split('_')[-4]), int(checkpoint_path.split('_')[-2])
        if resume:
            self.net_G.set_dict(net_G_dict)
            self.net_D.set_dict(net_D_dict)
            # self.opt_G.set_dict(opt_G_dict)
            self.opt_D.set_dict(opt_D_dict)
            print("Load from: {}".format(checkpoint_path))
        else:
            self.net_G.set_dict(net_G_dict)
            print("Load generator weights only.")
        
        print("Done with loading the checkpoint.")
        return current_epoch, current_iteration
    

    def start_of_epoch(self, current_epoch):
        self._start_of_epoch(current_epoch)
        self.current_epoch = current_epoch
        self.start_epoch_time = time.time()
    

    def start_of_iteration(self, data, current_iteration):
        data = self._start_of_iteration(data, current_iteration)
        self.current_iteration = current_iteration
        if not self.is_inference:
            self.net_D.train()
        self.net_G.train()
        self.start_iteration_time = time.time()
        return data
    

    def end_of_iteration(self, data, current_epoch, current_iteration):
        self.current_iteration = current_iteration
        self.current_epoch = current_epoch

        # Accumulate time
        self.elapsed_iteration_time += time.time() - self.start_iteration_time

        # Logging. 
        if current_iteration % self.cfg.logging_iter == 0:
            ave_t = self.elapsed_iteration_time / self.cfg.logging_iter
            self.time_iteration = ave_t
            print("Iteration: {}, average iter time: {:6f}.".format(current_iteration, ave_t))
            self.elapsed_iteration_time = 0

        self._end_of_iteration(data, current_epoch, current_iteration)

        # Compute image to be saved. 
        if current_iteration % self.cfg.image_save_iter == 0:
            self.save_image(self._get_save_path('images', 'jpg'), data)
        elif current_iteration % self.cfg.image_display_iter == 0:
            image_path = os.path.join(self.cfg.logdir, 'images', 'current.jpg')
            self.save_image(image_path, data)
        
        if current_iteration % self.cfg.logging_iter == 0:
            self._write_meters(current_iteration)
        

    def end_of_epoch(self, data, current_epoch, current_iteration):
        self.current_iteration = current_iteration
        self.current_epoch = current_epoch

        # Save everything to the checkpoint. 
        if current_iteration >= self.cfg.snapshot_save_start_iter and current_epoch % self.cfg.snapshot_save_epoch == 0:
            self.save_image(self._get_save_path('images', 'jpg'), data)
            self.save_checkpoint(current_epoch, current_iteration)
            self.write_metrics()

        elapsed_epoch_time = time.time() - self.start_epoch_time
        print("Epoch: {}, total time: {:6f}.".format(current_epoch, elapsed_epoch_time))

        self.time_epoch = elapsed_epoch_time
        self._end_of_epoch(data, current_epoch, current_iteration)
    

    def pre_process(self, data):
        data_cfg = self.cfg.data

        if hasattr(data_cfg, 'for_pose_dataset') and ('pose_maps-densepose' in data_cfg.input_types):
            pose_cfg = data_cfg.for_pose_dataset
            for i in range(len(data)):
                data[i]['tgt_label'] = self.pre_process_densepose(pose_cfg, data[i]['tgt_label'], self.is_reference)
                data[i]['ref_label'] = self.pre_process_densepose(pose_cfg, data[i]['ref_label'], self.is_reference)

        tgt_label = dg.to_variable(np.stack([item['tgt_label'] for item in data]))
        tgt_image = dg.to_variable(np.stack([item['tgt_image'] for item in data]))
        ref_label = dg.to_variable(np.stack([item['ref_label'] for item in data]))
        ref_image = dg.to_variable(np.stack([item['ref_image'] for item in data]))
        paths = [d['path'] for d in data]
        data = {}
        data['tgt_label'] = tgt_label
        data['tgt_image'] = tgt_image
        data['ref_label'] = ref_label
        data['ref_image'] = ref_image
        data['path'] = paths

        return data


    def pre_process_densepose(self, pose_cfg, pose_map, is_infer=False):
        part_map = pose_map[:, 2]
        assert (part_map >= 0).all() and (part_map < 25).all()

        if not is_infer:
            random_drop_prob = getattr(pose_cfg, 'random_drop_prob', 0) # 0.05
        else:
            random_drop_prob = 0
        if random_drop_prob > 0:
            densepose_map = pose_map[:, :3]
            for part_id in range(1, 25):
                if random.random() < random_drop_prob:
                    part_mask = abs(part_map - part_id) < 0.1
                    densepose_map[np.repeat(part_mask[:, np.newaxis], 3, axis=1)] = 0
            pose_map[:, :3] = densepose_map
        
        return pose_map
    

    def _start_of_epoch(self, current_epoch):
        cfg = self.cfg
        if current_epoch < cfg.single_frame_epoch:
            self.train_dataset.n_frames_total = 1
        elif current_epoch == cfg.single_frame_epoch:
            self.sequence_length = self.cfg.data.train.initial_sequence_length
            self.train_dataset.n_frames_total = self.sequence_length
            self.net_G.init_temporal_network()
            print("-------- Now start training %d frames --------" % self.sequence_length)
        
        temp_epoch = current_epoch - cfg.single_frame_epoch
        if temp_epoch > 0:
            sequence_length = cfg.data.train.initial_sequence_length * (2 ** (temp_epoch // cfg.num_epochs_temporal_step))
            sequence_length = min(sequence_length, self.sequence_length_max)
            if sequence_length > self.sequence_length:
                self.sequence_length = sequence_length
                self.train_dataset.n_frames_total = sequence_length
                print('-------- Updating sequence length to %d --------' % sequence_length)


    def _start_of_iteration(self, data, current_iteration):
        data = self.pre_process(data)
        return data


    def _end_of_iteration(self, data, current_epoch, current_iteration):
        if current_iteration % self.cfg.logging_iter == 0:
            message = '(epoch: %d, iters: %d) ' % (current_epoch, current_iteration)
            for k, v in self.gen_losses.items():
                if k != 'total':
                    message += '%s: %.3f,  ' % (k, v)
            message += '\n'
            for k, v in self.dis_losses.items():
                if k != 'total':
                    message += '%s: %.3f,  ' % (k, v)
            print(message)


    def _end_of_epoch(self, data, current_epoch, current_iteration):
        pass


    def step(self, data):
        # Whether to reuse generator output for both gen_update and dis_update. 
        # It saves time but comsumes a bit more memory.
        reuse_gen_output = getattr(self.cfg.trainer, 'reuse_gen_output', False)

        past_frames = [None, None]
        net_G_output = None
        data_prev = None
        for t in range(self.sequence_length):
            data_t = self.get_data_t(data, net_G_output, data_prev, t)
            data_prev = data_t

            # Discriminator update. 
            if reuse_gen_output:
                net_G_output = self.net_G(data_t)
            else:
                with dg.no_grad():
                    net_G_output = self.net_G(data_t)
            data_t, net_G_output = self.post_process(data_t, net_G_output)

            # Get losses and update D if image generated by network in training.
            if 'fake_images_source' not in net_G_output:
                net_G_output['fake_images_source'] = 'in_training'
            if net_G_output['fake_images_source'] != 'pretrained':
                net_D_output, _ = self.net_D(data_t, detach(net_G_output), past_frames)
                self.get_dis_losses(net_D_output)
            
            # Generator update. 
            if not reuse_gen_output:
                net_G_output = self.net_G(data_t)
                data_t, net_G_output = self.post_process(data_t, net_G_output)
            
            # Get losses and update G if image generated by network in training. 
            if 'fake_images_source' not in net_G_output:
                net_G_output['fake_images_source'] = 'in_training'
            if net_G_output['fake_images_source'] != 'pretrained':
                net_D_output, past_frames = self.net_D(data_t, net_G_output, past_frames)
                self.get_gen_losses(data_t, net_G_output, net_D_output)
    

    def get_data_t(self, data, net_G_output, data_prev, t):
        label = data['tgt_label'][:, t]
        image = data['tgt_image'][:, t]

        if data_prev is not None:
            nG = self.cfg.data.num_frames_G
            prev_labels = concat_frames(data_prev['prev_labels'], data_prev['label'], nG - 1)
            prev_images = concat_frames(data_prev['prev_images'], net_G_output['fake_images'].detach(), nG - 1)
        else:
            prev_labels = prev_images = None

        data_t = dict()
        data_t['label'] = label
        data_t['image'] = image
        data_t['ref_labels'] = data['ref_label']
        data_t['ref_images'] = data['ref_image']
        data_t['prev_labels'] = prev_labels
        data_t['prev_images'] = prev_images
        data_t['real_prev_image'] = data['tgt_image'][:, t - 1] if t > 0 else None

        return data_t

    
    def post_process(self, data, net_G_output):
        if self.has_fg:
            fg_mask = get_fg_mask(data['label'], self.has_fg)
            if net_G_output['fake_raw_images'] is not None:
                net_G_output['fake_raw_images'] = net_G_output['fake_raw_images'] * fg_mask
        
        return data, net_G_output


    def get_gen_losses(self, data_t, net_G_output, net_D_output):
        # Individual frame GAN loss and feature matching loss.
        self.gen_losses['GAN'], self.gen_losses['FeatureMatching'] = self.compute_GAN_losses(net_D_output['indv'], dis_update=False)

        # Perceptual loss. 
        self.gen_losses['Perceptual'] = self.criteria['Perceptual'](net_G_output['fake_images'], data_t['image'])

        # L1 loss. 
        if getattr(self.cfg.trainer.loss_weight, 'L1', 0) > 0:
            self.gen_losses['L1'] = self.criteria['L1'](net_G_output['fake_images'], data_t['image'])
        
        # Raw (hallucinated) output image losses (GAN and perceptual). 
        if 'raw' in net_D_output:
            raw_GAN_losses = self.compute_GAN_losses(net_D_output['raw'], dis_update=False)
            fg_mask = get_fg_mask(data_t['label'], self.has_fg)
            raw_perceptual_loss = self.criteria['Perceptual'](net_G_output['fake_raw_images'] * fg_mask, data_t['image'] * fg_mask)
            self.gen_losses['GAN'] += raw_GAN_losses[0]
            self.gen_losses['FeatureMatching'] += raw_GAN_losses[1]
            self.gen_losses['Perceptual'] += raw_perceptual_loss
        
        # Additional discriminator losses.
        if self.add_dis_cfg is not None:
            for name in self.add_dis_cfg:
                self.gen_losses['GAN_' + name], self.gen_losses['FeatureMatching_' + name] = \
                    self.compute_GAN_losses(net_D_output[name], dis_update=False)
        
        # Flow and mask loss. 
        if self.use_flow:
            self.gen_losses['Flow_L1'], self.gen_losses['Flow_Warp'], self.gen_losses['Flow_Mask'], self.gt_flow = \
                self.criteria['Flow'](data_t, net_G_output, self.current_epoch)
        
        # Temporal GAN loss and feature matching loss. 
        if self.cfg.trainer.loss_weight.temporal_gan > 0:
            if self.sequence_length > 1:
                for s in range(self.num_temporal_scales):
                    loss_GAN, loss_FM = self.compute_GAN_losses(
                        net_D_output['temporal_%d' % s], dis_update=False)
                    self.gen_losses['GAN_T%d' % s] = loss_GAN
                    self.gen_losses['FeatureMatching_T%d' % s] = loss_FM

        # Sum all losses together
        total_loss = dg.to_variable(np.zeros((1, )).astype("float32"))
        for key in self.gen_losses:
            if key != 'total':
                total_loss += self.gen_losses[key] * self.weights[key]
        
        self.gen_losses['total'] = total_loss

        total_loss.backward()
        self.opt_G.minimize(total_loss)
        self.opt_G.clear_gradients()
        # self.net_G.clear_gradients()
    

    def get_dis_losses(self, net_D_output):
        # Individual frame GAN loss. 
        self.dis_losses['GAN'] = self.compute_GAN_losses(net_D_output['indv'], dis_update=True)

        # Raw (hallucinated) output image GAN loss.
        if 'raw' in net_D_output:
            raw_loss = self.compute_GAN_losses(net_D_output['raw'], dis_update=True)
            self.dis_losses['GAN'] += raw_loss
        
        # Additional GAN loss. 
        if self.add_dis_cfg is not None:
            for name in self.add_dis_cfg:
                self.dis_losses['GAN_' + name] = self.compute_GAN_losses(net_D_output[name], dis_update=True)
        
        # Temporal GAN loss. 
        if self.cfg.trainer.loss_weight.temporal_gan > 0:
            if self.sequence_length > 1:
                for s in range(self.num_temporal_scales):
                    self.dis_losses['GAN_T%d' % s] = self.compute_GAN_losses(net_D_output['temporal_%d' % s], dis_update=True)
        
        # Sum all losses together. 
        total_loss = dg.to_variable(np.array((1, )).astype("float32"))
        for key in self.dis_losses:
            if key != 'total':
                total_loss += self.dis_losses[key] * self.weights[key]
        self.dis_losses['total'] = total_loss

        total_loss.backward()
        self.opt_D.minimize(total_loss)
        self.opt_D.clear_gradients()
        # self.net_D.clear_gradients()


    def compute_GAN_losses(self, net_D_output, dis_update):
        if net_D_output['pred_fake'] is None:
            return dg.to_variable(np.zeros((1, )).astype("float32")) if dis_update else [
                dg.to_variable(np.zeros((1, )).astype("float32")), dg.to_variable(np.zeros((1, )).astype("float32"))]

        if dis_update:
            # Get the GAN loss for real/fake outputs 
            GAN_loss = self.criteria['GAN'](net_D_output['pred_fake']['output'], False, dis_update=True) + \
                self.criteria['GAN'](net_D_output['pred_real']['output'], True, dis_update=True)
            return GAN_loss
        else:
            # Get the GAN loss and feature matching loss for fake output. 
            GAN_loss = self.criteria['GAN'](net_D_output['pred_fake']['output'], True, dis_update=False)
            FM_loss = self.criteria['FeatureMatching'](net_D_output['pred_fake']['features'], net_D_output['pred_real']['features'])
            return GAN_loss, FM_loss
    

    def _get_save_path(self, subdir, ext):
        subdir_path = os.path.join(self.cfg.logdir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path, exist_ok=True)
        return os.path.join(subdir_path, 'epoch_{:05}_iteration_{:09}.{}'.format(
            self.current_epoch, self.current_iteration, ext))


    def save_image(self, path, data):
        self.net_G.eval()

        self.net_G_output = None
        with dg.no_grad():
            first_net_G_output, last_net_G_output, _ = self.gen_frames(data)
        
        
        def get_images(data, net_G_output, return_first_frame=True, for_model_average=False):
            frame_idx = 0 if return_first_frame else -1
            warped_idx = 0 if return_first_frame else 1
            vis_images = []
            
            vis_images += [
                tensor2im(data['ref_image'][:, frame_idx]),
                self.visualize_label(data['tgt_label'][:, frame_idx]),
                tensor2im(data['tgt_image'][:, frame_idx])
            ]
            vis_images += [
                tensor2im(net_G_output['fake_images']),
                tensor2im(net_G_output['fake_raw_images'])
            ]
            vis_images += [
                # tensor2im(net_G_output['warped_images'][warped_idx]),
                # tensor2flow(net_G_output['fake_flow_maps'][warped_idx]),
                # tensor2flow(self.gt_flow[warped_idx]),
                # tensor2im(net_G_output['fake_occlusion_masks'][warped_idx])
            ]
            return vis_images
        
        vis_images_first = get_images(data, first_net_G_output)
        if self.sequence_length > 1:
            vis_images_last = get_images(data, last_net_G_output, return_first_frame=False)

            # If generating a video, the first row of each batch will be
            # the first generated frame and the flow/mask for warping the
            # reference image, and the second row will be the last generated
            # frame and the flow/mask for warping the previous frame.
            vis_images = [[np.vstack((im_first, im_last)) for im_first, im_last in zip(imgs_first, imgs_last)]
                for imgs_first, imgs_last in zip(vis_images_first, vis_images_last) if imgs_first is not None]
            
        else:
            vis_images = vis_images_first
        
        image_grid = np.hstack([np.vstack(im) for im in vis_images if im is not None])
        print("Save output images to {}".format(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.imwrite(path, image_grid)
        

    def gen_frames(self, data, use_model_average=False):
        net_G_output = None
        data_prev = None
        net_G = self.net_G

        # Iterate through the length of sequence. 
        all_info = {'inputs': [], 'outputs': []}
        for t in range(self.sequence_length):
            # Get the data at the current time frame. 
            data_t = self.get_data_t(data, net_G_output, data_prev, t)
            data_prev = data_t

            # Generator forward. 
            with dg.no_grad():
                net_G_output = net_G(data_t)
            
            # Do any postprocessing if necessary
            data_t, net_G_output = self.post_process(data_t, net_G_output)

            if t == 0:
                # Get the output at beginning of sequence for visualization. 
                first_net_G_output = net_G_output
            
            all_info['inputs'].append(data_t)
            all_info['outputs'].append(net_G_output)
        
        return first_net_G_output, net_G_output, all_info


    def visualize_label(self, label):
        cfgdata = self.cfg.data
        if hasattr(cfgdata, 'for_pose_dataset'):
            label = tensor2pose(self.cfg, label)
        else:
            label = tensor2im(label)
        return label


    def write_metrics(self):
        for k, v in self.meters.items():
            v.flush()
        regular_fid = self._compute_fid()
        if regular_fid is None:
            return
        if 'regular_fid' not in self.meters:
            self.meters['regular_fid'] = Meter('regular_fid', self.cfg.logdir)
        self.meters['regular_fid'].write(regular_fid, self.current_iteration)
        self.meters['regular_fid'].flush()


    def _write_meters(self, step):
        for update, losses in self.losses.items():
            # update is 'gen_update' or 'dis_update'. 
            assert update == 'gen_update' or update == 'dis_update'
            for loss_name, loss in losses.items():
                full_loss_name = update + '_' + loss_name
                if full_loss_name not in self.meters.keys():
                    self.meters[full_loss_name] = Meter(full_loss_name, self.cfg.logdir)
                self.meters[full_loss_name].write(loss.numpy()[0], step)


    def reset(self, ):
        self.net_G_output = self.data_prev = None
        self.t = 0

        self.net_G.reset()
    

    def test(self, test_data_loader, root_output_dir, inference_args):
        self.reset()
        args = inference_args
        test_data_loader.reset()
        test_data_loader.set_inference_sequence_idx(
            args.driving_seq_index, args.few_shot_seq_index, args.few_shot_frame_index)

        video = []
        max_length = min(test_data_loader.get_current_sequence_length(), args.max_seq_length)
        for it in range(max_length):
            data = [test_data_loader.get_items(it)]
            
            key = data[0]['path']
            filename = key.split('/')[-1]

            # Create output dir for this sequence. 
            if it == 0:
                seq_name = '%03d' % args.few_shot_seq_index
                output_dir = os.path.join(root_output_dir, seq_name)
                os.makedirs(output_dir, exist_ok=True)
                video_path = output_dir
            
            data = self.start_of_iteration(data, current_iteration=-1)
            # Get output and save images. 
            data['img_name'] = filename
            output = self.test_single(data, output_dir, inference_args)
            video.append(output)
        
        # Save output as mp4. 
        imageio.mimsave(video_path + '.mp4', video, fps=15)


    def test_single(self, data, output_dir=None, inference_args=None, return_fake_image=True):
        # if getattr(inference_args, 'finetune', False):
        #     if not getattr(self, 'has_fine_tuned', False):
        #         self.finetune(data, inference_args)
        
        net_G = self.net_G
        net_G.eval() 

        data_t = self.get_data_t(data, self.net_G_output, self.data_prev, 0)
        if self.is_inference or self.sequence_length > 1:
            self.data_prev = data_t
        
        # Generator forward. 
        with dg.no_grad():
            self.net_G_output = net_G(data_t)
        
        if output_dir is None:
            return self.net_G_output
        
        save_fake_only = getattr(inference_args, 'save_fake_only', False)
        if save_fake_only:
            ys, ye, xs, xe = get_face_bbox_for_output(None, data_t['label'][0:1], crop_smaller=0)
            image_grid = tensor2im(self.net_G_output['fake_images'])[0]
            h, w, _ = image_grid.shape
            face_mask = Image.open('/home/aistudio/vid2vid/test/images/face.png').resize((ye - ys, xe - xs))
            mask = np.zeros((h, w, 3)).astype("uint8")
            mask[ys:ye, xs:xe, :] = np.array(face_mask)[:, :, :3]
            image_grid[mask != 0] = 0
            image_grid += mask
            # image_grid = tensor2im(data_t['label'][:, 3:])[0]
        else:
            vis_images = self.get_test_output_images(data)
            image_grid = np.hstack([np.vstack(im) for im in vis_images if im is not None])
        
        if 'img_name' in data:
            save_name = data['img_name'].split('.')[0] + '.jpg'
        else:
            save_name = "%04d.jpg" % self.t 
        output_filename = os.path.join(output_dir, save_name)
        os.makedirs(output_dir, exist_ok=True)
        imageio.imwrite(output_filename, image_grid)
        self.t += 1

        if return_fake_image:
            return image_grid
        else:
            return self.net_G_output, image_grid
    

    def get_val_output_dir(self):
        return os.path.join(self.cfg.logdir, 'epoch_{:05}_iteration_{:09}'.format(
            self.current_epoch, self.current_iteration))


    def get_test_output_images(self, data):
        vis_images = [
            tensor2im(data['ref_image'][:, 0]),
            self.visualize_label(data['tgt_label'][:, -1]),
            tensor2im(data['tgt_image'][:, -1]),
            tensor2im(self.net_G_output['fake_images']),
            # tensor2im(self.net_G_output['warped_images'][0]),
        ]
        return vis_images


    def _compute_fid(self):
        self.net_G.eval()
        self.net_G_output = None
        trainer = self
        regular_fid_path = self._get_save_path('regular_fid', 'npy')
        few_shot = True if 'few_shot' in self.cfg.data.type else False
        regular_fid_value = compute_fid(regular_fid_path, self.val_dataset, trainer, preprocess=self.pre_process,
            sample_size=self.sample_size, is_video=True, few_shot_video=few_shot)
        print("Epoch {:05}, Iteration {:09}, Regular FID {}".format(
            self.current_epoch, self.current_iteration, regular_fid_value))
        return regular_fid_value
