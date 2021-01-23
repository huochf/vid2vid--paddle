# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------

import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg

from vid2vid.model.model_utils import extract_valid_pose_labels
from vid2vid.model.model_utils import resample, pick_image

from .reference_encoder import ReferenceEncoder
from .weight_generator import WeightGenerator
from .label_embedding import LabelEmbedding
from .generator import Generator
from .flow_generator import FlowGenerator

import vid2vid.utils.data as data_utils
import vid2vid.utils.misc as misc_utils


class FewShotGenerator(dg.Layer):

    def __init__(self, gen_cfg, data_cfg):
        super().__init__()
        self.gen_cfg = gen_cfg
        self.data_cfg = data_cfg
        self.flow_cfg = flow_cfg = gen_cfg.flow
        self.emb_cfg = emb_cfg = flow_cfg.multi_spade_combine.embed
        hyper_cfg = gen_cfg.hyper

        self.use_hyper_embed = hyper_cfg.is_hyper_embed # True
        self.num_frames_G = data_cfg.num_frames_G

        num_img_channels = data_utils.get_paired_input_image_channel_number(data_cfg)

        num_input_channels = data_utils.get_paired_input_label_channel_number(data_cfg)
        if num_input_channels == 0:
            num_input_channels = getattr(data_cfg, 'label_channels', 1)
        elif misc_utils.get_nested_attr(data_cfg, 'for_pose_dataset.pose_type', 'both') == 'open':
            num_input_channels -= 3
        
        # Number of hyper layers
        self.num_multi_spade_layers = getattr(flow_cfg.multi_spade_combine, 'num_layers', 3)

        # Whether to generate raw output for additional losses.
        self.generate_raw_output = getattr(flow_cfg, 'generate_raw_output', False)

        # For pose dataset. 
        self.is_pose_data = hasattr(data_cfg, 'for_pose_dataset')
        if self.is_pose_data:
            pose_cfg = data_cfg.for_pose_dataset
            self.pose_type = getattr(pose_cfg, 'pose_type', 'both')
            self.remove_face_labels = getattr(pose_cfg, 'remove_face_labels', False)

        self.main_generator = Generator(gen_cfg, data_cfg)
        self.reference_encoder = ReferenceEncoder(gen_cfg, data_cfg)
        self.weight_generator = WeightGenerator(gen_cfg, data_cfg)
        self.label_embedding = LabelEmbedding(gen_cfg, gen_cfg.embed, data_cfg, num_input_channels, num_hyper_layers=-1)

        # Flow estimation module. 
        # Whether to warp reference image and combine with the synthesized. 
        self.warp_ref = getattr(flow_cfg, 'warp_ref', True) # True
        if self.warp_ref:
            self.flow_network_ref = FlowGenerator(flow_cfg, data_cfg, 2)
            self.ref_image_embedding = LabelEmbedding(gen_cfg, emb_cfg, data_cfg, num_img_channels + 1)
        
        # At beginning of training, only train an image generator.
        # When starting training multiple frames, initialize the flow network. 
        self.temporal_initialized =  False
        if getattr(gen_cfg, 'init_temporal', False):
            self.init_temporal_network()
    

    def init_temporal_network(self, ):
        flow_cfg = self.flow_cfg
        data_cfg = self.data_cfg
        emb_cfg = self.emb_cfg
        gen_cfg = self.gen_cfg

        print("initialize temporal network")
        self.sep_prev_flownet = flow_cfg.sep_prev_flow or (self.num_frames_G != 2) or not flow_cfg.warp_ref
        if self.sep_prev_flownet: # False
            self.flow_network_temp = FlowGenerator(flow_cfg, data_cfg, self.num_frames_G)
        else:
            self.flow_network_temp = self.flow_network_ref
        
        self.sep_prev_embedding = emb_cfg.sep_warp_embed or not flow_cfg.warp_ref
        if self.sep_prev_embedding: # True
            num_img_channels = data_utils.get_paired_input_image_channel_number(self.data_cfg)
            self.prev_image_embedding = LabelEmbedding(gen_cfg, emb_cfg, data_cfg, num_img_channels + 1)
        else:
            self.prev_image_embedding = self.ref_image_embedding
        
        self.flow_temp_is_initalized = True

        self.temporal_initialized = True


    def forward(self, data):
        """
        few-shot vid2vid generator forward.
        """
        label = data['label'] # [b, 6, h ,w]
        ref_labels, ref_images = data['ref_labels'], data['ref_images']
        prev_labels, prev_images = data['prev_labels'], data['prev_images']
        is_first_frame = prev_labels is None

        if self.is_pose_data:
            label, prev_labels = extract_valid_pose_labels([label, prev_labels], 
                self.pose_type, self.remove_face_labels)
            ref_labels = extract_valid_pose_labels(ref_labels, self.pose_type, self.remove_face_labels)
        
        b, k, c, h, w = ref_images.shape
        ref_images = L.reshape(ref_images, (b * k, -1, h, w))
        if ref_labels is not None:
            ref_labels = L.reshape(ref_labels, (b * k, -1, h, w))
        
        # Encode the reference images to get the features.
        x, encoded_ref, atn, atn_vis, ref_idx = self.reference_encoder(ref_images, ref_labels, label, k)

        embedding_weights, norm_weights, conv_weights = self.weight_generator(encoded_ref, k, is_first_frame)
    
        # Encode the target label to get the encoded features.
        encoded_label = self.label_embedding(label, weights=(embedding_weights if self.use_hyper_embed else None)) 

        ref_images = L.reshape(ref_images, (b, k, -1, h, w))
        if ref_labels is not None:
            ref_labels = L.reshape(ref_labels, (b, k, -1, h, w))
        # Flow estimation.
        flow, flow_mask, img_warp, cond_inputs = self.flow_generation(label, ref_labels, ref_images, prev_labels, prev_images, ref_idx)
     
        encoded_ref = x
        for i in range(len(encoded_label)):
            encoded_label[i] = [encoded_label[i]]
        if self.generate_raw_output:
            encoded_label_raw = [encoded_label[i] for i in range(self.num_multi_spae_layers)]
        else:
            encoded_label_raw = None

        encoded_label = self.SPADE_combine(encoded_label, cond_inputs)

        img_final, img_raw = self.main_generator(encoded_ref, encoded_label, encoded_label_raw, conv_weights, norm_weights)

        # img_final = img_final * flow_mask[0] + img_warp[0] * (1 - flow_mask[0])
        output = dict()
        output['fake_images'] = img_final
        output['fake_flow_maps'] = flow
        output['fake_occlusion_masks'] = flow_mask
        output['fake_raw_images'] = img_raw
        output['warped_images'] = img_warp
        output['attention_visualization'] = atn_vis
        output['ref_idx'] = ref_idx

        return output


    def flow_generation(self, label, ref_labels, ref_images, prev_labels, prev_images, ref_idx):
        """
        Generates flows and masks for warping reference / previous images.

        Args:
            label (NxCxHxW): Target label map. 
            ref_labels (NxKxCxHxW): Reference label maps.
            ref_images (NxKx3xHxW): Reference images.
            prev_labels (NxTxCxHxW): Previous label maps.
            prev_images (NxTx3xHxW): Previous images.
            ref_idx (Nx1): index for which image to use from the reference images.

        Returns:
            - flow (list of Nx2xHxW): Optical flows.
            - occ_mask (list of Nx1xHxW): Occlusion masks.
            - img_warp (list of Nx3xHxW): Warped reference /previous images.
            - cond_inputs (list of Nx4xHxW): conditional inputs for SPADE combination
        """
        # Pick an image in the reference imagegs using ref_idx.
        ref_label, ref_image = pick_image([ref_labels, ref_images], ref_idx)

        # Only start using prev frames when enough prev frames are generated.
        has_prev = prev_labels is not None and prev_labels.shape[1] == self.num_frames_G - 1

        flow, occ_mask, img_warp, cond_inputs = [None] * 2, [None] * 2, [None] * 2, [None] * 2

        if self.warp_ref:
            # Generate flows / masks for warping the reference image.
            flow_ref, occ_mask_ref = self.flow_network_ref(label, ref_label, ref_image)
        
            ref_image_warp = resample(ref_image, flow_ref)
            flow[0], occ_mask[0], img_warp[0] = flow_ref, occ_mask_ref, ref_image_warp[:, :3]

            # Concat warped image and occlusion mask to form the conditional input.
            cond_inputs[0] = L.concat([img_warp[0], occ_mask[0]], axis=1)
        
        if self.temporal_initialized and has_prev:
            # Generate flows / masks for warping the previous image.
            b, t, c, h, w = prev_labels.shape
            prev_labels_concat = L.reshape(prev_labels, (b, -1, h, w))
            prev_images_concat = L.reshape(prev_images, (b, -1, h, w))
            flow_prev, occ_mask_prev = self.flow_network_temp(label, prev_labels_concat, prev_images_concat)

            img_prev_warp = resample(prev_images[:, -1], flow_prev)
            flow[1], occ_mask[1], img_warp[1] = flow_prev, occ_mask_prev, img_prev_warp
            cond_inputs[1] = L.concat([img_warp[1], occ_mask[1]], axis=1)

        return flow, occ_mask, img_warp, cond_inputs


    def SPADE_combine(self, encoded_label, cond_inputs):
        """
        Using Multi-SPADE to combine raw synthesized image with warped images.

        Args:
            encoded_label (list of tensors): Original label map embeddings.
            cond_inputs (list of tensors): New SPADE conditional inputs from the warped images.
        
        Returns:
            encoded_label (list of tensors): Combined conditional inputs.
        """
        # Generate the conditional embeddings from inputs.
        embedded_img_feat = [None, None]
        if cond_inputs[0] is not None:
            embedded_img_feat[0] = self.ref_image_embedding(cond_inputs[0])
        if cond_inputs[1] is not None:
            embedded_img_feat[1] = self.prev_image_embedding(cond_inputs[1])

         # Combone the original encoded label maps with new conditional embedding.
        for i in range(self.num_multi_spade_layers):
            encoded_label[i] += [w[i] if w is not None else None for w in embedded_img_feat]
        
        return encoded_label
    

    def reset(self, ):
        self.weight_generator.reset()

