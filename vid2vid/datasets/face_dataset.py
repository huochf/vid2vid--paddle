# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/few-shot-vid2vid)
# -----------------------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved. 
# Licensed under the Nvidia Source Code License.
# -----------------------------------------------------------------------
import os.path as path
from PIL import Image
import numpy as np 
import random

from vid2vid.datasets.base_dataset import BaseDataset
from vid2vid.datasets.image_folder import make_dataset, make_grouped_dataset, check_path_valid
from vid2vid.datasets.augmentation import get_img_params, get_video_params, get_transform
from vid2vid.datasets.keypoint2img import interp_points, draw_edge


class FaceDataset(BaseDataset):

    def initialize(self, cfg):
        self.cfg = cfg
        root = cfg.dataroot

        if cfg.isTrain:            
            self.L_paths = sorted(make_grouped_dataset(path.join(root, 'train_keypoints'))) 
            self.I_paths = sorted(make_grouped_dataset(path.join(root, 'train_images')))
            check_path_valid(self.L_paths, self.I_paths)
        else:
            self.L_paths = sorted(make_grouped_dataset(cfg.seq_path.replace('images', 'keypoints')))
            self.I_paths = sorted(make_grouped_dataset(cfg.seq_path))

            self.ref_L_paths = sorted(make_grouped_dataset(cfg.ref_img_path.replace('images', 'keypoints')))
            self.ref_I_paths = sorted(make_grouped_dataset(cfg.ref_img_path))

            self.inference_sequence_idx = 0
            self.inference_k_shot_sequence_idx = 0
            self.inference_k_shot_frame_idx = 0

        self.n_of_seqs = len(self.I_paths)                         # number of sequences to train 
        if cfg.isTrain: print('%d sequences' % self.n_of_seqs)        

        # mapping from keypoints to face part 
        self.add_upper_face = not cfg.no_upper_face
        self.part_list = [[list(range(0, 17)) + ((list(range(68, 83)) + [0]) if self.add_upper_face else [])], # face
                     [range(17, 22)],                                  # right eyebrow
                     [range(22, 27)],                                  # left eyebrow
                     [[28, 31], range(31, 36), [35, 28]],              # nose
                     [[36,37,38,39], [39,40,41,36]],                   # right eye
                     [[42,43,44,45], [45,46,47,42]],                   # left eye
                     [range(48, 55), [54,55,56,57,58,59,48], range(60, 65), [64,65,66,67,60]], # mouth and tongue
                    ]        
        self.ref_dist_x, self.ref_dist_y = [None] * 83, [None] * 83
        self.dist_scale_x, self.dist_scale_y = [None] * 83, [None] * 83        
        self.fix_crop_pos = True


    def num_inference_sequences(self, ):
        return self.n_of_seqs


    def set_inference_sequence_idx(self, sequence_idx, k_shot_index, k_shot_frame_index):
        assert not self.cfg.isTrain
        assert sequence_idx < self.n_of_seqs
        assert k_shot_index < self.n_of_seqs
        assert k_shot_frame_index < len(self.ref_I_paths[k_shot_index])
        self.inference_sequence_idx = sequence_idx
        self.inference_k_shot_sequence_idx = k_shot_index
        self.inference_k_shot_frame_idx = k_shot_frame_index


    def get_items(self, index): 
        cfg = self.cfg
        if cfg.isTrain:
            # np.random.seed()
            seq_idx = random.randrange(self.n_of_seqs)
            L_paths = self.L_paths[seq_idx]
            I_paths = self.I_paths[seq_idx]
            ref_L_paths, ref_I_paths = L_paths, I_paths
        else:
            seq_idx = self.inference_sequence_idx
            L_paths = self.L_paths[seq_idx]
            I_paths = self.I_paths[seq_idx]

            seq_idx = self.inference_k_shot_sequence_idx
            ref_L_paths, ref_I_paths = self.ref_L_paths[seq_idx], self.ref_I_paths[seq_idx]
                              
        n_frames_total, start_idx, t_step, ref_indices = get_video_params(cfg, self.n_frames_total, len(I_paths), index)
        w, h = cfg.fineSize, int(cfg.fineSize / cfg.aspect_ratio)
        img_params = get_img_params(cfg, (w, h))        
        is_first_frame = cfg.isTrain or True # index == 0
        
        transform_L = get_transform(cfg, img_params, method=Image.BILINEAR, normalize=False)        
        transform_I = get_transform(cfg, img_params, color_aug=cfg.isTrain)

        ### read in reference images
        Lr, Ir = self.Lr, self.Ir
        if is_first_frame:           
            # get crop coordinates and stroke width
            keypoints = self.read_data(ref_L_paths[ref_indices[0]], data_type='np')            
            ref_crop_coords = self.get_crop_coords(keypoints, for_ref=True)
            self.bw = max(1, (ref_crop_coords[1]-ref_crop_coords[0]) // 256)

            # get keypoints for all reference frames
            ref_L_paths = [ref_L_paths[idx] for idx in ref_indices]
            all_keypoints = self.read_all_keypoints(ref_L_paths, ref_crop_coords, is_ref=True)

            # read all reference images
            for i, idx in enumerate(ref_indices):
                keypoints = all_keypoints[i]
                ref_img = self.crop(self.read_data(ref_I_paths[idx]), ref_crop_coords)
                Li = self.get_face_image(keypoints, transform_L, ref_img.size, img_params)
                Ii = transform_I(ref_img, img_params)
                Lr = self.concat_frame(Lr, Li[np.newaxis, :])
                Ir = self.concat_frame(Ir, Ii[np.newaxis, :])
            if not cfg.isTrain:
                pass # self.Lr, self.Ir = Lr, Ir        

        ### read in target images  
        if is_first_frame:
            # get crop coordinates
            keypoints = self.read_data(L_paths[start_idx], data_type='np')
            crop_coords = self.get_crop_coords(keypoints)   
            if not cfg.isTrain: 
                if self.fix_crop_pos: self.crop_coords = crop_coords
                else: self.crop_size = crop_coords[1] - crop_coords[0], crop_coords[3] - crop_coords[2]
            self.bw = max(1, (crop_coords[1]-crop_coords[0]) // 256)

            # get keypoints for all frames
            end_idx = (start_idx + n_frames_total * t_step) if cfg.isTrain else (start_idx + self.cfg.how_many)
            L_paths = L_paths[start_idx : end_idx : t_step]             
            all_keypoints = self.read_all_keypoints(L_paths, crop_coords if self.fix_crop_pos else None, is_ref=False)        
            # if not cfg.isTrain: self.all_keypoints = all_keypoints
        else:
            # use same crop coordinates as previous frames
            if self.fix_crop_pos:
                crop_coords = self.crop_coords
            else:                
                keypoints = self.read_data(L_paths[start_idx], data_type='np')
                crop_coords = self.get_crop_coords(keypoints, self.crop_size)
            all_keypoints = self.all_keypoints

        L, I = self.L, self.I
        for t in range(n_frames_total):
            ti = t if cfg.isTrain else start_idx + t
            keypoints = all_keypoints[ti]
            I_path = I_paths[start_idx + t * t_step]                
            img = self.crop(self.read_data(I_path), crop_coords)
            Lt = self.get_face_image(keypoints, transform_L, img.size, img_params)
            It = transform_I(img, img_params)
            L = self.concat_frame(L, Lt[np.newaxis, :])                                
            I = self.concat_frame(I, It[np.newaxis, :])
        if not cfg.isTrain:
            pass # self.L, self.I = L, I        
        seq = path.basename(path.dirname(cfg.ref_img_path)) + '-' + str(cfg.ref_img_id) + '_' + path.basename(path.dirname(cfg.seq_path))

        return_list = {'tgt_label': L, 'tgt_image': I, 'ref_label': Lr, 'ref_image': Ir,
                       'path': I_path, 'seq': seq}
        return return_list

    def read_all_keypoints(self, L_paths, crop_coords, is_ref):
        all_keypoints = [self.read_keypoints(L_path, crop_coords) for L_path in L_paths]  
        if not self.cfg.isTrain or self.n_frames_total > 4:
            self.normalize_faces(all_keypoints, is_ref=is_ref)
        return all_keypoints
    
    def get_face_image(self, keypoints, transform_L, size, img_params):   
        w, h = size
        edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges
        # edge map for face region from keypoints
        im_edges = np.zeros((h, w), np.uint8) # edge map for all edges
        for edge_list in self.part_list:
            for edge in edge_list:
                im_edge = np.zeros((h, w), np.uint8) # edge map for the current edge
                for i in range(0, max(1, len(edge)-1), edge_len-1): # divide a long edge into multiple small edges when drawing
                    sub_edge = edge[i:i+edge_len]
                    x = keypoints[sub_edge, 0]
                    y = keypoints[sub_edge, 1]
                                    
                    curve_x, curve_y = interp_points(x, y) # interp keypoints to get the curve shape
                    draw_edge(im_edges, curve_x, curve_y, bw=self.bw)
        input_tensor = transform_L(Image.fromarray(im_edges), img_params)
        return input_tensor.astype("float32")

    def read_keypoints(self, L_path, crop_coords):                    
        keypoints = self.read_data(L_path, data_type='np')
     
        if crop_coords is None:
            crop_coords = self.get_crop_coords(keypoints) 
        keypoints[:, 0] -= crop_coords[2]
        keypoints[:, 1] -= crop_coords[0]
        
        # add upper half face by symmetry
        if self.add_upper_face:
            pts = keypoints[:17, :].astype(np.int32)
            baseline_y = (pts[0,1] + pts[-1,1]) / 2
            upper_pts = pts[1:-1,:].copy()
            upper_pts[:,1] = baseline_y + (baseline_y-upper_pts[:,1]) * 2 // 3
            keypoints = np.vstack((keypoints, upper_pts[::-1,:])) 

        return keypoints

    def get_crop_coords(self, keypoints, crop_size=None, for_ref=False):           
        min_y, max_y = int(keypoints[:,1].min()), int(keypoints[:,1].max())
        min_x, max_x = int(keypoints[:,0].min()), int(keypoints[:,0].max())
        x_cen, y_cen = (min_x + max_x) // 2, (min_y + max_y) // 2                
        w = h = (max_x - min_x)
        if crop_size is not None:
            h, w = crop_size[0] / 2, crop_size[1] / 2
        if self.cfg.isTrain and self.fix_crop_pos:
            offset_max = 0.2
            offset = [random.uniform(-offset_max, offset_max), 
                      random.uniform(-offset_max, offset_max)]
            if for_ref:
                scale_max = 0.2
                self.scale = [random.uniform(1 - scale_max, 1 + scale_max), 
                              random.uniform(1 - scale_max, 1 + scale_max)]
            w *= self.scale[0]
            h *= self.scale[1]
            x_cen += int(offset[0]*w)
            y_cen += int(offset[1]*h)
                        
        min_x = x_cen - w
        min_y = y_cen - h*1.25
        max_x = min_x + w*2        
        max_y = min_y + h*2

        return int(min_y), int(max_y), int(min_x), int(max_x)

    def normalize_faces(self, all_keypoints, is_ref=False):        
        central_keypoints = [8]
        face_centers = [np.mean(keypoints[central_keypoints,:], axis=0) for keypoints in all_keypoints]        
        compute_mean = not is_ref
        if compute_mean:
            if self.cfg.isTrain:
                img_scale = 1
            else:
                img_scale = self.img_scale / (all_keypoints[0][:,0].max() - all_keypoints[0][:,0].min())

        part_list = [[0,16], [1,15], [2,14], [3,13], [4,12], [5,11], [6,10], [7,9, 8], # face 17
                     [17,26], [18,25], [19,24], [20,23], [21,22], # eyebrows 10
                     [27], [28], [29], [30], [31,35], [32,34], [33], # nose 9
                     [36,45], [37,44], [38,43], [39,42], [40,47], [41,46], # eyes 12
                     [48,54], [49,53], [50,52], [51], [55,59], [56,58], [57], # mouth 12
                     [60,64], [61,63], [62], [65,67], [66], # tongue 8                     
                    ]
        if self.add_upper_face:
            part_list += [[68,82], [69,81], [70,80], [71,79], [72,78], [73,77], [74,76, 75]] # upper face 15

        for i, pts_idx in enumerate(part_list):            
            if compute_mean or is_ref:                
                mean_dists_x, mean_dists_y = [], []
                for k, keypoints in enumerate(all_keypoints):
                    pts = keypoints[pts_idx]
                    pts_cen = np.mean(pts, axis=0)
                    face_cen = face_centers[k]                    
                    for p, pt in enumerate(pts):                        
                        mean_dists_x.append(np.linalg.norm(pt - pts_cen))                        
                        mean_dists_y.append(np.linalg.norm(pts_cen - face_cen))
                mean_dist_x = sum(mean_dists_x) / len(mean_dists_x) + 1e-3                
                mean_dist_y = sum(mean_dists_y) / len(mean_dists_y) + 1e-3                
            if is_ref:
                self.ref_dist_x[i] = mean_dist_x
                self.ref_dist_y[i] = mean_dist_y
                self.img_scale = all_keypoints[0][:,0].max() - all_keypoints[0][:,0].min()
            else:
                if compute_mean:                    
                    self.dist_scale_x[i] = self.ref_dist_x[i] / mean_dist_x / img_scale
                    self.dist_scale_y[i] = self.ref_dist_y[i] / mean_dist_y / img_scale                    

                for k, keypoints in enumerate(all_keypoints):
                    pts = keypoints[pts_idx]                    
                    pts_cen = np.mean(pts, axis=0)
                    face_cen = face_centers[k]                    
                    pts = (pts - pts_cen) * self.dist_scale_x[i] + (pts_cen - face_cen) * self.dist_scale_y[i] + face_cen                    
                    all_keypoints[k][pts_idx] = pts

    def __len__(self):        
        if not self.cfg.isTrain: return len(self.L_paths)
        return max(10000, max([len(A) for A in self.L_paths]))  # max number of frames in the training sequences

    def name(self):
        return 'FaceDataset'

    def reset(self):
        pass
