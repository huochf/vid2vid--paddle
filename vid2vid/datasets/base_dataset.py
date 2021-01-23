# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/few-shot-vid2vid)
# -----------------------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved. 
# Licensed under the Nvidia Source Code License.
# -----------------------------------------------------------------------
from PIL import Image
import numpy as np
import random

import paddle
import paddle.fluid.layers as L


class BaseDataset():

    def __init__(self):
        super(BaseDataset, self).__init__()
        self.L = self.I = self.Lr = self.Ir = None
        self.n_frames_total = 1 # current number of frames to train in a single iteration
        self.use_lmdb = False
    

    def name(self):
        return "BaseDataset"
    

    def create_reader(self, ):
        def reader():
            for i in range(self.cfg.max_iter_per_epoch):
                index = len(self) // self.cfg.max_iter_per_epoch * i + random.randint(0, len(self) // self.cfg.max_iter_per_epoch)
                yield index
            # for count in range(len(self)):
            #     yield count
        
        return paddle.reader.xmap_readers(self.get_items, reader, 3, 512)
    

    def test_reader(self, ):
        def reader():
            for i in range(len(self.inference_sequence_idx)):
                yield i
            # for count in range(len(self)):
            #     yield count
        
        return paddle.reader.xmap_readers(self.get_items, reader, 1, 512)


    def batch_reader(self, batch_size):
        reader = self.create_reader()

        def _batch_reader():
            batch_out = []
            for data_list in reader():
                if data_list is None:
                    continue
                batch_out.append(data_list)
                if len(batch_out) == batch_size:
                    yield batch_out
                    batch_out = []
        return _batch_reader
    

    
    # def test_reader(self):
        
    #     def _reader():
    #         for i in range(12):
    #             print(i)
    #             yield i

    #     reader = paddle.reader.xmap_readers(self.get_items, _reader, 1, 512)

    #     def _test_reader():
    #         batch_out = []
    #         for data_list in reader():
    #             if data_list is None:
    #                 continue
    #             batch_out.append(data_list)
    #             yield batch_out
    #             batch_out = []
    #     return _test_reader


    def update_training_batch(self, ratio):
        # update the training sequence length to be longer
        seq_len_max = 30
        if self.n_frames_total < seq_len_max:
            self.n_frames_total = min(seq_len_max, self.cfg.n_frames_total * (2 ** ratio))
            print('-- Updating training sequence length to %d ---' % self.n_frames_total)
    

    def read_data(self, path, lmdb=None, data_type='img'):
        is_img = data_type == 'img'
        if self.use_lmdb and lmdb is not None:
            img, _ = lmdb.getitem_by_path(path.encode(), is_img)
            if is_img and len(img.mode) == 3:
                b, g, r = img.split()
                img = Image.merge("RGB", (r, g, b))
            elif data_type == 'np':
                img = img.decode()
                img = np.array([[int(j) for j in i.split(',')] for i in img.splitlines()])
        elif is_img:
            img = Image.open(path)
        elif data_type == 'np':
            img = np.loadtxt(path, delimiter=',')
        else:
            img = path

        return img
    

    def crop(self, img, coords):
        min_y, max_y, min_x, max_x = coords
        if isinstance(img, np.ndarray):
            return img[min_y:max_y, min_x:max_x]
        else:
            return img.crop((min_x, min_y, max_x, max_y))
    

    def concat_frame(self, A, Ai, n=100):
        if A is None or Ai.shape[0] >= n:
            return Ai[-n:]
        else:
            return np.concatenate([A, Ai])[-n:]
    

    def concat(self, tensors, dim=0):
        tensors = [t for t in tensors if t is not None]
        return np.concatenate(tensors, dim)
