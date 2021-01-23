import random
from PIL import Image
import numpy as np


class Compose():

    def __init__(self, transforms):
        self.transforms = transforms
    

    def __call__(self, image, params):
        for t in self.transforms:
            image = t(image, params)
        return image


class ScaleImage():

    def __init__(self, method=Image.NEAREST):
        self.method = method
    

    def __call__(self, image, params):
        w, h = params['new_size']
        return image.resize((w, h), self.method)


class Crop():

    def __call__(self, image, params):
        ow, oh = image.size
        x1, y1 = params['crop_pos']
        tw, th = params['crop_size']
        return image.crop((x1, y1, x1 + tw, y1 + th))


class Flip():

    def __call__(self, image, params):
        flip = params['flip']
        if flip:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return image


class ColorAUG():

    def __call__(self, image, params):
        colors = params['color_aug']
        h, s, v = image.convert('HSV').split()
        h = h.point(lambda i: (i + colors[0]) % 256)
        s = s.point(lambda i: min(255, max(0, i * colors[1] + colors[2])))
        v = v.point(lambda i: min(255, max(0, i * colors[3] + colors[4])))
        image = Image.merge('HSV', (h, s, v)).convert('RGB')

        return image


class Normalize():

    def __call__(self, image, params):
        image = np.array(image).astype("float32")
        image /= 255.0
        image -= np.array(params['mean'])
        image /= np.array(params['std'])
        image = image.transpose((2, 0, 1))
        return image


class ToTensor():

    def __call__(self, image, params):
        image = np.array(image)
        if len(image.shape) < 3:
            return image[np.newaxis, :]
        else:
            return image


def get_img_params(cfg, size):
    w, h = size
    new_w, new_h = w, h

    # resize input image
    if 'resize' in cfg.resize_or_crop:
        new_h = new_w = cfg.loadSize
    else:
        if 'scale_width' in cfg.resize_or_crop:
            new_w = cfg.loadSize
        elif 'random_scale' in cfg.resize_or_crop:
            new_w = random.randrange(int(cfg.fineSize), int(1.2 * cfg.fineSize))
        new_h = int(new_w * h) // w
    
    if 'crop' not in cfg.resize_or_crop:
        new_h = int(new_w // cfg.aspect_ratio)
    
    new_w = new_w // 4 * 4
    new_h = new_h // 4 * 4

    # crop resized image
    size_x = min(cfg.loadSize, cfg.fineSize)
    size_y = size_x // cfg.aspect_ratio

    if not cfg.isTrain: # crop central region
        pos_x = (new_w - size_x) // 2
        pos_y = (new_h - size_y) // 2
    else:
        pos_x = random.randrange(np.maximum(1, new_w - size_x))
        pos_y = random.randrange(np.maximum(1, new_h - size_y))
    
    # for color augmentation
    h_b = random.uniform(-30, 30)
    s_a = random.uniform(0.8, 1.2)
    s_b = random.uniform(-10, 10)
    v_a = random.uniform(0.8, 1.2)
    v_b = random.uniform(-10, 10)

    flip = random.random() > 0.5
    return {'new_size': (new_w, new_h), 'crop_pos': (pos_x, pos_y),
            'crop_size': (size_x, size_y), 'flip': flip, 
            'color_aug': (h_b, s_a, s_b, v_a, v_b),
            'mean': cfg.mean, 'std': cfg.std}


def get_video_params(cfg, n_frames_total, cur_seq_len, index):
    if cfg.isTrain:                
        n_frames_total = min(cur_seq_len, n_frames_total)             # total number of frames to load
        max_t_step = min(cfg.max_t_step, (cur_seq_len-1) // max(1, (n_frames_total-1)))        
        t_step = random.randrange(max_t_step) + 1                     # spacing between neighboring sampled frames                
        
        offset_max = max(1, cur_seq_len - (n_frames_total-1)*t_step)  # maximum possible frame index for the first frame
        if 'pose' in cfg.dataset_mode:
            start_idx = index % offset_max                            # offset for the first frame to load
            max_range, min_range = 60, 14                             # range for possible reference frames
        else:
            start_idx = random.randrange(offset_max)                  # offset for the first frame to load        
            max_range, min_range = 300, 14                            # range for possible reference frames
        
        ref_range = list(range(max(0, start_idx - max_range), max(1, start_idx - min_range))) \
                  + list(range(min(start_idx + min_range, cur_seq_len - 1), min(start_idx + max_range, cur_seq_len)))
        ref_indices = random.sample(ref_range, cfg.n_shot)       # indices for reference frames

    else:
        n_frames_total = 1
        start_idx = index
        t_step = 1
        ref_indices = cfg.ref_img_id   
        if isinstance(ref_indices, int):
            ref_indices = [ref_indices]  
            
    return n_frames_total, start_idx, t_step, ref_indices


def get_transform(cfg, params, method=Image.BICUBIC, normalize=True, toTensor=True, color_aug=False):
    transform_list = []
    transform_list.append(ScaleImage(method))

    if 'crop' in cfg.resize_or_crop:
        transform_list.append(Crop())
    
    if cfg.isTrain and color_aug:
        transform_list.append(ColorAUG())
    
    if cfg.isTrain and cfg.no_flip:
        transform_list.append(Flip())
    
    if toTensor:
        transform_list.append(ToTensor())
    
    if normalize:
        transform_list.append(Normalize())
    
    return Compose(transform_list)

