# Cell 1
import pickle

with open('/home/aistudio/vid2vid/pretrained_models/face_checkpoint.pkl', 'rb+') as f:
    weight_dict = pickle.load(f)

# Cell 2
weight_dict.keys()

# Cell 3
renamed_dict = {}

for k, v in weight_dict['net_G'].items():

    if 'averaged_model' in k:
        pass
    elif 'weight_generator' in k and 'ref' in k:
        header = 'reference_encoder'
        k_list = [header] + k.split('.')[3:]
        k_list.remove('layers')

        if 'weight_u' in k_list or 'weight_v' in k_list:
            k_list.insert(-1, 'spectral_norm')
        
        elif 'norm' in k_list and 'weight' in k_list:
            k_list[-1] = 'scale'
        
        elif 'bias' in k_list and 'conv' in k_list:
            k_list.insert(-1, 'layer')
        
        new_k = '.'.join(k_list)
        renamed_dict[new_k] = v
    elif 'weight_generator' in k and 'fc_spade' in k:
        header = 'weight_generator'
        k_list = [header] + k.split('.')[3:]

        k_list.remove('layers')

        if 'bias' in k_list:
            k_list.insert(-1, 'layer')
        elif 'weight_u' in k_list or 'weight_v' in k_list:
            k_list.insert(-1, 'spectral_norm')

            if 'weight_u' in k_list:
                k_list[-1] = 'weight_v'
            elif 'weight_v' in k_list:
                k_list[-1] = 'weight_u'
        
        new_k = '.'.join(k_list)
        if len(v.shape) == 2:
            renamed_dict[new_k] = v.transpose((1, 0))
        else:
            renamed_dict[new_k] = v
    
    elif 'label_embedding' in k:
        header = 'label_embedding'
        k_list = [header] + k.split('.')[4:]
        k_list.remove('layers')

        if 'bias' in k_list:
            k_list.insert(-1, 'layer')
        elif 'weight_u' in k_list or 'weight_v' in k_list:
            k_list.insert(-1, 'spectral_norm')
        
        new_k = '.'.join(k_list)
        renamed_dict[new_k] = v
    
    elif 'module.module.conv_img' in k:
        k_list = ['main_generator'] + k.split('.')[2:]
        k_list.remove('layers')
        new_k = '.'.join(k_list)

        renamed_dict[new_k] = v
    
    elif 'module.module.up_' in k:
        header = 'main_generator'
        k_list = [header] + k.split('.')[2:]
        k_list.remove('layers')

        if 'weight_u' in k_list or 'weight_v' in k_list:
            k_list.insert(-1, 'spectral_norm')
        elif 'mlp' in k:
            k_list.remove('layers')
        elif 'conv' in k_list and 'bias' in k_list:
            k_list.insert(-1, 'layer')
        
        new_k = '.'.join(k_list)
        renamed_dict[new_k] = v
    
    elif 'flow_network_ref' in k or 'flow_network_temp' in k:
        k_list = k.split('.')[2:]
        k_list.remove('layers')

        if 'norm' in k_list and 'weight' in k_list:
            k_list[-1] = 'scale'
        elif 'bias' in k_list and 'conv' in k_list and ('up' in k or 'res' in k or 'down' in k):
            k_list.insert(-1, 'layer')
        elif 'weight_u' in k_list or 'weight_v' in k_list:
            k_list.insert(-1, 'spectral_norm')

        new_k = '.'.join(k_list)
        renamed_dict[new_k] = v
    
    elif 'ref_image_embedding' in k or 'prev_image_embedding' in k: 
        k_list = k.split('.')[2:]
        k_list.remove('layers')

        if 'weight_u' in k_list or 'weight_v' in k_list:
            k_list.insert(-1, 'spectral_norm')
        elif 'bias' in k_list:
            k_list.insert(-1, 'layer')    
            
        new_k = '.'.join(k_list)
        renamed_dict[new_k] = v   
    else:
        print(k)
    

# Cell 4
import sys
sys.path.append('/home/aistudio')
import argparse

import paddle.fluid.dygraph as dg

from vid2vid.configs.config import Config
from vid2vid.model.generator.fewshot_gen_model import FewShotGenerator

cfg = Config('/home/aistudio/vid2vid/configs/face_ampO1.yaml')

model = FewShotGenerator(cfg.gen, cfg.data)

# Cell 5
model.load_dict(renamed_dict)