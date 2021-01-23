import paddle.fluid as F 

model_path = '/home/aistudio/vid2vid/model/backbones/InceptionV4_pretrained'
state_dict = F.io.load_program_state(model_path,)

new_state_dict = {}

renamed_dict = {
    'scale': 'weight',
    'offset': 'bias',
    'mean': '_mean',
    'variance': '_variance',
    'weights': 'weight',
}

for k, v in state_dict.items():
    if k[:4] == 'conv':
        new_name = ['stem']
        new_name.append(k[:12])
        if k.split('_')[-2] == 'bn':
            new_name.append('bn')
        else:
            new_name.append('conv')
        new_name.append(renamed_dict[k.split('_')[-1]])
        new_state_dict['.'.join(new_name)] = v
    
    elif k.split('_')[0] == 'inception' and 'stem' in k:
        new_name = ['stem']
        if '_bn' in k:
            offset = k.find('_bn')
        elif '_weights' in k:
            offset = k.find('_weights')
        else:
            exit(0)
        new_name.append(k[10: offset])
        if k.split('_')[-2] == 'bn':
            new_name.append('bn')
        else:
            new_name.append('conv')
        new_name.append(renamed_dict[k.split('_')[-1]])
        new_state_dict['.'.join(new_name)] = v

    elif k.split('_')[0] == 'inception':
        new_name = [k[:11]]
        new_name.append(str(int(k[11]) - 1))
        if '_bn' in k:
            offset = k.find('_bn')
        elif '_weights' in k:
            offset = k.find('_weights')
        else:
            exit(0)
        new_name.append('conv_' + k[13: offset])
        if k.split('_')[-2] == 'bn':
            new_name.append('bn')
        else:
            new_name.append('conv')
        new_name.append(renamed_dict[k.split('_')[-1]])
        new_state_dict['.'.join(new_name)] = v
    
    elif k.split('_')[0] == 'reduction':
        new_name = [k[:11]]
        if '_bn' in k:
            offset = k.find('_bn')
        elif '_weights' in k:
            offset = k.find('_weights')
        else:
            exit(0)
        new_name.append('conv_' + k[12: offset])
        if k.split('_')[-2] == 'bn':
            new_name.append('bn')
        else:
            new_name.append('conv')
        new_name.append(renamed_dict[k.split('_')[-1]])
        new_state_dict['.'.join(new_name)] = v
    else:
        print(k)

import sys
sys.path.append('/home/aistudio')
from vid2vid.model.backbones.inception_v4 import InceptionV4

model = InceptionV4()
model.set_dict(new_state_dict)

import paddle.fluid as F 
F.save_dygraph(model.state_dict(), '/home/aistudio/vid2vid/model/backbones/inceptionv4')
