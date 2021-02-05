import sys
sys.path.append('/home/aistudio')
import argparse
import numpy as np

import paddle.fluid.dygraph as dg

from vid2vid.configs.config import Config
from vid2vid.model.generator.fewshot_gen_model import FewShotGenerator
from vid2vid.model.discriminator.discriminator import Discriminator


def parse_args():
    parser = argparse.ArgumentParser(description='Test Model')
    parser.add_argument('--config', type=str, default='/home/aistudio/vid2vid/configs/dancing_ampO1.yaml',
        help='Path to the training config file.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config(args.config)

    with dg.guard():
        net_G = FewShotGenerator(cfg.gen, cfg.data)
        for k, v in net_G.state_dict().items():
            print(k + ": " + str(v.shape))
        
        data = {}
        data['image'] = dg.to_variable(np.ones((1, 3, 512, 256)).astype("float32"))
        data['label'] = dg.to_variable(np.ones((1, 6, 512, 256)).astype("float32"))
        data['ref_labels'] = dg.to_variable(np.ones((1, 1, 6, 512, 256)).astype("float32"))
        data['ref_images'] = dg.to_variable(np.ones((1, 1, 3, 512, 256)).astype("float32"))
        data['prev_labels'] = None # dg.to_variable(np.ones((1, 1, 6, 512, 256)).astype("float32"))
        data['prev_images'] = None # dg.to_variable(np.ones((1, 1, 3, 512, 256)).astype("float32"))

        net_G_output = net_G(data)

        for k, v in net_G_output.items():
            if isinstance(v, list):
                print(k)
                for v_ in v:
                    if v_ is not None:
                        print(v_.shape)
                    else:
                        print("None")
            else:
                if v is not None:
                    print(k + ": " + str(v.shape))
                else:
                    print(k + ": None")
        # fake_images: [1, 3, 512, 256]
        # fake_flow_maps [[1, 2, 512, 256], None]
        # fake_occlusion_masks [[1, 1, 512, 256], None]
        # fake_raw_images: None
        # warped_images [[1, 3, 512, 256], None]
        # attention_visualization: None
        # ref_idx: None

        net_D = Discriminator(cfg.dis, cfg.data)
        for k, v in net_D.state_dict().items():
            print(k + ": " + str(v.shape))
        
        net_D_output = net_D(data, net_G_output, net_G_output)
        net_D_output = net_D_output[0]
        print(net_D_output['indv']['pred_real']['output'][0].shape) # [1, 1, 31, 15]
        print(net_D_output['indv']['pred_real']['output'][1].shape) # [1, 1, 15,  7]
        print(net_D_output['indv']['pred_fake']['output'][0].shape) # [1, 1, 31, 15]
        print(net_D_output['indv']['pred_fake']['output'][1].shape) # [1, 1, 15,  7]
        print(net_D_output['face']['pred_real']['output'][0].shape) # [1, 1, 15, 15]
        print(net_D_output['face']['pred_fake']['output'][0].shape) # [1, 1, 15, 15]
        

        