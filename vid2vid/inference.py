# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------
import argparse
import sys
sys.path.append('/home/aistudio')

from vid2vid.configs.config import Config
from vid2vid.datasets.dataset import get_test_data_loader
from vid2vid.model.build_model import get_model_optimizer_and_scheduler
from vid2vid.utils.trainer import get_trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluating')
    parser.add_argument('--config', default='/home/aistudio/vid2vid/configs/dancing_ampO1.yaml', help='Path to the training config file.')
    parser.add_argument('--logdir', default='/home/aistudio/vid2vid/outputs/inference', help='Dir for saving evaluation results.')
    parser.add_argument('--checkpoint', help='Dir for loading models.')
    parser.add_argument('--output_dir', default='/home/aistudio/vid2vid/outputs/inference', help='Location to save the image outputs')
    parser.add_argument('--save_fake_only', default=True)
    parser.add_argument('--seq_path', type=str)
    parser.add_argument('--ref_img_path', type=str)
    parser.add_argument('--driving_seq_index', default=0, type=int)
    parser.add_argument('--few_shot_seq_index', default=0, type=int)
    parser.add_argument('--max_seq_length', default=128, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config(args.config)

    cfg.is_inference = True
    cfg.logdir = args.logdir

    cfg.inference_data.seq_path = args.seq_path
    cfg.inference_data.ref_img_path = args.ref_img_path
    cfg.inference_args.max_seq_length = args.max_seq_length
    cfg.inference_args.save_fake_only = args.save_fake_only
    cfg.inference_args.driving_seq_index = args.driving_seq_index
    cfg.inference_args.few_shot_seq_index = args.few_shot_seq_index
    
    test_data_loader = get_test_data_loader(cfg)
    net_G, net_D, opt_G, opt_D, sch_G, sch_D = get_model_optimizer_and_scheduler(cfg)
    trainer = get_trainer(cfg, net_G, net_D, opt_G, opt_D, sch_G, sch_D, None, test_data_loader)

    trainer.load_checkpoint(cfg, args.checkpoint)
    trainer.current_epoch = -1
    trainer.current_iteration = -1
    trainer.test(test_data_loader, args.output_dir, cfg.inference_args)

    print("Done with inference!!!")


if __name__ == '__main__':
    main()

