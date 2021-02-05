# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------
import sys
sys.path.append('/home/aistudio/')
import argparse
import os

from vid2vid.configs.config import Config
from vid2vid.datasets.dataset import get_train_and_val_dataloader, get_val_dataset
from vid2vid.model.build_model import get_model_optimizer_and_scheduler
from vid2vid.utils.trainer import get_trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluating')
    parser.add_argument('--eval_data_dir', type=str)
    parser.add_argument('--config', default='/home/aistudio/vid2vid/configs/dancing_ampO1.yaml', help='Path to the training config file.')
    parser.add_argument('--logdir', default='/home/aistudio/vid2vid/outputs/evaluation', help='Dir for saving evaluation results.')
    parser.add_argument('--checkpoint_logdir', help='Dir for loading models.')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config(args.config)

    cfg.is_inference = True
    cfg.logdir = args.logdir
    cfg.val_data.seq_path = cfg.val_data.ref_img_path = args.eval_data_dir
    
    val_dataset = get_val_dataset(cfg)
    net_G, net_D, opt_G, opt_D, sch_G, sch_D = get_model_optimizer_and_scheduler(cfg)
    trainer = get_trainer(cfg, net_G, net_D, opt_G, opt_D, sch_G, sch_D, None, val_dataset)

    checkpoints = sorted(os.listdir(args.checkpoint_logdir))
    for checkpoint in checkpoints:
        if checkpoint[0] == '.': # skip .ipynb_checkpoints files
            continue
        print(checkpoint)
        path = os.path.join(args.checkpoint_logdir, checkpoint, checkpoint)
        current_epoch, current_iteration = trainer.load_checkpoint(cfg, path, resume=False)
        if current_epoch < 0:
            continue
        trainer.current_epoch = current_epoch
        trainer.current_iteration = current_iteration
        print("Begin evaluate model %s" % path)
        trainer.write_metrics()
    print("Done with evaluation!!!")
    return


if __name__ == '__main__':
    main()

