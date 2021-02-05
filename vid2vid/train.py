# -----------------------------------------------------------------------
# Modified from imaginaire(https://github.com/NVlabs/imaginaire)
# -----------------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# -----------------------------------------------------------------------
import sys
sys.path.append('/home/aistudio')
import argparse

from vid2vid.configs.config import Config
from vid2vid.datasets.dataset import get_train_and_val_dataloader
from vid2vid.model.build_model import get_model_optimizer_and_scheduler
from vid2vid.utils.trainer import get_trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='/home/aistudio/vid2vid/configs/dancing_ampO1.yaml', help='Path to the training config file.')
    parser.add_argument('--logdir', default='/home/aistudio/vid2vid/outputs/train/logs', help='Dir for saving logs and models.')
    parser.add_argument('--max_epoch', default=2, type=int)
    parser.add_argument('--checkpoint', default='', help='Checkpoint path.')
    parser.add_argument('--max_iter_per_epoch', default=1000, type=int)
    parser.add_argument('--num_epochs_temporal_step', default=5, type=int)
    parser.add_argument('--train_data_root', default='/home/aistudio/data/0_109_pose')
    parser.add_argument('--val_data_root', default='/home/aistudio/data/0_109_pose')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config(args.config)

    cfg.max_epoch = int(args.max_epoch)
    cfg.logdir = args.logdir
    cfg.train_data.max_iter_per_epoch = int(args.max_iter_per_epoch)
    cfg.num_epochs_temporal_step = int(args.num_epochs_temporal_step)
    cfg.train_data.dataroot = args.train_data_root
    cfg.val_data.dataroot = args.val_data_root
    cfg.val_data.seq_path = cfg.val_data.ref_img_path = args.val_data_root + '/images'


    train_dataset, val_dataset = get_train_and_val_dataloader(cfg)
    net_G, net_D, opt_G, opt_D, sch_G, sch_D = get_model_optimizer_and_scheduler(cfg)
    trainer = get_trainer(cfg, net_G, net_D, opt_G, opt_D, sch_G, sch_D, train_dataset, val_dataset)

    current_epoch, current_iteration = trainer.load_checkpoint(cfg, args.checkpoint, resume=True)

    # Start training. 
    for epoch in range(current_epoch, cfg.max_epoch):
        print("Epoch {} ...".format(epoch))
        trainer.start_of_epoch(current_epoch)
        for it, data in enumerate(train_dataset.batch_reader(cfg.train_data.batch_size)()):
            data = trainer.start_of_iteration(data, current_iteration)
            
            for _ in range(cfg.trainer.gen_step):
                trainer.step(data)
            
            current_iteration += 1
            trainer.end_of_iteration(data, current_epoch, current_iteration)
            if current_iteration >= cfg.max_iter:
                print("Done with training!!!")
                return

        current_epoch += 1
        trainer.end_of_epoch(data, current_epoch, current_iteration)
    
    print("Done with training!!!")
    return


if __name__ == '__main__':
    main()
