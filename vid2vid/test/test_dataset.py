import sys
sys.path.append('/home/aistudio')
import argparse
import numpy as np

import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg

from vid2vid.configs.config import Config
from vid2vid.datasets.pose_dataset import PoseDataset
from vid2vid.datasets.face_dataset import FaceDataset
from vid2vid.model.generator.fewshot_gen_model import FewShotGenerator
from vid2vid.utils.visualize import visualize_dataset_image, visualize_dataset_face


def parse_args():
    parser = argparse.ArgumentParser(description='Test Model')
    parser.add_argument('--config', type=str, default='/home/aistudio/vid2vid/configs/face_ampO1.yaml',
        help='Path to the training config file.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config(args.config)

    # dataset = PoseDataset()
    # dataset.initialize(cfg.train_data)

    # reader = dataset.create_reader()

    # for i, data in enumerate(reader()):
    #     print(i)
    #     for k, v in data.items():
    #         if isinstance(v, str):
    #             print(k + ": " + v)
    #         else:
    #             print(k + ": " + str(v.shape))

    #     vis_path = '/home/aistudio/vid2vid/outputs/visualize/test_dataset/pose'
    #     visualize_dataset_image(data['tgt_image'], vis_path, 'target_image' + str(i))
    #     visualize_dataset_image(data['tgt_label'][:, :3], vis_path, 'densepose' + str(i))
    #     visualize_dataset_image(data['tgt_label'][:, 3:], vis_path, 'openpose' + str(i))

    # for i in range(dataset.num_inference_sequences()):
    #     dataset.set_inference_sequence_idx(i, i, 0)
    #     for i, data in enumerate(dataset.create_reader()()):
    #         print(i)
    #         for k, v in data.items():
    #             if isinstance(v, str):
    #                 print(k + ": " + v)
    #             else:
    #                 print(k + ": " + str(v.shape))

    dataset = FaceDataset()
    dataset.initialize(cfg.inference_data)
    reader = dataset.create_reader()

    # for i, data in enumerate(reader()):
    #     print(i)
    #     for k, v in data.items():
    #         if isinstance(v, str):
    #             print(k + ": " + v)
    #         else:
    #             print(k + ": " + str(v.shape))
    #     vis_path = '/home/aistudio/vid2vid/outputs/visualize/test_dataset/face'
    #     visualize_dataset_image(data['tgt_image'], vis_path, 'target_image' + str(i))
    #     visualize_dataset_face(data['tgt_label'], vis_path, 'face' + str(i))
    
    for i in range(dataset.num_inference_sequences()):
        dataset.set_inference_sequence_idx(i, i, 0)
        for i, data in enumerate(dataset.create_reader()()):
            print(i)
            for k, v in data.items():
                if isinstance(v, str):
                    print(k + ": " + v)
                else:
                    print(k + ": " + str(v.shape))
            vis_path = '/home/aistudio/vid2vid/outputs/visualize/test_dataset/face'
            visualize_dataset_image(data['tgt_image'], vis_path, 'target_image' + str(i))
            visualize_dataset_face(data['tgt_label'], vis_path, 'face' + str(i))

