from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.fluid as fluid

from paddle.fluid.dygraph import Conv2D, Pool2D
from paddle.fluid.param_attr import ParamAttr


__all__ = ['VGGNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19']


class Conv3x3(fluid.dygraph.Layer):

    def __init__(self, in_channels, out_channels, act=None, param_attr=None, bias_attr=None):
        super(Conv3x3, self).__init__()
        self.conv = Conv2D(in_channels, out_channels, filter_size=3, stride=1, padding=1,
            act=act, param_attr=param_attr, bias_attr=bias_attr)
    

    def forward(self, inputs):
        return self.conv(inputs)


class MaxPool(fluid.dygraph.Layer):

    def __init__(self, ):
        super(MaxPool, self).__init__()
        self.pool = Pool2D(pool_size=2, pool_type='max', pool_stride=2)
    

    def forward(self, inputs):
        return self.pool(inputs)


class VGGNet(fluid.dygraph.Layer):
    vgg_spec = {
        11: ([1, 1, 2, 2, 2]),
        13: ([2, 2, 2, 2, 2]),
        16: ([2, 2, 3, 3, 3]),
        19: ([2, 2, 4, 4, 4])
    }

    def __init__(self, layers=16):
        super(VGGNet, self).__init__()

        self.layers = layers
        assert layers in self.vgg_spec.keys(), \
            "supported layers are {} but input layer is {}".format(self.vgg_spec.keys(), layers)

        nums = self.vgg_spec[layers]
        self.conv1 = self.conv_block(3, 64, nums[0], name="conv1_")
        self.conv2 = self.conv_block(64, 128, nums[1], name="conv2_")
        self.conv3 = self.conv_block(128, 256, nums[2], name="conv3_")
        self.conv4 = self.conv_block(256, 512, nums[3], name="conv4_")
        self.conv5 = self.conv_block(512, 512, nums[4], name="conv5_")
    

    def forward(self, inputs):
        out_1 = self.conv1(inputs)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out_4 = self.conv4(out_3)
        out_5 = self.conv5(out_4)

        return [out_1, out_2, out_3, out_4, out_5]


    def conv_block(self, in_channels, out_channels, groups, name=None):
        convs = []
        for i in range(groups):
            convs.append((name + str(i + 1), Conv3x3(in_channels, out_channels,
                act='relu',
                param_attr=ParamAttr(name=name + str(i + 1) + '_weights'),
                bias_attr=False)))
            in_channels = out_channels
        convs.append((name + 'maxpool', MaxPool()))
        convs = fluid.dygraph.Sequential(*convs)
        return convs


def VGG11():
    return VGGNet(layers=11)


def VGG13():
    return VGGNet(layers=13)
    

def VGG16():
    return VGGNet(layers=16)
    

def VGG19():
    return VGGNet(layers=19)


def build_model():
    model = VGG19()
    model_path = '/home/aistudio/vid2vid/model/losses/vgg_pretrained/dygraph'
    param_dict, _ = fluid.dygraph.load_dygraph(model_path)
    model.set_dict(param_dict)
    print("load pretrained VGG19 model from " + model_path)
    return model


if __name__ == '__main__':
    import numpy as np
    from paddle.fluid.dygraph import to_variable

    with paddle.fluid.dygraph.guard():
        model = VGG19()
        model_path = '/home/aistudio/vid2vid/pretrained_models/vgg19/dygraph'
        param_dict, _ = fluid.dygraph.load_dygraph(model_path)
        model.load_dict(param_dict)
        print(param_dict.keys())
        print(model.state_dict().keys())

        image = to_variable(np.zeros((1, 3, 256, 256)).astype('float32'))
        feature_maps = model(image)
        for feature in feature_maps:
            print(feature.shape)
            # [1, 64, 128, 128]
            # [1, 128, 64, 64]
            # [1, 256, 32, 32]
            # [1, 512, 16, 16]
            # [1, 512, 8, 8]

