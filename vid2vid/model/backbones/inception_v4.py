# ------------------------------------------------------------------
# Modified from https://github.com/PaddlePaddle/models/blob/release/2.0-beta/PaddleCV/image_classification/models/inception_v4.py
# ------------------------------------------------------------------
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
# ------------------------------------------------------------------
import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg


def build():
    model = InceptionV4()
    path = '/home/aistudio/vid2vid/model/backbones/inceptionv4'
    state_dict, _ = dg.load_dygraph(path)
    model.set_dict(state_dict)
    print("load pretrained inception v4 models from path " + path)
    return model


class ConvBN(dg.Layer):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, act='relu'):
        super().__init__()
        self.conv = dg.Conv2D(num_channels=in_channels, num_filters=out_channels, filter_size=kernel_size, 
            stride=stride, padding=padding, bias_attr=False)
        self.bn = dg.BatchNorm(num_channels=out_channels, act=act)
    

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x        


class Stem(dg.Layer):

    def __init__(self, ):
        super().__init__()
        self.conv1_3x3_s2 = ConvBN( 3, 32, 3, 2, 0)
        self.conv2_3x3_s1 = ConvBN(32, 32, 3, 1, 0)
        self.conv3_3x3_s1 = ConvBN(32, 64, 3, 1, 1)

        self.pool1 = dg.Pool2D(pool_size=3, pool_stride=2, pool_type='max')
        self.stem1_3x3_s2 = ConvBN(64, 96, 3, 2, 0)

        self.stem2_3x3_reduce = ConvBN(96 + 64, 64, 1, 1, 0)
        self.stem2_3x3 = ConvBN(64, 96, 3, 1, 0)
        self.stem2_1x7_reduce = ConvBN(64 + 96, 64, 1, 1, 0)
        self.stem2_1x7 = ConvBN(64, 64, (7, 1), 1, (3, 0))
        self.stem2_7x1 = ConvBN(64, 64, (1, 7), 1, (0, 3))
        self.stem2_3x3_2 = ConvBN(64, 96, 3, 1, 0)

        self.stem3_3x3_s2 = ConvBN(192, 192, 3, 2, 0)
        self.pool2 = dg.Pool2D(pool_size=3, pool_stride=2, pool_type='max')
    

    def forward(self, x):
        conv = self.conv1_3x3_s2(x)    # 32
        conv = self.conv2_3x3_s1(conv) # 32
        conv = self.conv3_3x3_s1(conv) # 64

        pool1 = self.pool1(conv)  # 64
        conv1 = self.stem1_3x3_s2(conv) # 96
        concat1 = L.concat([pool1, conv1], 1) # 160

        conv2 = self.stem2_3x3_reduce(concat1) # 64
        conv2 = self.stem2_3x3(conv2)          # 96

        conv3 = self.stem2_1x7_reduce(concat1) # 64
        conv3 = self.stem2_1x7(conv3)   # 64
        conv3 = self.stem2_7x1(conv3)   # 64
        conv3 = self.stem2_3x3_2(conv3) # 96
        concat2 = L.concat([conv2, conv3], 1) # 192

        conv3 = self.stem3_3x3_s2(concat2)   # 192
        pool3 = self.pool2(concat2)           # 192
        concat3 = L.concat([conv3, pool3], 1) # 384

        return concat3


class InceptionA(dg.Layer):

    def __init__(self, in_channels, ):
        super().__init__()
        self.pool = dg.Pool2D(pool_size=3, pool_padding=1, pool_type='avg')
        self.conv_1x1 = ConvBN(in_channels, 96, 1, 1, 0)

        self.conv_1x1_2 = ConvBN(in_channels, 96, 1, 1, 0)

        self.conv_3x3_reduce = ConvBN(in_channels, 64, 1, 1, 0)
        self.conv_3x3 = ConvBN(64, 96, 3, 1, 1)

        self.conv_3x3_2_reduce = ConvBN(in_channels, 64, 1, 1, 0)
        self.conv_3x3_2 = ConvBN(64, 96, 3, 1, 1)
        self.conv_3x3_3 = ConvBN(96, 96, 3, 1, 1)
    

    def forward(self, x):
        pool1 = self.pool(x)
        conv1 = self.conv_1x1(pool1)

        conv2 = self.conv_1x1_2(x)

        conv3 = self.conv_3x3_reduce(x)
        conv3 = self.conv_3x3(conv3)

        conv4 = self.conv_3x3_2_reduce(x)
        conv4 = self.conv_3x3_2(conv4)
        conv4 = self.conv_3x3_3(conv4)

        concat = L.concat([conv1, conv2, conv3, conv4], 1) # 96 * 4

        return concat


class ReductionA(dg.Layer):

    def __init__(self, in_channels=384):
        super().__init__()
        self.pool = dg.Pool2D(pool_size=3, pool_stride=2, pool_type='max')
        self.conv_3x3 = ConvBN(in_channels, 384, 3, 2, 0)
        self.conv_3x3_2_reduce = ConvBN(in_channels, 192, 1, 1, 0)
        self.conv_3x3_2 = ConvBN(192, 224, 3, 1, 1)
        self.conv_3x3_3 = ConvBN(224, 256, 3, 2, 0)

    
    def forward(self, x):
        pool1 = self.pool(x)
        conv2 = self.conv_3x3(x)
        conv3 = self.conv_3x3_2_reduce(x)
        conv3 = self.conv_3x3_2(conv3)
        conv3 = self.conv_3x3_3(conv3)

        concat = L.concat([pool1, conv2, conv3], 1) # 1024
        return concat


class InceptionB(dg.Layer):

    def __init__(self, in_channels=1024):
        super().__init__()
        self.pool = dg.Pool2D(pool_size=3, pool_padding=1, pool_type='avg')
        self.conv_1x1 = ConvBN(in_channels, 128, 1, 1, 0)

        self.conv_1x1_2 = ConvBN(in_channels, 384, 1, 1, 0)

        self.conv_1x7_reduce = ConvBN(in_channels, 192, 1, 1, 0)
        self.conv_1x7 = ConvBN(192, 224, (1, 7), 1, (0, 3))
        self.conv_7x1 = ConvBN(224, 256, (7, 1), 1, (3, 0))

        self.conv_7x1_2_reduce = ConvBN(in_channels, 192, 1, 1, 0)
        self.conv_1x7_2 = ConvBN(192, 192, (1, 7), 1, (0, 3))
        self.conv_7x1_2 = ConvBN(192, 224, (7, 1), 1, (3, 0))
        self.conv_1x7_3 = ConvBN(224, 224, (1, 7), 1, (0, 3))
        self.conv_7x1_3 = ConvBN(224, 256, (7, 1), 1, (3, 0))
    

    def forward(self, x):
        pool1 = self.pool(x)
        conv1 = self.conv_1x1(pool1)     # 128
        conv2 = self.conv_1x1_2(x)      # 384
        conv3 = self.conv_1x7_reduce(x)
        conv3 = self.conv_1x7(conv3)
        conv3 = self.conv_7x1(conv3)    # 256
        conv4 = self.conv_7x1_2_reduce(x)
        conv4 = self.conv_1x7_2(conv4)
        conv4 = self.conv_7x1_2(conv4)
        conv4 = self.conv_1x7_3(conv4)
        conv4 = self.conv_7x1_3(conv4)  # 256

        concat = L.concat([conv1, conv2, conv3, conv4], 1) # 1024
        return concat


class ReductionB(dg.Layer):

    def __init__(self, in_channels=1024):
        super().__init__()
        self.pool = dg.Pool2D(pool_size=3, pool_stride=2, pool_type='max')
        self.conv_3x3_reduce = ConvBN(in_channels, 192, 1, 1, 0)
        self.conv_3x3 = ConvBN(192, 192, 3, 2, 0)

        self.conv_1x7_reduce = ConvBN(in_channels, 256, 1, 1, 0)
        self.conv_1x7 = ConvBN(256, 256, (1, 7), 1, (0, 3))
        self.conv_7x1 = ConvBN(256, 320, (7, 1), 1, (3, 0))
        self.conv_3x3_2 = ConvBN(320, 320, 3, 2, 0)
    

    def forward(self, x):
        pool1 = self.pool(x) # 1024
        conv2 = self.conv_3x3_reduce(x)
        conv2 = self.conv_3x3(conv2) # 192

        conv3 = self.conv_1x7_reduce(x)
        conv3 = self.conv_1x7(conv3)
        conv3 = self.conv_7x1(conv3)
        conv3 = self.conv_3x3_2(conv3) # 320

        concat = L.concat([pool1, conv2, conv3], 1) # 1536
        return concat


class InceptionC(dg.Layer):

    def __init__(self, in_channels=1536):
        super().__init__()
        self.pool = dg.Pool2D(pool_size=3, pool_padding=1, pool_type='avg')
        self.conv_1x1 = ConvBN(in_channels, 256, 1, 1, 0)

        self.conv_1x1_2 = ConvBN(in_channels, 256, 1, 1, 0)

        self.conv_1x1_3 = ConvBN(in_channels, 384, 1, 1, 0)
        self.conv_1x3 = ConvBN(384, 256, (1, 3), 1, (0, 1))
        self.conv_3x1 = ConvBN(384, 256, (3, 1), 1, (1, 0))

        self.conv_1x1_4 = ConvBN(in_channels, 384, 1, 1, 0)
        self.conv_1x3_2 = ConvBN(384, 448, (1, 3), 1, (0, 1))
        self.conv_3x1_2 = ConvBN(448, 512, (3, 1), 1, (1, 0))
        self.conv_1x3_3 = ConvBN(512, 256, (1, 3), 1, (0, 1))
        self.conv_3x1_3 = ConvBN(512, 256, (3, 1), 1, (1, 0))


    def forward(self, x):
        pool1 = self.pool(x)
        conv1 = self.conv_1x1(pool1) # 256
        conv2 = self.conv_1x1_2(x)   # 256
        conv3 = self.conv_1x1_3(x)
        conv3_1 = self.conv_1x3(conv3) # 256
        conv3_2 = self.conv_3x1(conv3) # 256
        conv4 = self.conv_1x1_4(x)
        conv4 = self.conv_1x3_2(conv4)
        conv4 = self.conv_3x1_2(conv4)
        conv4_1 = self.conv_1x3_3(conv4) # 256
        conv4_2 = self.conv_3x1_3(conv4) # 256

        concat = L.concat([conv1, conv2, conv3_1, conv3_2, conv4_1, conv4_2], 1) # 1536
        return concat


class InceptionV4(dg.Layer):

    def __init__(self):
        super().__init__()
        self.stem = Stem()

        inception_a = []
        for i in range(4):
            inception_a.append(InceptionA(384))
        self.inception_a = dg.LayerList(inception_a)
        self.reduction_a = ReductionA(384)

        inception_b = []
        for i in range(7):
            inception_b.append(InceptionB(1024))
        self.inception_b = dg.LayerList(inception_b)
        self.reduction_b = ReductionB(1024)

        inception_c = []
        for i in range(3):
            inception_c.append(InceptionC(1536))
        self.inception_c = dg.LayerList(inception_c)

        self.pool = dg.Pool2D(pool_type='avg', global_pooling=True)

    

    def forward(self, x):
        x = self.stem(x)

        for block_a in self.inception_a:
            x = block_a(x)
        x = self.reduction_a(x)

        for block_b in self.inception_b:
            x = block_b(x)
        x = self.reduction_b(x)

        for block_c in self.inception_c:
            x = block_c(x)
        
        x = self.pool(x)
        b, c, _, _ = x.shape
        x = L.reshape(x, (b, -1))
        return x


if __name__ == '__main__':
    import numpy as np
    inceptionv4 = InceptionV4()
    for k, v in inceptionv4.state_dict().items():
        print(k + ": " + str(v.shape))
    
    input = dg.to_variable(np.zeros((1, 3, 256, 256)).astype("float32"))
    print(inceptionv4(input).shape)
