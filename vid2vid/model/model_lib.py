
import paddle.fluid.layers as L 
import paddle.fluid.dygraph as dg


class Upsample(dg.Layer):
    def __init__(self, scale=2):
        super().__init__()
        self.scale_factor = scale
    

    def forward(self, x):
        return L.image_resize(x, scale=self.scale_factor, resample='NEAREST')
























