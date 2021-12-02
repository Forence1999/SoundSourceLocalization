import os
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, ReLU, Conv2D, AvgPool2D, Dense, Input, Flatten, Activation, \
    Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import max_norm

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
tf.config.experimental.set_memory_growth(gpus[0], True)

EPS = 1e-12
K.set_image_data_format('channels_first')
dropoutRate = 0.25


class ConvBlock2d(keras.Model):
    def __init__(self, out_channels, kernel_size=(3, 3), strides=(1, 1), dilation=(1, 1), norm=True,
                 activation=True, eps=EPS):
        super(ConvBlock2d, self).__init__()
        
        assert strides == (1, 1), "`strides` is expected (1,1)"
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.norm = norm
        self.activation = activation
        self.bn = BatchNormalization(axis=1, epsilon=eps, )
        self.relu = ReLU()
        self.conv2d = Conv2D(out_channels, kernel_size=kernel_size, strides=strides, dilation_rate=dilation,
                             use_bias=False, padding='same')
    
    def call(self, inputs, training=None, mask=None):
        """  channel-first
        Args:
            input (batch_size, in_channels, H, W)
        Returns:
            output (batch_size, out_channels, H, W)
        """
        x = inputs
        
        if self.norm:
            x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        output = self.conv2d(x)
        
        return output


class D2Block(keras.Model):
    def __init__(self, growth_rate, kernel_size=(3, 3), dilated=True, norm=True, activation=True, d2_depth=None,
                 eps=EPS, ):
        """
        Args:
            in_channels <int>: # of input channels
            growth_rate <int> or <list<int>>: # of output channels
            kernel_size <int> or <tuple<int>>: Kernel size
            dilated <bool> or <list<bool>>: Applies dilated convolution.
            norm <bool> or <list<bool>>: Applies batch normalization.
            activation <str> or <list<str>>: Applies activation function.
            d2_depth <int>: If `growth_rate` is given by list, len(growth_rate) must be equal to `d2_depth`.
        """
        super(D2Block, self).__init__()
        
        assert type(d2_depth) is int, "Type of `d2_depth` must be int"
        if type(growth_rate) is int:
            growth_rate = [growth_rate] * d2_depth
        elif type(growth_rate) is list:
            assert d2_depth == len(growth_rate), "`d2_depth` is different from `len(growth_rate)`"
        else:
            raise ValueError("Not support growth_rate={}".format(growth_rate))
        
        if type(dilated) is bool:
            dilated = [dilated] * d2_depth
        elif type(dilated) is list:
            assert d2_depth == len(dilated), "`d2_depth` is different from `len(dilated)`"
        else:
            raise ValueError("Not support dilated={}".format(dilated))
        
        if type(norm) is bool:
            norm = [norm] * d2_depth
        elif type(norm) is list:
            assert d2_depth == len(norm), "`d2_depth` is different from `len(norm)`"
        else:
            raise ValueError("Not support norm={}".format(norm))
        
        if type(activation) is bool:
            activation = [activation] * d2_depth
        elif type(activation) is list:
            assert d2_depth == len(activation), "`d2_depth` is different from `len(activation)`"
        else:
            raise ValueError("Not support activation={}".format(activation))
        
        self.growth_rate = growth_rate
        self.d2_depth = d2_depth
        self.strides = (1, 1)
        self.kernel_size = kernel_size
        self.sub_blocks = []
        # _in_channels = in_channels
        
        for idx in range(d2_depth):
            _out_channels = sum(growth_rate[idx:])
            if dilated[idx]:
                dilation = (2 ** idx, 2 ** idx)
            else:
                dilation = (1, 1)
            conv_block = ConvBlock2d(_out_channels, kernel_size=self.kernel_size, strides=self.strides,
                                     dilation=dilation,
                                     norm=norm[idx], activation=activation[idx], eps=eps)
            self.sub_blocks.append(conv_block)
            # _in_channels = growth_rate[idx]
    
    def call(self, inputs, training=None, mask=None):
        """
        Args:
            input: (batch_size, in_channels, H, W)
        Returns:
            output: (batch_size, out_channels, H, W), where `out_channels` is determined by ...
        """
        growth_rate, d2_depth = self.growth_rate, self.d2_depth
        
        x = inputs
        x_residual = 0
        
        for idx in range(d2_depth):
            x = self.sub_blocks[idx](x)
            x_residual = x_residual + x
            
            in_channels = growth_rate[idx]
            stacked_channels = sum(growth_rate[idx + 1:])
            sections = [in_channels, stacked_channels]
            x, x_residual = tf.split(x_residual, sections, axis=1)
        
        output = x
        return output


class D3Block(keras.Model):
    def __init__(self, growth_rate, kernel_size=(3, 3), num_d2block=None, dilated=True, norm=True, activation=True,
                 d2_depth=3, eps=EPS):
        """
        Args:
            in_channels <int>: # of input channels
            growth_rate <int> or <list<int>>: # of output channels, TODO: <list<list<int>>>
            kernel_size <int> or <tuple<int>>: Kernel size
            num_d2block <int>: If `growth_rate` is given by list, len(growth_rate) must be equal to `num_d2block`.
            dilated <str> or <bool> or <list<bool>>: Applies dilated convolution.
            norm <bool> or <list<bool>>: Applies batch normalization.
            activation <str> or <list<str>>: Applies activation function.
            d2_depth <int>:
        """
        super(D3Block, self).__init__()
        
        assert type(num_d2block) is int, "Type of `num_d2block` must be int"
        
        if type(growth_rate) is int:
            growth_rate = [growth_rate] * num_d2block
        elif type(growth_rate) is list:
            assert num_d2block == len(growth_rate), "`len(growth_rate)`is different from `num_d2block`"
        else:
            raise ValueError("Not support growth_rate={}".format(growth_rate))
        
        if type(d2_depth) is int:
            d2_depth = [d2_depth] * num_d2block
        elif type(d2_depth) is list:
            assert num_d2block == len(d2_depth), "`num_d2block` is different from `len(growth_rate)`"
        else:
            raise ValueError("Not support growth_rate={}".format(growth_rate))
        
        if type(dilated) is bool:
            dilated = [dilated] * num_d2block
        elif type(dilated) is list:
            assert num_d2block == len(dilated), "`num_d2block` is different from `len(dilated)`"
        else:
            raise ValueError("Not support dilated={}".format(dilated))
        
        if type(norm) is bool:
            norm = [norm] * num_d2block
        elif type(norm) is list:
            assert num_d2block == len(norm), "`num_d2block` is different from `len(norm)`"
        else:
            raise ValueError("Not support norm={}".format(norm))
        
        if type(activation) is bool:
            activation = [activation] * num_d2block
        elif type(activation) is list:
            assert num_d2block == len(activation), "`num_d2block` is different from `len(activation)`"
        else:
            raise ValueError("Not support activation={}".format(activation))
        
        self.growth_rate = growth_rate
        self.num_d2block = num_d2block
        self.out_channels = growth_rate[-1]
        
        self.sub_blocks = []
        
        for idx in range(num_d2block):
            _out_channels = sum(growth_rate[idx:])
            d2block = D2Block(_out_channels, kernel_size=kernel_size, dilated=dilated[idx],
                              norm=norm[idx], activation=activation[idx], d2_depth=d2_depth[idx], eps=eps)
            self.sub_blocks.append(d2block)
    
    def call(self, inputs, training=None, mask=None):
        """
        Args:
            input: (batch_size, in_channels, H, W)
        Returns:
            output: (batch_size, out_channels, H, W), where `out_channels` is determined by ...
        """
        growth_rate, num_d2block = self.growth_rate, self.num_d2block
        
        x = inputs
        x_residual = 0
        
        for idx in range(num_d2block):
            x = self.sub_blocks[idx](x)
            x_residual = x_residual + x
            
            in_channels = growth_rate[idx]
            stacked_channels = sum(growth_rate[idx + 1:])
            sections = [in_channels, stacked_channels]
            x, x_residual = tf.split(x_residual, sections, axis=1)
        
        output = x
        return output


class DownSampleD3Block(keras.Model):
    def __init__(self, growth_rate, kernel_size=(3, 3), num_d2block=None, dilated=True, norm=True, activation=True,
                 d2_depth=None, down_scale=(2, 2), eps=EPS):
        super(DownSampleD3Block, self).__init__()
        
        self.down_scale = down_scale
        self.d3block = D3Block(growth_rate, kernel_size=kernel_size, num_d2block=num_d2block, dilated=dilated,
                               norm=norm, activation=activation, d2_depth=d2_depth, eps=eps)
        self.downsample2d = AvgPool2D(pool_size=self.down_scale, padding="valid", )
        if dropoutRate is not None:
            self.dropout = Dropout(dropoutRate)
        
        self.out_channels = self.d3block.out_channels
    
    def call(self, inputs, training=None, mask=None):
        x = self.d3block(inputs)
        output = self.downsample2d(x)
        
        if dropoutRate is not None:
            output = self.dropout(output)
        
        return output


class Encoder(keras.Model):
    def __init__(self, num_d3block=3, growth_rate=16, kernel_size=(3, 3), num_d2block=None,
                 dilated=True, norm=True, activation=True, d2_depth=None, down_scale=(2, 2), eps=EPS):
        """
        Args:
            in_channels <int>:
            growth_rate <list<int>>:
            kernel_size <tuple<int>> or <int>:
            num_d2block <list<int>> or <int>:
            dilated <list<bool>> or <bool>:
            norm <list<bool>> or <bool>:
            activation <list<str>> or <str>:
        """
        super(Encoder, self).__init__()
        
        assert type(num_d3block) is int, "Type of `num_d3block` must be int"
        
        if type(growth_rate) is int:
            growth_rate = [growth_rate] * num_d3block
        elif type(growth_rate) is list:
            assert num_d3block == len(growth_rate), "`len(growth_rate)`is different from `num_d3block`"
        else:
            raise ValueError("Not support growth_rate={}".format(growth_rate))
        
        if type(num_d2block) is int:
            num_d2block = [num_d2block] * num_d3block
        elif type(num_d2block) is list:
            assert num_d3block == len(num_d2block), "Invalid length of `num_d2block`"
        else:
            raise ValueError("Invalid type of `num_d2block`.")
        
        if type(dilated) is bool:
            dilated = [dilated] * num_d3block
        elif type(dilated) is list:
            assert num_d3block == len(dilated), "Invalid length of `dilated`"
        else:
            raise ValueError("Invalid type of `dilated`.")
        
        if type(norm) is bool:
            norm = [norm] * num_d3block
        elif type(norm) is list:
            assert num_d3block == len(norm), "Invalid length of `norm`"
        else:
            raise ValueError("Invalid type of `norm`.")
        
        if type(activation) is bool:
            activation = [activation] * num_d3block
        elif type(activation) is list:
            assert num_d3block == len(activation), "Invalid length of `activation`"
        else:
            raise ValueError("Invalid type of `activation`.")
        
        if type(d2_depth) is int:
            d2_depth = [d2_depth] * num_d3block
        elif type(d2_depth) is list:
            assert num_d3block == len(d2_depth), "Invalid length of `d2_depth`"
        else:
            raise ValueError("Invalid type of `d2_depth`.")
        
        self.num_d3block = num_d3block
        self.ds_d3blocks = []
        for idx in range(num_d3block):
            downsample_block = DownSampleD3Block(growth_rate[idx], kernel_size=kernel_size,
                                                 num_d2block=num_d2block[idx], dilated=dilated[idx], norm=norm[idx],
                                                 activation=activation[idx], d2_depth=d2_depth[idx],
                                                 down_scale=down_scale, eps=eps)
            self.ds_d3blocks.append(downsample_block)
    
    def call(self, inputs, training=None, mask=None):
        num_d3block = self.num_d3block
        
        x = inputs
        for idx in range(num_d3block):
            x = self.ds_d3blocks[idx](x)
        output = x
        
        return output


def RD3Net(num_classes, Chans=64, SamplePoints=128, dropoutRate=None, norm_rate=0.25, classifier_units=128,
           num_d3block=2, growth_rate=4, kernel_size=(3, 3), num_d2block=2, dilated=True, norm=True, activation=True,
           d2_depth=2, down_scale=(2, 2), eps=EPS):
    """
    Args:
        in_channels <int>: # of input channels
        growth_rate <int> or <list<int>>: # of output channels, TODO: <list<list<int>>>
        kernel_size <int> or <tuple<int>>: Kernel size
        num_d3block <int>: If `growth_rate` is given by list, len(growth_rate) must be equal to `num_d3block`.
        dilated <str> or <bool> or <list<bool>>: Applies dilated convolution.
        norm <bool> or <list<bool>>: Applies batch normalization.
        activation <str> or <list<str>>: Applies activation function.
        num_d2block <int>:
    """
    
    input = Input(shape=(1, Chans, SamplePoints))
    encoder = Encoder(num_d3block=num_d3block, growth_rate=growth_rate, kernel_size=kernel_size,
                      num_d2block=num_d2block, dilated=dilated, norm=norm, activation=activation,
                      d2_depth=d2_depth, down_scale=down_scale, eps=eps)(input)
    bn = BatchNormalization(axis=1, epsilon=eps, )(encoder)
    relu = ReLU()(bn)
    flatten = Flatten()(relu)
    classifier = Dense(num_classes, activation='relu', kernel_constraint=max_norm(0.25))(flatten)
    # dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(0.25))(flatten)
    softmax = Activation('softmax', name='softmax')(classifier)
    
    return Model(inputs=input, outputs=softmax)


# class RD3Net(keras.Model):
#     def __init__(self, input_shape=(1, 6, 64), classifier_units=128, num_d3block=3, growth_rate=16, kernel_size=(3, 3),
#                  num_d2block=3, dilated=True, norm=True, activation=True, d2_depth=3, down_scale=(2, 2), eps=EPS):
#         """
#         Args:
#             in_channels <int>: # of input channels
#             growth_rate <int> or <list<int>>: # of output channels, TODO: <list<list<int>>>
#             kernel_size <int> or <tuple<int>>: Kernel size
#             num_d3block <int>: If `growth_rate` is given by list, len(growth_rate) must be equal to `num_d3block`.
#             dilated <str> or <bool> or <list<bool>>: Applies dilated convolution.
#             norm <bool> or <list<bool>>: Applies batch normalization.
#             activation <str> or <list<str>>: Applies activation function.
#             num_d2block <int>:
#         """
#         super(RD3Net, self).__init__()
#
#         # self.inputs = Input(shape=input_shape)
#         self.encoder = Encoder(num_d3block=num_d3block, growth_rate=growth_rate, kernel_size=kernel_size,
#                                num_d2block=num_d2block, dilated=dilated, norm=norm, activation=activation,
#                                d2_depth=d2_depth, down_scale=down_scale, eps=eps)
#         self.bn = BatchNormalization(axis=1, epsilon=eps, )
#         self.relu = ReLU()
#         self.flatten = Flatten()
#         self.classifier = Dense(classifier_units, activation='softmax', )
#
#     def call(self, inputs, training=None, mask=None):
#         """
#         Args:
#             input: (batch_size, in_channels, H, W)
#         Returns:
#             output: (batch_size, out_channels, H, W), where `out_channels` is determined by ...
#         """
#
#         # inputs = self.inputs(inputs)
#         x = inputs
#         x = self.encoder(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.flatten(x)
#         output = self.classifier(x)
#
#         return output


def _test_d2block():
    print("-" * 10, "Test D2Block", "-" * 10)
    
    batch_size = 4
    n_bins, n_frames = 6, 64
    in_channels = 2
    kernel_size = (3, 3)
    d2_depth = 4
    
    input = np.random.rand(batch_size, in_channels, n_bins, n_frames)
    print(input.shape)
    
    growth_rate = [2, 2, 3, 4]
    dilated = True
    model = D2Block(growth_rate, kernel_size=kernel_size, dilated=dilated, norm=True, activation=True,
                    d2_depth=d2_depth, eps=EPS, )
    output = model(input)
    print(output.shape)
    print(model.summary())


def _test_d3block():
    print("-" * 10, "Test D3Block", "-" * 10)
    
    batch_size = 4
    n_bins, n_frames = 6, 64
    in_channels = 3
    kernel_size = (3, 3)
    d2_depth = 4
    
    input = np.random.rand(batch_size, in_channels, n_bins, n_frames)
    print(input.shape)
    
    print("-" * 10, "D2 Block when `growth_rate` is given as int and `dilated` is given as bool.", "-" * 10)
    
    growth_rate = [2, 2, 3, ]
    dilated = True
    model = D3Block(growth_rate, kernel_size=kernel_size, num_d2block=3, dilated=dilated, norm=True, activation=True,
                    d2_depth=d2_depth, eps=EPS, )
    
    output = model(input)
    
    print(output.shape)
    print(model.summary())


def _test_rd3net():
    print("-" * 10, "Test D3Block", "-" * 10)
    
    batch_size = 4
    n_bins, n_frames = 6, 64
    in_channels = 1
    
    input = np.random.rand(batch_size, in_channels, n_bins, n_frames)
    print(input.shape)
    
    model = RD3Net(input_shape=input.shape, classifier_units=8, num_d3block=2, growth_rate=[4, 6, ],
                   kernel_size=(3, 3), num_d2block=3, dilated=True, norm=True, activation=True, d2_depth=4,
                   down_scale=(2, 2), eps=EPS)
    
    output = model(input)
    
    print(output.shape)
    print(model.summary())
    print(np.argmax(output, axis=-1))


if __name__ == '__main__':
    
    os.environ["PYTHONHASHSEED"] = str(0)
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)
    print("=" * 10, "D2 Block", "=" * 10)
    # _test_d2block()
    # _test_d3block()
    _test_rd3net()
