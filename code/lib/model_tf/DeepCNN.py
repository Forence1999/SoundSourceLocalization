# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: Sound Source Localization
# @File: DeepCNN.py
# @Time: 2021/11/29/11:00
# @Software: PyCharm
import os
import sys
import time
import random
import warnings
import numpy as np
from copy import deepcopy
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Permute, Dropout, Reshape, Permute, Lambda, \
    Conv2D, MaxPooling2D, AveragePooling2D, SeparableConv2D, DepthwiseConv2D, BatchNormalization, SpatialDropout2D, \
    MaxPool1D, Conv1D
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import backend as K
from model_tf.RD3Net import *
from model_tf.ResCNN_4_STFT_DOA import *


def DeepConvNet(num_classes, Chans=64, SamplePoints=256, dropoutRate=0.5):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.

    This implementation assumes the input is a 2-second EEG signal sampled at
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10

    Note that this implementation has not been verified by the original
    authors.

    """
    
    # start the model
    input_main = Input((1, Chans, SamplePoints))
    block1 = Conv2D(25, (1, 5), input_shape=(1, Chans, SamplePoints),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(25, (Chans, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    if dropoutRate is not None:
        block1 = Dropout(dropoutRate)(block1)
    
    block2 = Conv2D(50, (1, 5), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    if dropoutRate is not None:
        block2 = Dropout(dropoutRate)(block2)
    
    block3 = Conv2D(100, (1, 5), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    if dropoutRate is not None:
        block3 = Dropout(dropoutRate)(block3)
    
    block4 = Conv2D(200, (1, 4), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu', name='last_conv_out')(block4)
    # block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    if dropoutRate is not None:
        block4 = Dropout(dropoutRate)(block4)
    
    flatten = Flatten()(block4)
    
    dense = Dense(num_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)


if __name__ == '__main__':
    
    print('Hello World!')
