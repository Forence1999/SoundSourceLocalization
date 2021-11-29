# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: Sound Source Localization
# @File: EEGNet.py
# @Time: 2021/11/29/11:01
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
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Dropout, Conv2D, AveragePooling2D, \
    SeparableConv2D, DepthwiseConv2D, BatchNormalization, SpatialDropout2D
from tensorflow.keras.constraints import max_norm


def EEGNet(num_classes, Chans=64, SamplePoints=128, dropoutRate=None, kernLength=128, F1=4,
           D=2, F2=8, norm_rate=0.25, dropoutType='Dropout'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1 = Input(shape=(1, Chans, SamplePoints))
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(1, Chans, SamplePoints),
                    use_bias=False)(input1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 2))(block1)
    # if dropoutRate is not None:
    #     block1 = dropoutType(dropoutRate)(block1)
    
    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 2))(block2)
    if dropoutRate is not None:
        block2 = dropoutType(dropoutRate)(block2)
    
    flatten = Flatten(name='flatten')(block2)
    
    dense = Dense(num_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)


if __name__ == '__main__':
    
    print('Hello World!')
