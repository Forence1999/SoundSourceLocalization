# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: Sound Source Localization
# @File: ResCNN_4_STFT_DOA.py
# @Time: 2021/11/29/10:56
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


class ResCNN_4_STFT_DOA(tf.keras.Model):
    
    def __init__(self, num_classes, name=None, **kwargs):
        if name is None:
            name = self.__class__.__name__
        
        super(ResCNN_4_STFT_DOA, self).__init__(name=name, **kwargs)
        
        self.conv1 = Conv1D(6, 5, activation='relu', padding='valid')
        self.pool1 = MaxPool1D()
        self.conv2 = Conv1D(16, 5, activation='relu', padding='valid')
        self.pool2 = MaxPool1D()
        self.flatten = Flatten()
        self.dense1 = Dense(120, activation='relu')
        self.dense2 = Dense(84, activation='relu')
        self.out = Dense(num_classes, activation='linear')
    
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.out(x)
        
        if training is False:
            out = tf.nn.softmax(out, axis=-1)
        
        return out





if __name__ == '__main__':
    
    print('Hello World!')