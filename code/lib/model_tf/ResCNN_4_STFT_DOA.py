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
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Permute, Dropout, Reshape, Permute, Lambda, \
    Conv2D, MaxPooling2D, AveragePooling2D, SeparableConv2D, DepthwiseConv2D, BatchNormalization, SpatialDropout2D, \
    MaxPool1D, Conv1D, ReLU
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.activations import softmax
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import backend as K


class ResCNN_4_STFT_DOA(tf.keras.Model):  # channel-first
    
    def __init__(self, num_classes=8, name=None, **kwargs):
        if name is None:
            name = self.__class__.__name__
        super(ResCNN_4_STFT_DOA, self).__init__(name=name, **kwargs)
        print('-' * 20, 'This is a channel-first model.', '-' * 20, )
        # the input shape should be: (Channel=8, Time=7, Freq=337)
        
        # self.model_input = Input(shape=(8, 7, 337))
        
        self.conv_1 = Sequential([
            Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 3), padding='valid', ),  # (32, 7, 110)
            BatchNormalization(axis=1), Activation('relu'),
            Conv2D(filters=128, kernel_size=(1, 5), strides=(1, 2), padding='valid', ),  # (32, 7, 52)
            BatchNormalization(axis=1), Activation('relu')]
        )
        
        self.resconv_1 = Sequential([
            Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same'),
            BatchNormalization(axis=1), Activation('relu'),
            Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(axis=1), Activation('relu'),
            Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same'),
            BatchNormalization(axis=1)]
        )
        self.resconv_2 = Sequential([
            Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same'),
            BatchNormalization(axis=1), Activation('relu'),
            Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(axis=1), Activation('relu'),
            Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same'),
            BatchNormalization(axis=1)]
        )
        self.resconv_3 = Sequential([
            Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same'),
            BatchNormalization(axis=1), Activation('relu'),
            Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(axis=1), Activation('relu'),
            Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same'),
            BatchNormalization(axis=1)]
        )
        self.resconv_4 = Sequential([
            Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same'),
            BatchNormalization(axis=1), Activation('relu'),
            Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(axis=1), Activation('relu'),
            Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same'),
            BatchNormalization(axis=1)]
        )
        self.resconv_5 = Sequential([
            Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same'),
            BatchNormalization(axis=1), Activation('relu'),
            Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(axis=1), Activation('relu'),
            Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same'),
            BatchNormalization(axis=1)]
        )
        
        self.conv_6 = Sequential([
            Conv2D(12, kernel_size=(1, 1), strides=(1, 1), padding='valid'),
            BatchNormalization(axis=1), Activation('relu')]
        )
        
        self.conv_7 = Sequential([
            Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='valid'),
            BatchNormalization(axis=1), Activation('relu')]
        )
        
        self.conv_8 = Sequential([
            Conv2D(1, kernel_size=(7, 5), strides=(1, 1), padding='valid'),
            BatchNormalization(axis=1), Reshape((-1,) ), Activation('softmax')]
        )
        
        self.relu = Activation('relu')
        # self.flatten = Flatten(name='flatten')
    
    def call(self, inputs, training=None, mask=None):
        # a.shape: [B, 8, 7, 337]
        # inputs = self.model_input(inputs)
        a = self.conv_1(inputs)  # [B, 128, 7, 54]
        a_azi = self.relu(a + self.resconv_1(a))  # [B, 128, 7, 54]
        a_azi = self.relu(a_azi + self.resconv_2(a_azi))  # [B, 128, 7, 54]
        a_azi = self.relu(a_azi + self.resconv_3(a_azi))  # [B, 128, 7, 54]
        a_azi = self.relu(a_azi + self.resconv_4(a_azi))  # [B, 128, 7, 54]
        a_azi = self.relu(a_azi + self.resconv_5(a_azi))  # [B, 128, 7, 54]
        a_azi0 = self.conv_6(a_azi)  # [B, 360, 7, 54]
        a_azi = K.permute_dimensions(a_azi0, (0, 3, 2, 1))  # [B, 54, 7, 360]
        a_azi = self.conv_7(a_azi)  # [B, 500, 7, 360]
        a_azi = self.conv_8(a_azi)  # [B, 1, 1, 360]
        # a_azi = a_azi.view(a_azi.size(0), -1)  # [B, 360]
        # a_azi = self.flatten(a_azi)
        # a_azi = tf.squeeze(a_azi, axis=(1, 2), )
        return a_azi


if __name__ == '__main__':
    K.set_image_data_format('channels_first')
    model = ResCNN_4_STFT_DOA(num_classes=8)
    model.build(input_shape=(None, 8, 7, 337))
    model.summary()
    rand_input = np.random.random((3, 8, 7, 337))
    y = model(rand_input)
    print('Hello World!')
