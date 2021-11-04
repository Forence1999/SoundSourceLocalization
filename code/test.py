# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: SSL
# @File: test.py
# @Time: 2021/10/28/11:12
# @Software: PyCharm


import os
import sys
from copy import deepcopy
import numpy as np
from sklearn.preprocessing import LabelBinarizer


def smooth_labels_mean(labels, factor):
    labels = np.array(labels)
    labels *= (1 - factor)
    delta = factor / (labels.shape[1] - 1)
    for label in labels:
        label[np.isclose(label, 0)] += delta
    return labels


def smooth_labels_gauss(labels, sigma):
    labels = np.array(labels)
    num_cls = labels.shape[-1]
    gauss_template = np.roll([np.exp(- ((i - 0) ** 2) / (2 * (sigma ** 2))) / (np.sqrt(2 * np.pi) * sigma)
                              for i in range(num_cls // 2 - num_cls + 1, num_cls // 2 + 1)],
                             shift=num_cls // 2 - num_cls + 1, axis=0)
    print('\ngauss_template: ', np.around(gauss_template, 3))
    for i, label in enumerate(labels):
        labels[i] = np.roll(gauss_template, shift=np.argmax(label), axis=0)
    
    return labels


def one_hot_encoder(y, num_classes=None, dtype='float32'):
    """  copied from  tf.keras.utils.to_categorical"""
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


if __name__ == '__main__':
    y = np.load('./test_label.npz')['test_label']
    
    y1 = one_hot_encoder(y)
    y2 = smooth_labels_gauss(y1, sigma=1)
    
    lb = LabelBinarizer()
    y1 = lb.fit_transform(y)
    # testY = lb.transform(y)
