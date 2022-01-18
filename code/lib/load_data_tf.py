# !/usr/bin/env python
# -*- coding:utf-8 _*-
# @Author: swang
# @Contact: wang00sheng@gmail.com
# @Project Name: Sound Source Localization
# @File: load_data_tf.py
# @Time: 2022/01/18/12:00
# @Software: PyCharm
import os
import sys

sys.path.append(os.path.dirname(__file__))
import time
import random
import warnings
import numpy as np
from copy import deepcopy
import tensorflow as tf
import utils
from pathlib import Path


def CYC_PathDatasetGenerator(src_paths, ):
    x_paths = []
    for src_path in src_paths:
        x_paths.extend(Path(src_path).rglob(pattern='*.npz'))
    y_s = [int(i.parts[-3]) // 45 for i in x_paths]
    x_paths = [str(i) for i in x_paths]
    
    return list(zip(x_paths, y_s))


def CYC_map_func(x_path: tf.Tensor, y: tf.Tensor):
    x = np.load(x_path.numpy())['data']
    return (x, y)


def upper_case_fn(t: tf.Tensor):
    return t.numpy().decode('utf-8').upper()


def CYC_DataGenerator(path_ds, map_parallel_calls=None, map_deterministic=None,
                      shuffle_buffer_size=1024, shuffle_seed=None, reshuffle_each_iteration=True, num_prefetch=1, ):
    '''
    return a data generator for CYC dataset based on a list of (x_path, y) pair
    Args:
        path_ds:
    Returns: a data generator for CYC dataset
    '''
    xs, ys = list(zip(*path_ds))
    # xs, ys = tf.constant(xs), tf.constant(ys)
    xs, ys = tf.constant(xs), tf.constant(ys)
    dataset = tf.data.Dataset.from_tensor_slices((xs, ys))
    
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=shuffle_seed,
                              reshuffle_each_iteration=reshuffle_each_iteration)
    # dataset = dataset.map(map_func=CYC_map_func, num_parallel_calls=map_parallel_calls, deterministic=map_deterministic)
    dataset = dataset.map(
        map_func=lambda x_path, y: tf.py_function(func=CYC_map_func, inp=[x_path, y], Tout=(tf.float64, tf.int32)),
        num_parallel_calls=map_parallel_calls, deterministic=map_deterministic)
    dataset = dataset.prefetch(num_prefetch)
    
    return dataset


if __name__ == '__main__':
    x = list(range(10))
    y = list(range(10))
    path_ds = list(zip(x, y))
    
    CYC_DataGenerator(path_ds)
    
    print('Hello World!')
