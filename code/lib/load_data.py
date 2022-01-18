import scipy.io as scio
import h5py
import numpy as np
import time
from .utils import standard_normalizaion, standard_normalization_wise, shuffle_data, split_data
from scipy.signal import resample
import pickle


def smooth_labels_mean(labels, factor):
    labels = np.array(labels)
    labels *= factor
    delta = (1 - factor) / (labels.shape[1] - 1)
    for label in labels:
        label[np.isclose(label, 0)] += delta
    return labels


def smooth_labels_gauss(labels, sigma):
    labels = np.array(labels)
    num_cls = labels.shape[-1]
    gauss_template = np.roll([np.exp(- ((i - 0) ** 2) / (2 * (sigma ** 2))) / (np.sqrt(2 * np.pi) * sigma)
                              for i in range(num_cls // 2 - num_cls + 1, num_cls // 2 + 1)],
                             shift=num_cls // 2 - num_cls + 1, axis=0)
    print('gauss_template: ', np.around(gauss_template, 3))
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


def load_hole_dataset(sbj_idx, ds_path, shuffle=True, normalization=None, split=None, one_hot=False):
    ds = np.load(ds_path, allow_pickle=True)
    x = ds['x']
    y = ds['y']
    del ds
    
    x = np.concatenate(x[sbj_idx], axis=0)
    x = np.expand_dims(x, axis=1)
    y = np.concatenate(y[sbj_idx], axis=0)[:, -1] // 45
    if one_hot:
        y = one_hot_encoder(y)
    if normalization is not None:
        for i in range(len(x)):
            x[i] = standard_normalization_wise(x[i], normalization)
    if shuffle:
        x, y = shuffle_data([x, y])
    if split is not None:
        split_idx = int(len(y) * split)
        return x[:split_idx], y[:split_idx], x[split_idx:], y[split_idx:]
    return x, y


def load_CYC_dataset(sbj_idx, ds_path, shuffle=True, normalization=None, split=None, label_preprocess=None,
                     label_smooth_para=None):
    with open(ds_path, "rb") as fo:
        ds = pickle.load(fo)
    x = ds['x']
    y = ds['y']
    del ds
    
    x = np.concatenate(x[sbj_idx], axis=0)
    y = np.concatenate(y[sbj_idx], axis=0)
    y = np.array(y[:, -2], dtype=np.int) // 45
    
    if label_preprocess is None:
        pass
    elif label_preprocess == 'one_hot':
        y = one_hot_encoder(y)
    elif label_preprocess == 'mean_smooth':
        y = one_hot_encoder(y)
        y = smooth_labels_mean(y, factor=label_smooth_para)
    elif label_preprocess == 'gauss_smooth':
        y = one_hot_encoder(y)
        y = smooth_labels_gauss(y, sigma=label_smooth_para)
    else:
        raise ValueError('label_preprocess is not set right!')
    
    if normalization is not None:
        for i in range(len(x)):
            x[i] = standard_normalization_wise(x[i], normalization)
    if shuffle:
        x, y = shuffle_data([x, y])
    if split is not None:
        split_idx = int(len(y) * split)
        return x[:split_idx], y[:split_idx], x[split_idx:], y[split_idx:]
    return x, y
