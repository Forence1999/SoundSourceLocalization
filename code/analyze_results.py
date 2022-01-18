import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from lib.utils_plot import plot_2d_heatmap
import os
import sys
import pickle
import six
import random
from lib.confusion_matrix_pretty_print import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.metrics
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.python.keras.losses import Loss
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
from lib import utils, models_tf
from lib.load_data import load_hole_dataset, one_hot_encoder, load_CYC_dataset
from collections import Counter
from tensorflow.python.platform import tf_logging as logging


def create_multi_bars(data, color=None, title=None, x_labels=None, y_label=None, y_lim=None, tick_step=1.,
                      group_gap=0.2,
                      bar_gap=0., plt_show=True, value_show=True, dpi=300, value_fontsize=5, value_interval=0.01,
                      value_format='%.2f', save_path=None):
    '''
    x_labels: x轴坐标标签序列
    data: 二维列表，每一行为同一颜色的各个bar值，每一列为同一个横坐标的各个bar值
    tick_step: 默认x轴刻度步长为1，通过tick_step可调整x轴刻度步长。
    group_gap: 组与组之间的间隙，最好为正值，否则组与组之间重叠
    bar_gap ：每组中柱子间的空隙，默认为0，每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠
    '''
    data = np.asarray(data)
    bar_per_group_num = len(data[0])
    if color is not None:
        assert len(color) >= bar_per_group_num
    x_ticks = np.arange(bar_per_group_num) * tick_step
    group_num = len(data)
    group_width = tick_step - group_gap
    bar_span = group_width / group_num  # 组内每个bar的宽度
    bar_width = bar_span - bar_gap
    baseline_x = x_ticks - (group_width - bar_span) / 2  # baseline_x为每组柱子第一个柱子的基准x轴位置
    plt.figure(dpi=dpi)
    for index, y in enumerate(data):
        x = baseline_x + index * bar_span
        if color is not None:
            plt.bar(x, y, bar_width, color=color[index])
        else:
            plt.bar(x, y, bar_width)
        if value_show:
            for x0, y0 in zip(x, y):
                plt.text(x0, y0 + value_interval, value_format % y0, ha='center', va='bottom', fontsize=value_fontsize)
    if title is not None:
        plt.title(title)
    if x_labels is not None:
        plt.xticks(x_ticks, x_labels)
    if y_label is not None:
        plt.ylabel(y_label)
    if y_lim is not None:
        plt.ylim(y_lim)
    if save_path is not None:
        plt.savefig(save_path)
    if plt_show:
        plt.show()


def plot_num_res_block_vs_fold():
    train_accs, val_accs, test_accs, train_acc3s, val_acc3s, test_acc3s = [], [], [], [], [], []
    
    for fold in range(6):
        res_path = '/home/swang/project/SmartWalker/SSL/model/ResCNN_' + str(
            fold) + '_256ms_norm_drop_denoised_norm_ini_hann_np_STFT_clip_ms_64_overlap_0.5_epoch_20/_norm_whole_label_None/res.npz'
        res = np.load(res_path)
        train_accs.append(res['train_accs'])
        val_accs.append(res['val_accs'])
        test_accs.append(res['test_accs'])
        train_acc3s.append(res['train_acc3s'])
        val_acc3s.append(res['val_acc3s'])
        test_acc3s.append(res['test_acc3s'])
    train_accs, val_accs, test_accs, train_acc3s, val_acc3s, test_acc3s = np.asarray(train_accs), np.asarray(
        val_accs), np.asarray(test_accs), np.asarray(train_acc3s), np.asarray(val_acc3s), np.asarray(test_acc3s)
    
    train_accs = np.concatenate((train_accs, np.mean(train_accs, axis=1).reshape((-1, 1))), axis=1)
    val_accs = np.concatenate((val_accs, np.mean(val_accs, axis=1).reshape((-1, 1))), axis=1)
    test_accs = np.concatenate((test_accs, np.mean(test_accs, axis=1).reshape((-1, 1))), axis=1)
    
    x_labels = ['Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5', 'Mean', ]
    create_multi_bars(train_accs, title='Fold-wise Train Acc of ResConv_i with Whole Norm', x_labels=x_labels,
                      y_label='acc', y_lim=(0, 1), tick_step=1, group_gap=0.2, bar_gap=0.05, save_path=None)


def print_accs():
    def print_acc(res_path):
        res = np.load(res_path)
        train_accs = res['train_accs']
        val_accs = res['val_accs']
        test_accs = res['test_accs']
        train_acc3s = res['train_acc3s']
        val_acc3s = res['val_acc3s']
        test_acc3s = res['test_acc3s']
        
        print('train_accs: ', [round(i, 3) for i in train_accs], '-' * 5, round(np.mean(train_accs), 3))
        print('val_accs: ', [round(i, 3) for i in val_accs], '-' * 5, round(np.mean(val_accs), 3))
        print('test_accs: ', [round(i, 3) for i in test_accs], '-' * 5, round(np.mean(test_accs), 3))
        print('train_accs: ', [round(i, 3) for i in train_acc3s], '-' * 5, round(np.mean(train_acc3s), 3))
        print('val_accs: ', [round(i, 3) for i in val_acc3s], '-' * 5, round(np.mean(val_acc3s), 3))
        print('test_accs: ', [round(i, 3) for i in test_acc3s], '-' * 5, round(np.mean(test_acc3s), 3))
        print('res_path:', res_path)
    
    num_filter_ls = [8, 16, 32, 64, 128, 256, 512]
    num_res_block_ls = [4, 3, 2, 1, 0]
    
    for res_idx in num_res_block_ls:
        for filter_idx in num_filter_ls:
            try:
                res_path = '/home/swang/project/SmartWalker/SSL/model/ResCNN_' + str(
                    res_idx) + '_256ms_norm_drop_denoised_norm_ini_hann_np_STFT_clip_ms_64_overlap_0.5_epoch_20/_num_filter_' + str(
                    filter_idx) + '_norm_None_label_None/res.npz'
                print_acc(res_path)
            except:
                pass


def print_models():
    def print_model(num_res_block, num_filter):
        K.clear_session()
        os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        K.set_image_data_format('channels_first')
        model = models_tf.ResCNN_4_STFT_DOA(num_classes=8, num_res_block=num_res_block, num_filter=num_filter)
        model.build(input_shape=(None, 8, 7, 508))
        model.summary()
        print('\n', 'num_res_block:', num_res_block, 'num_filter:', num_filter, '\n', )
    
    num_filter_ls = [8, 16, 32, 64, 128, 256, 512]
    num_res_block_ls = [4, 3, 2, 1, 0]
    
    for res_idx in num_res_block_ls:
        for filter_idx in num_filter_ls:
            try:
                print_model(num_res_block=res_idx, num_filter=filter_idx)
            except:
                pass


def plot_num_res_block_vs_num_filter():
    num_filter_ls = [8, 16, 32, 64, 128, 256, 512]  # x_axis
    num_res_block_ls = [4, 3, 2, 1, 0]  # y_axis
    
    acc_names = ['train_accs', 'val_accs', 'test_accs', 'train_acc3s', 'val_acc3s', 'test_acc3s']
    for acc_name in acc_names:
        accs = []
        for res_idx in num_res_block_ls:
            temp_accs = []
            for filter_idx in num_filter_ls:
                try:
                    res_path = '/home/swang/project/SmartWalker/SSL/model/ResCNN_' + str(
                        res_idx) + '_256ms_norm_drop_denoised_norm_ini_hann_np_STFT_clip_ms_64_overlap_0.5_epoch_20/_num_filter_' + str(
                        filter_idx) + '_norm_None_label_None/res.npz'
                    res = np.load(res_path)
                    acc = res[acc_name]
                    temp_accs.append(np.mean(acc))
                except:
                    temp_accs.append(0)
            
            accs.append(temp_accs)
        accs = np.asarray(accs)
        accs[-1] = accs[-1].max()
        
        x_ticks = list(map(str, num_filter_ls))
        x_label = 'num_filter'
        y_ticks = list(map(str, num_res_block_ls))
        y_label = 'num_res_block'
        title = 'Acc of Models with different num_res_block & num_filter' + ' ' + acc_name
        plot_2d_heatmap(data=accs, x_ticks=x_ticks, x_label=x_label, y_ticks=y_ticks, y_label=y_label, title=title,
                        insert_summary=False, show_percent=True, show_null_value=False)
        print(accs)


if __name__ == "__main__":
    
    # print_accs()
    print_models()
    # plot_num_res_block_vs_num_filter()
