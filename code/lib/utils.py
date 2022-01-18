import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image
import cv2
import random
from collections import Counter


def set_random_seed(seed=0, fix_np=False, fix_tf=False, fix_torch=False, ):
    ''' setting random seed '''
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if fix_np:
        np.random.seed(seed)
    if fix_tf:
        import tensorflow as tf
        tf.random.set_seed(seed)
    if fix_torch:
        import torch
        import torch.backends.cudnn as cudnn
        
        torch.manual_seed(seed)
        cudnn.deterministic = True
        print('Warning:', 'You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')


def add_prefix_and_suffix_4_basename(path, prefix=None, suffix=None):
    '''
    add prefix and/or suffix string(s) to a path's basename
    :param path:
    :param prefix:
    :param suffix:
    :return:
    '''
    dir_path, basename = os.path.split(path)
    filename, ext = os.path.splitext(basename)
    filename = str(prefix if prefix is not None else '') + filename + str(suffix if suffix is not None else '') + ext
    
    return os.path.join(dir_path, filename)


def standard_normalizaion(x):
    return (x - np.mean(x)) / np.std(x)


def standard_normalization_wise(data, normalization=None):
    ''' channel-first '''
    data = np.array(data)
    if normalization is None:
        return data
    assert normalization in ['whole', 'sample-wise', 'channel-wise', 'samplepoint-wise']
    for i in range(len(data)):
        if normalization == 'whole':
            data = standard_normalizaion(data)
        elif normalization == 'sample-wise':
            data[i, :, :] = standard_normalizaion(data[i, :, :])
        elif normalization == 'channel-wise':
            data[i, :, :] = [standard_normalizaion(data[i, j, :]) for j in range(data.shape[-2])]
        elif normalization == 'samplepoint-wise':
            data[i, :, :] = np.array([standard_normalizaion(data[i, :, j]) for j in range(data.shape[-1])]).T
        else:
            print('-' * 20, 'normalization is incorrectly assigned', '-' * 20)
    
    return np.array(data)


def split_data(data, split=0.8, shuffle=True):
    x = data[0]
    y = data[1]
    data_size = len(x)
    split_index = int(data_size * split)
    indices = np.arange(data_size)
    if shuffle:
        indices = np.random.permutation(indices)
    x_train = x[indices[:split_index]]
    y_train = y[indices[:split_index]]
    x_test = x[indices[split_index:]]
    y_test = y[indices[split_index:]]
    return x_train, y_train, x_test, y_test


def split_data_wid(data, split=0.8, shuffle=True):
    x = data[0]
    y = data[1]
    s = data[2]
    data_size = len(x)
    split_index = int(data_size * split)
    indices = np.arange(data_size)
    if shuffle:
        indices = np.random.permutation(indices)
    x_train = x[indices[:split_index]]
    y_train = y[indices[:split_index]]
    s_train = s[indices[:split_index]]
    x_test = x[indices[split_index:]]
    y_test = y[indices[split_index:]]
    return x_train, y_train, s_train, x_test, y_test


def split_data_both(data, split=0.8, shuffle=True):
    x = data[0]
    x_poison = data[1]
    y = data[2]
    s = data[3]
    data_size = len(x)
    split_index = int(data_size * split)
    indices = np.arange(data_size)
    if shuffle:
        indices = np.random.permutation(indices)
    x_train = x[indices[:split_index]]
    x_train_poison = x_poison[indices[:split_index]]
    y_train = y[indices[:split_index]]
    s_train = s[indices[:split_index]]
    x_test = x[indices[split_index:]]
    y_test = y[indices[split_index:]]
    return x_train, x_train_poison, y_train, s_train, x_test, y_test


def shuffle_data(data, random_seed=None):
    '''
    data: [x, y]   type: numpy
    '''
    x, y = data
    data_size = x.shape[0]
    shuffle_index = get_shuffle_index(data_size, random_seed=random_seed)
    
    return x[shuffle_index], y[shuffle_index]


def get_shuffle_index(data_size, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    return np.random.permutation(np.arange(data_size))


def gen_cross_val_idx(num_sbj, num_fold, num_subfold, random_seed=None, ):
    if random_seed is not None:
        os.environ["PYTHONHASHSEED"] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    sbj_rand_idx = get_shuffle_index(num_sbj)
    split_ds_idx = [sbj_rand_idx[i::num_fold] for i in range(num_fold)] * num_fold
    
    all_split_fold = []
    for i in range(num_fold):
        train_idx = split_ds_idx[i:i + num_subfold[0]]
        val_idx = split_ds_idx[i + num_subfold[0]:i + num_subfold[0] + num_subfold[1]]
        test_idx = split_ds_idx[
                   i + num_subfold[0] + num_subfold[1]:i + num_subfold[0] + num_subfold[1] + num_subfold[2]]
        train_idx = sorted(np.concatenate(train_idx) if num_subfold[0] > 1 else train_idx[0])  #
        val_idx = sorted(np.concatenate(val_idx) if num_subfold[1] > 1 else val_idx[0])  #
        test_idx = sorted(np.concatenate(test_idx) if num_subfold[2] > 1 else test_idx[0])  #
        all_split_fold.append([train_idx, val_idx, test_idx])
    return all_split_fold


def statistic_label_proportion(train_y, val_y, test_y, do_print=True):
    train_y, val_y, test_y, = np.array(train_y), np.array(val_y), np.array(test_y),
    if train_y.ndim > 1:
        train_y = np.argmax(train_y, axis=-1)
    if val_y.ndim > 1:
        val_y = np.argmax(val_y, axis=-1)
    if test_y.ndim > 1:
        test_y = np.argmax(test_y, axis=-1)
    total_y = np.concatenate((train_y, val_y, test_y), axis=0)
    train, val, test, total = Counter(train_y), Counter(val_y), Counter(test_y), Counter(total_y)
    
    keys = sorted(np.unique(total_y))
    func = lambda key, dict: dict[key] if key in dict.keys() else 0
    sorted_train = [func(key, train) for key in keys]
    sorted_val = [func(key, val) for key in keys]
    sorted_test = [func(key, test) for key in keys]
    sorted_total = [func(key, total) for key in keys]
    if do_print:
        print('\n', '-' * 20, 'Statistical info of dataset labels', '-' * 20)
        print('label: ', keys, )
        print('train: ', sorted_train, )
        print('val:   ', sorted_val, )
        print('test:  ', sorted_test, )
        print('total: ', sorted_total, )
    
    return {
        'train': sorted_train,
        'val'  : sorted_val,
        'test' : sorted_test,
        'total': sorted_total,
    }


def bca(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    numb = m.shape[0]
    acc_each_label = 0
    for i in range(numb):
        acc = m[i, i] / np.sum(m[i, :], keepdims=False).astype(np.float32)
        acc_each_label += acc
    return acc_each_label / numb


def get_split_indices(data_size, split=[9, 1], shuffle=True):
    if len(split) < 2:
        raise TypeError(
            'The length of split should be larger than 2 while the length of your split is {}!'.format(len(split)))
    split = np.array(split)
    split = split / np.sum(split)
    if shuffle:
        indices = get_shuffle_index(data_size)
    else:
        indices = np.arange(data_size)
    split_indices_list = []
    start = 0
    for i in range(len(split) - 1):
        end = start + int(np.floor(split[i] * data_size))
        split_indices_list.append(indices[start:end])
        start = end
    split_indices_list.append(indices[start:])
    return split_indices_list


def batch_iter(data, batchsize, shuffle=True, random_seed=None):
    # Example: batches = list(utils.batch_iter([x_train, y_train], batchsize=batchsize, shuffle=True, random_seed=None))
    
    '''split dataset into batches'''
    if shuffle:
        x, y = shuffle_data(data, random_seed=random_seed)
    else:
        x, y = data
    data_size = x.shape[0]
    nb_batches = np.ceil(data_size / batchsize).astype(np.int)
    
    for batch_id in range(nb_batches):
        start_index = batch_id * batchsize
        end_index = min((batch_id + 1) * batchsize, data_size)
        yield x[start_index:end_index], y[start_index:end_index]


# def batch_iter( data, batchsize, shuffle=True, random_seed=None  ):
#     data = np.array(list(data))
#     data_size = data.shape[0]
#     num_batches = np.ceil(data_size/batchsize).astype(np.int)
#     # Shuffle the data
#     if shuffle:
#         shuffle_indices = get_shuffle_index(data_size)
#         shuffled_data = data[shuffle_indices]
#     else:
#         shuffled_data = data
#     for batch_num in range(num_batches):
#         start_index = batch_num*batchsize
#         end_index = min((batch_num+1)*batchsize, data_size)
#         yield shuffled_data[start_index:end_index]
#


def calculate_accuracy(y, y_pred, target_id=None):
    """
    Computes the accuracy as well as num_adv of attack of the target class.

    Args:
        y: ground truth labels. Accepts one hot encodings or labels.
        y_pred: predicted labels. Accepts probabilities or labels.
        target_id: target class

    Returns:
        accuracy
        accuracy_nb: number of samples which are classified correctly
        target_rate:
        target_total: number of samples which changed their labels from others to target_id
    """
    y = checked_argmax(y, to_numpy=True)  # tf.argmax(y, axis=-1).numpy()
    y_pred = checked_argmax(y_pred, to_numpy=True)  # tf.argmax(y_pred, axis=-1).numpy()
    accuracy = np.mean(np.equal(y, y_pred))
    accuracy_nb = np.sum(np.equal(y, y_pred))
    if target_id is not None:
        non_target_idx = (y != target_id)
        target_total = np.sum((y_pred[non_target_idx] == target_id))
        target_rate = target_total / np.sum(non_target_idx)
        
        # Cases where non_target_idx is 0, so target_rate becomes nan
        if np.isnan(target_rate):
            target_rate = 1.  # 100% target num_adv for this batch
        
        return accuracy, accuracy_nb, target_rate, target_total
    else:
        return accuracy, accuracy_nb


def calculate_class_weighted_accuracy(y, y_pred, class_weight=None):
    """
    Computes the accuracy as well as num_adv of attack of the target class.

    Args:
        y: ground truth labels. Accepts one hot encodings or labels.
        y_pred: predicted labels. Accepts probabilities or labels.
        class_weight: dictionary mapping class indices (integers) to a weight (float) value
    Returns:
        accuracy
        accuracy_nb: number of samples which are classified correctly
        target_rate:
        target_total: number of samples which changed their labels from others to target_id
    """
    y = checked_argmax(y, to_numpy=True)  # tf.argmax(y, axis=-1).numpy()
    y_pred = checked_argmax(y_pred, to_numpy=True)  # tf.argmax(y_pred, axis=-1).numpy()
    if class_weight is None:
        acc = np.mean(np.equal(y, y_pred))
        acc_num = np.sum(np.equal(y, y_pred))
        
        return {
            'acc'    : acc,
            'acc_num': acc_num
        }
    else:
        total = np.sum(list(class_weight.values()))
        for key in list(class_weight.keys()):
            class_weight[key] = class_weight[key] / total
        
        weighted_acc = 0
        for key in list(class_weight.keys()):
            key_idx = (y == key)
            key_total = np.sum((y_pred[key_idx] == key))
            weighted_acc += key_total / np.sum(key_idx) * class_weight[key]
        
        return weighted_acc


def checked_argmax(y, to_numpy=False):
    """
    Performs an argmax after checking if the input is either a tensor
    or a numpy matrix of rank 2 at least.

    Should be used in most cases for conformity throughout the
    codebase.

    Args:
        y: an numpy array or tensorflow tensor
        to_numpy: bool, flag to convert a tensor to a numpy array.

    Returns:
        an argmaxed array if possible, otherwise original array.
    """
    if y.ndim > 1:
        y = np.argmax(y, axis=-1)
    if to_numpy:
        return np.array(y)
    else:
        return y


def extract_nb_from_str(str):
    pattern = re.compile(r'\d+')
    res = re.findall(pattern, str)
    return list(map(int, res))


def get_files_by_suffix(root, suffix=''):
    if isinstance(suffix, str):
        suffix = (suffix,)
    else:
        suffix = tuple(suffix)
    file_list = []
    for parent, dirs, files in os.walk(root):
        for f in files:
            path = os.path.normpath(os.path.join(parent, f))
            if path.endswith(suffix):
                # img: (('.jpg', '.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                file_list.append(path)
    return file_list


def get_files_by_prefix(root, prefix=''):
    if isinstance(prefix, str):
        prefix = (prefix,)
    else:
        prefix = tuple(prefix)
    
    file_list = []
    for parent, dirs, files in os.walk(root):
        for f in files:
            if f.startswith(prefix):
                path = os.path.normpath(os.path.join(parent, f))
                file_list.append(path)
    return file_list


def get_dirs_by_suffix(root, suffix=''):
    if isinstance(suffix, str):
        suffix = (suffix,)
    else:
        suffix = tuple(suffix)
    
    dir_list = []
    for parent, dirs, files in os.walk(root):
        for d in dirs:
            path = os.path.normpath(os.path.join(parent, d))
            if path.endswith(suffix):
                dir_list.append(path)
    return dir_list


def get_dirs_by_prefix(root, prefix=''):
    if isinstance(prefix, str):
        prefix = (prefix,)
    else:
        prefix = tuple(prefix)
    
    dir_list = []
    for parent, dirs, files in os.walk(root):
        for d in dirs:
            if d.startswith(prefix):
                path = os.path.normpath(os.path.join(parent, d))
                dir_list.append(path)
    return dir_list


def get_subfiles_by_suffix(root, suffix=''):
    if isinstance(suffix, str):
        suffix = (suffix,)
    else:
        suffix = tuple(suffix)
    
    file_list = []
    for file_basename in os.listdir(root):
        fpath = os.path.join(root, file_basename)
        if os.path.isfile(fpath) and file_basename.endswith(suffix):
            # img: (('.jpg', '.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            path = os.path.normpath(fpath)
            file_list.append(path)
    return file_list


def get_subfiles_by_prefix(root, prefix=''):
    if isinstance(prefix, str):
        prefix = (prefix,)
    else:
        prefix = tuple(prefix)
    
    file_list = []
    for file_basename in os.listdir(root):
        fpath = os.path.join(root, file_basename)
        if os.path.isfile(fpath) and file_basename.startswith(prefix):
            path = os.path.normpath(fpath)
            file_list.append(path)
    return file_list


def get_subdirs_by_suffix(root, suffix=''):
    if isinstance(suffix, str):
        suffix = (suffix,)
    else:
        suffix = tuple(suffix)
    
    dir_list = []
    for dir_basename in os.listdir(root):
        dpath = os.path.join(root, dir_basename)
        if os.path.isdir(dpath) and dir_basename.endswith(suffix):
            path = os.path.normpath(dpath)
            dir_list.append(path)
    return dir_list


def get_subdirs_by_prefix(root, prefix=''):
    if isinstance(prefix, str):
        prefix = (prefix,)
    else:
        prefix = tuple(prefix)
    
    dir_list = []
    for dir_basename in os.listdir(root):
        dpath = os.path.join(root, dir_basename)
        if os.path.isdir(dpath) and dir_basename.startswith(prefix):
            path = os.path.normpath(dpath)
            dir_list.append(path)
    return dir_list


def plot_curve(data, title=None, img_path=None, show=True, y_lim=None):
    '''
    data: tuple of every curve's label, data and color
    for example:
        curve_name = ['Training acc_t', 'Validation acc_t', 'Test acc_t']
        curve_data = [train_acc, val_acc, test_acc]
        color = ['r', 'y', 'cyan']
        utils.plot_curve(data=list(zip(curve_name, curve_data, color)), title=title, img_path=s_img_path)

    '''
    plt.figure()
    for i in data:
        x_len = len(i[1])
        x = list(range(0, x_len))
        plt.plot(x, i[1], i[2], label=i[0])
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.title(title)
    plt.legend()
    if img_path is not None:
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.savefig(img_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_hist(data, title=None, img_path=None, bins=100, show=True):
    '''
    data: tuple of every curve's label, data and color
    for example:
        curve_name = ['Training acc_t', 'Validation acc_t', 'Test acc_t']
        curve_data = [train_acc, val_acc, test_acc]
        color = ['r', 'y', 'cyan']
        utils.plot_curve(data=list(zip(curve_name, curve_data, color)), title=title, img_path=s_img_path)

    '''
    plt.figure()
    for i in data:
        plt.hist(i[1], bins, color=i[2], label=i[0])
    
    # plt.ylim(0, 1.1)
    plt.title(title)
    plt.legend()
    if img_path is not None:
        plt.savefig(img_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_multi_bars(data, color=None, title=None, x_labels=None, y_label=None, y_lim=None, tick_step=1.,
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


def img_splice(img_paths, save_path, sgl_img_size):
    '''
    img_paths: 2-D list storing the paths of images
    sgl_img_size: size of single image
    '''
    
    width, height = sgl_img_size
    nb_column = max([len(i) for i in img_paths])
    nb_row = len(img_paths)
    res_img = Image.new(mode='RGB', size=(width * nb_column, height * nb_row), color=(255, 255, 255))
    for i in range(len(img_paths)):
        for j in range(len(img_paths[i])):
            # load imgs
            img = Image.open(img_paths[i][j])
            
            res_img.paste(img, (width * j, height * (i),
                                width * (j + 1), height * (i + 1)))
    res_img.save(save_path)
    return res_img


def otsu_threshold(data):
    data = np.array([data], dtype=np.uint8)
    threshold, res_data, = cv2.threshold(data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return res_data[0], threshold
